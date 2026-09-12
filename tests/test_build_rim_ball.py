"""The ray description the ball selector rests on, and the fixture test."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_rim_ball as ball  # noqa: E402
from courtvision.court_camera import distort_points, look_at, ptz_matrix  # noqa: E402

CENTRE = np.array([114.0, 42.9, 26.4])
SIZE = (1280, 720)


def _pose_and_pixel(target, point_3d, focal=1400.0):
    """PTZ params looking at `target`, and where `point_3d` lands in the picture."""
    import cv2
    params = look_at(target, focal, CENTRE)
    rotation = cv2.Rodrigues(np.asarray(params[:3]))[0]
    focal = float(np.exp(params[3]))
    intrinsics = np.array([[focal, 0, SIZE[0] / 2], [0, focal, SIZE[1] / 2], [0, 0, 1.0]])
    projected = intrinsics @ rotation @ (np.asarray(point_3d, float) - CENTRE)
    return params, projected[:2] / projected[2], projected[2]


def test_a_pixel_maps_to_the_world_direction_it_came_from():
    """The whole ball selector rests on this: pixel -> where the camera looked."""
    for point in ([25.0, 5.25, 10.0], [25.0, 20.0, 0.0], [40.0, 14.0, 6.0]):
        params, pixel, depth = _pose_and_pixel((25.0, 14.0), point)
        assert depth > 0 and 0 <= pixel[0] <= SIZE[0] and 0 <= pixel[1] <= SIZE[1]
        direction = ball.ray_directions(params, CENTRE, SIZE, [pixel])[0]
        truth = np.asarray(point, float) - CENTRE
        truth /= np.linalg.norm(truth)
        assert ball.angle_between(direction, truth) < 1e-6


def test_the_lens_is_undone_before_the_ray_is_taken():
    """Taken at a corner, where the lens actually bends -- the centre barely moves."""
    params = look_at((25.0, 14.0), 1400.0, CENTRE)
    pixel = np.array([1210.0, 660.0])
    k1, k2 = 0.006, 0.0006
    recorded = distort_points(pixel.reshape(1, 2), k1, SIZE, k2)[0]
    assert np.linalg.norm(recorded - pixel) > 0.5          # the lens really moved it
    straight = ball.ray_directions(params, CENTRE, SIZE, [pixel])[0]
    through = ball.ray_directions(params, CENTRE, SIZE, [recorded], k1, k2)[0]
    assert ball.angle_between(straight, through) < 1e-6


def test_a_still_object_holds_its_ray_while_the_camera_pans():
    """The point of using rays: panning must not make furniture look like a ball."""
    point = [-6.0, 10.0, 3.0]
    directions = []
    for target in ((15.0, 14.0), (25.0, 30.0), (35.0, 60.0)):
        params, pixel, depth = _pose_and_pixel(target, point, focal=900.0)
        if depth <= 0:
            continue
        directions.append(ball.ray_directions(params, CENTRE, SIZE, [pixel])[0])
    assert len(directions) >= 2
    for other in directions[1:]:
        assert ball.angle_between(directions[0], other) < 1e-6


def test_a_direction_seen_all_game_is_a_fixture_and_a_passing_ball_is_not():
    still = (100, -20)
    frames = [[still, (i, i)] for i in range(200)]
    fixtures = ball.find_fixtures(frames, posed_frames=200)
    assert still in fixtures
    assert (5, 5) not in fixtures


def test_a_short_run_cannot_manufacture_a_fixture():
    """Below FIXTURE_MIN_FRAMES a handful of detections is not evidence of furniture."""
    frames = [[(3, 3)] for _ in range(ball.FIXTURE_MIN_FRAMES - 1)]
    assert ball.find_fixtures(frames, posed_frames=len(frames)) == set()


def _row(t, candidates, dirs=None):
    """A frame as choose_balls wants it: candidates, their cells and rays."""
    dirs = dirs if dirs is not None else [None] * len(candidates)
    usable = all(d is not None for d in dirs) and bool(candidates)
    return {"t": t, "candidates": candidates,
            "cells": [ball.cell_of(d) for d in dirs] if usable else [],
            "dirs": np.array(dirs) if usable else None}


def _dir(deg):
    rad = np.radians(deg)
    return [float(np.cos(rad)), float(np.sin(rad)), 0.0]


def test_the_most_confident_survivor_wins_when_nothing_precedes_it():
    rows = [_row(0.0, [{"centre": [1, 1], "conf": 0.3},
                       {"centre": [2, 2], "conf": 0.8}])]
    decided, _, _ = ball.choose_balls(rows, set(), 15.0)
    assert decided[0][0][1]["centre"] == [2, 2]


def test_continuity_prefers_the_nearer_ray_over_the_louder_candidate():
    rows = [_row(0.0, [{"centre": [1, 1], "conf": 0.9}], [_dir(0)]),
            _row(0.2, [{"centre": [9, 9], "conf": 0.9}, {"centre": [2, 2], "conf": 0.2}],
                 [_dir(60), _dir(3)])]
    decided, _, _ = ball.choose_balls(rows, set(), 15.0)
    assert decided[1][0][1]["centre"] == [2, 2]


def test_a_stale_choice_does_not_pull_across_a_gap():
    """After a cut the previous ball says nothing, so confidence decides again."""
    rows = [_row(0.0, [{"centre": [1, 1], "conf": 0.9}], [_dir(0)]),
            _row(9.0, [{"centre": [9, 9], "conf": 0.9}, {"centre": [2, 2], "conf": 0.2}],
                 [_dir(60), _dir(3)])]
    decided, _, _ = ball.choose_balls(rows, set(), 15.0)
    assert decided[1][0][1]["centre"] == [9, 9]


def test_continuity_is_a_preference_not_a_cage():
    """Nothing near the last ray still yields the best candidate, not nothing."""
    rows = [_row(0.0, [{"centre": [1, 1], "conf": 0.9}], [_dir(0)]),
            _row(0.2, [{"centre": [9, 9], "conf": 0.5}], [_dir(80)])]
    decided, _, _ = ball.choose_balls(rows, set(), 15.0)
    assert decided[1][0] is not None and decided[1][0][1]["centre"] == [9, 9]


def test_a_fixture_candidate_is_dropped_even_when_it_is_the_most_confident():
    fixture = ball.cell_of(_dir(0))
    rows = [_row(0.0, [{"centre": [1, 1], "conf": 0.9}, {"centre": [5, 5], "conf": 0.2}],
                 [_dir(0), _dir(40)])]
    decided, dropped, _ = ball.choose_balls(rows, {fixture}, 15.0)
    assert dropped == 1 and decided[0][0][1]["centre"] == [5, 5]


def test_a_frame_whose_only_candidate_is_furniture_reports_no_ball():
    fixture = ball.cell_of(_dir(0))
    rows = [_row(0.0, [{"centre": [1, 1], "conf": 0.9}], [_dir(0)])]
    decided, _, _ = ball.choose_balls(rows, {fixture}, 15.0)
    assert decided[0][0] is None


def test_a_busy_direction_at_the_basket_is_never_called_furniture():
    """The rim sits as still as the rack ball; the game ball visits it all game."""
    to_rim = ball.rim_directions(CENTRE)
    busy = ball.cell_of(to_rim[0])
    frames = [[busy] for _ in range(200)]
    assert busy in ball.find_fixtures(frames, 200)
    assert busy not in ball.find_fixtures(frames, 200, protect=to_rim)


def test_furniture_away_from_the_baskets_is_still_caught_with_the_guard_on():
    to_rim = ball.rim_directions(CENTRE)
    away = ball.cell_of(_dir(200))
    frames = [[away] for _ in range(200)]
    assert away in ball.find_fixtures(frames, 200, protect=to_rim)


def test_a_candidate_standing_still_is_not_the_ball():
    """The only answer to the ray ambiguity: a spectator does not move."""
    rows = [_row(0.0, [{"centre": [1, 1], "conf": 0.9, "still": True},
                       {"centre": [5, 5], "conf": 0.2, "still": False}])]
    decided, _, standing = ball.choose_balls(rows, set(), 15.0)
    assert standing == 1 and decided[0][0][1]["centre"] == [5, 5]


def test_a_frame_of_nothing_but_still_candidates_still_reports_its_best():
    """A held ball IS still; treating stillness as disqualifying lost them."""
    rows = [_row(0.0, [{"centre": [1, 1], "conf": 0.9, "still": True},
                       {"centre": [2, 2], "conf": 0.4, "still": True}])]
    decided, _, standing = ball.choose_balls(rows, set(), 15.0)
    assert decided[0][0] is not None and decided[0][0][1]["centre"] == [1, 1]
    assert standing == 0
