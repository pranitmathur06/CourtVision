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
    frames = [[(3, 3)] for _ in range(10)]
    assert ball.find_fixtures(frames, posed_frames=10) == set()
