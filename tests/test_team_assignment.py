import numpy as np

from courtvision.team_assignment import (
    assign_teams,
    collect_samples,
    mean_lab_color,
    torso_crop,
)
from courtvision.types import PLAYER, Box, Frame, Track
from tests.fixtures.synthetic import StubDetector


def test_torso_crop_is_inside_the_box():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = torso_crop(image, Box(10, 10, 50, 90))
    assert crop.size > 0
    # Narrower and shorter than the full box: it targets the jersey, not limbs.
    assert crop.shape[1] < 40
    assert crop.shape[0] < 80


def test_torso_crop_of_degenerate_box_is_empty():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert torso_crop(image, Box(10, 10, 10, 10)).size == 0


def test_mean_lab_color_distinguishes_red_from_blue():
    red = np.full((10, 10, 3), (40, 40, 200), dtype=np.uint8)   # BGR
    blue = np.full((10, 10, 3), (200, 60, 40), dtype=np.uint8)
    assert not np.allclose(mean_lab_color(red), mean_lab_color(blue), atol=5.0)


def test_assign_teams_splits_two_color_groups():
    reds = [(i, np.array([50.0, 60.0, 40.0])) for i in (0, 1)]
    blues = [(i, np.array([50.0, -20.0, -40.0])) for i in (2, 3)]
    teams = assign_teams(reds + blues)
    assert set(teams) == {0, 1, 2, 3}
    # Which group is called "A" is arbitrary; the partition is what matters.
    assert teams[0] == teams[1]
    assert teams[2] == teams[3]
    assert teams[0] != teams[2]


def test_assign_teams_is_deterministic():
    samples = [
        (0, np.array([50.0, 60.0, 40.0])),
        (1, np.array([50.0, -20.0, -40.0])),
    ]
    assert assign_teams(samples) == assign_teams(list(reversed(samples)))


def test_assign_teams_uses_majority_vote_per_track():
    # Track 0 has three red samples and one stray blue one: it must come out red.
    samples = [
        (0, np.array([50.0, 60.0, 40.0])),
        (0, np.array([50.0, 62.0, 41.0])),
        (0, np.array([50.0, 58.0, 39.0])),
        (0, np.array([50.0, -20.0, -40.0])),
        (1, np.array([50.0, -21.0, -41.0])),
        (1, np.array([50.0, -19.0, -39.0])),
    ]
    teams = assign_teams(samples)
    assert teams[0] != teams[1]


def test_assign_teams_on_synthetic_clip_recovers_the_true_partition(synthetic):
    """Team A players must share a label, team B players must share the other."""
    from courtvision.extraction import extract_frames

    stub = StubDetector(synthetic)
    images, frames = [], []
    for index, time_s, image in extract_frames(synthetic.path, synthetic.fps):
        detections = stub.detect_at(index)
        # The stub's box order matches the truth's player index order.
        tracks = tuple(
            Track(player_index, det.box, det.label, det.conf)
            for player_index, det in enumerate(detections)
            if det.label == PLAYER
        )
        images.append(image)
        frames.append(Frame(index, time_s, tracks))

    teams = assign_teams(collect_samples(images, frames))
    truth = synthetic.teams
    group_a = {t for t, label in truth.items() if label == "A"}
    group_b = {t for t, label in truth.items() if label == "B"}
    assert len({teams[t] for t in group_a}) == 1
    assert len({teams[t] for t in group_b}) == 1
    assert teams[next(iter(group_a))] != teams[next(iter(group_b))]
