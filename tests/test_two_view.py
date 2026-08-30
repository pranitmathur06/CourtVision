import numpy as np

from courtvision.two_view import TwoViewClassifier


class Fake:
    def __init__(self, results):
        self.results = results

    def classify_batch(self, clips):
        return self.results[: len(clips)]


def _clips(n):
    return [np.zeros((16, 8, 8, 3), np.uint8) for _ in range(n)]


def test_rim_model_decides_rebound_not_the_crop_model():
    """The crop model never saw a rebound; its vote on one is worthless."""
    crop = Fake([("dribble", 0.9)])
    rim = Fake([("rebound", 0.88)])
    out = TwoViewClassifier(crop, rim).classify_batch(_clips(1), _clips(1))
    assert out == [("rebound", 0.88)]


def test_crop_rebound_is_overruled_when_the_rim_view_disagrees():
    """This is the 82%-of-a-game case: generic crop, model shouts rebound."""
    crop = Fake([("rebound", 0.99)])
    rim = Fake([("background", 0.91)])
    label, _ = TwoViewClassifier(crop, rim).classify_batch(_clips(1), _clips(1))[0]
    assert label == "background", "an overruled rebound must not become an event"


def test_a_low_confidence_rim_rebound_is_not_taken():
    crop = Fake([("shot", 0.8)])
    rim = Fake([("rebound", 0.4)])
    out = TwoViewClassifier(crop, rim, rim_threshold=0.6).classify_batch(
        _clips(1), _clips(1))
    assert out == [("shot", 0.8)], "below threshold the crop model's answer stands"


def test_non_rebound_actions_pass_through_untouched():
    crop = Fake([("dribble", 0.7), ("block", 0.6)])
    rim = Fake([("background", 0.9), ("background", 0.9)])
    out = TwoViewClassifier(crop, rim).classify_batch(_clips(2), _clips(2))
    assert out == [("dribble", 0.7), ("block", 0.6)]


def test_without_frames_it_degrades_to_the_crop_model():
    crop = Fake([("shot", 0.5)])
    out = TwoViewClassifier(crop, Fake([("rebound", 0.9)])).classify_batch(_clips(1))
    assert out == [("shot", 0.5)]
