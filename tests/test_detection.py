import numpy as np
import pytest

from courtvision.detection import COCO_CLASS_MAP, boxes_from_result
from courtvision.types import BALL, PLAYER


class FakeBoxes:
    """Mimics ultralytics `result.boxes`: parallel tensors as numpy arrays."""

    def __init__(self, xyxy, cls, conf):
        self.xyxy = np.array(xyxy, dtype=np.float32)
        self.cls = np.array(cls, dtype=np.float32)
        self.conf = np.array(conf, dtype=np.float32)

    def __len__(self):
        return len(self.cls)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_converts_person_to_player():
    result = FakeResult(FakeBoxes([[1, 2, 3, 4]], [0], [0.9]))
    detections = boxes_from_result(result, COCO_CLASS_MAP, conf=0.25)
    assert len(detections) == 1
    assert detections[0].label == PLAYER
    assert detections[0].conf == pytest.approx(0.9)
    assert detections[0].box.x1 == pytest.approx(1.0)
    assert detections[0].box.y2 == pytest.approx(4.0)


def test_converts_sports_ball_to_ball():
    result = FakeResult(FakeBoxes([[0, 0, 5, 5]], [32], [0.5]))
    detections = boxes_from_result(result, COCO_CLASS_MAP, conf=0.25)
    assert detections[0].label == BALL


def test_drops_classes_not_in_map():
    # COCO 15 is "cat" — irrelevant to basketball, must be discarded.
    result = FakeResult(FakeBoxes([[0, 0, 5, 5]], [15], [0.99]))
    assert boxes_from_result(result, COCO_CLASS_MAP, conf=0.25) == []


def test_drops_detections_below_confidence():
    result = FakeResult(FakeBoxes([[0, 0, 5, 5]], [0], [0.10]))
    assert boxes_from_result(result, COCO_CLASS_MAP, conf=0.25) == []


def test_handles_empty_result():
    result = FakeResult(FakeBoxes(np.empty((0, 4)), [], []))
    assert boxes_from_result(result, COCO_CLASS_MAP, conf=0.25) == []
