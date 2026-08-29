"""v2 §7.2 disaggregated serving."""

import threading
import time

import numpy as np
import pytest

from courtvision.config import Config
from courtvision.serving import StagePlan, run_disaggregated
from courtvision.types import ActionWindow, Frame


def make_inputs(n: int):
    config = Config()
    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(n)]
    frames = [Frame(i, i / config.target_fps, ()) for i in range(n)]
    return config, images, frames


def fake_classify(images, frames, config, holders):
    return [
        ActionWindow(
            start_index=frames[0].index,
            end_index=frames[-1].index,
            start_time_s=frames[0].time_s,
            end_time_s=frames[-1].time_s,
            label="dribble",
            conf=0.9,
        )
    ]


def test_stage_plan_reports_disaggregation():
    assert StagePlan("cuda:0", "cuda:1").is_disaggregated
    assert not StagePlan.single("mps").is_disaggregated
    assert StagePlan.single("cpu").detection == "cpu"


def test_runs_every_window_exactly_once():
    config, images, frames = make_inputs(48)
    windows, _ = run_disaggregated(images, frames, [1] * 48, fake_classify, config)
    starts = sorted(w.start_index for w in windows)
    assert starts == [0, 8, 16, 24, 32]  # size 16, stride 8 over 48 frames


def test_returns_nothing_for_a_clip_shorter_than_one_window():
    config, images, frames = make_inputs(10)
    windows, timing = run_disaggregated(images, frames, [None] * 10, fake_classify, config)
    assert windows == []
    assert timing.busy == {}


def test_records_busy_and_waiting_per_stage():
    config, images, frames = make_inputs(48)
    _, timing = run_disaggregated(images, frames, [1] * 48, fake_classify, config)
    assert set(timing.busy) == {"windowing", "classification"}
    assert all(v >= 0.0 for v in timing.waiting.values())


def test_producer_and_consumer_actually_overlap():
    """The stages must run concurrently, not one after the other."""
    config, images, frames = make_inputs(80)
    active = {"n": 0, "max": 0}
    lock = threading.Lock()

    def slow_classify(images_, frames_, config_, holders_):
        with lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        time.sleep(0.01)
        with lock:
            active["n"] -= 1
        return fake_classify(images_, frames_, config_, holders_)

    windows, timing = run_disaggregated(
        images, frames, [1] * 80, slow_classify, config, queue_size=2
    )
    assert len(windows) == 9
    # A bounded queue means the producer blocks once the consumer falls behind,
    # which is the backpressure working rather than buffering the whole clip.
    assert timing.waiting["windowing"] > 0.0


def test_propagates_classifier_errors():
    config, images, frames = make_inputs(48)

    def boom(*_):
        raise RuntimeError("classifier exploded")

    with pytest.raises(RuntimeError, match="classifier exploded"):
        run_disaggregated(images, frames, [1] * 48, boom, config)


def test_windows_are_in_time_order():
    config, images, frames = make_inputs(64)
    windows, _ = run_disaggregated(images, frames, [1] * 64, fake_classify, config)
    times = [w.start_time_s for w in windows]
    assert times == sorted(times)
