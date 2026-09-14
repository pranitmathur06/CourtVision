"""Sequential decoding, the producer thread, and where fp16 is asked for."""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.fast_detect import (
    Decoded, decode_ahead, frame_positions, half_precision_ok, sequential_frames,
    skip_count,
)


class FakeCapture:
    """A video that counts how often it is SEEKED, which is the whole point."""

    def __init__(self, n=100, fps=30.0):
        self.n, self.fps, self.at = n, fps, 0
        self.seeks, self.grabs, self.reads = 0, 0, 0

    def get(self, prop):
        return self.fps

    def set(self, prop, value):
        self.seeks += 1
        self.at = int(round(value / 1000.0 * self.fps))
        return True

    def read(self):
        self.reads += 1
        if self.at >= self.n:
            return False, None
        frame = np.full((4, 4, 3), self.at % 255, np.uint8)
        self.at += 1
        return True, frame

    def grab(self):
        self.grabs += 1
        if self.at >= self.n:
            return False
        self.at += 1
        return True

    def release(self):
        pass


def test_the_wanted_instants_are_evenly_spaced():
    assert frame_positions(0, 1.0, 0.25) == [0.0, 0.25, 0.5, 0.75]


def test_an_empty_span_wants_nothing():
    assert frame_positions(5.0, 5.0, 0.2) == []


def test_a_zero_step_is_refused_rather_than_looping_forever():
    with pytest.raises(ValueError):
        frame_positions(0, 1, 0)


def test_the_skip_is_the_frame_interval():
    assert skip_count(30.0, 0.2) == 6
    assert skip_count(59.94, 0.5) == 30


def test_a_step_below_one_frame_still_advances():
    assert skip_count(30.0, 0.001) == 1


def test_reading_forward_seeks_at_most_once():
    # The measured fault: seeking per frame cost 97.9 ms against 6.5 ms.
    cap = FakeCapture(n=200, fps=30.0)
    got = list(sequential_frames("x", 0.0, 2.0, 0.2, capture=cap))
    assert len(got) == 10
    assert cap.seeks == 0        # start_s is 0, so not even once
    assert cap.grabs > 0         # it skipped rather than decoded


def test_it_seeks_once_when_the_pass_starts_late():
    cap = FakeCapture(n=400, fps=30.0)
    list(sequential_frames("x", 1.0, 2.0, 0.2, capture=cap))
    assert cap.seeks == 1


def test_it_stops_cleanly_at_the_end_of_the_video():
    cap = FakeCapture(n=12, fps=30.0)
    got = list(sequential_frames("x", 0.0, 100.0, 0.2, capture=cap))
    assert 0 < len(got) < 10


def test_every_frame_carries_the_instant_it_came_from():
    cap = FakeCapture(n=200, fps=30.0)
    got = list(sequential_frames("x", 0.0, 1.0, 0.25, capture=cap))
    assert [d.time_s for d in got] == [0.0, 0.25, 0.5, 0.75]
    assert all(isinstance(d, Decoded) for d in got)


def test_the_producer_thread_yields_the_same_thing_in_the_same_order(monkeypatch):
    import courtvision.fast_detect as fd
    frames = [Decoded(t, np.zeros((2, 2, 3), np.uint8)) for t in (0.0, 0.2, 0.4)]
    monkeypatch.setattr(fd, "sequential_frames",
                        lambda *a, **k: iter(frames))
    assert [d.time_s for d in fd.decode_ahead("x", 0, 1, 0.2)] == [0.0, 0.2, 0.4]


def test_a_decoder_failure_reaches_the_consumer(monkeypatch):
    import courtvision.fast_detect as fd

    def boom(*a, **k):
        raise RuntimeError("codec gave up")
        yield
    monkeypatch.setattr(fd, "sequential_frames", boom)
    with pytest.raises(RuntimeError, match="codec gave up"):
        list(fd.decode_ahead("x", 0, 1, 0.2))


def test_half_precision_is_asked_for_on_cuda_only():
    assert half_precision_ok("cuda")
    assert half_precision_ok("cuda:0")
    assert not half_precision_ok("mps")
    assert not half_precision_ok("cpu")
