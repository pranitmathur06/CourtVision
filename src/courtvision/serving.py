"""v2 §7.2 — disaggregated serving: run pipeline stages on separate devices.

v1 runs every stage sequentially on one device, so while the detector works the
action classifier sits idle and vice versa. Measured on a held-out clip, the two
model stages are 27.2% and 35.8% of runtime — nearly two thirds spent with one
model idle while the other runs.

The stages have different shapes, which is what makes splitting them worthwhile:

* **Detection** is per-frame and streaming. Frame N is independent of frame N+1.
* **Action classification** is per-window and needs 16 consecutive frames, so it
  cannot start until the detector has produced them.

That is a classic producer/consumer pipeline: while the detector works on frames
17-32, the classifier can already be running the window over frames 1-16. On one
device they contend; on two they genuinely overlap.

Device placement is data, not code. `StagePlan` says which device each stage
runs on, so the same pipeline is single-device locally and multi-GPU on a rented
box with no code change:

    StagePlan(detection="mps",    action="mps")       # this Mac
    StagePlan(detection="cuda:0", action="cuda:1")    # 2-GPU box

Threads rather than processes: torch releases the GIL inside device kernels, so
two threads driving two GPUs do overlap. Processes would add IPC cost for the
frame tensors, which is the opposite of what this optimisation is for.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from courtvision.config import Config
from courtvision.types import ActionWindow, Frame


@dataclass(frozen=True)
class StagePlan:
    """Which device each model stage runs on.

    `resolve()` collapses the plan to a single device, for machines with one
    accelerator; the pipeline is otherwise identical.
    """

    detection: str = "cuda:0"
    action: str = "cuda:1"

    @staticmethod
    def single(device: str) -> "StagePlan":
        return StagePlan(detection=device, action=device)

    @property
    def is_disaggregated(self) -> bool:
        return self.detection != self.action


@dataclass
class StageTiming:
    """Wall-clock spent inside each stage, plus how long each spent blocked.

    Queue-wait time is the number that matters: a stage starved at its input is a
    stage that would benefit from more upstream throughput, and a stage blocked
    on its output is one that is ahead of its consumer.
    """

    busy: dict[str, float] = field(default_factory=dict)
    waiting: dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, busy: float, waiting: float) -> None:
        self.busy[stage] = self.busy.get(stage, 0.0) + busy
        self.waiting[stage] = self.waiting.get(stage, 0.0) + waiting


_SENTINEL = object()


def run_disaggregated(
    images: Sequence[np.ndarray],
    frames: Sequence[Frame],
    holders: Sequence[int | None],
    classify_fn,
    config: Config,
    queue_size: int = 4,
) -> tuple[list[ActionWindow], StageTiming]:
    """Overlap window construction with classification via a bounded queue.

    `classify_fn(images, frames, config, holders)` performs the model work for one
    batch of windows. The bounded queue applies backpressure: if the classifier
    falls behind, window construction blocks rather than buffering the whole clip
    into memory.
    """
    from courtvision.action_classifier import plan_windows

    plans = plan_windows(
        len(images), config.action_window_frames, config.action_stride_frames
    )
    timing = StageTiming()
    if not plans:
        return [], timing

    work: queue.Queue = queue.Queue(maxsize=queue_size)
    results: list[ActionWindow] = []
    error: list[BaseException] = []
    # Set when either side gives up. Without it, a consumer that dies mid-clip
    # leaves the producer blocked forever on a full queue and join() never
    # returns — the deadlock this pipeline exists to avoid, not to create.
    stop = threading.Event()
    POLL = 0.05

    def put(item) -> float:
        """Blocking put that still notices the consumer has gone away."""
        waited = 0.0
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                work.put(item, timeout=POLL)
                return waited + (time.perf_counter() - t0)
            except queue.Full:
                waited += time.perf_counter() - t0
        return waited

    def producer() -> None:
        busy = waiting = 0.0
        try:
            for start, end in plans:
                if stop.is_set():
                    break
                t0 = time.perf_counter()
                item = (start, end)
                busy += time.perf_counter() - t0
                waiting += put(item)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller
            error.append(exc)
            stop.set()
        finally:
            put(_SENTINEL)
            timing.record("windowing", busy, waiting)

    def consumer() -> None:
        busy = waiting = 0.0
        try:
            while True:
                t0 = time.perf_counter()
                try:
                    item = work.get(timeout=POLL)
                except queue.Empty:
                    waiting += time.perf_counter() - t0
                    if stop.is_set():
                        break
                    continue
                waiting += time.perf_counter() - t0
                if item is _SENTINEL:
                    break
                start, end = item
                t1 = time.perf_counter()
                results.extend(
                    classify_fn(
                        images[start : end + 1],
                        frames[start : end + 1],
                        config,
                        holders[start : end + 1],
                    )
                )
                busy += time.perf_counter() - t1
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)
        finally:
            # Unblock the producer whatever happened, then drain so its final
            # put() cannot wedge on a full queue.
            stop.set()
            try:
                while True:
                    work.get_nowait()
            except queue.Empty:
                pass
            timing.record("classification", busy, waiting)

    threads = [
        threading.Thread(target=producer, name="windowing", daemon=True),
        threading.Thread(target=consumer, name="classification", daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if error:
        raise error[0]
    return results, timing
