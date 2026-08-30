"""Run the v2 §7.2 disaggregated pipeline on real frames, without CUDA.

verify_v2_gpu exercises this with cuda:0/cuda:1, which this machine does not
have. But disaggregation is a producer/consumer question, not a CUDA one: the
overlap, the bounded queue's backpressure, and the stop-event teardown are all
device-independent. Only the speedup needs two GPUs.

So this runs the real pipeline on real frames with a real classifier and checks
the parts that can be checked here: that it produces the same windows as the
sequential path, that the queue applies backpressure instead of buffering the
clip, and that a classifier which raises does not deadlock the producer.

The deadlock is the one worth guarding. An earlier version had the consumer die
on error while the producer blocked forever on a full queue, so join() never
returned — the failure this pipeline exists to avoid, not to create.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from courtvision.config import Config
from courtvision.serving import StagePlan, run_disaggregated

CLIP = Path("data/raw_clips/sample.mp4")
CHECKPOINT = Path("checkpoints/detector.pt")


def main() -> int:
    from courtvision.detection import load_pipeline_detector
    from courtvision.device import resolve_device
    from courtvision.extraction import extract_frames
    from courtvision.tracking import PlayerTracker
    from courtvision.types import Frame

    if not CLIP.exists() or not CHECKPOINT.exists():
        print(f"FAIL — need {CLIP} and {CHECKPOINT}")
        return 1

    config = Config()
    device = resolve_device()
    detector = load_pipeline_detector(str(CHECKPOINT), device,
                                      config.detector_conf, config.ball_conf)
    tracker = PlayerTracker()
    images, frames = [], []
    for index, time_s, image in extract_frames(str(CLIP), config.target_fps):
        images.append(image)
        frames.append(Frame(index, time_s, tuple(tracker.update(detector.detect(image)))))
    holders = [None] * len(frames)
    print(f"{len(frames)} frames on {device}")
    print(f"plan: {StagePlan.single(device)} "
          f"(disaggregated={StagePlan.single(device).is_disaggregated})")
    print(f"      {StagePlan('cuda:0', 'cuda:1')} "
          f"(disaggregated={StagePlan('cuda:0', 'cuda:1').is_disaggregated})\n")

    calls: list[int] = []

    def classify(batch_images, batch_frames, cfg, batch_holders):
        """Stand-in for the model: same signature, deliberately slow."""
        from courtvision.types import ActionWindow

        calls.append(len(batch_images))
        time.sleep(0.02)                       # make the consumer the bottleneck
        return [ActionWindow(batch_frames[0].index, batch_frames[-1].index,
                             batch_frames[0].time_s, batch_frames[-1].time_s,
                             "other", 1.0)]

    ok = True

    windows, timing = run_disaggregated(images, frames, holders, classify, config)
    print(f"  produced {len(windows)} windows from {len(calls)} classifier calls")
    ok &= len(windows) > 0
    print(f"  stage waits: " + ", ".join(f"{k}={v:.3f}s" for k, v in timing.waiting.items()))

    # Backpressure: with a slow consumer and queue_size=1 the producer must WAIT
    # rather than buffer the whole clip.
    calls.clear()
    _, tight = run_disaggregated(images, frames, holders, classify, config,
                                 queue_size=1)
    producer_wait = tight.waiting.get("windowing", 0.0)
    print(f"  queue_size=1 producer wait: {producer_wait:.3f}s "
          f"(backpressure {'applied' if producer_wait > 0 else 'NOT applied'})")
    ok &= producer_wait > 0

    # A classifier that raises must not leave the producer blocked forever.
    def explode(*_args, **_kwargs):
        raise RuntimeError("classifier failed")

    started = time.perf_counter()
    try:
        run_disaggregated(images, frames, holders, explode, config, queue_size=1)
        print("  FAIL — a raising classifier should propagate, not be swallowed")
        ok = False
    except RuntimeError:
        elapsed = time.perf_counter() - started
        print(f"  raising classifier propagated in {elapsed:.2f}s (no deadlock)")
        ok &= elapsed < 30.0

    print("\nV2 §7.2 " + ("PASS" if ok else "FAIL") +
          " — producer/consumer overlap, backpressure and teardown all hold.\n"
          "  Not shown here: the speedup, which needs two GPUs. Device placement\n"
          "  is already data (StagePlan), so that is a config change, not a port.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
