"""Per-stage wall-clock profile of the v1 pipeline (spec §7.1, step 1).

Spec §7.1 says: profile before writing any CUDA, because the bottleneck is often
somewhere unexpected. This measures each stage separately on whatever clip is
available, with a warmup pass first so Metal shader compilation is not counted.

Honest limits of this measurement, stated up front:
  * Run on MPS, not CUDA. Relative stage costs transfer; absolute numbers do not.
  * Uses stock/base weights. Fine-tuning changes accuracy, not layer shapes, so
    per-call timing is representative.
  * Uses the synthetic 640x360 clip unless a real clip exists. Only decode and
    crop costs scale with source resolution — YOLO and VideoMAE letterbox to a
    fixed size regardless.
  * Commentary (stage 8) is a network round-trip, not compute. Excluded here and
    measured separately; it is latency-bound and no kernel will help it.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from courtvision.action_classifier import FRAME_SIZE, plan_windows
from courtvision.config import Config
from courtvision.detection import COCO_CLASS_MAP, YoloDetector
from courtvision.device import resolve_device
from courtvision.events import build_events
from courtvision.extraction import extract_frames
from courtvision.possession import possession_timeline
from courtvision.render import render_video
from courtvision.team_assignment import assign_teams, collect_samples
from courtvision.tracking import PlayerTracker
from courtvision.types import ActionWindow, Frame

REAL_CLIP = Path("data/raw_clips/sample.mp4")
TIMINGS: dict[str, float] = {}


@contextmanager
def stage(name: str):
    start = time.perf_counter()
    yield
    TIMINGS[name] = TIMINGS.get(name, 0.0) + (time.perf_counter() - start)


def resolve_clip():
    """Return (path, is_real, truth). truth is None for a real clip."""
    if REAL_CLIP.exists():
        return str(REAL_CLIP), True, None
    from tests.fixtures.synthetic import make_clip

    path = str(Path(tempfile.mkdtemp()) / "synthetic.mp4")
    return path, False, make_clip(path, n_frames=50, fps=10)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile the v1 pipeline per stage")
    parser.add_argument("--skip-action", action="store_true",
                        help="skip VideoMAE (avoids a ~400 MB model download)")
    args = parser.parse_args()

    config = Config()
    device = resolve_device()
    clip_path, is_real, truth = resolve_clip()

    print(f"device={device}  clip={'REAL' if is_real else 'SYNTHETIC'} ({clip_path})")
    print("warming up (Metal shader compilation is not part of the measurement)...")

    detector = YoloDetector("yolo11n.pt", device, config.detector_conf, COCO_CLASS_MAP)
    warm = np.zeros((360, 640, 3), dtype=np.uint8)
    for _ in range(2):
        detector.detect(warm)

    # --- stage 1: extraction -------------------------------------------------
    images: list[np.ndarray] = []
    times: list[float] = []
    with stage("1 extraction"):
        for _, time_s, image in extract_frames(clip_path, config.target_fps):
            images.append(image)
            times.append(time_s)
    n = len(images)
    if n == 0:
        print("no frames decoded")
        return 1

    # --- stage 2: detection --------------------------------------------------
    per_frame_detections = []
    with stage("2 detection"):
        for image in images:
            per_frame_detections.append(detector.detect(image))

    # On the synthetic clip stock COCO YOLO detects nothing (rectangles are not
    # people), which would leave stages 3-5 timing empty input and reporting a
    # meaningless 0.000s. Feed them ground-truth boxes instead so they carry a
    # realistic load. Stage 2's timing above is still the real YOLO measurement.
    downstream = per_frame_detections
    if truth is not None:
        from tests.fixtures.synthetic import StubDetector

        stub = StubDetector(truth)
        downstream = [stub.detect_at(i) for i in range(n)]
        print(f"note: feeding ground-truth detections to stages 3-5 "
              f"({sum(len(d) for d in downstream)} boxes); "
              f"stock YOLO found {sum(len(d) for d in per_frame_detections)} "
              f"on synthetic input")

    # --- stage 3: tracking ---------------------------------------------------
    tracker = PlayerTracker()
    frames: list[Frame] = []
    with stage("3 tracking"):
        for index, detections in enumerate(downstream):
            frames.append(Frame(index, times[index], tuple(tracker.update(detections))))

    # --- stage 4: team assignment -------------------------------------------
    with stage("4 team assignment"):
        teams = assign_teams(collect_samples(images, frames))

    # --- stage 5: possession -------------------------------------------------
    with stage("5 possession"):
        holders = possession_timeline(frames, config)

    # --- stage 6: action classification -------------------------------------
    windows: list[ActionWindow] = []
    if args.skip_action:
        print("skipping stage 6 (--skip-action)")
    else:
        import cv2
        import torch
        from transformers import (
            VideoMAEForVideoClassification,
            VideoMAEImageProcessor,
        )
        from courtvision.types import ACTIONS

        base = "MCG-NJU/videomae-base"
        processor = VideoMAEImageProcessor.from_pretrained(base)
        model = VideoMAEForVideoClassification.from_pretrained(
            base, num_labels=len(ACTIONS), ignore_mismatched_sizes=True
        ).to(device).eval()

        plans = plan_windows(n, config.action_window_frames, config.action_stride_frames)
        if plans:  # warm up the video model too
            first = np.stack([
                cv2.cvtColor(cv2.resize(images[i], (FRAME_SIZE, FRAME_SIZE)),
                             cv2.COLOR_BGR2RGB)
                for i in range(plans[0][0], plans[0][1] + 1)])
            with torch.no_grad():
                model(**{k: v.to(device) for k, v in
                         processor(list(first), return_tensors="pt").items()})

        with stage("6 action classification"):
            for start, end in plans:
                clip = np.stack([
                    cv2.cvtColor(cv2.resize(images[i], (FRAME_SIZE, FRAME_SIZE)),
                                 cv2.COLOR_BGR2RGB)
                    for i in range(start, end + 1)])
                inputs = processor(list(clip), return_tensors="pt")
                with torch.no_grad():
                    logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
                idx = int(logits.softmax(dim=-1)[0].argmax())
                windows.append(ActionWindow(start, end, times[start], times[end],
                                            ACTIONS[idx], 1.0))

    # --- stage 7: event structuring -----------------------------------------
    with stage("7 events"):
        events = build_events(windows, holders, teams)

    # --- stage 9: render -----------------------------------------------------
    out_dir = Path("outputs/profile")
    out_dir.mkdir(parents=True, exist_ok=True)
    with stage("9 render"):
        render_video(images, frames, teams, holders,
                     str(out_dir / "profile.mp4"), config.target_fps)

    # --- report --------------------------------------------------------------
    total = sum(TIMINGS.values())
    print(f"\n{'stage':<26}{'seconds':>10}{'% total':>10}{'ms/frame':>12}")
    print("-" * 58)
    for name in sorted(TIMINGS):
        seconds = TIMINGS[name]
        print(f"{name:<26}{seconds:>10.3f}{100 * seconds / total:>9.1f}%"
              f"{1000 * seconds / n:>12.1f}")
    print("-" * 58)
    print(f"{'TOTAL':<26}{total:>10.3f}{100.0:>9.1f}%{1000 * total / n:>12.1f}")
    print(f"\n{n} frames at {config.target_fps}fps "
          f"= {n / config.target_fps:.1f}s of footage in {total:.1f}s wall clock "
          f"({(n / config.target_fps) / total:.2f}x realtime)")
    print(f"{len(windows)} action windows, {len(events)} events, {len(teams)} tracks")

    hottest = max(TIMINGS, key=TIMINGS.get)
    print(f"\nBOTTLENECK: {hottest} at {100 * TIMINGS[hottest] / total:.1f}% of total")
    print("Stage 8 (commentary) is excluded — it is one network round-trip, "
          "latency-bound, and no kernel work applies to it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
