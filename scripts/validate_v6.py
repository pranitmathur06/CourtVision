"""V6 — Possession heuristic sanity check (spec §6).

Scores the heuristic against a hand-built answer key of known "who has the ball"
moments. Create outputs/v6_answer_key.json first by watching the annotated V4
video and recording 10 timestamps with the track ID that visibly has the ball:

    [{"time_s": 1.4, "track_id": 3}, {"time_s": 2.8, "track_id": 7}, ...]

The key is judged against the SAME track IDs the tracker produced, so build it
from outputs/v4_tracking.mp4, not from jersey numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from courtvision.config import Config
from courtvision.detection import load_finetuned
from courtvision.device import resolve_device
from courtvision.extraction import extract_frames
from courtvision.possession import possession_timeline
from courtvision.tracking import PlayerTracker
from courtvision.types import Frame

CLIP = Path("data/raw_clips/sample.mp4")
CHECKPOINT = Path("checkpoints/detector.pt")
ANSWER_KEY = Path("outputs/v6_answer_key.json")
REQUIRED_CORRECT = 8


def main() -> int:
    if not ANSWER_KEY.exists():
        print(
            f"V6 FAIL — no answer key at {ANSWER_KEY}; watch outputs/v4_tracking.mp4 "
            'and record 10 moments as [{"time_s": float, "track_id": int}, ...]'
        )
        return 1
    if not CLIP.exists() or not CHECKPOINT.exists():
        print(f"V6 FAIL — need clip at {CLIP} and checkpoint at {CHECKPOINT}")
        return 1

    config = Config()
    detector = load_finetuned(str(CHECKPOINT), resolve_device(), config.detector_conf)
    tracker = PlayerTracker()

    frames: list[Frame] = []
    for index, time_s, image in extract_frames(str(CLIP), config.target_fps):
        frames.append(Frame(index, time_s, tuple(tracker.update(detector.detect(image)))))

    timeline = possession_timeline(frames, config)
    key = json.loads(ANSWER_KEY.read_text())

    correct = 0
    misses = []
    for entry in key:
        # Nearest sampled frame to the annotated moment.
        frame = min(frames, key=lambda f: abs(f.time_s - entry["time_s"]))
        predicted = timeline[frame.index]
        if predicted == entry["track_id"]:
            correct += 1
        else:
            misses.append(
                {"time_s": entry["time_s"], "want": entry["track_id"], "got": predicted}
            )

    ok = correct >= REQUIRED_CORRECT
    verdict = "PASS" if ok else "FAIL"
    print(f"V6 {verdict} — {correct}/{len(key)} correct (need >={REQUIRED_CORRECT})")
    for miss in misses:
        print(f"  miss at {miss['time_s']:.1f}s: wanted {miss['want']}, got {miss['got']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
