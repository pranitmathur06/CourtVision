"""V6 — Possession heuristic sanity check (spec §6).

Scores the heuristic against a hand-built answer key of known "who has the ball"
moments, recorded as IMAGE POSITIONS rather than track IDs:

    [{"time_s": 1.1, "x": 895, "y": 420, "note": "..."},
     {"time_s": 7.1, "x": null, "y": null, "note": "ball in flight"}]

Positions, not IDs, because a track ID is an artefact of one particular detector
and tracker run. An earlier version of this key stored IDs; retraining the
detector silently invalidated every entry, and the gate went on reporting a
number as if it still meant something. A point on the ball-handler's body is a
fact about the footage and survives any model change.

`x`/`y` null means nobody is in possession (ball in flight, loose ball).
Build the key by eye from outputs/v4_tracking.mp4 or the raw clip.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from courtvision.config import Config
from courtvision.detection import load_pipeline_detector
from courtvision.device import resolve_device
from courtvision.extraction import extract_frames
from courtvision.possession import possession_timeline
from courtvision.tracking import PlayerTracker
from courtvision.types import Frame

CLIP = Path("data/raw_clips/sample.mp4")
CHECKPOINT = Path("checkpoints/detector.pt")
# Hand-made ground truth, so it lives under version control. It sat in
# outputs/ for most of this project's life, which .gitignore excludes: it had
# never been committed, would have been lost with any clean of that directory,
# and was not present on a fresh checkout at all. Irreplaceable data does not
# belong in a directory named for disposable output.
ANSWER_KEY = Path("data/ground_truth/v6_possession.json")
LEGACY_ANSWER_KEY = Path("outputs/v6_answer_key.json")
# The spec asks for 8 of 10; this keeps that 80% ratio for whatever size the
# answer key actually is.
REQUIRED_RATIO = 0.8


def main() -> int:
    if not ANSWER_KEY.exists() and LEGACY_ANSWER_KEY.exists():
        print(f"V6 note — using the legacy key at {LEGACY_ANSWER_KEY}; "
              f"move it to {ANSWER_KEY} so it is version-controlled")
        globals()["ANSWER_KEY"] = LEGACY_ANSWER_KEY
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
    # Possession needs the ball, so this gate uses the composite detector
    # (fine-tuned players + large COCO model for the ball), not players alone.
    detector = load_pipeline_detector(
        str(CHECKPOINT), resolve_device(), config.detector_conf, config.ball_conf
    )
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

        # Resolve the annotated point to whichever track covers it now.
        x, y = entry.get("x"), entry.get("y")
        if x is None or y is None:
            expected = None
        else:
            expected = None
            for track in frame.players():
                if (track.box.x1 <= x <= track.box.x2
                        and track.box.y1 <= y <= track.box.y2):
                    expected = track.track_id
                    break
            if expected is None:
                misses.append({"time_s": entry["time_s"], "want": "no track at "
                               f"({x},{y})", "got": predicted})
                continue

        if predicted == expected:
            correct += 1
        else:
            misses.append(
                {"time_s": entry["time_s"], "want": expected, "got": predicted}
            )

    import math

    required = math.ceil(REQUIRED_RATIO * len(key))
    ok = correct >= required
    verdict = "PASS" if ok else "FAIL"
    print(f"V6 {verdict} — {correct}/{len(key)} correct (need >={required})")
    for miss in misses:
        print(f"  miss at {miss['time_s']:.1f}s: wanted {miss['want']}, got {miss['got']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
