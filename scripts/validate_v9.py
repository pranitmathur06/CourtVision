"""V9 — End-to-end run (spec §6).

Runs the full pipeline on a HELD-OUT clip that was not used in any fine-tuning,
and checks it completes without crashing and produces both outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from courtvision.config import Config
from scripts.run_pipeline import run_pipeline

CLIP = Path("data/raw_clips/holdout.mp4")
OUT_DIR = Path("outputs/v9")


def main() -> int:
    if not CLIP.exists():
        print(
            f"V9 FAIL — no held-out clip at {CLIP}. It must NOT be a clip used in "
            "V3 or V7 fine-tuning, or this checks nothing."
        )
        return 1

    try:
        summary = run_pipeline(str(CLIP), str(OUT_DIR), Config())
    except Exception as exc:  # noqa: BLE001 - a crash here is the failure under test
        print(f"V9 FAIL — pipeline raised {type(exc).__name__}: {exc}")
        return 1

    video_ok = Path(summary["video_path"]).stat().st_size > 0
    log = json.loads(Path(summary["log_path"]).read_text())
    log_ok = len(log["events"]) == summary["n_events"] and summary["n_events"] > 0
    ok = video_ok and log_ok and not summary["errors"]

    verdict = "PASS" if ok else "FAIL"
    print(
        f"\nV9 {verdict} — {summary['n_frames']} frames, {summary['n_tracks']} tracks, "
        f"{summary['n_events']} events, {summary['n_lines']} commentary lines, "
        f"{len(summary['errors'])} fabrication errors in {summary['seconds']:.1f}s"
    )
    if ok:
        print(
            f"  Now spot-check 5 random moments in {summary['video_path']} against "
            f"{summary['log_path']} — that comparison is the actual V9 criterion."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
