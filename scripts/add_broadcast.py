"""One command: a video file and an NBA game id in, a scored broadcast out.

    scripts/add_broadcast.py --game hou

THIS SCRIPT IS THE DEFINITION OF DONE. Everything else in the pipeline is a
stage inside it. Before it existed, adding a broadcast meant running nine
scripts in an order written down nowhere, with paths that had to agree across
eighteen files -- which is why the repository had three games in it and not
four, and why "a new broadcast fits right in" was a claim nobody could check.

IT IS A STAGE GRAPH, NOT A SHELL SCRIPT. Each stage declares what it produces,
and a stage whose outputs exist is skipped. So a run that dies in the sixth
stage -- and one will, because these stages are hours long -- is resumed by
running the identical command again. Nothing takes a path as an argument from
the caller: every path comes from `courtvision.games`, so a stage cannot be
pointed at another game's cache.

WHAT IT REFUSES TO DO. It will not invent a constant. Every threshold a stage
uses is that stage's own default, recorded in the manifest, and there is no
`--tune` anywhere in this file: the acceptance test for this pipeline is a
broadcast nothing was tuned on, and a driver that tunes per game would make
that test meaningless while appearing to pass it. If a stage needs a different
constant for a different arena, that is a finding and belongs in the stage.

THE MANIFEST IS THE DELIVERABLE, ALONGSIDE THE ARTEFACTS. It records the commit,
every stage's command line, its duration, its outputs and their sizes, and
whether it ran or was skipped -- so the question "was this number produced by
the code that is checked in now" has an answer that is not a memory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.candidates import COURT_ERODE_FILE  # noqa: E402
from courtvision.floor_colour import FLOOR_FILE  # noqa: E402
from courtvision.games import Broadcast, get, registry  # noqa: E402

PYTHON = sys.executable


def at(path) -> Path:
    """A registry path as an absolute one under the repository.

    Every path in the registry is written relative to the repository root, and
    the stages run with `cwd=ROOT` -- but the driver's own `exists()` checks ran
    against the CALLER's directory, so running this from anywhere else reported
    a video that is plainly on disk as missing and would have re-fetched every
    play-by-play from the network.
    """
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


@dataclass
class Stage:
    """One step: what it makes, and how."""
    name: str
    #: Run only if at least one of these is missing.
    produces: list[Path]
    #: argv after the interpreter. `None` means the stage is a python callable.
    argv: list[str] | None = None
    #: Stages that must have produced their outputs first.
    needs: list[str] = field(default_factory=list)
    #: A note printed when the stage is skipped for want of a prerequisite.
    why: str = ""
    #: True when the stage wants a GPU and will be slow without one.
    heavy: bool = False

    def done(self) -> bool:
        """Every output exists AND, if it is JSON, parses.

        No stage writes atomically. A stage killed inside `json.dump` leaves a
        truncated file that `exists()` calls complete -- so the driver skips it
        forever and the next stage dies on a JSONDecodeError, hours later and
        pointing at the wrong script. Parsing is cheap next to re-running any of
        these stages, and it is the only check that can tell a finished output
        from an interrupted one.
        """
        for path in self.produces:
            if not path.exists():
                return False
            if path.suffix == ".json":
                try:
                    json.loads(path.read_text())
                except (ValueError, OSError):
                    return False
        return True


def stages_for(game: Broadcast, detector: str, ball_detector: str | None,
               fps: float, skip: list[str],
               args_registry: str | None = None) -> list[Stage]:
    """The pipeline, in dependency order, for one broadcast.

    Ordered so the cheapest things that unblock a number come first: the clock
    read unblocks alignment, and alignment is the arm that clears 90%. Detection
    is last because it is the only part that wants a GPU, and a broadcast with
    no detections still has a scored, published, searchable timeline.
    """
    S = lambda p: str(at(p))                             # noqa: E731
    official = at(f"data/pbp/shots_{game.game_id}.json")
    on_video = at(f"outputs/shots_on_video_{game.game_id}.json")
    out: list[Stage] = [
        Stage("pbp", [at(game.pbp)],
              [S(ROOT / "scripts" / "cache_pbp.py"),
               "--game-id", game.game_id, "--out", S(game.pbp)]),
        Stage("clock", [at(game.clock)],
              [S(ROOT / "scripts" / "read_game_clock.py"),
               "--video", S(game.video), "--out", S(game.clock)]),
        Stage("align", [at(game.aligned)],
              [S(ROOT / "scripts" / "align_game_events.py"),
               "--game-id", game.game_id, "--clock", S(game.clock),
               "--out", S(game.aligned)],
              needs=["clock"]),
        Stage("roster", [at(game.roster)],
              [S(ROOT / "scripts" / "fetch_roster.py"),
               "--game-id", game.game_id, "--out", S(game.roster)]),
        Stage("clips", [at(game.clip_index)],
              [S(ROOT / "scripts" / "cut_event_clips.py"),
               "--video", S(game.video), "--events", S(game.aligned),
               "--out-dir", S(game.clip_dir),
               "--index-name", game.clip_index.name, "--prefix", game.prefix,
               "--skip", "Rebound"],
              needs=["align"]),
        Stage("scoreboard", [at(game.scoreboard)],
              [S(ROOT / "scripts" / "read_scoreboard.py"),
               "--video", S(game.video), "--clock", S(game.clock),
               "--out", S(game.scoreboard)],
              needs=["clock"]),
        Stage("detections", [at(game.detections)],
              [S(ROOT / "scripts" / "cache_detections.py"),
               "--video", S(game.video), "--detector", detector,
               "--fps", str(fps), "--out", S(game.detections)],
              heavy=True),
        # `detect_shots.py --shots` defaults to `shots_on_video_0042400407.json`
        # -- ONE GAME'S official shot times, with that game's number in the
        # filename. Without these two stages a fourth broadcast's shot detector
        # is scored against the 2025 Finals Game 7: every agrees-with-official
        # label and the first/second-half split come from the wrong game. And
        # nothing in the repository produced that file for any game; it was made
        # by hand, which is the one thing the acceptance test forbids.
        Stage("official_shots", [official],
              [S(ROOT / "scripts" / "official_shots.py"),
               "--game-id", game.game_id, "--out", str(official)],
              needs=["pbp"]),
        Stage("shots_on_video", [on_video],
              [S(ROOT / "scripts" / "align_shots_to_video.py"),
               "--official", str(official), "--clock", S(game.clock),
               "--out", str(on_video)],
              needs=["official_shots", "clock"]),
        Stage("vision_shots", [at(game.vision_shots)],
              [S(ROOT / "scripts" / "detect_shots.py"),
               "--detections", S(game.detections), "--clock", S(game.clock),
               "--shots", str(on_video), "--out", S(game.vision_shots)],
              needs=["detections", "clock", "shots_on_video"]),
        Stage("clip_detections", [at(game.clip_detections)],
              [S(ROOT / "scripts" / "clip_detect_raw.py"),
               "--video", S(game.video), "--index", S(game.clip_index),
               "--detector", detector, "--out", S(game.clip_detections)]
              + (["--ball-detector", ball_detector] if ball_detector else []),
              needs=["clips"], heavy=True),
        # THE FLOOR IS LEARNED FROM THE BROADCAST, AND THAT NEEDS THE BOXES,
        # so it cannot happen inside the detection pass. A player stands on the
        # floor, so the strip below his box is floor -- which means the floor
        # can only be learned once somebody has drawn the boxes. Hence three
        # stages after the detections rather than a flag on them: learn the
        # arena's floor, choose its erosion against the two bounds the sport
        # supplies, then rewrite the mask. None of the three runs a model and
        # none needs a label.
        Stage("floor_colour", [at(FLOOR_FILE)],
              [S(ROOT / "scripts" / "fit_floor_colour.py"),
               "--game", game.key],
              needs=["clip_detections"]),
        Stage("court_erode", [at(COURT_ERODE_FILE)],
              [S(ROOT / "scripts" / "fit_court_mask.py"),
               "--game", game.key, "--frames", "900"],
              needs=["floor_colour"]),
        Stage("remask", [at(str(game.clip_detections) + ".premask.json")],
              [S(ROOT / "scripts" / "remask_detections.py"),
               "--game", game.key, "--replace"],
              needs=["court_erode"]),
        Stage("overlays", [at(game.overlays)],
              [S(ROOT / "scripts" / "clip_boxes.py"),
               "--from-cache", S(game.clip_detections),
               "--out", S(game.overlays)],
              needs=["remask"]),
        Stage("score", [at(game.end_to_end)],
              [S(ROOT / "scripts" / "score_game_end_to_end.py"),
               "--game-id", game.game_id, "--aligned", S(game.aligned),
               "--clock", S(game.clock),
               # THE WHOLE-GAME DETECTION CACHE, not detect_shots' report.
               # `vision_shot_stream` reads `blob["frames"]` and re-runs
               # detect_shots' own rim tracking over them; handed the report
               # instead it found no `frames` key, built zero calls, and printed
               # a vision arm of 0.000 with no error anywhere.
               "--detections", S(game.detections),
               "--readings", S(game.scoreboard), "--out", S(game.end_to_end)],
              needs=["align"]),
        Stage("report", [at(game.report)],
              [S(ROOT / "scripts" / "eval_by_game.py"),
               "--game", game.key, "--out", S(game.report)]
              + (["--registry", args_registry] if args_registry else []),
              needs=["align"]),
    ]
    return [s for s in out if s.name not in skip]


def run(stage: Stage, force: bool, dry: bool) -> dict:
    """Run one stage, or say why it was not run. Returns its manifest row."""
    row = {"stage": stage.name,
           "produces": [str(p) for p in stage.produces],
           "argv": stage.argv}
    if stage.done() and not force:
        row["status"] = "skipped (outputs exist)"
        print(f"  [skip] {stage.name:16} {', '.join(str(p) for p in stage.produces)}")
        return row
    if dry:
        row["status"] = "would run"
        print(f"  [dry ] {stage.name:16} {' '.join(stage.argv or [])}")
        return row
    print(f"  [run ] {stage.name:16} {' '.join(stage.argv or [])}", flush=True)
    began = time.time()
    result = subprocess.run([PYTHON] + (stage.argv or []), cwd=ROOT,
                            env=_env())
    row["seconds"] = round(time.time() - began, 1)
    row["returncode"] = result.returncode
    missing = [str(p) for p in stage.produces if not p.exists()]
    row["missing"] = missing
    row["status"] = ("ok" if result.returncode == 0 and not missing
                     else "FAILED")
    row["sizes"] = {str(p): p.stat().st_size for p in stage.produces
                    if p.exists()}
    if row["status"] == "FAILED":
        # A stage that failed after writing SOMETHING leaves an output the next
        # run would treat as finished. Nothing here writes atomically, so the
        # only safe thing is to take the half-written file away and let the
        # stage run again.
        for path in stage.produces:
            if path.exists() and not stage.done():
                path.unlink()
                row.setdefault("removed", []).append(str(path))
    mark = "ok" if row["status"] == "ok" else "FAILED"
    print(f"  [{mark:4}] {stage.name:16} {row['seconds']}s"
          + (f"  missing {missing}" if missing else ""))
    return row


def _env() -> dict:
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:                                    # pragma: no cover
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True,
                        help="registry key, official game id, or video stem")
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--ball-detector", default=None)
    parser.add_argument("--fps", type=float, default=5.0,
                        help="sampling rate of the whole-game detection cache")
    parser.add_argument("--only", action="append", default=[],
                        help="run only these stages (repeatable)")
    parser.add_argument("--skip", action="append", default=[],
                        help="do not run these stages (repeatable)")
    parser.add_argument("--force", action="store_true",
                        help="re-run stages whose outputs already exist")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--registry", default=None,
                        help="a games.json other than the shipped one. Exists so "
                             "an integration test can build a broadcast out of a "
                             "few small files and drive this whole path.")
    parser.add_argument("--list", action="store_true",
                        help="print the registry and exit")
    args = parser.parse_args()

    if args.list:
        for key, g in registry(args.registry).items():
            flag = " (unseen)" if g.unseen else ""
            print(f"{key:6} {g.game_id}  {g.label}{flag}\n       {g.video}")
        return 0

    game = get(args.game, args.registry)
    if not at(game.video).exists():
        print(f"FAIL - {game.video} is not on disk")
        return 1
    print(f"{game.label}  ({game.key}, official {game.game_id})")
    print(f"  video {game.video}")
    if game.unseen:
        print("  UNSEEN: this broadcast has chosen no threshold and trained "
              "no model. Its numbers are the acceptance test.")

    plan = stages_for(game, args.detector, args.ball_detector, args.fps,
                      args.skip, args.registry)
    if args.only:
        plan = [s for s in plan if s.name in args.only]
    if not plan:
        print("nothing to do")
        return 0

    # What a stage produces is on the stage. There used to be a second,
    # parallel mapping here, and `all([])` made any name missing from it read as
    # satisfied -- so a typo in a `needs` list, or a new stage, would silently
    # run with its prerequisite absent.
    produced = {s.name: s for s in stages_for(game, args.detector,
                                              args.ball_detector, args.fps, [],
                                              args.registry)}
    rows, satisfied = [], set()
    for stage in plan:
        blocked = [n for n in stage.needs
                   if n not in satisfied and not _satisfied(n, produced)]
        if blocked:
            rows.append({"stage": stage.name, "status": f"blocked on {blocked}"})
            print(f"  [----] {stage.name:16} blocked on {', '.join(blocked)}")
            continue
        row = run(stage, args.force, args.dry_run)
        rows.append(row)
        if row["status"] in ("ok", "skipped (outputs exist)", "would run"):
            satisfied.add(stage.name)

    manifest = {
        "game": {"key": game.key, "game_id": game.game_id, "label": game.label,
                 "video": game.video, "unseen": game.unseen},
        "commit": commit(),
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": PYTHON,
        "constants": {"detector": args.detector,
                      "ball_detector": args.ball_detector,
                      "detection_fps": args.fps},
        "stages": rows,
    }
    path = at(args.manifest or f"outputs/games/manifest_{game.key}.json")
    if not args.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=1))
        print(f"\nmanifest -> {path}")

    failed = [r["stage"] for r in rows if r.get("status", "").startswith(("FAIL", "blocked"))]
    if failed:
        print(f"INCOMPLETE - {', '.join(failed)}")
        return 1
    print("complete")
    return 0


def _satisfied(name: str, produced: dict[str, "Stage"]) -> bool:
    """Has this stage's outputs on disk, from this run or an earlier one.

    An UNKNOWN name is not satisfied. The previous version looked the name up in
    a dict and fell back to an empty list, and `all([])` is True -- so every
    stage name it did not know about was reported as done.
    """
    stage = produced.get(name)
    return stage.done() if stage else False


if __name__ == "__main__":
    raise SystemExit(main())
