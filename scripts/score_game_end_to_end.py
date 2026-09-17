"""Given a broadcast, what fraction of its plays does the stack get right?

The measurement this project has been deferring. `docs/continuous-game-accuracy.md`
defines an 85% bar in two readings and answers both -- 0.751 captured, 0.879
said-and-true -- on 25 Hz SportVU tracking coordinates, with perfect ball height
and stable player identities handed over free. That document says so itself, and
its last statement of the ledger reads "85% emitted-commentary precision, real
video -- not close". There has never been a composed number on video, and there
is no event-weighted-F1 code anywhere in the repo, so neither figure is
reproducible by anyone, including us.

THE TWO READINGS, kept separate because they are not close to each other:

    (a) what fraction of the game's plays does it CAPTURE   event-weighted F1
    (b) what fraction of what it SAYS is true               precision on the
                                                            emitted stream

ONE STREAM FORMAT, so vision, the scoreboard, the official feed and a
tracking-data run all pass through identical code and their numbers can be
compared. Every stream declares which of its inputs are VISION-DERIVED and which
are FEED-DERIVED, and that line prints above its numbers. It is the direct answer
to the confound that has made this project's demo read at 95%: the event text
comes from the official play-by-play, and vision's contribution is the timestamp.

AN ASSERTION BUDGET, so a system cannot score well by saying nothing specific.
Each call lists what it asserts. A call claiming only `kind` cannot lose points
on `made`, and the report prints how many of the emitted calls assert each
attribute -- so "0.95 precision" next to "asserts an outcome on 0% of calls"
reads as what it is.

MATCHING IS COARSE-CLASS, ONE-TO-ONE, GLOBALLY GREEDY, AT +-3.0 s.
Three seconds because that is `event_alignment.DEFAULT_TOLERANCE_S` and
`detect_shots.TOLERANCE_S`, and because the clock reader samples at 1 s so the
truth's own placement is quantised there; a tighter tolerance would mostly
measure sampling density. Globally greedy rather than in list order, so the
answer does not depend on the order calls arrive in.

IDENTITY IS NEVER A MATCHING KEY. Rebound attribution measures 45-47% against a
72% ceiling and jersey identity 45%, so requiring a name would multiply every
number by about a half and make this a jersey-OCR benchmark wearing an
end-to-end costume. No stream currently asserts a player at all, which the
assertion budget prints as "a player on 0% of its calls"; when one does, that
is where its identity accuracy will appear.

THE LIVE-PLAY FILTER RESTRICTS THE DENOMINATOR TOO. Official plays can only be
aligned where the clock is READABLE; the filter admits calls where the clock is
readable AND RUNNING. Scoring filtered calls against an unrestricted denominator
would credit the filter for suppressing false alarms in regions the truth could
never occupy. One region mask is applied to both sides, so the circular cell
cannot be printed. Both settings are additionally restricted to the
clock-readable span, so the only difference between them is running versus held
-- a comparison the truth can occupy on both sides.

PRECISION NEVER PRINTS WITHOUT ITS COVERAGE. `docs/continuous-game-accuracy.md`
contains "field goals only, 0.941 at coverage 0.437" as a warning, and the
printer here has no path that emits one without the other.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from courtvision.stats import (block_bootstrap, mcnemar,  # noqa: E402
                               wilson)

#: Seconds either side that count as the same instant. See the docstring.
TOLERANCE_S = 3.0
#: Also reported, so the reader can see how much the tolerance is buying.
SENSITIVITY_S = (2.0, 3.0, 5.0)
#: Free throws at one stoppage share a clock value. A trip of two is two of the
#: game's plays and one moment on the video, so matching is on trips and
#: weighting on attempts, and both counts are printed.
FT_TRIP_MERGE_S = 12.0
#: Blocks for the bootstrap over the timeline. Round 65 established this.
BLOCK_S = 120.0

#: Official label -> the coarse class matching happens on. Fine attributes
#: (made, points) are scored on the matched subset, never used to match.
CLASSES = {
    "Made Shot (2PT)": "field_goal",
    "Made Shot (3PT)": "field_goal",
    "Missed Shot": "field_goal",
    "Free Throw (made)": "free_throw",
    "Free Throw (miss)": "free_throw",
    "Rebound": "rebound",
    "Turnover": "turnover",
    "Foul": "foul",
    "Steal": "steal",
    "Block": "block",
    "Violation": "violation",
}
#: Not plays. `Assist` is excluded because `align_game_events.classify` appends
#: it to the SAME row as the basket it assisted, so counting it would score one
#: basket twice; it is reported as an attribute of a matched field goal instead.
NOT_A_PLAY = frozenset({"Substitution", "Timeout", "Instant Replay", "Jump Ball",
                        "Ejection", "period", "Assist"})


@dataclass(frozen=True)
class Play:
    """One official play, already placed on the video clock."""
    video_s: float
    kind: str
    label: str
    made: bool | None
    points: int | None
    align_error_s: float
    description: str


@dataclass(frozen=True)
class Call:
    """One thing a system says happened."""
    video_s: float
    kind: str
    asserts: frozenset[str] = frozenset({"kind"})
    made: bool | None = None
    points: int | None = None
    player: str | None = None


@dataclass(frozen=True)
class Stream:
    """What a system emitted, and what it was allowed to look at."""
    source: str
    vision_derived: tuple[str, ...]
    feed_derived: tuple[str, ...]
    calls: tuple[Call, ...]
    runnable: bool = True
    blocked_because: str = ""


@dataclass
class Scored:
    """Everything one mode produced, ready to print or serialise."""
    label: str
    stream: Stream
    per_class: dict = field(default_factory=dict)
    captured: float = 0.0
    captured_interval: tuple[float, float] = (0.0, 1.0)
    timestamped: int = 0
    described: int = 0
    emitted: int = 0
    architectural_coverage: float = 0.0
    captured_coverage: float = 0.0
    assertion_budget: dict = field(default_factory=dict)
    false_calls: dict = field(default_factory=dict)
    capture_vector: list = field(default_factory=list)
    #: play video_s -> captured. Keyed by the play so two modes under different
    #: masks can still be paired on the plays they both saw.
    captured_plays: dict = field(default_factory=dict)


def outcome(label: str, description: str):
    """(made, points) for a play, or (None, None) when it does not say."""
    if label.startswith("Made Shot"):
        return True, 3 if "3PT" in label else 2
    if label == "Missed Shot":
        return False, 0
    if label == "Free Throw (made)":
        return True, 1
    if label == "Free Throw (miss)":
        return False, 0
    return None, None


def load_truth(path: str) -> tuple[list[Play], dict]:
    """The official plays on the video clock, and what was dropped getting here."""
    blob = json.load(open(path))
    plays, skipped = [], {}
    for event in blob["events"]:
        label = event["action"]
        if label in NOT_A_PLAY or label not in CLASSES:
            skipped[label] = skipped.get(label, 0) + 1
            continue
        made, points = outcome(label, event.get("description", ""))
        plays.append(Play(video_s=float(event["video_s"]), kind=CLASSES[label],
                          label=label, made=made, points=points,
                          align_error_s=float(event.get("error_s", 0.0)),
                          description=event.get("description", "")))
    plays.sort(key=lambda p: p.video_s)
    meta = {"game_id": blob.get("game_id"),
            "located": len(blob["events"]),
            "overall_rate": blob.get("overall_rate"),
            "not_scored": skipped,
            "scored_plays": len(plays)}
    return plays, meta


def free_throw_trips(plays: Sequence[Play], merge_s: float = FT_TRIP_MERGE_S):
    """Collapse free throws at one stoppage into a trip; keep both counts.

    The clock is frozen for a trip, so two attempts share a video second. A
    system cannot be asked to emit two calls a tenth of a second apart, and
    scoring it as though it should caps free-throw recall by construction --
    which is exactly what Round 4 measured at 0.53.
    """
    out, attempts = [], 0
    for play in plays:
        if play.kind != "free_throw":
            out.append(play)
            continue
        attempts += 1
        if out and out[-1].kind == "free_throw" \
                and play.video_s - out[-1].video_s <= merge_s:
            continue
        out.append(play)
    trips = sum(1 for p in out if p.kind == "free_throw")
    return out, attempts, trips


def match(calls: Sequence[Call], plays: Sequence[Play],
          tolerance_s: float = TOLERANCE_S):
    """One-to-one, globally greedy by |dt|. Order-independent by construction.

    `detect_shots.score` and `run_broadcast.match` both walk their inputs in list
    order, so their answer depends on the order calls arrive in. Here every
    admissible pair is built, sorted by how close it is, and taken greedily.
    """
    pairs = []
    for i, call in enumerate(calls):
        for j, play in enumerate(plays):
            if call.kind != play.kind:
                continue
            gap = abs(call.video_s - play.video_s)
            if gap > tolerance_s:
                continue
            pairs.append((gap, j, i))
    pairs.sort()
    call_to_play: dict[int, int] = {}
    used_plays, used_calls = set(), set()
    for _, j, i in pairs:
        if i in used_calls or j in used_plays:
            continue
        used_calls.add(i)
        used_plays.add(j)
        call_to_play[i] = j
    return call_to_play


def describes(call: Call, play: Play) -> bool:
    """Is every attribute the call ASSERTED also right?"""
    if "made" in call.asserts and call.made != play.made:
        return False
    if "points" in call.asserts and call.points != play.points:
        return False
    return True


def weighted_f1(calls: Sequence[Call], plays: Sequence[Play],
                tolerance_s: float, weights: dict[str, float]) -> float:
    """Event-weighted F1 of one call set against one play set.

    `weights` come from the WHOLE game rather than from the resample, so a
    bootstrap draw that happens to contain no rebounds does not quietly
    reweight the statistic it is meant to put an interval around.

    The honest consequence, which is not the same thing: a draw that contains
    no instances of a class scores that class 0 while it keeps its full weight,
    so the interval is slightly pessimistic for rare classes rather than
    reweighted. That is why feed-assisted, which is exactly 1.000, reports a
    lower bound of 0.995 rather than 1.000.
    """
    pairing = match(calls, plays, tolerance_s)
    counts: dict[str, list[int]] = {}
    for play in plays:
        counts.setdefault(play.kind, [0, 0, 0])[0] += 1
    for call in calls:
        counts.setdefault(call.kind, [0, 0, 0])[1] += 1
    for i in pairing:
        counts[calls[i].kind][2] += 1
    total = 0.0
    for kind, weight in weights.items():
        official, emitted, hits = counts.get(kind, [0, 0, 0])
        precision = hits / emitted if emitted else 0.0
        recall = hits / official if official else 0.0
        if precision + recall:
            total += weight * 2 * precision * recall / (precision + recall)
    return total


def bootstrap_weighted_f1(calls: Sequence[Call], plays: Sequence[Play],
                          tolerance_s: float, weights: dict[str, float],
                          block_s: float = BLOCK_S, draws: int = 400,
                          seed: int = 0) -> tuple[float, float]:
    """Interval for the event-weighted F1, resampling whole blocks of TIME.

    Both the calls and the plays inside a drawn block travel together, which is
    what preserves the matching. Round 65 measured what resampling individual
    attempts does to this data: it reported a meaningless 0.375-0.456.

    An earlier version here resampled the per-play CAPTURE vector instead, which
    is a bootstrap of recall wearing an F1's label -- and it showed, because the
    interval did not contain its own point estimate.
    """
    import numpy as np

    if not plays:
        return 0.0, 1.0
    start = min([p.video_s for p in plays] + [c.video_s for c in calls] or [0.0])
    def block_of(when):
        return int((when - start) // block_s)
    play_blocks: dict[int, list[Play]] = {}
    call_blocks: dict[int, list[Call]] = {}
    for play in plays:
        play_blocks.setdefault(block_of(play.video_s), []).append(play)
    for call in calls:
        call_blocks.setdefault(block_of(call.video_s), []).append(call)
    # Blocks holding CALLS but no plays are resampled too. Skipping them meant
    # 22 of G7's false alarms -- pregame and halftime -- could never be drawn,
    # and the bootstrap mean sat above the point estimate, which is the tell.
    keys = sorted(set(play_blocks) | set(call_blocks))
    if len(keys) < 3:
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    draws_out = []
    for _ in range(draws):
        pick = rng.integers(0, len(keys), size=len(keys))
        drawn_plays, drawn_calls = [], []
        for offset, index in enumerate(pick):
            key = keys[index]
            shift = offset * block_s - key * block_s      # keep blocks apart
            for play in play_blocks.get(key, []):
                drawn_plays.append(Play(play.video_s + shift, play.kind,
                                        play.label, play.made, play.points,
                                        play.align_error_s, play.description))
            for call in call_blocks.get(key, []):
                drawn_calls.append(Call(call.video_s + shift, call.kind,
                                        call.asserts, call.made, call.points,
                                        call.player))
        draws_out.append(weighted_f1(drawn_calls, drawn_plays, tolerance_s,
                                     weights))
    return (float(np.percentile(draws_out, 2.5)),
            float(np.percentile(draws_out, 97.5)))


def score_mode(stream: Stream, plays: Sequence[Play], *, label: str,
               tolerance_s: float = TOLERANCE_S, span_s: float = 0.0,
               live_mask=None, running_mask=None,
               clock_running_s: float = 0.0,
               clock_held_s: float = 0.0) -> Scored:
    """One stream against one set of plays. The region restricts BOTH sides."""
    calls = list(stream.calls)
    kept_plays = list(plays)
    if live_mask is not None:
        calls = [c for c in calls if live_mask(c.video_s)]
        kept_plays = [p for p in kept_plays if live_mask(p.video_s)]

    result = Scored(label=label, stream=stream)
    if not stream.runnable:
        return result

    pairing = match(calls, kept_plays, tolerance_s)
    counts: dict[str, dict] = {}
    for play in kept_plays:
        counts.setdefault(play.kind, {"official": 0, "emitted": 0, "hits": 0})
        counts[play.kind]["official"] += 1
    for call in calls:
        counts.setdefault(call.kind, {"official": 0, "emitted": 0, "hits": 0})
        counts[call.kind]["emitted"] += 1

    described = 0
    matched_play = set()
    for i, j in pairing.items():
        counts[calls[i].kind]["hits"] += 1
        matched_play.add(j)
        if describes(calls[i], kept_plays[j]):
            described += 1

    total_official = sum(v["official"] for v in counts.values()) or 1
    weighted, emittable = 0.0, 0.0
    for kind, v in counts.items():
        weight = v["official"] / total_official
        precision = v["hits"] / v["emitted"] if v["emitted"] else 0.0
        recall = v["hits"] / v["official"] if v["official"] else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        v.update({"weight": weight, "precision": precision, "recall": recall,
                  "f1": f1,
                  "precision_ci": wilson(v["hits"], v["emitted"]),
                  "recall_ci": wilson(v["hits"], v["official"])})
        weighted += weight * f1
        if v["emitted"]:
            emittable += weight
    result.per_class = counts
    result.captured = weighted
    result.architectural_coverage = emittable
    result.captured_coverage = sum(v["weight"] * v["recall"]
                                   for v in counts.values())
    result.timestamped = len(pairing)
    result.described = described
    result.emitted = len(calls)
    result.capture_vector = [j in matched_play for j in range(len(kept_plays))]
    result.captured_plays = {round(p.video_s, 3): (j in matched_play)
                             for j, p in enumerate(kept_plays)}

    budget: dict[str, float] = {}
    for attribute in ("kind", "made", "points", "player"):
        budget[attribute] = (sum(1 for c in calls if attribute in c.asserts)
                             / len(calls)) if calls else 0.0
    result.assertion_budget = budget

    wrong = [(c.video_s, i in pairing) for i, c in enumerate(calls)]
    result.captured_interval = bootstrap_weighted_f1(
        calls, kept_plays, tolerance_s,
        {kind: v["weight"] for kind, v in counts.items()})

    # The split needs the LIVE mask, not the mask this mode was scored under.
    # Passing the mode's own mask made every miss count as clock-running
    # whenever the mode was ungated -- G1 printed 2.37 running / 0.00 held when
    # the truth was 1.14 / 2.91, which reads as though the dead ball were free.
    misses = sum(1 for _, right in wrong if not right)
    if clock_running_s or clock_held_s:
        running_misses = sum(1 for when, right in wrong
                             if not right and (running_mask is None
                                               or running_mask(when)))
        held_misses = misses - running_misses
        result.false_calls = {
            "clock_running_per_min": (running_misses / (clock_running_s / 60.0)
                                      if clock_running_s else 0.0),
            "clock_held_per_min": (held_misses / (clock_held_s / 60.0)
                                   if clock_held_s else 0.0)}
    return result


# ---- streams ---------------------------------------------------------------

def feed_stream(plays: Sequence[Play], path: str, jitter_s: float = 0.0) -> Stream:
    """The official record, placed on the video by the clock reader.

    This mode says nothing vision found. Its event identity, type, outcome and
    player all come from the feed; the only thing vision contributes is WHEN. Its
    precision is 1.000 by construction and means only that the feed agrees with
    itself, which is why the printer refuses to report it as an accuracy.

    IT TAKES THE ALREADY-MERGED PLAYS, not the file. Built from the file it
    emitted one call per free-throw ATTEMPT against a truth merged into TRIPS,
    and scored 0.622 on the one mode that is right by definition -- which is
    exactly what a mode that is right by definition is for: it caught the
    scorer's own inconsistency before any vision number was reported.
    """
    rng = None
    if jitter_s:
        import numpy as np
        rng = np.random.default_rng(0)
    calls = []
    for play in plays:
        when = play.video_s + (float(rng.normal(0, jitter_s)) if rng else 0.0)
        calls.append(Call(video_s=when, kind=play.kind,
                          asserts=frozenset({"kind", "made", "points"}),
                          made=play.made, points=play.points))
    return Stream(source=f"the official play-by-play on the video clock ({path})",
                  vision_derived=("game clock",),
                  feed_derived=("event identity", "type", "outcome", "player"),
                  calls=tuple(calls))


def vision_shot_stream(detections: str, clock: str | None,
                       live_only: bool) -> Stream:
    """Field goals from ball-to-rim geometry. Imports detect_shots, never copies.

    Copying its rim tracking or its live-play rule would let the composed number
    drift away from the component number it is supposed to compose.
    """
    if not detections or not Path(detections).exists():
        return Stream(source="scripts/detect_shots.py", vision_derived=(),
                      feed_derived=(), calls=(), runnable=False,
                      blocked_because=f"no detection cache at {detections}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import detect_shots

    blob = json.load(open(detections))
    frames = blob.get("frames") or blob.get("rows") or []
    # The same calls detect_shots makes, with its own frozen constants, by
    # importing it. Copying them would let this number drift away from the
    # component number it is supposed to compose.
    rims = detect_shots.rim_track(frames, detect_shots.GAP_S)
    gaps = detect_shots.distances(frames, rims)
    shots = detect_shots.shots(gaps, detect_shots.APPROACH, detect_shots.FAR,
                               detect_shots.MERGE_S)
    keep = None
    if live_only and clock and Path(clock).exists():
        keep = detect_shots.live_play(json.load(open(clock)).get("readings", []))
    calls = [Call(video_s=float(t), kind="field_goal",
                  asserts=frozenset({"kind"}))
             for t in shots if keep is None or keep(float(t))]
    return Stream(source="ball-to-rim geometry, scripts/detect_shots.py",
                  vision_derived=("ball boxes", "rim boxes")
                  + (("game clock",) if keep is not None else ()),
                  feed_derived=(), calls=tuple(calls))


def scoreboard_stream(readings: str | None, clock: str | None) -> Stream:
    """Typed makes and free throws from the score, when a reader exists.

    `scoreboard_events.py` reports F1 0.918 for any make and 0.864 for free
    throws on a real uncut broadcast -- the best numbers in this project -- and
    NO SCRIPT IN THIS REPOSITORY PRODUCES THE READINGS THOSE CAME FROM.
    `run_broadcast.py` requires `--readings` and nothing writes it. Reporting
    this rung as zero would hide that; it is reported as blocked.
    """
    if not readings or not Path(readings).exists():
        return Stream(
            source="scoreboard_events.py", vision_derived=(), feed_derived=(),
            calls=(), runnable=False,
            blocked_because=(
                "no score/shot-clock readings. scoreboard_events.py records "
                "F1 0.918 for any make and 0.864 for free throws on this kind "
                "of broadcast, but nothing in this repository writes the "
                "readings those numbers were computed from. Supply --readings, "
                "or accept that the one architecture measured to reach 85% is "
                "unmeasured here."))
    from courtvision.scoreboard_events import score_events
    blob = json.load(open(readings))
    calls = []
    for event in score_events(blob.get("readings", [])):
        calls.append(Call(video_s=float(event.get("video_s", event.get("t", 0.0))),
                          kind="field_goal" if event.get("points", 2) > 1
                          else "free_throw",
                          asserts=frozenset({"kind", "made", "points"}),
                          made=True, points=int(event.get("points", 2))))
    return Stream(source="events the scoreboard STATES, scoreboard_events.py",
                  vision_derived=("score digits", "shot clock", "game clock"),
                  feed_derived=(), calls=tuple(calls))


def clock_spans(path: str | None, readable_gap_s: float = 3.0):
    """(span, running s, held s, readable mask, live mask) from the readings.

    TWO MASKS, NOT ONE, and the first is the fix for a real circularity this
    file claimed to prevent and did not.

    `readable` is "the clock could be read near here at all". Official plays
    reach the video ONLY through the clock reader, so truth cannot exist
    outside it -- and an adversarial check found 31 of 286 vision calls on
    Finals G1, and 92 of 317 on G7, lying outside it and counted as false
    alarms. Every one was a miss by construction. Scoring them credited the
    clock-gated mode with suppressing calls in regions the truth could never
    occupy, which is precisely the circular result the module docstring says is
    unprintable. It was worth about +0.05 of the "+0.18 precision from gating".

    `live` is "and the clock was RUNNING", which is the honest comparison,
    because truth occupies both running and held regions.
    """
    if not path or not Path(path).exists():
        return 0.0, 0.0, 0.0, None, None
    readings = json.load(open(path)).get("readings", [])
    if not readings:
        return 0.0, 0.0, 0.0, None, None
    import bisect
    times = sorted(float(r["t"]) for r in readings)
    span = times[-1] - times[0]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import detect_shots
    live = detect_shots.live_play(readings)

    def readable(when: float) -> bool:
        i = bisect.bisect_left(times, when)
        for j in (i - 1, i):
            if 0 <= j < len(times) and abs(times[j] - when) <= readable_gap_s:
                return True
        return False

    step = (span / max(len(times) - 1, 1)) or 1.0
    running = sum(step for t in times if live(t))
    return span, running, span - running, readable, live


def provenance() -> dict:
    def run(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  cwd=Path(__file__).resolve().parent.parent
                                  ).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("git", "rev-parse", "--short", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def report(summary: dict) -> None:
    truth = summary["truth"]
    print(f"\n  score_game_end_to_end.py   game {summary['game_id']}")
    print(f"  truth: {truth['path']}")
    print(f"         {truth['located']} official rows placed on the video clock; "
          f"{truth['scored_plays']} of them are PLAYS")
    dropped = ", ".join(f"{k} {v}" for k, v in
                        sorted(truth["not_scored"].items(), key=lambda kv: -kv[1]))
    print(f"         not scored: {dropped}")
    print(f"         free throws: {truth['ft_attempts']} attempts in "
          f"{truth['ft_trips']} trips; BOTH matching and weighting are on trips")
    clock = summary["clock"]
    if clock["span_s"]:
        print(f"  clock: {clock['span_s']:.0f}s readable, running on "
              f"{clock['running_share']:.1%} of it")
    print(f"  match: coarse class, one-to-one, +/-{summary['tolerance_s']:.1f}s. "
          f"Identity is NOT a matching key.")

    for mode in summary["modes"]:
        print(f"\n  MODE {mode['label']}")
        if not mode["runnable"]:
            print(f"    BLOCKED: {mode['blocked_because']}")
            continue
        print(f"    vision-derived: {', '.join(mode['vision_derived']) or 'nothing'}"
              f"  |  feed-derived: {', '.join(mode['feed_derived']) or 'nothing'}")
        print(f"\n    {'class':<13}{'weight':>8}{'emitted':>9}{'official':>10}"
              f"{'P':>7}{'R':>7}{'F1':>7}")
        for kind, v in sorted(mode["per_class"].items(),
                              key=lambda kv: -kv[1]["weight"]):
            # A class the mode never emits has no precision -- not a precision
            # of zero -- and prints a dash. Its F1 is still 0 and it still
            # carries its full weight, which is the point.
            shown = "-" if not v["emitted"] else format(v["precision"], ".3f")
            print(f"    {kind:<13}{v['weight']:>8.3f}{v['emitted']:>9}"
                  f"{v['official']:>10}{shown:>7}"
                  f"{v['recall']:>7.3f}{v['f1']:>7.3f}")
        low, high = mode["captured_interval"]
        print(f"\n    (a) FRACTION OF THE GAME'S PLAYS CAPTURED   "
              f"event-weighted F1  {mode['captured']:.3f}")
        print(f"        block bootstrap over {BLOCK_S:.0f}s blocks: "
              f"{low:.3f}-{high:.3f}")
        print(f"        architectural coverage {mode['architectural_coverage']:.3f}"
              f" -- this mode cannot emit "
              f"{1 - mode['architectural_coverage']:.0%} of the game's plays at "
              f"all, and those classes keep their full weight above")
        if mode["tautological"]:
            print(f"\n    (b) NOT REPORTED. {mode['label']} says nothing vision "
                  f"found: its event identity,\n        type, outcome and player "
                  f"all come from the official feed, so its precision\n        is "
                  f"1.000 by construction and means only that the feed agrees "
                  f"with itself.\n        Read the timing instead: "
                  f"{mode['timing']}")
        else:
            hits, total = mode["timestamped"], mode["emitted"]
            plow, phigh = wilson(hits, total)
            print(f"\n    (b) FRACTION OF WHAT IT SAYS THAT IS TRUE   "
                  f"{hits}/{total} = {hits / max(total, 1):.3f}"
                  f"  (95% CI {plow:.2f}-{phigh:.2f})")
            print(f"        described (also right on every attribute asserted)  "
                  f"{mode['described']}/{total}")
            print(f"        captured coverage {mode['captured_coverage']:.3f}   "
                  f"architectural coverage {mode['architectural_coverage']:.3f}")
            budget = mode["assertion_budget"]
            print(f"        it asserts an outcome on {budget.get('made', 0):.0%} "
                  f"of its calls and a player on {budget.get('player', 0):.0%}")
        if mode.get("false_calls"):
            fc = mode["false_calls"]
            print(f"\n    false calls per minute   clock running "
                  f"{fc['clock_running_per_min']:.2f}   held "
                  f"{fc['clock_held_per_min']:.2f}")

    if summary["paired"]:
        print("\n  PAIRED, on the official plays both modes were scored against "
              "(exact McNemar):")
        for row in summary["paired"]:
            mark = "significant" if row["p"] < 0.05 else "not significant"
            print(f"    {row['a']} vs {row['b']}   {row['only_a']} plays only "
                  f"{row['a']} captures, {row['only_b']} only {row['b']}   "
                  f"p = {row['p']:.4f}  ({mark}, on {row['shared_plays']} shared "
                  f"plays)")
        print("    precision is NOT compared this way: the modes emit different "
              "events, so\n    there is no shared denominator to pair on. Only "
              "capture is paired.")

    print("\n  CAVEATS, printed with the numbers rather than after them")
    for n, line in enumerate(summary["caveats"], start=1):
        print(f"   {n} {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--detections", default=None)
    parser.add_argument("--clock", default=None)
    parser.add_argument("--readings", default=None)
    parser.add_argument("--tolerance-s", type=float, default=TOLERANCE_S)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    plays, meta = load_truth(args.aligned)
    plays, ft_attempts, ft_trips = free_throw_trips(plays)
    span, running, held, readable, live = clock_spans(args.clock)
    # Every mode is restricted to the readable span. The live mask is applied
    # ON TOP of it, so the only difference between the gated and ungated modes
    # is running versus held -- a comparison the truth can occupy on both sides.
    def gated(when):
        return (readable is None or readable(when)) and (live is None or live(when))
    game_span = (max(p.video_s for p in plays) - min(p.video_s for p in plays)
                 if plays else 1.0)

    modes = [
        ("vision", vision_shot_stream(args.detections or "", args.clock, False),
         readable),
        ("vision+clock", vision_shot_stream(args.detections or "", args.clock,
                                            True), gated),
        ("vision+scoreboard", scoreboard_stream(args.readings, args.clock), gated),
        ("feed-assisted", feed_stream(plays, args.aligned), readable),
    ]
    scored = []
    for label, stream, mask in modes:
        scored.append(score_mode(stream, plays, label=label,
                                 tolerance_s=args.tolerance_s,
                                 span_s=game_span, live_mask=mask,
                                 running_mask=live,
                                 clock_running_s=running, clock_held_s=held))

    errors = sorted(abs(p.align_error_s) for p in plays)
    def pct(q):
        return errors[min(int(q * len(errors)), len(errors) - 1)] if errors else 0.0
    timing = (f"p50 {pct(0.5):.2f}s  p90 {pct(0.9):.2f}s  "
              f"within 1s {sum(1 for e in errors if e <= 1) / max(len(errors), 1):.0%}")

    # Pair on the plays BOTH modes were scored against, keyed by the play
    # itself rather than by position. Two modes under different masks have
    # different denominators, and comparing the vectors by length silently
    # skipped every comparison that mattered -- the only pair that ever printed
    # was vision against feed-assisted, which is trivially significant and
    # answers nothing.
    paired = []
    runnable = [s for s in scored if s.stream.runnable and s.captured_plays]
    for i in range(len(runnable)):
        for j in range(i + 1, len(runnable)):
            a, b = runnable[i], runnable[j]
            shared = sorted(set(a.captured_plays) & set(b.captured_plays))
            if not shared:
                continue
            av = [a.captured_plays[k] for k in shared]
            bv = [b.captured_plays[k] for k in shared]
            only_a, only_b, p = mcnemar(av, bv)
            paired.append({"a": a.label, "b": b.label, "only_a": only_a,
                           "only_b": only_b, "p": p, "shared_plays": len(shared)})

    caveats = [
        "the truth is itself vision-derived: official plays reach the video "
        "through the same\n     clock reader the live-play gate uses, so a few "
        "percent never arrive and the two\n     error sources are correlated. "
        "This is capture WITHIN the span the clock could read.",
        "steal and block ride on the same official row as the turnover or "
        "missed shot they\n     belong to, so they are counted twice in the "
        "denominator. They are declared, not removed.",
        "the live-play rule and the period rule were both written after looking "
        "at these\n     games, so a filtered number on a game they were tuned on "
        "is optimistic.",
        f"free throws: {ft_attempts} attempts in {ft_trips} trips, and BOTH the "
        "matching and the\n     weighting are on trips -- a frozen clock puts a "
        "whole trip at one video second, so a\n     system cannot be asked to "
        "emit two calls a tenth of a second apart. Weighting on\n     attempts "
        "instead would move the headline by about 0.01; an earlier version of "
        "this\n     file claimed attempts and did trips.",
        "the vision mode has no score reader behind it, so it never claims a "
        "make or a miss.\n     It is scored on attempts, and its assertion "
        "budget says so.",
    ]

    summary = {
        "script": "scripts/score_game_end_to_end.py",
        "provenance": provenance(),
        "game_id": args.game_id,
        "tolerance_s": args.tolerance_s,
        "truth": {"path": args.aligned, **meta,
                  "ft_attempts": ft_attempts, "ft_trips": ft_trips},
        "clock": {"span_s": span, "running_share": running / span if span else 0.0},
        "timing": timing,
        "modes": [{"label": s.label, "runnable": s.stream.runnable,
                   "blocked_because": s.stream.blocked_because,
                   "vision_derived": list(s.stream.vision_derived),
                   "feed_derived": list(s.stream.feed_derived),
                   "tautological": not s.stream.vision_derived
                   or bool(s.stream.feed_derived),
                   "timing": timing,
                   "per_class": s.per_class, "captured": s.captured,
                   "captured_interval": s.captured_interval,
                   "timestamped": s.timestamped, "described": s.described,
                   "emitted": s.emitted,
                   "architectural_coverage": s.architectural_coverage,
                   "captured_coverage": s.captured_coverage,
                   "assertion_budget": s.assertion_budget,
                   "false_calls": s.false_calls} for s in scored],
        "paired": paired,
        "reference_points": {
            "tracking_ceiling_event_weighted_f1": 0.751,
            "tracking_ceiling_precision_drop_steal": 0.879,
            "measured_on": "25 Hz SportVU with perfect ball height and stable "
                           "player ids, NOT on video"},
        "caveats": caveats,
    }
    report(summary)
    out = args.out or f"outputs/end_to_end_{args.game_id}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=1, default=str)
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
