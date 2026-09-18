"""How good is the stack on ONE broadcast? Every arm, with intervals.

WHY PER GAME AT ALL. The pooled handler number is 59.2%, and inside it the
detector's miss rate on the three broadcasts is 6.3%, 9.1% and 18.8% -- a
three-fold spread that a single pooled figure cannot show and that changes which
work is worth doing. A number that cannot be attributed to a game cannot tell
you a game regressed.

THE ARMS ARE SPLIT BY WHETHER THEY NEED LABELS, BECAUSE THAT DECIDES WHAT A NEW
BROADCAST GETS ON ARRIVAL.

    label-free   clock coverage, alignment against the official feed, clip
                 placement, and two-path registration consistency
    labelled     ball selection, handler attribution

A broadcast nobody has labelled gets every label-free arm in full, and the
labelled arms print as "no labels" -- which is an interval of 0.00-1.00, not a
score of zero. `stats.wilson` answers an empty denominator with the whole width
for exactly this reason, and this file is the caller that made it matter.

SIZING, STATED BEFORE THE NUMBERS RATHER THAN AFTER. 200 uniform frames per game
resolves a 10-point move as a paired test and nothing smaller. A 5-point per-game
gate would need about 780 frames per game, which is not going to be labelled, so
the 5-point regime stays POOLED but stratified by game, with `--compare` running
a chi-square homogeneity test across games as the alarm. It asks "is one game's
rate different from the others", which is a far more powerful question on this data
than asking each game to carry its own tight interval, and it is the question
"did this game regress" actually is.

THE BAR IS DECLARED HERE, PER ARM, AND SOME ARMS CANNOT REACH IT. `--bar 0.90`
is applied to every arm and the report prints PASS, FAIL, or -- for an arm whose
architecture caps it below the bar -- CAPPED, with the cap. Vision-only play
capture is arithmetically limited to the share of a game's plays that are shots
(0.43 on Finals G1); printing FAIL against 0.90 there would suggest a bug where
there is a design limit, and printing PASS against a lowered bar would be worse.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.games import Broadcast, get, registry  # noqa: E402
from courtvision.stats import homogeneity, wilson  # noqa: E402

#: A clip is placed if the official instant lands within this of where it was cut.
CLIP_TOLERANCE_S = 1.0
#: Two registrations of one instant may differ by this much and still agree. A
#: player moves under 3 ft in the 0.2 s between the two frames, so a
#: disagreement above this is the registration moving, not the sport.
REGISTRATION_GATE_FT = 2.0
#: The clock is "readable near here" within this of a reading.
READABLE_GAP_S = 3.0
#: Overlap at which the detector's handler box is the labelled player. 0.5
#: because that is what `eval_handler.py` has always used, and a per-game arm
#: that scored at 0.3 would print a number nobody could compare to the pooled
#: 49.7% on record.
HANDLER_IOU = 0.5
#: `eval_handler.py` predicts at this floor. The cached detections were written
#: at 0.08, so candidates below it are dropped here rather than silently making
#: this arm a different, more permissive measurement than the one on record.
HANDLER_CONF = 0.25


class Arm:
    """One measured thing: a rate, an interval, and what limits it.

    `cap` is an ARCHITECTURAL ceiling -- the best this arm could do if its model
    were perfect -- and is None when there is no such limit. `n` of zero means
    nothing was measured, which prints as an interval of the whole width.
    """

    def __init__(self, name: str, hits: int, total: int, *, cap: float | None = None,
                 note: str = "", labelled: bool = False, detail: dict | None = None,
                 descriptive: bool = False, weighted: float | None = None):
        self.name, self.hits, self.total = name, hits, total
        self.cap, self.note, self.labelled = cap, note, labelled
        self.detail = detail or {}
        #: A property of the BROADCAST, not of the stack. "The clock was running
        #: on 70% of readings" is a fact about a basketball game -- roughly a
        #: third of a broadcast is dead ball -- and giving it a pass/fail against
        #: an accuracy bar would report the sport as a defect. Descriptors print
        #: their value and no verdict.
        self.descriptive = descriptive
        #: An event-WEIGHTED F1 has no integer numerator and no n. It was
        #: briefly given a fabricated `hits=round(rate*1000), total=1000` so it
        #: could live in an Arm, and that fake denominator printed as "n=1000"
        #: and was written into the report JSON -- a made-up sample size beside
        #: a real one, in a file whose whole purpose is that numbers carry the
        #: n they were taken over. It carries its rate directly instead, prints
        #: no n, and takes its interval from the bootstrap the scorer computed.
        self.weighted = weighted
        self.low, self.high = ((0.0, 1.0) if weighted is not None
                               else wilson(hits, total))

    @property
    def rate(self) -> float | None:
        if self.weighted is not None:
            return self.weighted
        return self.hits / self.total if self.total else None

    def verdict(self, bar: float) -> str:
        if not self.total and self.weighted is None:
            return "NO DATA"
        if self.descriptive:
            return "(descriptive)"
        if self.cap is not None and self.cap < bar:
            # The share of its OWN ceiling this arm reaches. "CAPPED at 0.43"
            # alone read identically for an arm at 0.43 and an arm at 0.01, so
            # a mode performing at 2% of what its architecture allows looked
            # like one performing at its limit.
            share = (self.rate or 0.0) / self.cap if self.cap else 0.0
            return f"CAPPED {share:.0%} of {self.cap:.2f}"
        if self.low >= bar:
            return "PASS"
        if self.rate is not None and self.rate >= bar:
            return "PASS (point)"
        return "FAIL"

    def row(self, bar: float) -> str:
        if not self.total and self.weighted is None:
            return (f"  {self.name:<34}{'--':>9}{'':>9}"
                    f"   0.00-1.00   {'NO DATA':<20}{self.note}")
        # A weighted rate prints no n, because it has none. An empty column is
        # the honest thing there; a number would be an invented sample size.
        count = "" if self.weighted is not None else f"n={self.total}"
        return (f"  {self.name:<34}{self.rate:>9.3f}{count:>9}"
                f"   {self.low:.2f}-{self.high:.2f}   "
                f"{self.verdict(bar):<20}{self.note}")

    def as_dict(self) -> dict:
        return {"name": self.name,
                "hits": None if self.weighted is not None else self.hits,
                "total": None if self.weighted is not None else self.total,
                "rate": self.rate, "low": self.low, "high": self.high,
                "cap": self.cap, "labelled": self.labelled,
                "descriptive": self.descriptive, "weighted": self.weighted,
                "note": self.note,
                **({"detail": self.detail} if self.detail else {})}


# -- label-free arms --------------------------------------------------------

def _official(game: Broadcast) -> list[dict]:
    """This game's official actions, from the cache. Never the network."""
    return json.loads(game.pbp.read_text()) if game.pbp.exists() else []


def clock_arm(game: Broadcast) -> list[Arm]:
    """How much of the broadcast the clock could be read on, and when it ran.

    Not an accuracy: it is the DENOMINATOR every other arm lives inside. Official
    plays reach the video only through this reader, so a play outside the
    readable span is a play no arm can be scored on, and a report that did not
    print this span would be quoting rates over an unstated fraction of a game.
    """
    if not game.clock.exists():
        return [Arm("clock readable", 0, 0, note="no clock read")]
    data = json.loads(game.clock.read_text())
    readings = data.get("readings", data if isinstance(data, list) else [])
    if not readings:
        return [Arm("clock readable", 0, 0, note="clock read produced nothing")]
    times = sorted(float(r["t"]) for r in readings)
    step = float(data.get("step", 1.0)) if isinstance(data, dict) else 1.0
    span = times[-1] - times[0]
    # A game is 48 minutes of clock. What share of THAT the reader saw is the
    # honest coverage question; the share of the video file is not, because a
    # video file is mostly not basketball.
    elapsed = sorted({round(float(r["elapsed"])) for r in readings
                      if r.get("elapsed") is not None})
    periods = sorted({r["period"] for r in readings if r.get("period")})
    # HOW LONG THE GAME WAS COMES FROM THE OFFICIAL FEED, NOT FROM THE READER.
    # Taking it from the periods the reader SAW makes the denominator depend on
    # the numerator: one misread period number in a four-period game adds five
    # minutes of overtime that never happened and drops the coverage figure by
    # seven points, and a reader that missed period 3 entirely would be scored
    # against a shorter game and look better for it.
    official = max((a.get("period") or 0) for a in _official(game)) if \
        game.pbp.exists() else max(periods or [4])
    game_seconds = 720 * 4 + 300 * max(official - 4, 0)
    sys.path.insert(0, str(ROOT / "scripts"))
    import detect_shots
    live = detect_shots.live_play(readings)
    running = sum(1 for t in times if live(t))
    unexpected = [p for p in periods if p > official]
    return [
        Arm("clock: game seconds seen", len(elapsed), game_seconds,
            note=f"periods {periods} against the feed's {official}; "
                 f"{span / 60:.0f} min of video span"
                 + (f"  <-- READER SAW PERIODS THE GAME DID NOT HAVE: "
                    f"{unexpected}" if unexpected else "")),
        Arm("clock: running share", running, len(times), descriptive=True,
            note=f"of {len(times)} readings at {step:g}s steps -- a fact about "
                 f"the game, not an accuracy"),
    ]


def alignment_arm(game: Broadcast) -> tuple[list[Arm], dict[str, list[bool]]]:
    """Does the official feed land where the clock reader says it does?

    Label-free: the official feed IS the truth, and the only thing being
    measured is whether the clock read carries it onto the video. This is the
    arm that clears 90% today, and it is the one a new broadcast gets for free.

    Returns per-action-type right/wrong vectors too, so `--compare` can run
    the homogeneity test across games on them.
    """
    if not game.aligned.exists():
        return [Arm("alignment: overall", 0, 0, note="not aligned")], {}
    data = json.loads(game.aligned.read_text())
    per = data.get("per_action", {})
    # The denominator is LABEL ROWS, not plays. `classify` gives a made shot
    # that was assisted two labels, so 7-9% of these rows are a second label on
    # a play already counted. The rate moves by at most 0.2 points either way --
    # measured on all three broadcasts -- but the n is not a count of plays and
    # printing it as one would be the kind of quiet denominator error this file
    # exists to catch.
    arms = [Arm("alignment: overall",
                sum(v["located"] for v in per.values()),
                sum(v["total"] for v in per.values()),
                note=f"{len(data.get('events', []))} located rows. The n is "
                     f"LABEL rows, ~8% above the play count: a made shot that "
                     f"was assisted carries two labels")]
    vectors: dict[str, list[bool]] = {}
    for action, v in sorted(per.items(), key=lambda kv: -kv[1]["total"]):
        if v["total"] < 5:
            continue
        arms.append(Arm(f"  align {action}", v["located"], v["total"]))
        vectors[action] = [True] * v["located"] + [False] * (v["total"] - v["located"])
    return arms, vectors


def clips_arm(game: Broadcast) -> list[Arm]:
    """Do the cut clips still correspond to the alignment that is on disk?

    THIS IS NOT AN INDEPENDENT ACCURACY AND IT USED TO CLAIM TO BE. `error_s` in
    the clip index is copied straight from the aligner, where it is the gap
    between the event's game clock and the nearest clock READING -- the same
    quantity the alignment arm above is built from. Reported as "clips within
    1.0 s" it read as a second, independent check on placement, and on two of
    the three broadcasts it is exactly `P(error <= 1 | located)`: a conditional
    slice of the arm above it, printed as though it were new evidence.

    What is NOT in the arm above, and is worth an arm of its own, is whether the
    clips were cut from the alignment that is on disk NOW. On Finals G7 they
    were not: 150 of 454 indexed rows are absent from the current
    `aligned_events.json` and 100 currently cut-worthy rows are absent from the
    index, because the clock resolver was fixed after the clips were cut. Two
    rows of one report were being computed against two different truths, with
    nothing saying so.
    """
    if not game.clip_index.exists():
        return [Arm("clips: match the alignment", 0, 0, note="no clips cut")]
    clips = json.loads(game.clip_index.read_text()).get("clips", [])
    if not clips:
        return [Arm("clips: match the alignment", 0, 0,
                    note="index has no clips")]
    cut = sum(1 for c in clips if c.get("clip"))
    within = sum(1 for c in clips
                 if c.get("error_s") is not None
                 and abs(float(c["error_s"])) <= CLIP_TOLERANCE_S)
    arms = [Arm("clips: within 1.0s", within, len(clips), descriptive=True,
                note=f"{cut} of {len(clips)} rows carry footage. NOT an "
                     f"independent check -- this is the alignment arm's own "
                     f"error, conditioned on being located")]
    if not game.aligned.exists():
        return arms
    live = {(round(float(e["video_s"]), 1), e["action"])
            for e in json.loads(game.aligned.read_text()).get("events", [])}
    fresh = sum(1 for c in clips
                if (round(float(c["video_s"]), 1), c["action"]) in live)
    arms.append(Arm("clips: still match the alignment", fresh, len(clips),
                    note=f"indexed rows present in {game.aligned} as it stands "
                         f"now; anything less than 1.000 means the clips were "
                         f"cut from an alignment that has since changed"))
    return arms


def registration_arm(game: Broadcast, samples: int, weights: str,
                     device: str | None) -> list[Arm]:
    """Two independent registrations of one instant must agree. No labels.

    This is the arm that makes "a fourth broadcast gets a report on arrival"
    true: it needs the video and a pose model and nothing a person wrote down.
    Its failure mode is also the one this project has already been burned by --
    coverage bought with hallucinated landmarks -- so the arm reports COVERAGE
    and AGREEMENT separately and never multiplies them into one number.
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO

    sys.path.insert(0, str(ROOT / "scripts"))
    from check_registration_consistency import GAP_S, _register
    from courtvision.court_keypoints import COURT_DETECTION_CONF

    if not Path(weights).exists():
        return [Arm("registration: agreement", 0, 0,
                    note=f"no pose weights at {weights}")]
    capture = cv2.VideoCapture(str(game.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    duration = (capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    model = YOLO(weights)
    lo, hi = 0.10 * duration, 0.95 * duration
    errors, both, tried = [], 0, 0
    for k in range(samples):
        when = lo + (hi - lo) * k / max(samples - 1, 1)
        frames = []
        for offset in (0.0, GAP_S):
            capture.set(cv2.CAP_PROP_POS_MSEC, (when + offset) * 1000.0)
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
        if len(frames) != 2:
            continue
        tried += 1
        a = _register(model, frames[0], device, 0.5, 4, COURT_DETECTION_CONF)
        b = _register(model, frames[1], device, 0.5, 4, COURT_DETECTION_CONF)
        if a is None or b is None:
            continue
        shift = _orb(frames[0], frames[1])
        if shift is None:
            continue
        both += 1
        errors.append(_disagreement(a, b, shift, frames[0].shape))
    capture.release()
    if not errors:
        return [Arm("registration: coverage", both, tried, descriptive=True,
                    note="no instant registered on both paths"),
                Arm("registration: agreement", 0, 0, note="nothing to compare")]
    errors_v = np.array(errors, dtype=float)
    finite = errors_v[np.isfinite(errors_v)]
    p50 = float(np.percentile(finite, 50)) if len(finite) else float("inf")
    p90 = float(np.percentile(finite, 90)) if len(finite) else float("inf")
    agreed = int((errors_v <= REGISTRATION_GATE_FT).sum())
    return [
        Arm("registration: coverage", both, tried, descriptive=True,
            note=f"instants where both paths registered, of {tried} sampled -- "
                 f"read WITH the agreement below, never multiplied into it"),
        Arm("registration: agreement", agreed, len(errors_v),
            note=f"within {REGISTRATION_GATE_FT:g} ft; p50 {p50:.2f} ft, "
                 f"p90 {p90:.2f} ft",
            detail={"p50_ft": p50, "p90_ft": p90}),
    ]


def _orb(a, b):
    """Homography carrying a point in frame a to frame b, or None."""
    import cv2
    import numpy as np
    orb = cv2.ORB_create(nfeatures=4000)
    ka, da = orb.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return matrix


def _disagreement(a, b, shift, shape) -> float:
    """Feet between the two paths' answers for a grid of floor points."""
    import cv2
    import numpy as np
    h, w = shape[:2]
    points = np.float32([[x, y] for x in np.linspace(w * 0.2, w * 0.8, 4)
                         for y in np.linspace(h * 0.55, h * 0.9, 3)]
                        ).reshape(-1, 1, 2)
    direct = cv2.perspectiveTransform(points, a)
    carried = cv2.perspectiveTransform(
        cv2.perspectiveTransform(points, shift), b)
    gaps = np.linalg.norm(direct - carried, axis=2).ravel()
    return float(np.median(gaps))


# -- labelled arms ----------------------------------------------------------

#: The labelling pages computed a frame's time as `start_s + f / 30.0` -- the
#: literal 30.0, not the encode's true 29.97. They use the cache's own rate now,
#: because 30.0 puts every frame of a 60 fps broadcast at twice its real time,
#: but the labels already collected were placed with the literal. Both
#: conventions are tried and the BOXES decide which row is right, so this file
#: does not have to know which page wrote which label.
PAGE_FPS = 30.0
#: Player boxes the pages offered for the handler choice.
PLAYER_CONF = 0.35
#: The two labelling manifests, which record the boxes each page actually drew.
MANIFESTS = ("data/labeling/handler/manifest.json",
             "data/labeling/possession/manifest.json")


def _fingerprints(label: str) -> dict[float, list[list[list[float]]]]:
    """{frame time: the box sets the labeller was shown at that time}.

    A time alone does not identify a frame. Clips overlap and the caches sample
    every second frame at 30 Hz, so 52% to 60% of rounded times hold two to six
    different cache rows -- and keying on time alone scored about a third of
    every labelled frame against a NEIGHBOURING frame's detections. The boxes
    the page drew are recorded in its manifest, and they are a fingerprint.
    """
    out: dict[float, list] = {}
    for name in MANIFESTS:
        path = ROOT / name
        if not path.exists():
            continue
        for row in json.loads(path.read_text())["frames"]:
            if row.get("game") != label:
                continue
            # Sorted here because the two pages store their boxes in different
            # orders -- the handler page sorts by x, the possession page keeps
            # detection order -- and an order-sensitive comparison matched
            # every handler frame and NONE of the possession ones, which reads
            # as "this broadcast has no ball labels".
            out.setdefault(round(float(row["t"]), 1), []).append(
                sorted([list(b) for b in row["boxes"]], key=lambda b: b[0]))
    return out


def _same(a, b, tolerance: float = 0.6) -> bool:
    return len(a) == len(b) and all(
        abs(x - y) <= tolerance for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def _cached_frames(game: Broadcast) -> tuple[dict[float, list], int]:
    """{frame time: the detections the labeller was actually shown}, and misses.

    THE FRAME IS IDENTIFIED BY ITS BOXES, NOT BY ITS TIMESTAMP. See
    `_fingerprints`. A labelled frame whose boxes match no cache row at its time
    is one the cache no longer holds -- the clips were recut, or the detector
    was re-run since -- and it is EXCLUDED and counted, never scored as a miss.
    Scoring it would compare a person's answer about one set of boxes against a
    model's answer about a different set.

    CHECKED AGAINST THE EVALUATOR IT REPLACES. Pooled over the three labelled
    broadcasts on the uniform split, scored the way `eval_handler.py` scores --
    same IoU 0.5, same 0.25 confidence floor, same denominator including the
    frames the detector missed -- this agrees with that script to within a frame,
    and it needs no detector pass to do it.
    """
    if not game.clip_detections.exists() or not game.clip_index.exists():
        return {}, 0
    wanted = _fingerprints(game.label)
    if not wanted:
        return {}, 0
    data = json.loads(game.clip_detections.read_text())
    index = {c["clip"]: c for c in
             json.loads(game.clip_index.read_text())["clips"] if c.get("clip")}
    rates = {PAGE_FPS, float(data.get("fps") or PAGE_FPS)}
    found: dict[float, list] = {}
    for clip, frames in data.get("clips", {}).items():
        start = index.get(clip, {}).get("start_s")
        if start is None:
            continue
        for frame in frames:
            times = {round(float(start) + frame["f"] / rate, 1) for rate in rates}
            times = [t for t in times if t in wanted]
            if not times:
                continue
            when = times[0]
            on = frame.get("on") or []
            people = [b for b in frame["d"] if b[0] in ("p", "h")]
            boxes = sorted([b[2:] for n, b in enumerate(people)
                            if b[1] >= PLAYER_CONF and (n >= len(on) or on[n])],
                           key=lambda b: b[0])
            if any(_same(boxes, w) for w in wanted[when]):
                found[when] = frame["d"]
    return found, len(wanted) - len(found)


def _labels(path: Path, label: str) -> list[dict]:
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text())["frames"]
            if r.get("game") == label]


#: `pick` values that make a frame part of the MEASURING set. Everything else
#: -- "hard" in the possession round, "disagree" in the handler round -- was
#: chosen BECAUSE the model was failing there, and belongs to training.
UNIFORM = ("random",)


def _split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(uniform, hard). Never blended, and the uniform half is the headline.

    POOLING THEM MOVES EVERY LABELLED NUMBER BY 13 TO 30 POINTS, in the
    direction that flatters nothing and misleads everything: the hard half is a
    set of frames selected for the detector having failed on them, so a ceiling
    computed over the pool is a ceiling on a hard-case set wearing an in-game
    label. Measured with the frame matching this file now uses, on every
    labelled broadcast:

                                    pooled   uniform     hard
        handler, winnable   G7       0.368     0.500    0.263
                            G1       0.330     0.564    0.172
                            ECF      0.419     0.657    0.233
        ball, any rank      G7       0.600     0.902    0.318
                            G1       0.613     0.824    0.357
                            ECF      0.812     0.964    0.480

    The pooled column describes no population at all: it is an average over two
    sets in a ratio nobody chose, which changes the moment another labelling
    round is run with a different hard share.
    """
    return ([r for r in rows if r.get("pick") in UNIFORM],
            [r for r in rows if r.get("pick") not in UNIFORM])


def handler_arm(game: Broadcast) -> list[Arm]:
    """Does the detector's handler class name the player a person named?

    THE HEADLINE MIXES TWO PROBLEMS AND THIS SPLITS THEM. A frame where the
    labeller saw the handler and the detector drew NO BOX for him is a
    guaranteed miss for every possession method ever written, and it is a
    measurement of the player detector. Reporting only the pooled rate hides a
    detector-miss rate that runs 6.3% to 18.8% across these three broadcasts.

    AND THE MISS RATE'S DENOMINATOR IS FRAMES WHERE SOMEBODY HAD THE BALL.
    Frames answered "nobody has it" -- a ball in flight, a loose ball, a dead
    ball -- are 40% of the labelled set and there is no handler in them to draw
    or to miss. Counting them made the detector look better than it is.
    """
    frames, unmatched = _cached_frames(game)
    rows = (_labels(ROOT / "data/labels/handler_labels.json", game.label)
            + _labels(ROOT / "data/labels/possession_labels.json", game.label))
    uniform, hard = _split([r for r in rows
                            if (r.get("handler_verdict") or r.get("verdict"))
                            in ("box", "missing", "nobody")])
    if not uniform and not hard:
        return [Arm("handler: detector drew him", 0, 0, labelled=True,
                    note="no handler labels for this broadcast"),
                Arm("handler: winnable frames", 0, 0, labelled=True)]

    def score(rows_in):
        held = [r for r in rows_in
                if (r.get("handler_verdict") or r.get("verdict")) in
                ("box", "missing")]
        missing = sum(1 for r in held
                      if (r.get("handler_verdict") or r.get("verdict")) == "missing")
        right = seen = 0
        for row in held:
            if (row.get("handler_verdict") or row.get("verdict")) != "box" \
                    or not row.get("handler_box"):
                continue
            dets = frames.get(round(float(row["t"]), 1))
            if dets is None:
                continue
            seen += 1
            handlers = [d for d in dets if d[0] == "h" and d[1] >= HANDLER_CONF]
            if handlers and _iou(max(handlers, key=lambda d: d[1])[2:6],
                                 row["handler_box"]) >= HANDLER_IOU:
                right += 1
        return len(held) - missing, len(held), right, seen

    drew, held, right, seen = score(uniform)
    _, _, hard_right, hard_seen = score(hard)
    note = (f"of frames where SOMEBODY had it; "
            f"{held - drew} the detector drew no box for")
    if unmatched:
        note += f"; {unmatched} labels excluded (boxes no longer in the cache)"
    return [
        Arm("handler: detector drew him", drew, held, labelled=True, note=note),
        Arm("handler: winnable frames", right, seen, labelled=True,
            note=f"uniform frames where he HAS a box. On the HARD half, chosen "
                 f"because the model was failing: "
                 f"{hard_right}/{hard_seen} = "
                 f"{hard_right / max(hard_seen, 1):.3f} -- never blended in"),
    ]


def ball_arm(game: Broadcast) -> list[Arm]:
    """Is the ball where a person said it is -- top-1, and at any rank?

    PROPOSED-AT-ANY-RANK IS THE CEILING OF EVERY SELECTOR, TRACKER AND KERNEL
    that will ever be built on this detector. If it is 0.88, no amount of
    selection work can be worth more than 12 points and the effort belongs in
    detection. It costs nothing to print and it re-ranks the roadmap -- which is
    exactly why it has to be computed on the UNIFORM half. Half the possession
    round's frames were drawn because the ball model had no confident candidate
    there, and a ceiling averaged over those is not the ceiling in a game.
    """
    frames, unmatched = _cached_frames(game)
    rows = _labels(ROOT / "data/labels/possession_labels.json", game.label)
    uniform, hard = _split([r for r in rows
                            if r.get("ball_verdict") == "ball" and r.get("ball")])
    if not uniform and not hard:
        return [Arm("ball: top-1", 0, 0, labelled=True,
                    note="no located-ball labels for this broadcast"),
                Arm("ball: proposed at any rank", 0, 0, labelled=True)]

    def score(rows_in):
        top1 = any_rank = seen = 0
        for row in rows_in:
            dets = frames.get(round(float(row["t"]), 1))
            if dets is None:
                continue
            seen += 1
            balls = sorted([d for d in dets if d[0] == "b"], key=lambda d: -d[1])
            radius = max(float(row.get("radius") or 10.0), 10.0)
            hit = [_inside(b[2:6], row["ball"], radius) for b in balls]
            if hit and hit[0]:
                top1 += 1
            if any(hit):
                any_rank += 1
        return top1, any_rank, seen

    top1, any_rank, seen = score(uniform)
    _, hard_any, hard_seen = score(hard)
    note = "uniform frames only"
    if unmatched:
        note += f"; {unmatched} labels excluded (boxes no longer in the cache)"
    return [
        Arm("ball: top-1", top1, seen, labelled=True,
            note=f"most confident candidate is the ball, on {note}"),
        Arm("ball: proposed at any rank", any_rank, seen, labelled=True,
            note=f"THE CEILING of every selector built on this detector. On the "
                 f"HARD half: {hard_any}/{hard_seen} = "
                 f"{hard_any / max(hard_seen, 1):.3f}, never blended in"),
    ]


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _inside(box, point, radius: float) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return math.hypot(cx - point[0], cy - point[1]) <= radius + 6.0


# -- end to end -------------------------------------------------------------

def end_to_end_arms(game: Broadcast) -> list[Arm]:
    """The composed number, if `score_game_end_to_end.py` has been run.

    Read rather than recomputed, because that script is the one place the
    matching rules live and a second implementation of them would eventually
    disagree with the first.
    """
    if not game.end_to_end.exists():
        return [Arm("end to end: captured", 0, 0,
                    note="score_game_end_to_end.py has not run for this game")]
    summary = json.loads(game.end_to_end.read_text())
    arms = []
    for mode in summary.get("modes", []):
        if not mode.get("runnable"):
            continue
        label = mode["label"]
        cap = mode.get("architectural_coverage")
        captured = mode.get("captured") or 0.0
        # Captured is an event-WEIGHTED F1, not a count of anything, so it has
        # no integer numerator. Its interval is the block bootstrap the scorer
        # already computed; an Arm built from a fake denominator would print a
        # Wilson interval that means nothing.
        low, high = mode.get("captured_interval", (0.0, 1.0))
        arm = Arm(f"e2e {label}: captured", 0, 0, cap=cap, weighted=captured)
        arm.low, arm.high = low, high
        arm.note = (f"event-weighted F1, block bootstrap; this mode can emit at "
                    f"most {cap:.0%} of the game's plays" if cap else
                    "event-weighted F1, block bootstrap")
        arms.append(arm)
        if not mode.get("tautological"):
            arms.append(Arm(f"e2e {label}: said and true",
                            mode.get("timestamped", 0), mode.get("emitted", 0),
                            note="precision on the emitted stream"))
    return arms


# -- the report -------------------------------------------------------------

#: Where `eval_kits.py` leaves its per-broadcast result.
KITS_REPORT = "outputs/games/kits.json"


def kits_arm(game: Broadcast) -> list[Arm]:
    """The kit model, scored by a rule of the sport and no labels at all.

    Label-free, so a broadcast nobody has touched gets this in full on arrival
    -- the same property that makes the registration and clock arms worth
    having. It is READ from `eval_kits.py`'s output rather than recomputed,
    because the measurement decodes every clip and this report must stay fast.
    """
    path = ROOT / KITS_REPORT
    if not path.exists():
        return [Arm("kits: never six a side", 0, 0,
                    note=f"{KITS_REPORT} has not been written; run eval_kits.py")]
    rows = {row["game"]: row for row in json.loads(path.read_text())}
    row = rows.get(game.key)
    if row is None or not row.get("fitted"):
        return [Arm("kits: never six a side", 0, 0,
                    note="no kit model for this broadcast")]
    total = int(row["rule_n"])
    hits = round(row["obeys_five_a_side"] * total)
    return [
        Arm("kits: never six a side", hits, total,
            note=f"frames where neither kit was given six players; "
                 f"CIELAB separation {row['separation_lab']:.0f}"),
        Arm("kits: called officials", round(row["rejected_share"] * 1000), 1000,
            descriptive=True,
            note="share of boxes; three officials among thirteen people is "
                 "0.231, and far above it means the rule was bought by "
                 "abstaining rather than earned"),
    ]


def evaluate(game: Broadcast, args) -> dict:
    arms: list[Arm] = []
    arms += clock_arm(game)
    align, vectors = alignment_arm(game)
    arms += align
    arms += clips_arm(game)
    if not args.no_registration:
        try:
            arms += registration_arm(game, args.registration_samples,
                                     args.pose_weights, args.device)
        except Exception as error:                        # pragma: no cover
            arms.append(Arm("registration: agreement", 0, 0,
                            note=f"failed: {type(error).__name__}: {error}"))
    arms += kits_arm(game)
    arms += handler_arm(game)
    arms += ball_arm(game)
    arms += end_to_end_arms(game)
    held = sorted(k for k, v in (game.held_out or {}).items() if v)
    return {"game": game.key, "game_id": game.game_id, "label": game.label,
            "unseen": game.unseen, "bar": args.bar,
            "held_out": held,
            "not_held_out": sorted(k for k, v in (game.held_out or {}).items()
                                   if not v),
            "held_out_note": game.held_out_note,
            "arms": [a.as_dict() for a in arms],
            "alignment_vectors": {k: [int(b) for b in v] for k, v in vectors.items()},
            "_arms": arms}


def print_report(result: dict, bar: float) -> None:
    game = result["label"]
    print(f"\n  eval_by_game.py   {game}  ({result['game']}, "
          f"official {result['game_id']})")
    held, not_held, note = (result["held_out"], result["not_held_out"],
                            result["held_out_note"])
    if held:
        print(f"  HELD OUT for: {', '.join(held)}")
        print("  -- no threshold in this repository was chosen on this "
              "broadcast for those arms\n  and no model was trained on it. "
              "Those numbers are an acceptance test.")
    if not_held:
        print(f"  NOT held out for: {', '.join(not_held)}. Numbers on those arms "
              f"are not\n  acceptance results and must not be quoted as any.")
    if note:
        print(f"  {note}")
    print(f"  bar {bar:.2f}. PASS means the LOWER end of the interval clears it; "
          f"'PASS (point)'\n  means the point estimate does and the interval "
          f"does not. CAPPED means the arm's\n  architecture cannot reach the "
          f"bar however good its model gets, and prints how much\n  of its "
          f"OWN ceiling it reaches.\n")
    print(f"  {'arm':<34}{'rate':>9}{'n':>9}   95% CI      verdict")
    print(f"  {'-' * 84}")
    labelled_seen = False
    for arm in result["_arms"]:
        if arm.labelled and not labelled_seen:
            print("  " + "-- needs labels " + "-" * 68)
            labelled_seen = True
        print(arm.row(bar))
    missing = [a.name for a in result["_arms"]
               if not a.total and a.weighted is None]
    if missing:
        print(f"\n  not measured on this broadcast: {', '.join(missing)}")
        print("  an arm with no data prints 0.00-1.00, which is a claim of no "
              "idea and not\n  a claim of zero. See stats.wilson.")


def compare(results: list[dict], bar: float) -> None:
    """Is one game's alignment different from the others?

    The per-game regression alarm. Three broadcasts of a few hundred rows each
    cannot each carry an interval tight enough to see a five-point move, but
    asking whether ONE game's rate differs from the others is a far more
    powerful question on the same data -- and it is the question "did this game
    regress" actually is.
    """
    print("\n  ACROSS GAMES -- chi-square homogeneity on alignment, "
          "per action type")
    print(f"  {'action':<24}{'games':>7}{'Q':>9}{'df':>5}{'p':>9}   per-game rates")
    actions = defaultdict(dict)
    for r in results:
        for action, vector in r["alignment_vectors"].items():
            actions[action][r["game"]] = [bool(v) for v in vector]
    for action, by_game in sorted(actions.items(),
                                  key=lambda kv: -sum(len(v) for v in kv[1].values())):
        if len(by_game) < 2:
            continue
        q, df, p = homogeneity(list(by_game.values()))
        rates = "  ".join(f"{k} {sum(v) / len(v):.3f}" for k, v in sorted(by_game.items()))
        flag = "  <-- HETEROGENEOUS" if p < 0.05 else ""
        print(f"  {action:<24}{len(by_game):>7}{q:>9.2f}{df:>5}{p:>9.4f}   "
              f"{rates}{flag}")
    print("\n  a significant p means these games do NOT share one rate for that "
          "action, and a\n  pooled figure for it is an average of different "
          "numbers rather than one number\n  measured more precisely.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", action="append", default=[],
                        help="registry key, game id or video stem; repeatable. "
                             "Omit for every registered broadcast.")
    parser.add_argument("--bar", type=float, default=0.90)
    parser.add_argument("--registration-samples", type=int, default=60)
    parser.add_argument("--pose-weights",
                        default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-registration", action="store_true",
                        help="skip the arm that needs the video and a GPU")
    parser.add_argument("--compare", action="store_true",
                        help="also test whether the games evaluated share "
                             "one rate, per action type")
    parser.add_argument("--registry", default=None,
                        help="a games.json other than the shipped one. Exists so "
                             "an integration test can build a broadcast out of "
                             "three small files and check this whole path, "
                             "without which the seam between the driver's "
                             "outputs and this report's inputs is untested.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    keys = args.game or list(registry(args.registry))
    results = []
    for key in keys:
        game = get(key, args.registry)
        result = evaluate(game, args)
        print_report(result, args.bar)
        results.append(result)
    if args.compare and len(results) > 1:
        compare(results, args.bar)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [{k: v for k, v in r.items() if k != "_arms"} for r in results]
        path.write_text(json.dumps(payload[0] if len(payload) == 1 else payload,
                                   indent=1))
        print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
