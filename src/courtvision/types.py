"""Data structures that cross stage boundaries. Geometry only, no pipeline logic."""

from __future__ import annotations

from dataclasses import dataclass

PLAYER = "player"
BALL = "ball"
RIM = "rim"
# A player the detector judges to be controlling the ball. Distinct from PLAYER
# so stage 5 can use a learned signal instead of a pure distance heuristic.
HANDLER = "handler"
CLASSES = (PLAYER, BALL, RIM, HANDLER)

# The spec (§1) names dribble/pass/shot/rebound/other. `block` and `steal` were
# added on request. Data reality, recorded so nobody re-derives it:
#   dribble 3,490 | pass 1,070 | block 996 | shot 426 | other plenty  (SpaceJam, clean)
#   rebound  — SpaceJam has none; BARD has 227 headline clips, cross-domain
#   steal    — BARD has 435 clips that are exactly {Steal, Turnover}. That pair is
#              ONE steal seen from both sides (different players in 891 of 929
#              cases), not two confounded actions.
ACTIONS = ("dribble", "pass", "shot", "rebound", "block", "steal", "other")
TEAMS = ("A", "B")


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates, top-left origin."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class Detection:
    """A single-frame detection, before tracking assigns an identity."""

    box: Box
    label: str
    conf: float


@dataclass(frozen=True)
class Track:
    """A detection carrying an identity. track_id is -1 for non-player classes."""

    track_id: int
    box: Box
    label: str
    conf: float


@dataclass(frozen=True)
class Frame:
    """Everything known about one sampled frame."""

    index: int
    time_s: float
    tracks: tuple[Track, ...]

    def players(self) -> tuple[Track, ...]:
        """Every person on court, whether or not they hold the ball."""
        return tuple(t for t in self.tracks if t.label in (PLAYER, HANDLER))

    def handler(self) -> Track | None:
        """The tracked player the detector marked as controlling the ball."""
        handlers = [t for t in self.tracks if t.label == HANDLER]
        if not handlers:
            return None
        return max(handlers, key=lambda t: t.conf)

    def ball(self) -> Track | None:
        balls = [t for t in self.tracks if t.label == BALL]
        if not balls:
            return None
        return max(balls, key=lambda t: t.conf)


@dataclass(frozen=True)
class ActionWindow:
    """A classified span of frames. Bounds are inclusive."""

    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float
    label: str
    conf: float


@dataclass(frozen=True)
class Event:
    """A discrete play-by-play event; the input to commentary generation.

    `player_name` is filled in by courtvision.enrichment when official
    play-by-play data is available and aligns confidently. It stays None
    otherwise, and commentary then refers to the anonymous `Player <track_id>` —
    naming a real person on a guess is worse than not naming them.
    """

    time_s: float
    track_id: int | None
    team: str | None
    action: str
    possession_change: bool
    player_name: str | None = None


def clip_source(path) -> str:
    """Which corpus a labeled action clip came from, per CLIP not per class.

    Attribution used to be `action in ("rebound", "steal") -> BARD`, which was
    correct only while each class drew from exactly one corpus. Once `shot` and
    `other` are populated from both — the fix for the source/label confound —
    that rule silently mislabels every BARD shot as SpaceJam and reports zero
    cross-source confusions no matter what the model does.

    SpaceJam names clips by zero-padded index (`0000053.mp4`, `0000026_flipped`);
    BARD names them after the game (`bkn-vs-det-0022400861__284.mp4`).
    """
    stem = getattr(path, "stem", None) or str(path).rsplit("/", 1)[-1].split(".")[0]
    return "SpaceJam" if stem.removesuffix("_flipped").isdigit() else "BARD"


def balanced_subset(clips, limit: int) -> list:
    """The first `limit` clips, drawn evenly from each corpus.

    Plain truncation is corpus-ordered: SpaceJam names are zero-padded digits
    and sort before BARD's game-derived names, so `clips[:limit]` on a class
    populated from both corpora returns SpaceJam only. That silently empties
    the cross-source comparison the subset exists to make, while still printing
    a confident per-class accuracy.
    """
    clips = sorted(clips)
    if not limit or limit >= len(clips):
        return clips
    groups: dict[str, list] = {}
    for clip in clips:
        groups.setdefault(clip_source(clip), []).append(clip)
    ordered = [groups[key] for key in sorted(groups)]
    picked: list = []
    for index in range(max((len(g) for g in ordered), default=0)):
        for group in ordered:
            if index < len(group) and len(picked) < limit:
                picked.append(group[index])
        if len(picked) >= limit:
            break
    return sorted(picked)



def stratified_split(per_class: dict, val_fraction: float = 0.2, seed: int = 0):
    """Split each class independently, so one class's size cannot move another's.

    The previous split concatenated every class into one list in ACTIONS order
    and shuffled the whole thing. That made the split fragile in a way that
    silently invalidated comparisons: changing rebound from 226 clips to 223
    shifted every sample ordered AFTER rebound into different positions, so
    block, steal and other were reshuffled while dribble, pass and shot — which
    come before it — kept identical validation sets. Only 15 of 79 block
    validation clips survived a change to a different class, and a per-class
    accuracy that looks like a 23-point regression is then partly a different
    set of clips.

    Splitting per class also makes the validation set stratified, so a rare
    class cannot land mostly on one side by luck.

    Returns (train, val) as lists of (path, label_index).
    """
    import random as _random

    train: list = []
    val: list = []
    for label_index, action in enumerate(sorted(per_class)):
        clips = sorted(per_class[action])
        _random.Random(seed + label_index).shuffle(clips)
        cut = int(len(clips) * (1 - val_fraction))
        train.extend((c, label_index) for c in clips[:cut])
        val.extend((c, label_index) for c in clips[cut:])
    return train, val
