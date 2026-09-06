"""Offensive and defensive structure, from court positions.

Everything here needs positions in feet, which only became trustworthy once the
registration reached 1.72 ft median / 2.43 ft at p75. At the earlier 7-10 ft
error none of these quantities meant anything: a defender 4 ft away and one
12 ft away were indistinguishable, so "contested" could not be measured at all.

Deliberately NOT here: anything requiring player identity. Jersey OCR does not
exist in this project, so these functions describe configurations -- how many
defenders are near, how spread the offense is, whether the floor is balanced --
and never who was where. A coaching tool may say "the nearest defender was
4.2 ft away"; it must not say a name it does not know.

Team assignment is by SIDE OF BALL, not by jersey colour: the five players
nearest the basket being attacked are treated as the defense. That is an
approximation and is stated as one wherever it matters.

CALLERS MUST DETECT AT conf 0.25, NOT HIGHER. Splitting five-and-five requires
most of the ten to be found, and the detector's threshold decides that:

    conf 0.35    6 players on court    8% of frames reach 8
    conf 0.25    9 players on court   75%
    conf 0.15   14 players on court  100%   (referees and false positives)

Run at 0.35 first, this layer reported a 13.9 ft median nearest defender
against a real 4-5 ft, called 42 of 61 shots open, and inverted the
open-versus-contested relationship -- all downstream of splitting five-and-five
when only six people had been found.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from courtvision.court import BASKET, COURT_WIDTH, FREE_THROW_LINE_Y

COURT_LENGTH_FT = 94.0
# Inside this, a defender is contesting rather than merely nearby. NBA tracking
# work has used 4 ft for "tightly contested" and 6 ft for "open"; both are
# reported so a caller can choose.
CONTEST_FT = 4.0
OPEN_FT = 6.0
# The paint, for counting help defenders.
PAINT_HALF_WIDTH_FT = 8.0
PAINT_DEPTH_FT = 19.0
# Corner threes sit 22 ft from the basket, everything else 23.75.
THREE_POINT_RADIUS_FT = 23.75
CORNER_THREE_X_FT = 3.0


@dataclass(frozen=True)
class Spacing:
    """How the offense occupies the floor at one moment."""

    players: int
    width_ft: float
    depth_ft: float
    nearest_pair_ft: float
    in_paint: int


def distance_to_basket(points: np.ndarray) -> np.ndarray:
    """Feet from each court position to the basket being attacked."""
    points = np.asarray(points, dtype=float)
    return np.hypot(points[:, 0] - BASKET[0], points[:, 1] - BASKET[1])


def is_three_point_distance(point) -> bool:
    """Whether a spot is beyond the arc, corners included."""
    x, y = float(point[0]), float(point[1])
    if x <= CORNER_THREE_X_FT or x >= COURT_WIDTH - CORNER_THREE_X_FT:
        # In the corners the line is a straight 22 ft, not the arc.
        return float(np.hypot(x - BASKET[0], y - BASKET[1])) >= 22.0
    return float(np.hypot(x - BASKET[0], y - BASKET[1])) >= THREE_POINT_RADIUS_FT


def in_paint(points: np.ndarray) -> np.ndarray:
    """Mask of positions inside the painted lane."""
    points = np.asarray(points, dtype=float)
    return ((np.abs(points[:, 0] - COURT_WIDTH / 2.0) <= PAINT_HALF_WIDTH_FT)
            & (points[:, 1] >= 0.0) & (points[:, 1] <= PAINT_DEPTH_FT))


def defenders_near(spot, defenders: np.ndarray,
                   radius_ft: float = CONTEST_FT) -> int:
    """How many defenders are within `radius_ft` of a spot."""
    if defenders is None or len(defenders) == 0:
        return 0
    defenders = np.asarray(defenders, dtype=float)
    return int((np.hypot(defenders[:, 0] - spot[0],
                         defenders[:, 1] - spot[1]) <= radius_ft).sum())


def nearest_defender_ft(spot, defenders: np.ndarray) -> float | None:
    """Distance to the closest defender, or None when none are known."""
    if defenders is None or len(defenders) == 0:
        return None
    defenders = np.asarray(defenders, dtype=float)
    return float(np.hypot(defenders[:, 0] - spot[0],
                          defenders[:, 1] - spot[1]).min())


def contest_level(spot, defenders: np.ndarray) -> str:
    """'open', 'guarded' or 'contested' -- the vocabulary a coach uses."""
    nearest = nearest_defender_ft(spot, defenders)
    if nearest is None:
        return "unknown"
    if nearest >= OPEN_FT:
        return "open"
    if nearest >= CONTEST_FT:
        return "guarded"
    return "contested"


def spacing(offense: np.ndarray) -> Spacing | None:
    """How well the offense is spread.

    Width and depth are the extents actually occupied; `nearest_pair_ft` is the
    tightest pair, which is what "bunched up" means concretely.
    """
    if offense is None or len(offense) < 2:
        return None
    offense = np.asarray(offense, dtype=float)
    gaps = []
    for index in range(len(offense)):
        others = np.delete(offense, index, axis=0)
        gaps.append(np.hypot(others[:, 0] - offense[index, 0],
                             others[:, 1] - offense[index, 1]).min())
    return Spacing(
        players=len(offense),
        width_ft=float(offense[:, 0].max() - offense[:, 0].min()),
        depth_ft=float(offense[:, 1].max() - offense[:, 1].min()),
        nearest_pair_ft=float(min(gaps)),
        in_paint=int(in_paint(offense).sum()),
    )


def help_defenders(spot, defenders: np.ndarray,
                   shooter_radius_ft: float = OPEN_FT) -> int:
    """Defenders in the paint who are NOT guarding this spot.

    This is what "the help sank" means: bodies committed to the rim while the
    ball is somewhere else, which is the reason a shooter ends up open.
    """
    if defenders is None or len(defenders) == 0:
        return 0
    defenders = np.asarray(defenders, dtype=float)
    away = np.hypot(defenders[:, 0] - spot[0],
                    defenders[:, 1] - spot[1]) > shooter_radius_ft
    return int((in_paint(defenders) & away).sum())


def split_by_side(points: np.ndarray,
                  ball_spot=None) -> tuple[np.ndarray, np.ndarray]:
    """Split people into (offense, defense) WITHOUT jersey colour.

    The five nearest the attacked basket are taken as the defense. This is an
    approximation -- a rolling big or a crashing offensive rebounder breaks it
    -- and callers must treat the split as a heuristic, not a fact. It exists
    because jersey OCR does not, and it fails most often in exactly the
    situations a coach cares about, so it is reported alongside its own
    uncertainty rather than silently.
    """
    if points is None or len(points) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    points = np.asarray(points, dtype=float)
    if len(points) <= 5:
        return points, np.empty((0, 2))
    order = np.argsort(distance_to_basket(points))
    defense = points[order[:5]]
    offense = points[order[5:]]
    if ball_spot is not None and len(offense):
        # The ball handler is offense by definition; if he sorted into the
        # defensive five, move him across.
        ball_spot = np.asarray(ball_spot, dtype=float)
        gaps = np.hypot(defense[:, 0] - ball_spot[0],
                        defense[:, 1] - ball_spot[1])
        if gaps.min() < 3.0:
            index = int(np.argmin(gaps))
            offense = np.vstack([offense, defense[index]])
            defense = np.delete(defense, index, axis=0)
    return offense, defense


# ---------------------------------------------------------------------------
# Finding the release.
#
# Everything above is measured AT a moment, so choosing the wrong moment
# invalidates all of it. The play-by-play clock trails the action -- it is
# logged after the play resolves -- so sampling at the recorded shot time, or a
# fixed offset from it, lands on a frame where nobody is at the shot location
# yet. Players cover 10-15 ft/s, and measuring defenders around an empty patch
# of floor produced a 13.7 ft "nearest defender" against a real 4-5 ft, and
# made tight defense appear to HELP shooting.
#
# The release frame is the one where somebody is actually standing at the shot
# location the feed reports. Selecting on that is not circular for what is
# measured afterwards: selection uses the distance from a PLAYER to the shot
# spot; the measurement is that player's distance to the NEAREST OTHER player.
#
# With this, the nearest player to the shooter measures p25 2.8, p50 4.7,
# p75 8.4 ft -- which is what NBA tracking reports.


def pick_release(frames: "list[tuple[float, np.ndarray]]",
                 shot_spot) -> "tuple[float, np.ndarray, int] | None":
    """Choose the frame at which the shot was released.

    `frames` is [(time, positions)] over a window around the logged shot time,
    positions being (N, 2) court coordinates. `shot_spot` is the feed's court
    location for the shot. Returns (time, positions, shooter_index) for the
    frame where someone stands closest to that spot, or None.
    """
    spot = np.asarray(shot_spot, dtype=float)
    best = None
    for time, points in frames:
        if points is None or len(points) == 0:
            continue
        points = np.asarray(points, dtype=float)
        gaps = np.hypot(points[:, 0] - spot[0], points[:, 1] - spot[1])
        index = int(np.argmin(gaps))
        if best is None or gaps[index] < best[0]:
            best = (float(gaps[index]), time, points, index)
    if best is None:
        return None
    _, time, points, index = best
    return time, points, index


def shot_context(points: np.ndarray, shooter_index: int) -> dict:
    """What was true around the shooter at the release.

    Anonymous by construction: describes the configuration, never who was in
    it. The nearest other player is reported without asserting he is a defender
    -- with no jersey identity a team-mate setting a screen looks identical.
    """
    points = np.asarray(points, dtype=float)
    if len(points) <= shooter_index:
        return {}
    shooter = points[shooter_index]
    others = np.delete(points, shooter_index, axis=0)
    record = {
        "people_on_floor": int(len(points)),
        "shooter_ft_from_basket": round(float(np.hypot(
            shooter[0] - BASKET[0], shooter[1] - BASKET[1])), 1),
        "three_point_range": bool(is_three_point_distance(shooter)),
    }
    if len(others):
        gaps = np.hypot(others[:, 0] - shooter[0], others[:, 1] - shooter[1])
        record["nearest_other_ft"] = round(float(gaps.min()), 1)
        record["within_4ft"] = int((gaps <= CONTEST_FT).sum())
        record["within_6ft"] = int((gaps <= OPEN_FT).sum())
        record["in_paint_away_from_ball"] = int(
            (in_paint(others) & (gaps > OPEN_FT)).sum())
    return record


# ---------------------------------------------------------------------------
# Teams by jersey, not by distance to the basket.
#
# split_by_side takes the five nearest the rim as the defense, which is wrong
# whenever a big rolls, a guard drives, or an offensive rebounder crashes --
# exactly the moments a coach cares about. The jerseys say it directly: ten
# players wear two colours, five each.
#
# Clustering torso colour into two groups is the reliable half. Deciding WHICH
# group is on offense is the part that has to survive a live game, where the
# play-by-play feed is not available yet, so it is kept separate and given
# several fallbacks in `offense_first`.

MIN_TORSO_PIXELS = 40
# Minimum CIELAB distance between two kits before they count as two teams. Two
# NBA teams never wear colours closer than this; noise within one kit does.
SEPARATION_FLOOR_LAB = 12.0


def torso_colours(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Mean CIELAB colour of each player's jersey region.

    LAB because Euclidean distance in it tracks perceived difference, which is
    what separates two teams' kits; RGB does not.
    """
    import cv2

    out = []
    height, width = image.shape[:2]
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=float)[:, :4]:
        box_w, box_h = x2 - x1, y2 - y1
        a = int(max(0, x1 + 0.25 * box_w))
        b = int(max(0, y1 + 0.20 * box_h))
        c = int(min(width, x1 + 0.75 * box_w))
        d = int(min(height, y1 + 0.55 * box_h))
        if c - a < 2 or d - b < 2:
            out.append(np.full(3, np.nan))
            continue
        crop = image[b:d, a:c]
        if crop.size < MIN_TORSO_PIXELS:
            out.append(np.full(3, np.nan))
            continue
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        out.append(lab.reshape(-1, 3).mean(axis=0).astype(float))
    return np.vstack(out) if out else np.empty((0, 3))


def split_by_jersey(image: np.ndarray, boxes: np.ndarray,
                    positions: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Split players into two teams by jersey colour.

    Returns (team_one, team_two) as court positions, or None when the colours
    do not separate cleanly. Refusing is deliberate: a wrong team split turns
    every defensive statement into its opposite, so "unknown" is the honest
    answer when the evidence is weak.

    Referees are the known contaminant -- striped kit, and a third colour --
    so the two clusters are required to be roughly balanced before being
    trusted.
    """
    colours = torso_colours(image, boxes)
    positions = np.asarray(positions, dtype=float)
    usable = np.isfinite(colours).all(axis=1)
    if usable.sum() < 6 or len(positions) != len(colours):
        return None
    values = colours[usable]
    points = positions[usable]

    # Two-means on three dimensions, by hand: the split is one-dimensional in
    # practice and this avoids a scikit-learn dependency in the hot path.
    spread = values.std(axis=0)
    axis = int(np.argmax(spread))
    order = np.argsort(values[:, axis])
    ordered = values[order]
    best = None
    # From 1, not 2. Starting at 2 makes the correct split unreachable when one
    # kit has a single representative: the best cut then lands INSIDE the large
    # group, mixing the two colours and passing every downstream guard.
    for cut in range(1, len(ordered)):
        left, right = ordered[:cut], ordered[cut:]
        within = (left.var(axis=0).sum() * len(left)
                  + right.var(axis=0).sum() * len(right))
        balance = abs(len(left) - len(right))
        if best is None or (within, balance) < best[0]:
            best = ((within, balance), cut)
    if best is None:
        return None
    cut = best[1]
    left, right = ordered[:cut], ordered[cut:]
    # Balance alone does not prove two kits are present. Given seven players in
    # one colour and one in another, the best cut lands INSIDE the large group
    # and returns a tidy-looking 6-2 split of a single team. The clusters must
    # also be further apart than they are wide.
    between = float(np.linalg.norm(left.mean(axis=0) - right.mean(axis=0)))
    within = float(np.sqrt(left.var(axis=0).sum()) + np.sqrt(right.var(axis=0).sum()))
    if between < max(SEPARATION_FLOOR_LAB, within):
        return None
    one = points[order[:cut]]
    two = points[order[cut:]]
    # Ten players, five a side: a split more lopsided than 7-3 is a colour
    # failure, not a formation.
    if min(len(one), len(two)) < max(2, int(0.3 * (len(one) + len(two)))):
        return None
    return one, two


def offense_first(team_one: np.ndarray, team_two: np.ndarray,
                  ball_spot=None) -> tuple[np.ndarray, np.ndarray]:
    """Order two teams as (offense, defense).

    With a ball position, the team holding it is the offense -- the only
    definition that is always true, and the one a LIVE system must use because
    the play-by-play feed has not arrived yet.

    Without one, falls back to mean distance from the attacked basket: the
    defense sits nearer its own rim. That is a tendency rather than a law, so
    a caller that needs certainty should supply the ball.
    """
    if ball_spot is not None:
        spot = np.asarray(ball_spot, dtype=float)
        near_one = (np.hypot(team_one[:, 0] - spot[0],
                             team_one[:, 1] - spot[1]).min()
                    if len(team_one) else np.inf)
        near_two = (np.hypot(team_two[:, 0] - spot[0],
                             team_two[:, 1] - spot[1]).min()
                    if len(team_two) else np.inf)
        return ((team_one, team_two) if near_one <= near_two
                else (team_two, team_one))
    mean_one = (distance_to_basket(team_one).mean()
                if len(team_one) else np.inf)
    mean_two = (distance_to_basket(team_two).mean()
                if len(team_two) else np.inf)
    return ((team_one, team_two) if mean_one >= mean_two
            else (team_two, team_one))


# Ten players are on the floor and no more. Everything else a detector finds
# inside the court bounds -- three referees, a coach who stepped out, a player
# waiting to be subbed at the scorer's table -- is not in the play.
#
# Measured before enforcing this, the jersey split succeeded on 96.3% of frames
# but produced 6v4, 8v4, 7v6 and 5v3 far more often than 5v5 (3 of 26). The
# clustering was right; the roster was not.
TEAM_SIZE = 5


def enforce_five(team_one: np.ndarray, team_two: np.ndarray,
                 focus) -> tuple[np.ndarray, np.ndarray]:
    """Trim each side to the five closest to the action.

    `focus` is where the play is -- the ball if it is known, otherwise the
    centroid of everyone on court. Referees drift away from it, which is what
    makes this work without ever identifying one.
    """
    focus = np.asarray(focus, dtype=float)

    def nearest_five(team):
        if len(team) <= TEAM_SIZE:
            return team
        gaps = np.hypot(team[:, 0] - focus[0], team[:, 1] - focus[1])
        return team[np.argsort(gaps)[:TEAM_SIZE]]

    return nearest_five(team_one), nearest_five(team_two)


def teams_at(image: np.ndarray, boxes: np.ndarray, positions: np.ndarray,
             ball_spot=None) -> "tuple[np.ndarray, np.ndarray] | None":
    """(offense, defense) at one moment, five a side, from jerseys and the ball.

    This is the function a live system calls. It needs no play-by-play feed:
    jerseys give the two teams, and the ball says which of them is attacking.
    Both are available from the current frame alone.

    Returns None when the jerseys do not separate -- a wrong team split inverts
    every defensive statement made afterwards, so "unknown" is the safe answer.
    """
    split = split_by_jersey(image, boxes, positions)
    if split is None:
        return None
    focus = (ball_spot if ball_spot is not None
             else np.asarray(positions, dtype=float).mean(axis=0))
    one, two = enforce_five(split[0], split[1], focus)
    return offense_first(one, two, ball_spot)


# ---------------------------------------------------------------------------
# Naming players.
#
# Jersey OCR was measured on this broadcast and is not usable: of 244 jersey
# crops, easyocr returned a number on 8.6%, and only 66.7% of those were on
# either roster -- an effective 5.7% correct, with the single most-read number
# not belonging to any player in the game. The limit is resolution: 720p
# source, players about 160 px tall, numbers about 50 px. It would need 1080p+
# or a digit-specific model, and voting across a track cannot rescue a 5.7%
# base rate.
#
# For a RECORDED game none of that matters, because the feed names the shooter
# and gives his court location. Pairing that with the release frame attributes
# a name to a detected player without reading anything.
#
# For a LIVE game it does matter, and this is the honest gap: until the feed
# arrives, players are anonymous. The tactical functions were written to work
# anonymously for exactly this reason.


def name_shooter(positions: np.ndarray, shooter_index: int,
                 shot_spot, player_name: str,
                 max_gap_ft: float = 6.0) -> "dict | None":
    """Attach a fed name to a detected player at the release.

    Returns None when the nearest player is too far from the reported shot
    location to be the shooter -- attributing an action to the wrong player is
    the one error a coaching tool cannot make, so distance has to be checked
    rather than assumed.
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) <= shooter_index or not player_name:
        return None
    spot = np.asarray(shot_spot, dtype=float)
    where = positions[shooter_index]
    gap = float(np.hypot(where[0] - spot[0], where[1] - spot[1]))
    if gap > max_gap_ft:
        return None
    return {"name": player_name,
            "court_xy": [round(float(where[0]), 1), round(float(where[1]), 1)],
            "gap_to_reported_ft": round(gap, 1)}
