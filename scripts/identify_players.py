"""Name the players in each clip, so a question about one can box that one.

Asking "how many shots did Jalen Williams make" and getting a list is half an
answer; boxing him in the footage is the other half. That needs identity, and
identity on a broadcast is the hardest thing in this repository.

WHAT MAKES IT POSSIBLE AT ALL is that a clip is six seconds. Per frame the
digit reader answers on about a fifth of torso crops -- players turn, arms
cross the chest, the number blurs -- but a player is on screen for the whole
clip, so thirty frames give a handful of reads to vote over. `JerseyVoter` is
built for exactly that.

TEAM COLOUR IS NOT OPTIONAL, and skipping it is how a first attempt read every
player as an OKC player. Numbers repeat across teams: in this game #2 is both
Nembhard and Gilgeous-Alexander, #9 is both McConnell and Caruso. A number
alone is ambiguous; a number plus a kit is not. `kit_members` splits the boxes
into the two kits and each cluster is assigned to whichever team's roster its
reads actually match -- so the mapping comes from the reads themselves rather
than from a colour I picked.

TRACKLETS, NOT TRACKS. The repository's tracker is known to fragment badly on
broadcast -- 463 identities for ten players over five minutes. Over six seconds
that does not matter: linking each box to the nearest box in the next frame by
overlap is enough to keep a player together long enough to vote, and a tracklet
that breaks simply votes twice.

WHAT THIS CANNOT DO, MEASURED: it does not work well enough to use. On 25
clips whose play-by-play names a rostered player -- so we know who is on
screen -- that player was among the identified names in 4 of them, 16%. The
names it does produce are frequently wrong: the clip of Dort's foul came back
with Nesmith, Nembhard and Holmgren.

The chain is why. The digit reader answers on 13% of crops here, sampling at
0.2 s gives about thirty crops a clip across ten players, and the kit split
then has to be right as well. Three weak stages multiply.

THE DENSER SAMPLING WAS THEN RUN AND MEASURED, by `clip_detect_dense.py`:
every frame detected at 30 fps with a jersey read every third frame, 16,384
crops over 30 clips against 8,468 over 25.

    5 fps sampling    the named player identified in   4/25   (16%)
    30 fps sampling                                   13/29   (45%)

Nearly three times better, and still not enough to draw a name on a box. More
than half the time the player the play-by-play names is not among those
identified, and the names that ARE assigned have no measured precision -- that
would need per-player ground-truth boxes, which do not exist here.

What the second measurement settles is WHERE the limit is: 16% of crops give a
number at all, and that is the digit reader on 720p broadcast torsos, not the
linking or the voting. Beating it needs a better reader, not more frames --
the same answer the ball detection reached, and for the same reason.

So the page draws the detector's boxes WITHOUT names, and says so.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

STEP_S = 0.2
#: Two boxes in consecutive frames are the same player above this overlap.
LINK_IOU = 0.35
#: A tracklet needs this many agreeing reads before it is named.
MIN_VOTES = 2
#: ...and the winner must beat the runner-up by this much.
MIN_MARGIN = 2


def iou(a, b):
    """Overlap of two [x1, y1, x2, y2] boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    area_b = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def link(frames, min_iou=LINK_IOU):
    """Assign a tracklet id to every box.

    `frames` is [(t, [box, ...]), ...]. Returns the same shape with an id
    alongside each box. Greedy nearest-overlap against the previous frame,
    which is all six seconds needs.
    """
    out, previous, next_id = [], [], 0
    for t, boxes in frames:
        assigned, taken = [], set()
        for box in boxes:
            best, best_score = None, min_iou
            for pid, pbox in previous:
                if pid in taken:
                    continue
                score = iou(box, pbox)
                if score >= best_score:
                    best, best_score = pid, score
            if best is None:
                best = next_id
                next_id += 1
            taken.add(best)
            assigned.append((best, box))
        out.append((t, assigned))
        previous = assigned
    return out


def team_for_clusters(reads_by_cluster, roster):
    """Map each kit cluster to the team its reads actually match.

    `reads_by_cluster` is {cluster: Counter(number)}. The cluster is given to
    whichever team's roster explains more of its reads, so the colour never has
    to be named in advance.
    """
    out = {}
    for cluster, reads in reads_by_cluster.items():
        score: Counter = Counter()
        for number, n in reads.items():
            for team in roster.get(number, {}):
                score[team] += n
        out[cluster] = score.most_common(1)[0][0] if score else None
    return out


def name_tracklets(votes, teams_by_tracklet, roster,
                   min_votes=MIN_VOTES, margin=MIN_MARGIN):
    """{tracklet: name} for the tracklets the evidence actually supports."""
    named = {}
    for tid, reads in votes.items():
        team = teams_by_tracklet.get(tid)
        if team is None:
            continue
        allowed = Counter({n: c for n, c in reads.items()
                           if team in roster.get(n, {})})
        if not allowed:
            continue
        ranked = allowed.most_common(2)
        number, count = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        if count < min_votes or count - runner_up < margin:
            continue
        named[tid] = roster[number][team]
    return named


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--roster", required=True, help="number -> {team: name}")
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    import numpy as np

    from courtvision.candidates import kit_members
    from courtvision.digit_net import DigitReader
    from courtvision.jersey import jersey_crop

    roster = json.load(open(args.roster))
    cache = json.load(open(args.detections))
    by_time = {round(r["t"], 1): r["boxes"] for r in cache["frames"]}
    reader = DigitReader("models/jersey_digits.pt")

    clips = {}
    for path in args.index:
        for c in json.load(open(path))["clips"]:
            clips.setdefault(c["clip"], round(float(c.get("start_s",
                             float(c["video_s"]) - 3.0)), 1))
    names = sorted(clips)
    if args.limit:
        names = names[:args.limit]

    capture = cv2.VideoCapture(args.video)
    out, read_count, crop_count = {}, 0, 0
    for n, clip in enumerate(names):
        start = clips[clip]
        frames = []
        for i in range(int(round(args.duration / STEP_S))):
            t = round(start + i * STEP_S, 1)
            boxes = [b["xyxy"] for b in by_time.get(t, [])
                     if b["cls"] in ("player", "handler")]
            if boxes:
                frames.append((round(i * STEP_S, 1), boxes))
        if not frames:
            continue

        linked = link(frames)
        votes: dict[int, Counter] = defaultdict(Counter)
        kit_of: dict[int, Counter] = defaultdict(Counter)
        cluster_reads: dict[int, Counter] = defaultdict(Counter)

        for offset, assigned in linked:
            capture.set(cv2.CAP_PROP_POS_MSEC, (start + offset) * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            boxes = np.array([b for _, b in assigned], float)
            inkit = kit_members(frame, boxes)
            colours = None
            if inkit.any():
                from courtvision.tactics import torso_colours
                colours = torso_colours(frame, boxes)
            for k, (tid, box) in enumerate(assigned):
                crop = jersey_crop(frame, box)
                if crop is None or crop.size == 0:
                    continue
                crop_count += 1
                got = reader.read(crop)
                if not got:
                    continue
                number = got[0]
                read_count += 1
                votes[tid][number] += 1
                # Kit side: the sign of the torso colour along its own spread
                # axis, which is what kit_members clusters on.
                side = 0
                if colours is not None and np.isfinite(colours[k]).all():
                    axis = int(np.argmax(np.nanstd(colours, axis=0)))
                    mid = float(np.nanmedian(colours[:, axis]))
                    side = 1 if colours[k][axis] >= mid else 0
                kit_of[tid][side] += 1
                cluster_reads[side][number] += 1

        cluster_team = team_for_clusters(cluster_reads, roster)
        teams_by_tracklet = {tid: cluster_team.get(c.most_common(1)[0][0])
                             for tid, c in kit_of.items() if c}
        named = name_tracklets(votes, teams_by_tracklet, roster)

        boxes_out = []
        for offset, assigned in linked:
            row = [[named.get(tid, ""), *[int(round(v)) for v in box]]
                   for tid, box in assigned]
            boxes_out.append([offset, row])
        out[clip] = boxes_out
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(names)} clips, {read_count}/{crop_count} crops read",
                  flush=True)
    capture.release()

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"note": "per-clip player boxes with a name where the jersey could be "
                       "read and the kit agreed. An empty name means the number was "
                       "never legible -- it is left blank rather than guessed.",
               "source_size": [1280, 720], "step_s": STEP_S,
               "clips": out}, open(target, "w"), separators=(",", ":"))
    named_boxes = sum(1 for v in out.values() for _, row in v for b in row if b[0])
    total_boxes = sum(len(row) for v in out.values() for _, row in v)
    print(f"{len(out)} clips; {read_count}/{crop_count} crops gave a number "
          f"({read_count/max(crop_count,1):.0%})")
    print(f"  {named_boxes}/{total_boxes} boxes carry a name "
          f"({named_boxes/max(total_boxes,1):.0%})")
    print(f"  {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
