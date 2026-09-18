"""Does a wrist separate the ball's owner from the man guarding him?

THE RECORDED FAILURE THIS TESTS, quoted from the handler labelling page: "on a
dribble a defender's hands are often nearer the ball than the holder's". Every
possession rule this project has tried scores the player by the distance from
the ball to his BOX -- centre, edge, body-heights, any of them -- and they all
land between 42% and 49%, with a measured ceiling of 58.7% even when the ball
position is supplied by hand. The kernels reached 59.2%, which is that ceiling.

A box centre is the middle of a torso. A ball is held in hands, and hands are
where a pose model puts wrists. `yolo11s-pose.pt` has been in this repository
the whole time and nothing has ever asked it this question.

WHAT IS MEASURED, on the frames where a person labelled BOTH the ball and the
player holding it:

    box centre   is the labelled handler the player whose box centre is
                 nearest the labelled ball?
    box edge     ...whose box is nearest, measured to its edge?
    wrist        ...who has a wrist nearest it?

All three are ORACLE measurements: they are handed the true ball position, so
each is an upper bound on any method that has to find the ball first. The
question is not whether wrists are good enough to ship -- it is whether the
ceiling moves at all when hands replace torsos, because 58.7% is where every
proximity rule has stopped.

Reported on the uniform half and the hard half separately, because the hard
half was drawn where the ball model was already failing.
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

from courtvision.stats import mcnemar, wilson  # noqa: E402

#: COCO pose keypoint indices. 9 and 10 are the wrists.
WRISTS = (9, 10)
#: ...and the elbows, so a wrist the model did not see has a fallback that is
#: still an arm rather than a torso.
ELBOWS = (7, 8)
LABELS = ("data/labels/possession_labels.json",)
IMAGES = ROOT / "data" / "labeling" / "possession" / "images"


def rows():
    out = []
    for name in LABELS:
        path = ROOT / name
        if not path.exists():
            continue
        for row in json.loads(path.read_text())["frames"]:
            verdict = row.get("handler_verdict") or row.get("verdict")
            if (row.get("ball_verdict") == "ball" and row.get("ball")
                    and verdict == "box" and row.get("handler_box")):
                out.append(row)
    return out


def centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def to_box(point, box) -> float:
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


def overlaps(a, b, threshold: float = 0.5) -> bool:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return False
    inter = (x2 - x1) * (y2 - y1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(smaller, 1e-6) >= threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pose", default="yolo11s-pose.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device

    labelled = rows()
    print(f"\n  {len(labelled)} frames where a person labelled BOTH the ball "
          f"and the player holding it\n")
    model, device = YOLO(args.pose), resolve_device()

    got = {k: defaultdict(list) for k in ("centre", "edge", "wrist")}
    seen = no_pose = 0
    for row in labelled:
        image = cv2.imread(str(IMAGES / row["file"]))
        if image is None:
            continue
        found = model.predict(image, device=device, verbose=False,
                              imgsz=args.imgsz, conf=args.conf)[0]
        if found.boxes is None or not len(found.boxes):
            no_pose += 1
            continue
        boxes = [[float(v) for v in b] for b in found.boxes.xyxy.cpu().numpy()]
        points = (found.keypoints.xy.cpu().numpy()
                  if found.keypoints is not None else None)
        # Which pose box is the labelled handler? The one that overlaps his
        # labelled box most -- the pose model detects people, not the handler.
        truth = row["handler_box"]
        target = next((i for i, b in enumerate(boxes) if overlaps(b, truth)), None)
        if target is None:
            continue
        seen += 1
        ball = tuple(row["ball"])
        split = "uniform" if row.get("pick") == "random" else "hard"

        nearest_centre = min(range(len(boxes)),
                             key=lambda i: math.dist(ball, centre(boxes[i])))
        nearest_edge = min(range(len(boxes)), key=lambda i: to_box(ball, boxes[i]))
        got["centre"][split].append(nearest_centre == target)
        got["edge"][split].append(nearest_edge == target)

        if points is None:
            got["wrist"][split].append(False)
            continue

        def hand_distance(i: int) -> float:
            person = points[i]
            hands = [person[j] for j in WRISTS
                     if j < len(person) and (person[j] > 0).all()]
            if not hands:
                hands = [person[j] for j in ELBOWS
                         if j < len(person) and (person[j] > 0).all()]
            if not hands:
                return to_box(ball, boxes[i])
            return min(math.dist(ball, (float(h[0]), float(h[1]))) for h in hands)

        nearest_hand = min(range(len(boxes)), key=hand_distance)
        got["wrist"][split].append(nearest_hand == target)

    print(f"  {seen} scored; {no_pose} frames where the pose model found nobody\n")
    print(f"  {'rule':<12}{'uniform':>20}{'hard':>20}")
    summary = {}
    for rule in ("centre", "edge", "wrist"):
        line = f"  {rule:<12}"
        summary[rule] = {}
        for split in ("uniform", "hard"):
            v = got[rule][split]
            hits, n = sum(v), len(v)
            low, high = wilson(hits, n)
            summary[rule][split] = {"hits": hits, "n": n}
            line += f"{f'{hits}/{n} = {hits / max(n, 1):.3f}':>20}"
        print(line)
        for split in ("uniform", "hard"):
            v = got[rule][split]
            if v:
                low, high = wilson(sum(v), len(v))
                print(f"      {split:<8} 95% CI {low:.3f}-{high:.3f}")

    print("\n  PAIRED against the box-centre rule, on the uniform half "
          "(exact McNemar):")
    base = got["centre"]["uniform"]
    for rule in ("edge", "wrist"):
        only_a, only_b, p = mcnemar(got[rule]["uniform"], base)
        verdict = ("better" if only_a > only_b else
                   "worse" if only_b > only_a else "level")
        print(f"    {rule:<8} {only_a:>3} frames only it gets, {only_b:>3} only "
              f"the box centre   p = {p:.4f}   ({verdict})")
    print("\n  Every rule here is an ORACLE: it is handed the true ball "
          "position, so each\n  is an upper bound on any method that has to "
          "find the ball first. The measured\n  proximity ceiling this is "
          "tested against is 58.7%.")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
        print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
