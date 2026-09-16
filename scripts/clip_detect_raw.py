"""Cache the detector's raw output for every clip, once, so selection is free.

Choosing which boxes to draw is where the mistakes are -- a ball on a
spectator, a rim on a hoarding, a box on somebody in row four -- and every
attempt at fixing that used to cost a full detection pass over 63,000 frames,
about three and a half hours. So the pass is separated from the choosing: this
writes down everything the detector saw, at every threshold worth considering,
and `clip_boxes.py --from-cache` decides from the file in seconds.

Two things are cached alongside the boxes because they need the pixels and the
chooser does not:

  on_court   whether each player box's feet land on the floor, from
             candidates.stands_on_court -- the test that separates a player
             from a spectator when both wear the home kit's colour.
  court      the bounding box of the floor itself, which bounds where a ball
             can plausibly be: one in the tenth row is not in play.

Detection runs every STEP frames rather than every frame. The ball moves a
median 3.4 px between frames at 30 fps, so the frames in between are recovered
by interpolation at a cost that is not visible and a saving that is half the
pass.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

#: Everything at or above this is written down; the chooser decides from there.
CONF_FLOOR = 0.08
#: Detect every Nth frame of the clip.
STEP = 2
#: Recompute the floor mask this often, in detected frames. It moves slowly.
COURT_EVERY = 3
#: Frames handed to the detector at once.
BATCH = 8

CODE = {"player": "p", "handler": "h", "ball": "b", "rim": "r"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    import numpy as np
    from ultralytics import YOLO

    from courtvision.candidates import court_region, stands_on_court
    from courtvision.device import resolve_device

    clips = {}
    for path in args.index:
        for c in json.load(open(path))["clips"]:
            if not c.get("clip"):
                continue        # indexed, deliberately not cut
            clips.setdefault(c["clip"], round(float(c.get("start_s",
                             float(c["video_s"]) - 3.0)), 1))
    names = sorted(clips)
    if args.limit:
        names = names[:args.limit]

    model, device = YOLO(args.detector), resolve_device()
    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(round(args.duration * fps))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = {}
    started = time.time()
    for n, clip in enumerate(names):
        capture.set(cv2.CAP_PROP_POS_MSEC, clips[clip] * 1000)
        wanted, frames = [], []
        for f in range(count):
            ok, frame = capture.read()
            if not ok:
                break
            if f % args.step == 0:
                wanted.append(f)
                frames.append(frame)

        rows, region, court = [], None, None
        for start in range(0, len(frames), BATCH):
            chunk = frames[start:start + BATCH]
            results = model.predict(chunk, device=device, verbose=False,
                                    imgsz=args.imgsz, conf=CONF_FLOOR)
            for offset, result in enumerate(results):
                i = start + offset
                boxes = result.boxes
                here = []
                if boxes is not None and len(boxes):
                    for cls, conf, box in zip(boxes.cls.cpu().numpy(),
                                              boxes.conf.cpu().numpy(),
                                              boxes.xyxy.cpu().numpy()):
                        kind = CODE.get(model.names[int(cls)])
                        if kind is None:
                            continue
                        here.append([kind, round(float(conf), 3)]
                                    + [round(float(v), 1) for v in box])
                # The floor moves slowly, so it is re-found every few frames and
                # reused between -- but whether a box stands on it depends on
                # that frame's boxes and is answered for every one of them.
                if i % COURT_EVERY == 0 or region is None:
                    region = court_region(chunk[offset])
                    if region is not None:
                        ys, xs = np.nonzero(region)
                        court = ([int(xs.min()), int(ys.min()),
                                  int(xs.max()), int(ys.max())]
                                 if len(xs) else None)
                people = [b[2:] for b in here if b[0] in ("p", "h")]
                on = (stands_on_court(region, np.array(people, float)).tolist()
                      if people and region is not None else [])
                rows.append({"f": wanted[i], "d": here,
                             "court": court, "on": on})
        out[clip] = rows
        if (n + 1) % 20 == 0:
            done = (n + 1) / len(names)
            left = (time.time() - started) * (1 - done) / max(done, 1e-6) / 60
            print(f"  {n + 1}/{len(names)} clips, {left:.0f} min left", flush=True)
    capture.release()

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "detector": args.detector,
               "imgsz": args.imgsz, "conf_floor": CONF_FLOOR,
               "step": args.step, "fps": fps, "frames_per_clip": count,
               "source_size": [width, height], "clips": out},
              open(target, "w"), separators=(",", ":"))
    print(f"{len(out)} clips -> {target} "
          f"({target.stat().st_size / 1e6:.0f} MB, "
          f"{(time.time() - started) / 60:.0f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
