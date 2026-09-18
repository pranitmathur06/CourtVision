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
#: Detect every Nth frame of the clip. IN FRAMES, WHICH IS THE PROBLEM: 2 is
#: 15 Hz on the 30 fps broadcasts this was written for and 30 Hz on a 60 fps
#: one, so a fourth broadcast would silently get twice the temporal resolution
#: of the other three, at twice the cost, and its cached detections would not be
#: comparable with theirs. `RATE` is the quantity that was meant -- the same
#: mistake `TrackerConfig` was built to stop making with `TRACK_MAX_AGE = 15`.
STEP = 2
#: Detected frames per second of clip. 15 Hz is exactly `STEP = 2` at 30 fps,
#: so the three broadcasts already cached are unchanged.
RATE = 15.0
#: Recompute the floor mask this often, in detected frames. It moves slowly.
COURT_EVERY = 3
#: Frames handed to the detector at once.
BATCH = 8
#: How far the floor mask is eroded before feet are tested against it. The
#: library's own default is 45 px, which is tuned for a different question and
#: costs real players here: on the Finals G1 broadcast it kept 4 of the 10
#: players on the floor, because a player standing near a sideline has his feet
#: at the very edge of a mask that has just been pulled 45 px inward. Measured
#: over 39 frames of each of two broadcasts, at 15 px the count goes 4 -> 7 on
#: that game and 7 -> 8 on G7 while still dropping about two boxes a frame --
#: the bench and the front row, which is what the test is for. A convex hull
#: over the floor recovers everything and filters nothing, so it is not used.
COURT_ERODE_PX = 15

CODE = {"player": "p", "handler": "h", "ball": "b", "rim": "r"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--ball-detector", default=None,
                        help="a single-class ball model whose boxes REPLACE the "
                             "4-class detector's ball class. The two are trained "
                             "on different things: the 4-class model learned the "
                             "ball from labels that were its own false positives, "
                             "and offers 17 candidates a frame; the specialist is "
                             "trained on tiles at the ball's own scale against "
                             "hard negatives cut from the crowd, and offers 2.6.")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--rate", type=float, default=RATE,
                        help="detected frames per second of clip. Converted to "
                             "a frame step against the video's own rate, so a "
                             "60 fps broadcast is sampled at the same instants "
                             "as a 30 fps one instead of twice as often.")
    parser.add_argument("--step", type=int, default=None,
                        help="override --rate with a literal frame step. Only "
                             "for reproducing a cache written before --rate "
                             "existed.")
    parser.add_argument("--court-erode", type=int, default=None,
                        help="how far the floor mask is pulled in, in PIXELS. "
                             "Overrides --court-erode-share. Only for "
                             "reproducing a cache written before the share "
                             "existed.")
    parser.add_argument("--court-erode-share", type=float, default=None,
                        help="how far the floor mask is pulled in, as a share "
                             "of frame height. Defaults to this broadcast's "
                             "entry in outputs/games/court_erode.json, which "
                             "fit_court_mask.py chooses against two label-free "
                             "bounds, and to the shipped 45/720 if there is "
                             "none. A share rather than pixels because 45 px "
                             "is 6.25% of a 720p frame and 4.2% of a 1080p "
                             "one, and the constant otherwise means two "
                             "different things on two broadcasts.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    erode_share = args.court_erode_share
    if erode_share is None and args.court_erode is None:
        fitted = ROOT / "outputs" / "games" / "court_erode.json"
        if fitted.exists():
            picked = json.loads(fitted.read_text()).get(args.game)
            if picked is not None:
                erode_share = float(picked)
                print(f"  court erosion {erode_share:.4f} of frame height, "
                      f"fitted for {args.game} by fit_court_mask.py")
    if erode_share is None and args.court_erode is None:
        erode_share = COURT_ERODE_SHARE

    import cv2
    import numpy as np
    from ultralytics import YOLO

    from courtvision.candidates import court_region, stands_on_court
    from courtvision.device import describe, resolve_device
    from courtvision.fast_detect import half_precision_ok

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
    ball_model = YOLO(args.ball_detector) if args.ball_detector else None
    # fp16 where it pays and nowhere else. `half_precision_ok` is CUDA-only by
    # measurement, not by assumption: asked for on this Mac's MPS the same pass
    # ran 446 ms a frame against 202 in fp32, and every timing in this
    # repository was taken on MPS in fp32. On a rented CUDA box it is close to
    # free speed, which is the difference between a three-hour pass and a
    # twenty-minute one.
    precision = {"quantize": 16} if half_precision_ok(device) else {}
    print(f"  {describe()}"
          f"{'  fp16' if precision else '  fp32'}", flush=True)
    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = args.step if args.step else max(1, int(round(fps / args.rate)))
    count = int(round(args.duration * fps))
    print(f"  {fps:.2f} fps source, step {step} -> {fps / step:.1f} detected "
          f"frames a second", flush=True)
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
            if f % step == 0:
                wanted.append(f)
                frames.append(frame)

        rows, region, court = [], None, None
        for start in range(0, len(frames), BATCH):
            chunk = frames[start:start + BATCH]
            results = model.predict(chunk, device=device, verbose=False,
                                    imgsz=args.imgsz, conf=CONF_FLOOR, **precision)
            ball_results = (ball_model.predict(chunk, device=device, verbose=False,
                                               imgsz=args.imgsz, conf=CONF_FLOOR,
                                               **precision)
                            if ball_model is not None else None)
            for offset, result in enumerate(results):
                i = start + offset
                boxes = result.boxes
                here = []
                if boxes is not None and len(boxes):
                    for cls, conf, box in zip(boxes.cls.cpu().numpy(),
                                              boxes.conf.cpu().numpy(),
                                              boxes.xyxy.cpu().numpy()):
                        kind = CODE.get(model.names[int(cls)])
                        if kind is None or (kind == "b" and ball_model is not None):
                            continue
                        here.append([kind, round(float(conf), 3)]
                                    + [round(float(v), 1) for v in box])
                if ball_results is not None:
                    found = ball_results[offset].boxes
                    if found is not None and len(found):
                        for conf, box in zip(found.conf.cpu().numpy(),
                                             found.xyxy.cpu().numpy()):
                            here.append(["b", round(float(conf), 3)]
                                        + [round(float(v), 1) for v in box])
                # The floor moves slowly, so it is re-found every few frames and
                # reused between -- but whether a box stands on it depends on
                # that frame's boxes and is answered for every one of them.
                if i % COURT_EVERY == 0 or region is None:
                    region = court_region(chunk[offset],
                                          erode_px=args.court_erode,
                                          erode_share=erode_share)
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
               "step": step, "rate": fps / step, "fps": fps,
               "frames_per_clip": count,
               "source_size": [width, height], "clips": out},
              open(target, "w"), separators=(",", ":"))
    print(f"{len(out)} clips -> {target} "
          f"({target.stat().st_size / 1e6:.0f} MB, "
          f"{(time.time() - started) / 60:.0f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
