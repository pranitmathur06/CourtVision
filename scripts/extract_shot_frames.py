"""Extract the frames worth hand-labelling: the moments around each official shot.

Everything automatic supplies EASY frames. The teacher (yolo11x) fires on ~20%
of frames and higher inference resolution recovers ordinary play -- neither
supplies the ball at the rim during a shot, where it is fastest, most
motion-blurred, and most occluded by net, backboard and hands. Those frames are
what the detector misses and what the pipeline is bottlenecked on, so they are
the only ones worth a person's hour.

Frames are seeded with the student detector's best guess so labelling is mostly
confirmation rather than clicking.
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

import cv2

sys.path.insert(0, "src")

# The ball is a near-constant size at broadcast distance, so a click is enough:
# the box is this wide, centred on it. Measured median across teacher labels.
BALL_PX = 26


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/raw_clips/quarter.mp4")
    ap.add_argument("--dets", required=True, help="cached detections with clock")
    ap.add_argument("--game", default="0022401223")
    ap.add_argument("--anchor-video-s", type=float, default=900.0)
    ap.add_argument("--anchor-period", type=int, default=3)
    ap.add_argument("--anchor-clock", default="4:38")
    ap.add_argument("--out", default="data/labeling/shots")
    ap.add_argument("--window-s", type=float, default=1.5)
    ap.add_argument("--fps", type=float, default=5.0, help="labelling rate")
    ap.add_argument("--weights", default="checkpoints/ball_student.pt")
    ap.add_argument("--imgsz", type=int, default=1920)
    args = ap.parse_args()

    mins, secs = args.anchor_clock.split(":")
    anchor_elapsed = (args.anchor_period - 1) * 720 + (720 - (int(mins) * 60 + float(secs)))

    data = json.loads(Path(args.dets).read_text())
    advanced = {t: v for t, v in data["clock"]}
    marks = sorted(advanced)

    def elapsed_at(video_s: float) -> float:
        """Video time to game elapsed, by counting seconds the clock advanced.

        The clock only runs during live play, so game time cannot be a linear
        function of video time; counting ticks is the mapping.
        """
        step = 1 if video_s >= args.anchor_video_s else -1
        low, high = sorted((args.anchor_video_s, video_s))
        ticks = sum(1 for m in marks if low < m <= high and advanced.get(m) is True)
        return anchor_elapsed + step * ticks

    from nba_api.stats.endpoints import playbyplayv3
    actions = playbyplayv3.PlayByPlayV3(game_id=args.game, timeout=60).get_dict()["game"]["actions"]

    def clock_seconds(text: str) -> float | None:
        match = re.fullmatch(r"PT(\d+)M([\d.]+)S", (text or "").strip())
        return int(match.group(1)) * 60 + float(match.group(2)) if match else None

    shots = []
    for entry in actions:
        kind = (entry.get("actionType") or "").strip()
        clock = clock_seconds(entry.get("clock") or "")
        period = entry.get("period")
        if clock is None or period is None or kind not in ("Made Shot", "Missed Shot"):
            continue
        shots.append(((period - 1) * 720 + (720 - clock), kind,
                      (entry.get("description") or "")[:60]))

    frames = [r["t"] for r in data["frames"]]
    lo, hi = elapsed_at(frames[0]), elapsed_at(frames[-1])
    inside = [s for s in shots if lo <= s[0] <= hi]
    print(f"  {len(inside)} official shots inside the footage")

    # invert the mapping once: game elapsed -> the video second that reaches it
    reverse: dict[int, float] = {}
    for video_s in frames:
        reverse.setdefault(int(elapsed_at(video_s)), video_s)

    from ultralytics import YOLO
    from courtvision.device import resolve_device
    model = YOLO(args.weights)
    device = resolve_device()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.video)
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    step = 1.0 / args.fps
    manifest = []
    for index, (elapsed, kind, description) in enumerate(sorted(inside)):
        centre = reverse.get(int(elapsed))
        if centre is None:
            continue
        offset = -args.window_s
        while offset <= args.window_s + 1e-6:
            t = centre + offset
            offset += step
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(t * video_fps))
            ok, image = capture.read()
            if not ok:
                continue
            name = f"s{index:03d}_{int(t*100):07d}.jpg"
            cv2.imwrite(str(out / "images" / name), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            result = model.predict(image, device=device, conf=0.02,
                                   imgsz=args.imgsz, verbose=False)[0]
            guess = None
            if len(result.boxes):
                best = max(zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()),
                           key=lambda pair: pair[1])
                (x1, y1, x2, y2), conf = best
                guess = {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "conf": round(float(conf), 3)}
            manifest.append({"file": name, "shot": index, "kind": kind,
                             "desc": description, "video_s": round(t, 2),
                             "guess": guess,
                             "w": image.shape[1], "h": image.shape[0]})
        if index and index % 10 == 0:
            print(f"    {index}/{len(inside)} shots, {len(manifest)} frames", flush=True)
    capture.release()
    (out / "manifest.json").write_text(json.dumps(manifest))
    seeded = sum(1 for m in manifest if m["guess"])
    print(f"  {len(manifest)} frames -> {out}")
    print(f"  {seeded} ({seeded/max(len(manifest),1):.0%}) come with a proposal to confirm; "
          f"{len(manifest)-seeded} need a click")
    return 0


if __name__ == "__main__":
    sys.exit(main())
