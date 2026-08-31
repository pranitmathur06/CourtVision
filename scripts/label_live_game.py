"""Label windows in a continuous broadcast by joining the game clock to play-by-play.

Every previous attempt failed the same way: whichever corpus supplied the
negative class became trivially identifiable, so the model learned "which
broadcast is this" instead of "what is happening". Trained with BARD negatives
it called 80-92% of a game a rebound; trained with live negatives it called
100% of a game background. At serve time every window is live, so the only way
out is live-corpus examples of EVERY class.

That needs labels on live footage, and the join key is the clock on screen:

    video frame -> (period, clock) -> official play-by-play -> action

Periods come from clock RESETS rather than from reading "2ND"/"3RD" text — the
clock only counts down, so a large jump upward is a new period. That is the
same monotonicity the reader already validates itself with.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np

# Every network draws its own scoreboard, so the clock crop and the frames used
# to learn its digits are per broadcast. Templates built from a single frame
# cover only the digits in that one reading and read 1 frame in 60; several
# frames spread across the game cover enough of 0-9 to read four in five.
#
# `anchors` are (fraction through the video, digits with no colon). Readings
# under a minute are skipped: broadcasts switch to a decimal "56.9" there, which
# is a different format rather than a different value.
BROADCASTS = {
    "tnt": {
        "roi": (598, 634, 1098, 1170),
        "anchors": [(0.12, "150"), (0.18, "1118"), (0.24, "655"), (0.30, "251"),
                    (0.60, "835"), (0.66, "500"), (0.72, "104")],
    },
    "espn": {
        "roi": (632, 668, 812, 912),
        "anchors": [(0.10, "636"), (0.28, "813"), (0.34, "420"),
                    (0.52, "840"), (0.58, "552")],
    },
}
GAME_BROADCAST = {
    "0042400301": "tnt",
    "0042400302": "tnt",
    "0042400311": "espn",
    "0042400312": "espn",
    "0042400313": "espn",
}
CLOCK_ROI = BROADCASTS["tnt"]["roi"]     # default
N_FRAMES = 16
SIZE = 224
PERIOD_SECONDS = 12 * 60


def build_broadcast_templates(video: str, profile: dict):
    """Learn this broadcast's digits from frames whose clock value is known."""
    import cv2

    from courtvision.scoreboard import build_templates_from_many

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    top, bottom, left, right = profile["roi"]
    pairs = []
    for fraction, digits in profile["anchors"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * fraction))
        ok, frame = cap.read()
        if ok:
            pairs.append((frame[top:bottom, left:right], digits))
    cap.release()
    if not pairs:
        raise ValueError("no anchor frames could be read")
    return build_templates_from_many(pairs)


def read_clock_track(video: str, templates, sample_every_s: float = 2.0,
                     roi: tuple[int, int, int, int] | None = None):
    """[(frame_index, period, clock_seconds)] wherever the clock is legible."""
    import cv2

    from courtvision.scoreboard import clock_to_seconds, read_clock

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(fps * sample_every_s), 1)
    top, bottom, left, right = roi or CLOCK_ROI

    track: list[tuple[int, int, int]] = []
    period, previous, dropped = 1, None, 0
    for index in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            break
        text, _ = read_clock(frame[top:bottom, left:right], templates)
        if text is None:
            continue
        seconds = clock_to_seconds(text)
        if seconds > PERIOD_SECONDS:            # 12:00 is the maximum
            continue
        # A clock only counts down. A real period boundary is specific: the
        # clock was near zero and is now near twelve minutes. "Any jump upward
        # over a minute" is far too loose — it fired on misreads and produced 28
        # periods for a five-period game, which then stranded 84% of the
        # play-by-play with nowhere to align to.
        if previous is not None and seconds > previous + 60:
            if seconds >= 600 and previous <= 150:
                period += 1
                previous, dropped = seconds, 0
                track.append((index, period, seconds))
                continue
            # A misread — unless we keep saying that. One bad LOW reading sets
            # `previous` too low and then every correct reading after it looks
            # like a jump upward, so the whole rest of the game is discarded.
            # Several rejections in a row mean the reference is wrong, not the
            # readings, so resynchronise onto what we are actually seeing.
            dropped += 1
            if dropped >= 4:
                previous, dropped = seconds, 0
                track.append((index, period, seconds))
            continue
        if previous is not None and seconds > previous + 2:
            dropped += 1
            if dropped >= 4:
                previous, dropped = seconds, 0
                track.append((index, period, seconds))
            continue
        previous, dropped = seconds, 0
        track.append((index, period, seconds))
    cap.release()
    return track, fps


def frame_for(track, period: int, clock_seconds: int) -> int | None:
    """Video frame whose clock is closest to this play's, within the period."""
    same = [t for t in track if t[1] == period]
    if not same:
        return None
    clocks = [t[2] for t in same]
    # clocks descend through a period; bisect wants ascending
    ascending = clocks[::-1]
    position = bisect.bisect_left(ascending, clock_seconds)
    position = min(max(position, 0), len(ascending) - 1)
    chosen = same[len(same) - 1 - position]
    return chosen[0] if abs(chosen[2] - clock_seconds) <= 3 else None


def write_window(cap, start: int, stride: int, out_dir: Path, name: str,
                 detector, full_frame: bool) -> bool:
    import cv2

    from courtvision.action_classifier import crop_player
    from courtvision.types import HANDLER, PLAYER

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(start, 0))
    frames = []
    for k in range(N_FRAMES * stride):
        ok, f = cap.read()
        if not ok:
            break
        if k % stride == 0:
            frames.append(f)
    if len(frames) < N_FRAMES:
        return False
    frames = frames[:N_FRAMES]
    if full_frame:
        images = [cv2.resize(f, (SIZE, SIZE)) for f in frames]
    else:
        box = None
        for f in frames[:5]:
            found = detector.detect(f)
            cands = ([d for d in found if d.label == HANDLER]
                     or [d for d in found if d.label == PLAYER])
            if cands:
                box = max(cands, key=lambda d: d.conf).box
                break
        images = [crop_player(f, box) if box is not None
                  else cv2.resize(f, (SIZE, SIZE)) for f in frames]
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = images[0].shape[:2]
    writer = cv2.VideoWriter(str(out_dir / f"{name}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for image in images:
        writer.write(image)
    writer.release()
    return True


def main() -> int:
    import cv2

    from courtvision.config import Config
    from courtvision.detection import load_pipeline_detector
    from courtvision.device import resolve_device
    from courtvision.nba_feed import fetch_game_plays

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/workspace/game.mp4")
    parser.add_argument("--game-id", default="0042400301")
    parser.add_argument("--broadcast", default=None,
                        help="scoreboard profile; inferred from --game-id if unset")
    parser.add_argument("--crop-out", default="data/live/actions")
    parser.add_argument("--full-out", default="data/live/rim")
    parser.add_argument("--backgrounds", type=int, default=700)
    args = parser.parse_args()

    name = args.broadcast or GAME_BROADCAST.get(args.game_id, "tnt")
    profile = BROADCASTS[name]
    templates = build_broadcast_templates(args.video, profile)
    print(f"  broadcast {name}: templates for {sorted(templates)}", flush=True)

    track, fps = read_clock_track(args.video, templates, roi=profile["roi"])
    periods = sorted({p for _, p, _ in track})
    print(f"  clock read at {len(track)} sample points, periods {periods}", flush=True)
    if len(track) < 200:
        print("FAIL — too few clock reads to align")
        return 1

    plays = fetch_game_plays(args.game_id)
    print(f"  {len(plays)} official plays", flush=True)

    cfg = Config()
    detector = load_pipeline_detector("checkpoints/detector.pt", resolve_device(),
                                      cfg.detector_conf, cfg.ball_conf)
    cap = cv2.VideoCapture(args.video)
    stride = max(int(fps / 10), 1)
    crop_root, full_root = Path(args.crop_out), Path(args.full_out)

    import collections
    written = collections.Counter()
    event_frames: list[int] = []
    for i, play in enumerate(plays):
        if play.period is None or play.clock_seconds is None:
            continue
        frame = frame_for(track, play.period, play.clock_seconds)
        if frame is None:
            continue
        # Centre the window on the event.
        start = frame - (N_FRAMES * stride) // 2
        event_frames.append(frame)
        action = play.action
        if write_window(cap, start, stride, crop_root / action,
                        f"live_{i:05d}", detector, full_frame=False):
            written[action] += 1
        if action in ("rebound", "block", "shot"):
            write_window(cap, start, stride, full_root / action,
                         f"live_{i:05d}", detector, full_frame=True)

    # Background: far from every aligned event.
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    guard = int(fps * 4)
    events = sorted(event_frames)
    candidates = np.linspace(int(total * 0.06), int(total * 0.94),
                             args.backgrounds * 2).astype(int)
    made = 0
    for c in candidates:
        if made >= args.backgrounds:
            break
        position = bisect.bisect_left(events, int(c))
        near = min([abs(int(c) - events[j])
                    for j in (position - 1, position) if 0 <= j < len(events)]
                   or [10 ** 9])
        if near < guard:
            continue
        if write_window(cap, int(c) - (N_FRAMES * stride) // 2, stride,
                        crop_root / "background", f"live_bg_{made:05d}",
                        detector, full_frame=False):
            write_window(cap, int(c) - (N_FRAMES * stride) // 2, stride,
                         full_root / "background", f"live_bg_{made:05d}",
                         detector, full_frame=True)
            made += 1
    cap.release()
    written["background"] = made
    print(f"\n  live-labelled windows: {dict(written.most_common())}")
    print(f"  aligned {len(event_frames)}/{len(plays)} official plays to video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
