"""Cut a short clip around every officially-logged event, so the stream links to footage.

The Film Room answers from tracking coordinates, and coordinates are the one
thing a person cannot check. "Nikola Vucevic set 11 screens" is a claim about a
number; "here is the 6 seconds where he set this one" is a claim anyone can
falsify by watching.

The two halves of this project never meet on one game -- SportVU tracking runs
2015-10-27 to 2016-01-23 and no release pairs it with broadcast video. So the
footage comes from the other side: `align_shots_to_video.py` put the official
play-by-play for 0042400407 onto the video's own clock, 94.6% of 545 events
located, and 517 of them exact to the second. That alignment is the index.

WHAT IS AND IS NOT CLAIMED HERE. These clips are cut at the time the NBA's own
play-by-play says the event happened, carried onto the video by a clock read
off the scoreboard. They are not the output of the vision stack, and they are
not evidence that the vision stack found anything. They are ground truth with a
timestamp, which is what makes them a fair place to check a claim.

LEAD_S before the logged instant, because a logged rebound is the moment the
ball is secured and the shot that caused it is already in the air. TAIL_S after,
because a made shot's consequence -- the inbound, the celebration -- is how a
viewer confirms it went in.

Events that are not things to watch are skipped: substitutions, timeouts,
period markers and instant-replay stoppages. They are 91 of the 545 and none of
them has anything on screen worth six seconds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

#: Seconds either side of the logged instant.
LEAD_S, TAIL_S = 3.0, 3.0
#: 426x240 at crf 36 is 92 KB for six seconds. The whole set fits in the tens of
#: megabytes, which is what makes it hostable on a static site at all.
WIDTH, CRF = 426, 36
#: Logged rows with nothing to see.
SKIP = {"Substitution", "Timeout", "period", "Instant Replay"}
#: An event the aligner placed this far from its clock reading is not worth
#: cutting: the clip would show a different possession.
MAX_ERROR_S = 4.0


def wanted(event, skip=SKIP, max_error=MAX_ERROR_S):
    """Is this row worth six seconds of video?"""
    if event.get("action") in skip:
        return False
    if abs(float(event.get("error_s", 0.0))) > max_error:
        return False
    return event.get("video_s") is not None


def clip_name(event):
    """A stable name from the instant, so re-running does not duplicate work."""
    return f"e{int(round(float(event['video_s']) * 10)):06d}.mp4"


def cut(video, start, duration, out, width=WIDTH, crf=CRF):
    """One clip. Seek BEFORE -i so ffmpeg jumps rather than decoding to there."""
    return subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{start:.2f}",
         "-t", f"{duration:.2f}", "-i", str(video),
         "-vf", f"scale={width}:-2", "-c:v", "libx264", "-crf", str(crf),
         "-preset", "veryfast", "-an", "-movflags", "+faststart",
         str(out), "-y"], capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--events", required=True, help="align_shots_to_video.py output")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lead", type=float, default=LEAD_S)
    parser.add_argument("--tail", type=float, default=TAIL_S)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data = json.load(open(args.events))
    events = [e for e in data["events"] if wanted(e)]
    if args.limit:
        events = events[:args.limit]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    index, made, failed, skipped = [], 0, 0, 0
    for n, event in enumerate(events):
        name = clip_name(event)
        target = out / name
        start = max(float(event["video_s"]) - args.lead, 0.0)
        if not target.exists():
            result = cut(args.video, start, args.lead + args.tail, target)
            if result.returncode != 0 or not target.exists():
                failed += 1
                continue
            made += 1
        else:
            skipped += 1
        index.append({"clip": name, "video_s": round(float(event["video_s"]), 1),
                      "start_s": round(start, 1),
                      "action": event["action"],
                      "description": event.get("description", ""),
                      "error_s": event.get("error_s", 0.0),
                      "elapsed_s": event.get("elapsed_s"),
                      "period": event.get("period")})
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(events)} cut", flush=True)

    (out / "index.json").write_text(json.dumps(
        {"note": "clips cut at the instant the NBA's own play-by-play says the event "
                 "happened, carried onto the video by a scoreboard clock read. Ground "
                 "truth with a timestamp -- NOT the vision stack's output.",
         "game_id": data.get("game_id"), "video": args.video,
         "lead_s": args.lead, "tail_s": args.tail,
         "clips": index}, indent=1))
    size = sum(f.stat().st_size for f in out.glob("*.mp4"))
    print(f"{len(index)} clips ({made} cut now, {skipped} already there, {failed} failed)")
    print(f"  {size/1e6:.1f} MB total, {size/max(len(index),1)/1e3:.0f} KB each")
    print(f"  {out / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
