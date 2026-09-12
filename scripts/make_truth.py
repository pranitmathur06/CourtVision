"""Turn a labeller's judgements on the rendered sheets into scoreable truth.

The sheets draw what the system claims; the labeller answers, per frame, what
is actually there. Confirming a proposal is the cheap case and the common one,
so a confirmed proposal BECOMES the truth coordinate -- it was judged to sit on
the object, which is exactly what the tolerance asks. Anything not confirmed is
given its own coordinate by hand, read off the panel.

Judgement lines, one per frame, in a plain text file:

    12 rim=ok ball=ok           both claims confirmed
    13 rim=- ball=-             neither object is in this picture
    14 rim=ok;ok ball=-         both baskets shown and both claims right
    15 rim=x:120,88 ball=ok     a rim IS there, at panel (120, 88); no right claim
    16 rim=ok;+x:520,90 ball=-  one claim right, a SECOND rim unclaimed at ...
    17 rim=ok ball=x:301,204    the ball is there but the claim missed it
    18 skip                     unreadable render, dropped from the grid

Objects within one judgement are separated by ";" -- the comma belongs to the
coordinate. `ok` consumes the system's next unused report for that object, in
the order the system listed them. Coordinates are PANEL pixels -- what the
labeller actually looked at -- scaled up by --scale to the frame the system
works in.

Widths come with the truth because the tolerance is one object width: a rim is
1.5 ft and a ball 0.79 ft, and their apparent size follows the camera, so the
width is taken from the detector's own box for that object where there is one
and from DEFAULT_*_PX otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

#: Apparent sizes at 720p on a main-camera view, used when nothing better is on
#: offer. Deliberately generous to the SYSTEM only in that a larger width is a
#: looser tolerance, so these are set at the small end of what was measured.
DEFAULT_RIM_PX = 34.0
DEFAULT_BALL_PX = 18.0

POINT = re.compile(r"x:(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)")


def parse_spec(spec, reports, scale, default_px, widths):
    """One object's judgement -> [{centre, width}] in frame pixels."""
    out, used = [], 0
    if spec in ("-", ""):
        return out
    for piece in spec.split(";"):
        piece = piece.strip().lstrip("+")
        if piece == "ok":
            if used >= len(reports):
                raise ValueError(f"'ok' with no report left to confirm: {spec!r}")
            centre = [float(v) for v in reports[used]]
            width = widths[used] if used < len(widths) else default_px
            out.append({"centre": centre, "width": float(width)})
            used += 1
            continue
        found = POINT.fullmatch(piece)
        if not found:
            raise ValueError(f"cannot read {piece!r} in {spec!r}")
        out.append({"centre": [float(found.group(1)) * scale,
                               float(found.group(2)) * scale],
                    "width": float(default_px)})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgements", required=True)
    parser.add_argument("--system", required=True, help="what the sheets were drawn from")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="panel pixels -> frame pixels (640-wide panel of 1280)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = json.load(open(args.system))["frames"]
    frames, skipped = [], 0
    for line in open(args.judgements):
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        index = int(parts[0])
        row = rows[index]
        if len(parts) == 2 and parts[1] == "skip":
            skipped += 1
            continue
        specs = {}
        for token in parts[1:]:
            key, _, value = token.partition("=")
            specs[key] = value
        rim_reports = row.get("rim") or []
        ball_reports = [row["ball"]] if row.get("ball") else []
        frames.append({
            "t": row["t"],
            "rim": parse_spec(specs.get("rim", "-"), rim_reports, args.scale,
                              DEFAULT_RIM_PX, []),
            "ball": (parse_spec(specs.get("ball", "-"), ball_reports, args.scale,
                                DEFAULT_BALL_PX, []) or [None])[0],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"system": args.system, "scale": args.scale, "skipped": skipped,
               "frames": frames}, open(out, "w"), indent=0)
    rims = sum(len(f["rim"]) for f in frames)
    balls = sum(1 for f in frames if f["ball"])
    print(f"{len(frames)} frames labelled ({skipped} skipped); "
          f"{rims} rims visible, {balls} balls visible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
