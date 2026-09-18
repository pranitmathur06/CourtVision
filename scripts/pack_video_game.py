"""Fold the video-backed game into the Film Room's stream, honestly labelled.

The Film Room answers from tracking coordinates, and coordinates are the one
thing a reader cannot check. This adds the game where the claim CAN be checked:
Finals G7, whose official play-by-play `align_shots_to_video.py` put onto the
video's own clock -- 545 events, 94.6% located, 517 of them exact to the second.

WHAT THESE ROWS ARE, stated here because the distinction decides whether the
whole page is honest. The other twenty games are DERIVED: possession, screens
and transition computed from 25 Hz positions, each row carrying a confidence
because the pipeline might be wrong. These rows are the NBA'S OWN RECORD with a
timestamp attached. They are not the vision stack's output and they are not
evidence that it found anything. They are there so a reader can watch the thing
the other twenty games can only assert, and the page says so rather than
letting the reader assume a video means the system saw it.

So `confidence` here is not a vision score. It is 1.0 when the aligner placed
the event exactly on its scoreboard reading and falls with the placement error,
because that is the only thing about these rows that can be wrong.

Player names come from the description's leading surname, which is how the
play-by-play writes them ("Nembhard 14' Pullup Jump Shot"). A description that
does not start with a name -- a jump ball, a violation -- yields no player
rather than a guess.
"""

from __future__ import annotations

import argparse
import json
import re

#: The play-by-play's vocabulary mapped onto the stream's, so a question about
#: "shots" reaches both the derived games and this one.
ACTION = {
    "Made Shot (2PT)": "made shot", "Made Shot (3PT)": "made shot",
    "Missed Shot": "shot", "Rebound": "rebound", "Assist": "assist",
    "Steal": "steal", "Turnover": "turnover", "Block": "block",
    "Foul": "foul", "Free Throw (made)": "free throw",
    "Free Throw (miss)": "free throw", "Jump Ball": "jump ball",
    "Violation": "violation",
}
#: An event placed this far from its clock reading loses all confidence.
ERROR_FLOOR_S = 4.0
#: Twelve-minute periods; overtime is five.
PERIOD_S, OVERTIME_S = 720.0, 300.0


def player_from(description):
    """The leading surname the play-by-play writes, or None.

    "MISS Jal. Williams 10' Step Back" -> "Jal. Williams"
    "Nesmith REBOUND (Off:0 Def:1)"    -> "Nesmith"
    "Jump Ball Hartenstein vs. Turner" -> None, because the leading token is
    not a name and guessing one would put a wrong player in the evidence.
    """
    if not description:
        return None
    text = description.strip()
    text = re.sub(r"^MISS\s+", "", text)
    if re.match(r"^(Jump Ball|Timeout|Instant Replay)", text):
        return None
    # "Pacers Timeout", "THUNDER Rebound" -- a club is not a player, and letting
    # one through puts a team in the player filter.
    if re.match(r"^(Pacers|Thunder|THUNDER|PACERS)\b", text):
        return None
    # A name is capitalised words, possibly with an abbreviated first name
    # ("Jal. Williams") or a particle, stopping at the first lowercase word or
    # an all-caps verb like REBOUND / BLOCK / STEAL.
    match = re.match(r"^((?:[A-Z][A-Za-z'’.\-]*\.?\s)*?[A-Z][A-Za-z'’\-]+)(?=\s|$)", text)
    if not match:
        return None
    name = match.group(1).strip()
    name = re.sub(r"\s+(REBOUND|BLOCK|STEAL|TURNOVER|FOUL|Free|Throw).*$", "", name)
    return name or None


def clock_of(elapsed_s, period_s=PERIOD_S, overtime_s=OVERTIME_S):
    """(period, "M:SS" remaining) from seconds elapsed in the game."""
    if elapsed_s < 4 * period_s:
        period = int(elapsed_s // period_s) + 1
        left = period_s - (elapsed_s - (period - 1) * period_s)
    else:
        over = elapsed_s - 4 * period_s
        period = 5 + int(over // overtime_s)
        left = overtime_s - (over - (period - 5) * overtime_s)
    left = max(0.0, left)
    return period, f"{int(left // 60)}:{int(left % 60):02d}"


def confidence(error_s, floor=ERROR_FLOOR_S):
    """1.0 when the aligner placed it exactly; 0 at the floor."""
    return round(max(0.0, 1.0 - abs(float(error_s)) / floor), 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, help="the packed stream to extend")
    parser.add_argument("--clips", required=True, help="cut_event_clips.py index.json")
    parser.add_argument("--game", default=None,
                        help="registry key or official game id; fills --game-id "
                             "and --date from it. The two defaults below name "
                             "one broadcast and are kept only so the command "
                             "recorded in the accuracy log still reproduces.")
    parser.add_argument("--game-id", default="0042400407")
    parser.add_argument("--date", default="2025 Finals G7")
    parser.add_argument("--roster", default=None,
                        help="number -> {team: full name}. The play-by-play writes "
                             "surnames only, so 'Gilgeous-Alexander' cannot be found by "
                             "someone who types 'shai gilgeous alexander'. The roster "
                             "supplies the full name.")
    parser.add_argument("--vision", default=None,
                        help="cut_event_clips vision index -- what the pipeline called "
                             "from PIXELS, added as its own game so the reader can watch "
                             "it succeed and fail against the official record")
    parser.add_argument("--tip-off-s", type=float, default=524.0,
                        help="video seconds at which the game clock starts. It "
                             "was the bare literal 524.0 in the middle of a "
                             "list comprehension, with nothing saying it "
                             "belonged to one broadcast; a second game packed "
                             "with it would have every vision row's period and "
                             "clock wrong by however far its tip-off differs.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.game:
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
        from courtvision.games import get
        chosen = get(args.game)
        args.game_id, args.date = chosen.game_id, chosen.label

    stream = json.load(open(args.stream))
    clips = json.load(open(args.clips))

    # Surname -> full names. A LIST, because two players share a surname here
    # and keeping one collapsed "Jal. Williams" into Kenrich Williams -- the
    # same collision the jersey numbers had. The play-by-play writes the
    # initial exactly so this can be resolved, so it is used.
    full_by_last: dict[str, list[str]] = {}
    if args.roster:
        for teams in json.load(open(args.roster)).values():
            for who in teams.values():
                full_by_last.setdefault(who.split()[-1].lower(), []).append(who)

    actions, details, names = stream["actions"], stream["details"], stream["players"]
    action_index = {a: i for i, a in enumerate(actions)}
    detail_index = {d: i for i, d in enumerate(details)}
    by_name = {v: k for k, v in names.items()}

    rows, unnamed = [], 0
    for c in clips["clips"]:
        action = ACTION.get(c["action"], c["action"].lower())
        if action not in action_index:
            action_index[action] = len(actions)
            actions.append(action)

        who = player_from(c.get("description", ""))
        if who:
            options = full_by_last.get(who.split()[-1].lower(), [])
            if len(options) == 1:
                who = options[0]
            elif len(options) > 1:
                # "Jal. Williams" vs "K. Williams": match the initial.
                initial = who.split()[0].rstrip(".").lower() if len(who.split()) > 1 else ""
                picked = [o for o in options if o.split()[0].lower().startswith(initial)] \
                    if initial else []
                if len(picked) == 1:
                    who = picked[0]
        if who is None:
            unnamed += 1
            pid = 0
        else:
            pid = by_name.get(who)
            if pid is None:
                pid = "v%d" % len(by_name)
                names[pid] = who
                by_name[who] = pid

        detail = c.get("description", "") or ""
        if detail and detail not in detail_index:
            detail_index[detail] = len(details)
            details.append(detail)

        period, clock = clock_of(float(c.get("elapsed_s") or 0.0))
        rows.append([round(float(c["video_s"]), 1), period, clock,
                     action_index[action], pid, -1, None, None,
                     confidence(c.get("error_s", 0.0)),
                     detail_index.get(detail, -1), 0, c["clip"]])

    stream["games"].insert(0, {"id": args.game_id, "date": args.date,
                               "video": True, "rows": rows})

    if args.vision:
        vision = json.load(open(args.vision))
        vrows = []
        for c in vision["clips"]:
            action = c["action"]
            if action not in action_index:
                action_index[action] = len(actions)
                actions.append(action)
            detail = c.get("description", "")
            if detail and detail not in detail_index:
                detail_index[detail] = len(details)
                details.append(detail)
            # Not a per-call score -- the detector has none. This is the class's
            # MEASURED precision on the held-out half, which is exactly how much
            # any single call is worth, and 0 for a shot it never made.
            conf = 0.0 if action == "vision miss" else round(vision.get("precision", 0.52), 2)
            period, clock = clock_of(max(c["video_s"] - args.tip_off_s, 0.0))
            vrows.append([round(float(c["video_s"]), 1), period, clock,
                          action_index[action], 0, -1, None, None, conf,
                          detail_index.get(detail, -1), 0, c["clip"]])
        stream["games"].insert(1, {"id": args.game_id + "-vision",
                                   "date": "2025 Finals G7 \u00b7 what the VISION pipeline saw",
                                   "video": True, "vision": True, "rows": vrows})
        stream["vision_note"] = (
            "These rows are the vision pipeline's own calls, made from pixels: ball and "
            "rim detected per frame, no play-by-play involved. Held-out F1 0.619, "
            "precision 0.522, recall 0.762 -- so about half its calls are wrong, and "
            "those are in here to be watched rather than hidden.")
    # ONCE, not once per game. This appended unconditionally, so packing three
    # games produced `[..., "clip", "clip", "clip"]` -- the shipped stream has
    # exactly that -- and every row carries one clip value at index 11, leaving
    # the extra names pointing at columns that do not exist.
    if "clip" not in stream["fields"]:
        stream["fields"] = stream["fields"] + ["clip"]
    stream["video_note"] = (
        "Rows for this game are the NBA's own play-by-play placed on the video's "
        "clock, not the vision stack's output. Confidence here is how exactly the "
        "aligner placed the event, not how sure a detector was.")
    json.dump(stream, open(args.out, "w"), separators=(",", ":"))

    import collections
    kinds = collections.Counter(actions[r[3]] for r in rows)
    print(f"{len(rows)} video-backed rows added as game {args.game_id}")
    print(f"  {len(set(r[11] for r in rows))} distinct clips")
    print(f"  {unnamed} rows carry no player (the description does not start with a name)")
    print(f"  actions: {dict(kinds)}")
    print(f"  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
