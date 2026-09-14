"""Compact the action stream into one payload the web page carries.

The page has to answer questions about twenty games without a server, so the
whole stream ships inside it. Written as records with names on every field that
is 2.4 MB; as parallel arrays against a shared vocabulary it is a fraction of
that, and the page expands it once on load.

`detail` is the only free text and it is highly repetitive -- most rows say one
of a few dozen sentences -- so it is stored as a vocabulary plus an index
rather than repeated in full.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pack(streams):
    actions, teams, details = [], [], []
    a_idx, t_idx, d_idx = {}, {}, {}

    def intern(value, table, index):
        if value is None:
            return -1
        if value not in index:
            index[value] = len(table)
            table.append(value)
        return index[value]

    games, players = [], {}
    for s in streams:
        players.update(s.get("players", {}))
        rows = []
        for r in s["rows"]:
            court = r.get("court") or [None, None]
            rows.append([
                round(r["t"], 1),
                r.get("period") or 0,
                r.get("clock") or "",
                intern(r["action"], actions, a_idx),
                r.get("player_id") or 0,
                intern(r.get("team"), teams, t_idx),
                None if court[0] is None else round(court[0], 1),
                None if court[1] is None else round(court[1], 1),
                round(float(r.get("confidence") or 0), 2),
                intern(r.get("detail") or "", details, d_idx),
                r.get("screener_id") or 0,
            ])
        games.append({"id": s["game_id"], "date": s["date"], "rows": rows})
    return {"actions": actions, "teams": teams, "details": details,
            "players": players, "games": games,
            "fields": ["t", "period", "clock", "action", "player_id", "team",
                       "x", "y", "confidence", "detail", "screener_id"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default="outputs/stream")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    streams = [json.load(open(p)) for p in sorted(Path(args.stream).glob("*.json"))]
    packed = pack(streams)
    text = json.dumps(packed, separators=(",", ":"))
    Path(args.out).write_text(text)
    rows = sum(len(g["rows"]) for g in packed["games"])
    print(f"{len(packed['games'])} games, {rows} rows, "
          f"{len(packed['actions'])} action types, {len(packed['players'])} players")
    print(f"  {len(text)/1e6:.2f} MB packed -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
