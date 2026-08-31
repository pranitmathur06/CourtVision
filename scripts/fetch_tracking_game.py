"""Fetch one SportVU tracking game and cache the extracted JSON.

The public NBA tracking feed was pulled from stats.nba.com in June 2016, so
2015-16 is the last season with coordinates. The corpus survives in community
mirrors; `sealneaward/nba-movement-data` is the one that carries a licence (MIT
on its scripts — the NBA data underneath is unlicensed re-hosting, so this is
for research only and must never be redistributed).

Coverage is 2015-10-27 to 2016-01-23: 636 games, about half a season. Not the
whole season, whatever the dataset cards say.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

MIRROR = ("https://raw.githubusercontent.com/sealneaward/nba-movement-data"
          "/master/data/{name}")
INDEX = ("https://api.github.com/repos/sealneaward/nba-movement-data"
         "/contents/data")
CACHE = Path("data/tracking")


def available() -> list[str]:
    """Every game archive in the mirror, as `MM.DD.YYYY.AWAY.at.HOME.7z`."""
    with urllib.request.urlopen(INDEX, timeout=60) as response:
        listing = json.load(response)
    return sorted(entry["name"] for entry in listing
                  if entry["name"].endswith(".7z"))


def fetch(name: str, cache: Path = CACHE) -> Path:
    """Download and extract one game; returns the path to its JSON."""
    import py7zr

    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / name
    if not archive.exists():
        url = MIRROR.format(name=name)
        print(f"  downloading {name}", flush=True)
        with urllib.request.urlopen(url, timeout=300) as response:
            archive.write_bytes(response.read())
    with py7zr.SevenZipFile(archive, "r") as handle:
        members = handle.getnames()
        target = [m for m in members if m.endswith(".json")]
        if not target:
            raise ValueError(f"{name} contains no json: {members}")
        extracted = cache / target[0]
        if not extracted.exists():
            handle.extract(path=cache, targets=target)
    archive.unlink(missing_ok=True)      # the JSON is what we keep
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true",
                        help="show what the mirror holds and exit")
    parser.add_argument("--count", type=int, default=1,
                        help="how many games to fetch, from the start")
    parser.add_argument("--name", default=None, help="one specific archive")
    args = parser.parse_args()

    names = available()
    if args.list:
        print(f"  {len(names)} games, {names[0]} .. {names[-1]}")
        for name in names[:10]:
            print(f"    {name}")
        return 0

    wanted = [args.name] if args.name else names[: args.count]
    for name in wanted:
        path = fetch(name)
        size = path.stat().st_size
        print(f"  {name} -> {path} ({size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
