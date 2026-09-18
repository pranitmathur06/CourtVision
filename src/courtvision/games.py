"""Which broadcasts exist, and where every artefact about one lives.

THE REGISTRY IS THE POINT, AND SO IS WHAT IT REPLACES. Three scripts carried a
hardcoded `GAMES` dict -- `make_possession_label_page.py`, `make_handler_label_page.py`,
`build_possession_windows.py` -- and around fifteen more carried
`default="data/raw_clips/fullgame.mp4"`. Adding a fourth broadcast meant editing
eighteen files and getting every one of them to agree, which is another way of
saying a fourth broadcast could not be added. It is the single blocker on "a new
game can be scored on arrival", and it is the smallest item on the list.

PATHS ARE DERIVED, NEVER LISTED. A registry that stored `"detections":
"outputs/clip_detections_g1.json"` would just be the same eighteen strings in one
file, and one of them would eventually point at another game's cache with nothing
to say so. So the entry holds only what cannot be computed -- the video, the
official game id, a human label, a one-letter clip prefix -- and every artefact
path falls out of the key. Two games cannot collide unless their keys collide,
and keys are the dict's own keys.

THE ONE-LETTER PREFIX IS A REAL CEILING. Clip files are named `f001110.mp4`, and
the letter separates games in a directory they share; twenty-six letters is
twenty-six games. It is kept because three published games already use e/f/g and
renaming their files would break every permalink -- but `prefix` is validated for
uniqueness here, so the twenty-seventh broadcast fails loudly at registration
instead of quietly overwriting the first.

"HELD OUT" IS PER ARM, BECAUSE A BROADCAST IS NOT ONE THING. The obvious design
is a boolean, and it was a boolean for about an hour, and it was WRONG. The
Houston footage has never been near the event pipeline -- no clock read, no
alignment, no clip, no possession or handler or ball label, no detector training
frame -- and it is thoroughly inside the COURT REGISTRATION work:
`docs/continuous-game-accuracy.md` says in as many words that "Toyota Center is
no longer a clean unseen arena -- it was diagnosed on", four directories of hand
labels sit on this exact file, and `court_register.SEARCH_MIN_SAMPLES`,
`court_camera.FLOOR_LANE_LAB` and `court_refine.PAINT_POLARITY` were each set
"with those values in view".

A single `unseen: true` would have published an acceptance-test result over a
registration number the broadcast helped choose. So the entry records which ARMS
it is held out for, `held_out_for` answers per arm, and a claim about an arm not
listed has to say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: The registry, relative to the repository root.
REGISTRY = Path(__file__).resolve().parents[2] / "data" / "games.json"

#: The only path names a `published` map may override. A typo in one of these
#: used to fall through to the derived path with no complaint, which is the
#: quietest possible way for a game to be scored against the wrong file.
_PUBLISHABLE = frozenset({
    "clock", "scoreboard", "detections", "clip_detections", "aligned",
    "clip_dir", "clip_index", "overlays", "vision_shots", "roster"})


@dataclass(frozen=True)
class Broadcast:
    """One game: what it is, and where everything derived from it goes.

    Only the first four fields are data. Everything else is a path computed
    from `key`, so a stale path cannot be stored.
    """
    key: str
    game_id: str
    label: str
    video: str
    #: One letter, unique across the registry. Prefixes clip filenames.
    prefix: str
    #: Which arms this broadcast is held out for: {arm: True}. An arm that is
    #: absent is NOT held out and must not be reported as an acceptance result.
    #: See the module docstring for why this is not a boolean.
    held_out: dict[str, bool] | None = None
    #: Why, in one line, so a reader does not have to find the round in the log.
    held_out_note: str = ""
    #: Frames per second of the source, for the record.
    fps: float | None = None
    #: What the official feed says the final score was, so a scoreboard reader
    #: has something to be wrong against without a network.
    final_score: dict[str, int] | None = None
    #: Paths that were PUBLISHED before this registry existed and so cannot be
    #: derived. `docs/clips/index.json` is fetched by the live page and its name
    #: is part of a URL; renaming it to `index_g7.json` would be a registry
    #: tidying itself by breaking the site. Only these three games have any, and
    #: a new broadcast must have none -- which `add_broadcast.py` enforces.
    published: dict[str, str] | None = None

    def held_out_for(self, arm: str) -> bool:
        """Is a number from this broadcast, on this arm, an honest hold-out?

        Unknown arms answer False. A broadcast is guilty until the registry says
        otherwise, because the failure this guards against is publishing a
        contaminated number as an acceptance test, and the quiet default has to
        be the one that cannot do that.
        """
        return bool((self.held_out or {}).get(arm, False))

    @property
    def unseen(self) -> bool:
        """Held out for EVERY arm the registry names. Rarely true."""
        return bool(self.held_out) and all(self.held_out.values())

    # -- derived paths -----------------------------------------------------
    def _path(self, name: str, default: str) -> Path:
        """A published name if this game has one, otherwise the derived one."""
        return Path((self.published or {}).get(name, default))

    @property
    def stem(self) -> str:
        """The video's filename stem, which is what the clock reader keys on."""
        return Path(self.video).stem

    @property
    def clock(self) -> Path:
        return self._path("clock", f"outputs/clock/{self.stem}.json")

    @property
    def scoreboard(self) -> Path:
        return self._path("scoreboard", f"outputs/scoreboard/{self.stem}.json")

    @property
    def detections(self) -> Path:
        """The whole-game detection cache, sampled at a low rate."""
        return self._path("detections", f"outputs/detections/{self.stem}.json")

    @property
    def clip_detections(self) -> Path:
        """Dense detections inside the cut clips, at full width."""
        return self._path("clip_detections",
                          f"outputs/clip_detections_{self.key}.json")

    @property
    def aligned(self) -> Path:
        return self._path("aligned", f"outputs/games/aligned_{self.key}.json")

    @property
    def pbp(self) -> Path:
        return Path("data/pbp_cache") / f"{self.game_id}.json"

    @property
    def roster(self) -> Path:
        return self._path("roster", f"outputs/games/roster_{self.game_id}.json")

    @property
    def clip_dir(self) -> Path:
        """Where this game's clip files live.

        The three published games share `docs/clips/`, which is TRACKED IN GIT
        and is 12% of what a clone costs. A fourth game's 400 clips would be
        another few hundred megabytes of binaries in version control, and the
        plan already has them moving to object storage. So a new broadcast cuts
        into `data/clips/<key>/`, which is ignored -- the pipeline is identical
        and the repository does not grow by a game every time one arrives.
        """
        return self._path("clip_dir", f"data/clips/{self.key}")

    @property
    def clip_index(self) -> Path:
        return self._path("clip_index", str(self.clip_dir / "index.json"))

    @property
    def overlays(self) -> Path:
        """Per-clip box tracks. Lives beside the clips, for the same reason."""
        return self._path("overlays", str(self.clip_dir / "overlays.json"))

    @property
    def vision_shots(self) -> Path:
        return self._path("vision_shots", f"outputs/vision_shots_{self.key}.json")

    @property
    def report(self) -> Path:
        return Path("outputs/games") / f"report_{self.key}.json"

    @property
    def end_to_end(self) -> Path:
        return Path("outputs/games") / f"end_to_end_{self.key}.json"


def _load(path: Path | str | None = None) -> dict[str, Broadcast]:
    raw = json.loads(Path(path or REGISTRY).read_text())
    games = {key: Broadcast(key=key, **entry) for key, entry in raw.items()}
    prefixes: dict[str, str] = {}
    stems: dict[str, str] = {}
    ids: dict[str, str] = {}
    for key, game in games.items():
        if len(game.prefix) != 1 or not game.prefix.isalpha():
            raise ValueError(f"{key}: prefix must be one letter, got {game.prefix!r}")
        if game.prefix in prefixes:
            raise ValueError(f"{key} and {prefixes[game.prefix]} share clip prefix "
                             f"{game.prefix!r}; clip filenames would collide")
        prefixes[game.prefix] = key
        # The stem, not the path: `data/raw_clips/fullgame.mp4` and
        # `data/games/fullgame.mp4` are different files and the SAME clock,
        # detection and scoreboard artefacts, because those are keyed on the
        # encode. The docstring above claims two games cannot collide unless
        # their keys do; without this check that claim was simply false.
        if game.stem in stems:
            raise ValueError(f"{key} and {stems[game.stem]} have video stem "
                             f"{game.stem!r}; they would share clock, detection "
                             f"and scoreboard artefacts")
        stems[game.stem] = key
        # `get()` matches keys first, so an entry KEYED with another game's
        # official id would shadow it silently.
        if game.game_id in ids:
            raise ValueError(f"{key} and {ids[game.game_id]} both claim official "
                             f"game {game.game_id}")
        ids[game.game_id] = key
        if key in ids and ids[key] != key:
            raise ValueError(f"{key} is keyed with another game's official id")
        unknown = set(game.published or {}) - _PUBLISHABLE
        if unknown:
            raise ValueError(f"{key}: published names {sorted(unknown)}, which "
                             f"are not paths; a typo here silently falls back "
                             f"to the derived path")
    return games


@lru_cache(maxsize=4)
def _cached(path: str) -> dict[str, Broadcast]:
    return _load(path)


def registry(path: Path | str | None = None) -> dict[str, Broadcast]:
    """Every registered broadcast, keyed by its short key."""
    return dict(_cached(str(path or REGISTRY)))


def get(key_or_id: str, path: Path | str | None = None) -> Broadcast:
    """One broadcast, by registry key or by official game id.

    Accepting both is deliberate: scripts that already take `--game-id` keep
    their flag and gain the registry, and nobody has to learn a second name for
    a game they can already name.
    """
    games = registry(path)
    if key_or_id in games:
        return games[key_or_id]
    for game in games.values():
        if game.game_id == key_or_id:
            return game
        if game.stem == key_or_id:
            return game
    raise KeyError(f"no broadcast {key_or_id!r} in {path or REGISTRY}; "
                   f"known: {', '.join(sorted(games))}")
