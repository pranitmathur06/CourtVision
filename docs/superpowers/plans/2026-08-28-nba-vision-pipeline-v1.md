# NBA Vision-to-Text Pipeline v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-machine pipeline that turns a short basketball clip into tracked players, team labels, ball possession, classified actions, and LLM-generated play-by-play commentary, plus an annotated output video.

**Architecture:** Nine sequential stages (extract → detect → track → team → possession → action → events → commentary → render), each a separate importable module behind a narrow interface. Pure-logic stages (possession, events, commentary validation, windowing, render helpers) are built test-first against a deterministic synthetic clip with known ground truth. Model-dependent stages (detector fine-tuning, action classifier) are gated by `scripts/validate_vN.py` scripts that print an unambiguous PASS/FAIL plus a metric, exactly as the spec requires.

**Tech Stack:** Python 3.11.11 (pyenv), PyTorch on Apple **MPS**, `ultralytics` (YOLO11), `supervision` (ByteTrack), `scikit-learn` (KMeans), `transformers` (VideoMAE), `opencv-python`, `langgraph` 1.0.7, `anthropic` SDK (`claude-opus-5`), `pytest`.

**Spec:** [README.md](README.md) — "NBA Vision-to-Text Pipeline — v1 Implementation Spec"

---

## Execution status (2026-08-28)

**Tasks 1–11 complete. 79 tests passing. Task 12 blocked, correctly, on spec §9.3.**

Every stage is built, unit-tested against the synthetic clip, and committed on
branch `feat/v1-pipeline`. All nine `validate_vN.py` scripts are written and run
end-to-end; each currently reports FAIL for one reason only — a missing external
input, not a code defect.

| Needed | Blocks | Who |
|---|---|---|
| A basketball clip at `data/raw_clips/sample.mp4` (10–60s, single angle) | V1, V2, V4, V5, V6 | user |
| A labeled detector subset (50–200 frames, player/ball/rim) at `data/labeled/detector/` | V3 → then V4, V5, V6 | user (licence call) |
| Labeled action clips at `data/labeled/actions/<action>/*.mp4` (≥20) | V7 | user (licence call) |
| `ANTHROPIC_API_KEY`, or `ant auth login` | V8 | user |
| A held-out clip at `data/raw_clips/holdout.mp4` | V9 | user |

`scripts/run_pipeline.py` and `scripts/validate_v9.py` are deliberately **not
written yet**: spec §9.3 forbids the end-to-end script until V1–V8 pass
individually, and Task 12's own first step enforces that. Resume at Task 12 once
the inputs above land.

---

## Global Constraints

- **Python 3.13.0**, from `/Library/Frameworks/Python.framework/Versions/3.13`. All work happens inside the project venv at `.venv` — always invoke `./.venv/bin/python`. **Not** the pyenv 3.11.11 that `python3` resolves to by default: that build is missing the `_lzma` C extension, which breaks `import torchvision` and therefore the VideoMAE import Task 8 needs. Rebuilding the user's global pyenv interpreter would affect their other projects, so the venv was rebased onto the framework 3.13 build instead — isolated and reversible. *(Amended during execution; the plan originally specified 3.11.11.)*
- **Device is `mps`, not `cuda`.** This machine is an Apple M2 (10-core GPU, 16 GB unified memory). There is no NVIDIA GPU. Every model call resolves its device through `courtvision.device.resolve_device()`. Never hardcode `"cuda"` and never call `.cuda()`. This keeps the code portable: it selects `cuda` unmodified on the rented GPU box v2 will need.
- **Disk is no longer a constraint.** The machine had 6.6 GB free when this plan was written, which drove a 4 GB budget; it now has ~170 GB. Full datasets are viable. The "small subset first" sequencing in Tasks 4 and 8 is still correct — but for the spec's own reason (prove the loop before scaling), not for disk. *(Amended during execution.)*
- **Model ID is `claude-opus-5`.** Exact string, no date suffix. Do not substitute a cheaper model.
- **Every `scripts/validate_vN.py` must print a single final line** of the form `V<N> PASS — <metric>` or `V<N> FAIL — <metric>` and exit `0` on pass, `1` on fail. Pass/fail is never a judgment call buried in output.
- **Every task ends with a commit.** Small, frequent commits.
- **Spec §9.6 — stop, don't work around.** If any `validate_vN.py` fails twice in a row after a genuine fix attempt, stop and flag it for review rather than loosening the threshold, editing the answer key, or skipping the gate. Several of these (possession accuracy, action-classifier accuracy) have real difficulty ceilings, and a repeat failure may mean the approach needs rethinking rather than debugging.
- **The five action labels are exactly** `("dribble", "pass", "shot", "rebound", "other")` — defined once in `courtvision.types.ACTIONS`, never re-spelled inline.
- **Teams are the strings `"A"` and `"B"`.** Which physical team is A is arbitrary and must not be asserted anywhere.

## Deviations from the spec (deliberate, with rationale)

These differ from README.md and are intentional. Do not "fix" them back.

1. **`src/courtvision/` package + `pyproject.toml`, not flat `src/*.py`** (spec §8). A real package installed with `pip install -e .` means tests and scripts import the same way with zero `sys.path` hacks.
2. **pytest unit tests in addition to the `validate_vN.py` scripts.** The spec only asks for the scripts. But possession, event structuring, commentary validation, windowing, and render helpers are deterministic pure functions — those get real TDD. The scripts remain for the empirical, model-dependent checks where a unit test can't express "the boxes look right."
3. **A synthetic clip fixture with known ground truth** (`tests/fixtures/synthetic.py`) — not in the spec. It renders colored rectangles on deterministic paths with a scripted ball-holder schedule. This makes tracking, team assignment, possession, events, and render testable with **zero downloaded data**, which the 6.6 GB disk constraint makes close to mandatory.
4. **MPS, not CUDA** — forced by the hardware. Spec §7 item 1 (custom CUDA kernel) is not merely deferred, it is **impossible on this machine** and will need a rented GPU.
5. **The `anthropic` SDK is called directly inside LangGraph nodes**, rather than via `langchain-anthropic`. LangGraph still owns the graph (that's the spec's stated goal of reusing LangGraph experience); skipping the LangChain adapter layer saves a large dependency tree on a disk-constrained machine.

## Known blockers to resolve before the tasks that need them

- **Task 4 (V3) and Task 8 (V7) need labeled data.** No dataset is on disk. SpaceJam is the smallest option. Check licence terms and free space before pulling.
- **Task 10 (V8) needs Claude credentials.** Neither `ANTHROPIC_API_KEY` nor the `ant` CLI is present. Set `ANTHROPIC_API_KEY` in the environment, or install the `ant` CLI and run `ant auth login`, before starting Task 10. Tasks 1–9 and 11–12 need no credentials.
- **Task 2 needs one real basketball clip** at `data/raw_clips/sample.mp4` (10–60s, single camera angle). Every automated test uses the synthetic clip instead, so this only gates the `validate_vN.py` scripts.

---
## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, pytest config. Enables `pip install -e .` |
| `src/courtvision/types.py` | Every dataclass crossing a stage boundary: `Box`, `Detection`, `Track`, `Frame`, `ActionWindow`, `Event`. No logic beyond geometry. |
| `src/courtvision/config.py` | One frozen `Config` dataclass holding every tunable threshold. No magic numbers anywhere else. |
| `src/courtvision/device.py` | `resolve_device()` — the single place that knows about MPS. |
| `src/courtvision/extraction.py` | Stage 1. Decode a video, yield frames at a target FPS. |
| `src/courtvision/detection.py` | Stages 2–3. `Detector` protocol + `YoloDetector`. |
| `src/courtvision/tracking.py` | Stage 3. ByteTrack wrapper: `Detection` list in, `Track` list out. |
| `src/courtvision/team_assignment.py` | Stage 4. Torso crops → Lab colors → KMeans(2) → stable per-track `"A"`/`"B"`. |
| `src/courtvision/possession.py` | Stage 5. Pure. Nearest-player rule + temporal smoothing. |
| `src/courtvision/action_classifier.py` | Stage 6. Pure windowing + VideoMAE classifier. |
| `src/courtvision/events.py` | Stage 7. Pure. Merge possession + teams + actions into `Event`s. |
| `src/courtvision/commentary.py` | Stage 8. LangGraph graph, Anthropic narrator, and the anti-fabrication validator. |
| `src/courtvision/render.py` | Stage 9. OpenCV overlays + JSON log. |
| `tests/fixtures/synthetic.py` | Deterministic clip generator with ground truth + `StubDetector`. Test-only. |
| `tests/test_*.py` | One test module per source module. |
| `scripts/validate_v1.py` … `validate_v9.py` | The spec's §6 checkpoints. One per gate. |
| `scripts/run_pipeline.py` | End-to-end driver. Written last, in Task 12. |

---

### Task 1: Project scaffold, core types, device resolution

Sets up everything later tasks import. Ends with a green test run proving the package is importable and the geometry is right.

**Files:**
- Create: `pyproject.toml`, `.gitignore` (append), `src/courtvision/__init__.py`, `src/courtvision/types.py`, `src/courtvision/config.py`, `src/courtvision/device.py`
- Create: `tests/__init__.py`, `tests/test_types.py`
- Create (empty dirs with `.gitkeep`): `data/raw_clips/`, `data/labeled/`, `checkpoints/`, `outputs/`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `Box(x1,y1,x2,y2)` with `.width`, `.height`, `.center` → `tuple[float,float]`; `Detection(box, label, conf)`; `Track(track_id, box, label, conf)`; `Frame(index, time_s, tracks)` with `.players()` → `tuple[Track,...]` and `.ball()` → `Track | None`; `ActionWindow(start_index, end_index, start_time_s, end_time_s, label, conf)`; `Event(time_s, track_id, team, action, possession_change)`; constants `PLAYER`, `BALL`, `RIM`, `ACTIONS`, `TEAMS`; `Config` frozen dataclass; `resolve_device() -> str`.

- [ ] **Step 1: Verify free disk before installing anything**

```bash
df -h . | tail -1
```
Expected: at least 4 GB in the `Avail` column. If under 4 GB, **stop and tell the user** — the venv alone is ~1.5–2 GB. Do not proceed, and do not delete user files to make room.

- [ ] **Step 2: Create the venv and directory skeleton**

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
mkdir -p src/courtvision tests/fixtures scripts data/raw_clips data/labeled checkpoints outputs
touch data/raw_clips/.gitkeep data/labeled/.gitkeep checkpoints/.gitkeep outputs/.gitkeep
touch src/courtvision/__init__.py tests/__init__.py tests/fixtures/__init__.py
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "courtvision"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.0",
    "opencv-python>=4.10",
    "torch>=2.4",
    "torchvision>=0.19",
    "ultralytics>=8.3",
    "supervision>=0.25",
    "scikit-learn>=1.5",
    "transformers>=4.44",
    "langgraph>=1.0",
    "anthropic>=1.0",
    "pydantic>=2.8",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Extend `.gitignore`**

```bash
cat >> .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
data/raw_clips/*
data/labeled/*
checkpoints/*
outputs/*
!**/.gitkeep
EOF
```

Weights, clips and outputs are large and must never enter git — the disk budget depends on it.

- [ ] **Step 5: Write the failing test for types**

Create `tests/test_types.py`:

```python
from courtvision.types import ACTIONS, BALL, PLAYER, Box, Frame, Track


def test_box_geometry():
    box = Box(10.0, 20.0, 30.0, 60.0)
    assert box.width == 20.0
    assert box.height == 40.0
    assert box.center == (20.0, 40.0)


def test_frame_players_excludes_ball():
    player = Track(1, Box(0, 0, 10, 20), PLAYER, 0.9)
    ball = Track(-1, Box(5, 5, 7, 7), BALL, 0.8)
    frame = Frame(0, 0.0, (player, ball))
    assert frame.players() == (player,)


def test_frame_ball_returns_highest_confidence_ball():
    low = Track(-1, Box(0, 0, 2, 2), BALL, 0.3)
    high = Track(-1, Box(9, 9, 11, 11), BALL, 0.7)
    frame = Frame(0, 0.0, (low, high))
    assert frame.ball() is high


def test_frame_ball_returns_none_when_absent():
    frame = Frame(0, 0.0, (Track(1, Box(0, 0, 10, 20), PLAYER, 0.9),))
    assert frame.ball() is None


def test_actions_are_the_five_spec_labels():
    assert ACTIONS == ("dribble", "pass", "shot", "rebound", "other")
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'courtvision'`

- [ ] **Step 7: Write `src/courtvision/types.py`**

```python
"""Data structures that cross stage boundaries. Geometry only, no pipeline logic."""

from __future__ import annotations

from dataclasses import dataclass

PLAYER = "player"
BALL = "ball"
RIM = "rim"
CLASSES = (PLAYER, BALL, RIM)

ACTIONS = ("dribble", "pass", "shot", "rebound", "other")
TEAMS = ("A", "B")


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates, top-left origin."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class Detection:
    """A single-frame detection, before tracking assigns an identity."""

    box: Box
    label: str
    conf: float


@dataclass(frozen=True)
class Track:
    """A detection carrying an identity. track_id is -1 for non-player classes."""

    track_id: int
    box: Box
    label: str
    conf: float


@dataclass(frozen=True)
class Frame:
    """Everything known about one sampled frame."""

    index: int
    time_s: float
    tracks: tuple[Track, ...]

    def players(self) -> tuple[Track, ...]:
        return tuple(t for t in self.tracks if t.label == PLAYER)

    def ball(self) -> Track | None:
        balls = [t for t in self.tracks if t.label == BALL]
        if not balls:
            return None
        return max(balls, key=lambda t: t.conf)


@dataclass(frozen=True)
class ActionWindow:
    """A classified span of frames. Bounds are inclusive."""

    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float
    label: str
    conf: float


@dataclass(frozen=True)
class Event:
    """A discrete play-by-play event; the input to commentary generation."""

    time_s: float
    track_id: int | None
    team: str | None
    action: str
    possession_change: bool
```

- [ ] **Step 8: Install the package and run the tests**

```bash
./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/python -m pytest tests/test_types.py -v
```
Expected: PASS, 5 passed. This install pulls torch and friends — expect several minutes and ~1.5–2 GB.

- [ ] **Step 9: Write `src/courtvision/device.py`**

```python
"""Single source of truth for which torch device to use.

This machine is an Apple M2 — there is no CUDA. `cuda` is checked anyway so the
same code runs unmodified on a rented GPU box for v2.
"""

from __future__ import annotations


def resolve_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

- [ ] **Step 10: Write `src/courtvision/config.py`**

```python
"""Every tunable threshold in the pipeline. No magic numbers elsewhere."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Stage 1 — extraction
    target_fps: int = 10

    # Stage 2 — detection
    detector_conf: float = 0.25

    # Stage 5 — possession.
    # Distance from ball centre to player centre, divided by that player's box
    # height, so the threshold is scale-invariant as players move up/down court.
    possession_max_norm_dist: float = 0.8
    # A challenger must hold the ball this many consecutive frames to take over.
    possession_min_hold_frames: int = 3
    # Ball may vanish (occlusion, in flight) this many frames before the holder is dropped.
    possession_max_gap_frames: int = 5

    # Stage 6 — action classification. VideoMAE expects exactly 16 frames.
    action_window_frames: int = 16
    action_stride_frames: int = 8

    # Stage 8 — commentary
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = 16000
    llm_max_attempts: int = 2
```

- [ ] **Step 11: Verify device resolution reports MPS**

```bash
./.venv/bin/python -c "from courtvision.device import resolve_device; print(resolve_device())"
```
Expected: `mps`. If it prints `cpu`, torch was built without MPS support — reinstall torch before continuing, since every later stage will be far slower.

- [ ] **Step 12: Commit**

```bash
git add pyproject.toml .gitignore src/ tests/ data/ checkpoints/ outputs/
git commit -m "feat: scaffold courtvision package with core types, config, device resolution"
```

---
### Task 2: Synthetic clip fixture + frame extraction (V1)

The synthetic clip is the backbone of every later test. It renders four players as solid colored rectangles moving on deterministic, non-crossing paths, plus a ball that follows a scripted holder schedule with two "in flight" gaps. Because the ground truth is known exactly, tracking, team assignment, possession, events, and render can all be tested without downloading a single byte of data.

**Files:**
- Create: `tests/fixtures/synthetic.py`, `tests/conftest.py`, `src/courtvision/extraction.py`, `tests/test_extraction.py`, `scripts/validate_v1.py`

**Interfaces:**
- Consumes: `Box`, `Detection`, `PLAYER`, `BALL` from `courtvision.types` (Task 1).
- Produces:
  - `make_clip(path: str, n_frames: int = 50, fps: int = 10) -> SyntheticTruth`
  - `SyntheticTruth` with fields `path: str`, `fps: int`, `n_frames: int`, `teams: dict[int, str]`, `holder_by_frame: list[int | None]`, `player_boxes: list[dict[int, Box]]`, `ball_boxes: list[Box | None]`
  - `StubDetector(truth: SyntheticTruth)` with `.detect(frame_index: int) -> list[Detection]`
  - pytest fixture `synthetic` (session-scoped) yielding a `SyntheticTruth`
  - `probe(video_path: str) -> tuple[float, int]` returning `(source_fps, source_frame_count)`
  - `expected_frame_count(source_frame_count: int, source_fps: float, target_fps: int) -> int`
  - `extract_frames(video_path: str, target_fps: int) -> Iterator[tuple[int, float, np.ndarray]]` yielding `(output_index, time_s, bgr_image)`

- [ ] **Step 1: Write the synthetic clip generator**

Create `tests/fixtures/synthetic.py`:

```python
"""Deterministic synthetic basketball clip with exact ground truth.

Four players move on straight, non-crossing horizontal lanes so a correct tracker
produces zero ID switches. The ball is drawn at the current holder's centre, with
two scripted "in flight" gaps where no player holds it — those exercise the
possession smoother's gap-bridging.

Everything here is test-only and must never be imported by src/courtvision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from courtvision.types import BALL, PLAYER, Box, Detection

WIDTH, HEIGHT = 640, 360
COURT_COLOR = (60, 90, 130)          # BGR, a dull brown-ish court
TEAM_COLORS = {"A": (40, 40, 200), "B": (200, 60, 40)}   # BGR: A red, B blue
BALL_COLOR = (30, 150, 240)          # BGR orange
PLAYER_W, PLAYER_H = 28, 72
BALL_R = 7

# player index -> team. Players 0,1 are team A; 2,3 are team B.
TEAMS: dict[int, str] = {0: "A", 1: "A", 2: "B", 3: "B"}

# (first_frame, last_frame_inclusive, holder). holder None means ball in flight.
HOLDER_SCHEDULE: list[tuple[int, int, int | None]] = [
    (0, 19, 0),
    (20, 21, None),
    (22, 34, 2),
    (35, 36, None),
    (37, 49, 1),
]


@dataclass(frozen=True)
class SyntheticTruth:
    path: str
    fps: int
    n_frames: int
    teams: dict[int, str]
    holder_by_frame: list[int | None]
    player_boxes: list[dict[int, Box]]
    ball_boxes: list[Box | None]


def _holder_at(frame_index: int) -> int | None:
    for start, end, holder in HOLDER_SCHEDULE:
        if start <= frame_index <= end:
            return holder
    return None


def _player_box(player_index: int, frame_index: int, n_frames: int) -> Box:
    """Each player owns a horizontal lane and oscillates within it. Lanes never overlap."""
    lane_x = 60 + player_index * 150
    phase = 2.0 * math.pi * frame_index / max(n_frames, 1)
    x = lane_x + 25.0 * math.sin(phase + player_index)
    y = 120.0 + 40.0 * math.sin(phase * 2.0 + player_index)
    return Box(x, y, x + PLAYER_W, y + PLAYER_H)


def make_clip(path: str, n_frames: int = 50, fps: int = 10) -> SyntheticTruth:
    """Render the clip to `path` and return its exact ground truth."""
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")

    holder_by_frame: list[int | None] = []
    player_boxes: list[dict[int, Box]] = []
    ball_boxes: list[Box | None] = []

    for frame_index in range(n_frames):
        image = np.full((HEIGHT, WIDTH, 3), COURT_COLOR, dtype=np.uint8)

        boxes = {i: _player_box(i, frame_index, n_frames) for i in TEAMS}
        for player_index, box in boxes.items():
            cv2.rectangle(
                image,
                (int(box.x1), int(box.y1)),
                (int(box.x2), int(box.y2)),
                TEAM_COLORS[TEAMS[player_index]],
                thickness=-1,
            )

        holder = _holder_at(frame_index)
        if holder is None:
            # Ball in flight: park it high above the court, far from every player.
            bx, by = WIDTH / 2.0, 30.0
        else:
            bx, by = boxes[holder].center
        cv2.circle(image, (int(bx), int(by)), BALL_R, BALL_COLOR, thickness=-1)

        writer.write(image)
        holder_by_frame.append(holder)
        player_boxes.append(boxes)
        ball_boxes.append(Box(bx - BALL_R, by - BALL_R, bx + BALL_R, by + BALL_R))

    writer.release()
    return SyntheticTruth(
        path=path,
        fps=fps,
        n_frames=n_frames,
        teams=dict(TEAMS),
        holder_by_frame=holder_by_frame,
        player_boxes=player_boxes,
        ball_boxes=ball_boxes,
    )


class StubDetector:
    """Replays ground-truth boxes as if a perfect detector produced them.

    Lets every downstream stage be tested without model weights. Note it is
    indexed by frame number, not by image content — it is a fixture, not a model.
    """

    def __init__(self, truth: SyntheticTruth) -> None:
        self._truth = truth

    def detect_at(self, frame_index: int) -> list[Detection]:
        detections = [
            Detection(box, PLAYER, 0.99)
            for box in self._truth.player_boxes[frame_index].values()
        ]
        ball = self._truth.ball_boxes[frame_index]
        if ball is not None:
            detections.append(Detection(ball, BALL, 0.95))
        return detections
```

- [ ] **Step 2: Write the pytest fixture**

Create `tests/conftest.py`:

```python
import pytest

from tests.fixtures.synthetic import SyntheticTruth, make_clip


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory) -> SyntheticTruth:
    """One rendered synthetic clip shared by the whole test session."""
    path = tmp_path_factory.mktemp("clips") / "synthetic.mp4"
    return make_clip(str(path))
```

- [ ] **Step 3: Write the failing extraction tests**

Create `tests/test_extraction.py`:

```python
import numpy as np

from courtvision.extraction import expected_frame_count, extract_frames, probe


def test_probe_reports_source_fps_and_count(synthetic):
    source_fps, source_count = probe(synthetic.path)
    assert round(source_fps) == synthetic.fps
    assert source_count == synthetic.n_frames


def test_expected_frame_count_at_native_fps():
    assert expected_frame_count(50, 10.0, 10) == 50


def test_expected_frame_count_downsamples_by_half():
    assert expected_frame_count(50, 10.0, 5) == 25


def test_expected_frame_count_never_upsamples_past_source():
    # Asking for more frames than exist just returns every source frame.
    assert expected_frame_count(50, 10.0, 30) == 50


def test_extract_frames_yields_expected_count_at_native_fps(synthetic):
    frames = list(extract_frames(synthetic.path, synthetic.fps))
    assert len(frames) == synthetic.n_frames


def test_extract_frames_downsamples(synthetic):
    frames = list(extract_frames(synthetic.path, synthetic.fps // 2))
    assert len(frames) == synthetic.n_frames // 2


def test_extract_frames_yields_increasing_indices_and_times(synthetic):
    frames = list(extract_frames(synthetic.path, synthetic.fps))
    indices = [i for i, _, _ in frames]
    times = [t for _, t, _ in frames]
    assert indices == list(range(len(frames)))
    assert times == sorted(times)
    assert times[0] == 0.0


def test_extract_frames_yields_decoded_images(synthetic):
    _, _, image = next(iter(extract_frames(synthetic.path, synthetic.fps)))
    assert isinstance(image, np.ndarray)
    assert image.shape == (360, 640, 3)
    assert image.dtype == np.uint8
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_extraction.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'courtvision.extraction'`

- [ ] **Step 5: Write `src/courtvision/extraction.py`**

```python
"""Stage 1 — decode a video and yield frames at a target sampling rate.

Sampling is done by keeping a source frame whenever its scaled index advances,
which spreads the kept frames evenly instead of taking a contiguous prefix.
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


def probe(video_path: str) -> tuple[float, int]:
    """Return (source_fps, source_frame_count) without decoding the whole file."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return fps, count


def expected_frame_count(
    source_frame_count: int, source_fps: float, target_fps: int
) -> int:
    """How many frames `extract_frames` will yield. Never upsamples."""
    if target_fps >= source_fps:
        return source_frame_count
    return int(source_frame_count * target_fps / source_fps)


def extract_frames(
    video_path: str, target_fps: int
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (output_index, time_s, bgr_image) at approximately `target_fps`.

    time_s is derived from the *source* frame index, so timestamps stay true to
    the original clip regardless of the sampling rate.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(target_fps)
    ratio = min(target_fps / source_fps, 1.0)

    try:
        source_index = 0
        output_index = 0
        kept = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            # Keep this frame if the running quota of kept frames has advanced.
            if int((source_index + 1) * ratio) > kept:
                kept += 1
                yield output_index, source_index / source_fps, image
                output_index += 1
            source_index += 1
    finally:
        capture.release()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_extraction.py tests/test_types.py -v`
Expected: PASS, 13 passed.

- [ ] **Step 7: Write `scripts/validate_v1.py`**

```python
"""V1 — Frame extraction sanity check (spec §6).

Extracts frames from a real clip at the target FPS, confirms the count matches
duration x fps within rounding, and saves a few frames as images for eyeballing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from courtvision.config import Config
from courtvision.extraction import expected_frame_count, extract_frames, probe

CLIP = Path("data/raw_clips/sample.mp4")
OUT_DIR = Path("outputs/v1_frames")


def main() -> int:
    if not CLIP.exists():
        print(f"V1 FAIL — no clip at {CLIP}; place a 10-60s basketball clip there")
        return 1

    config = Config()
    source_fps, source_count = probe(str(CLIP))
    duration_s = source_count / source_fps
    expected = expected_frame_count(source_count, source_fps, config.target_fps)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    actual = 0
    for index, time_s, image in extract_frames(str(CLIP), config.target_fps):
        actual += 1
        if index < 5:
            cv2.imwrite(str(OUT_DIR / f"frame_{index:03d}_t{time_s:.2f}.png"), image)

    # Allow one frame of slack for rounding at the tail.
    ok = abs(actual - expected) <= 1
    verdict = "PASS" if ok else "FAIL"
    print(
        f"V1 {verdict} — source {source_fps:.2f}fps x {duration_s:.2f}s; "
        f"expected {expected} frames at {config.target_fps}fps, got {actual}; "
        f"sample images in {OUT_DIR}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8: Run V1 against a real clip**

```bash
./.venv/bin/python scripts/validate_v1.py
```
Expected: `V1 PASS — ...`. If it reports no clip, ask the user for a 10–60s single-angle basketball clip at `data/raw_clips/sample.mp4`, then re-run. Open two or three PNGs in `outputs/v1_frames/` and confirm they look like basketball, not garbage — a decoded-but-corrupt video still produces the right frame *count*.

- [ ] **Step 9: Commit**

```bash
git add tests/ src/courtvision/extraction.py scripts/validate_v1.py
git commit -m "feat: synthetic clip fixture and frame extraction (V1)"
```

---
### Task 3: Detector interface + stock YOLO baseline (V2)

Proves the detection library and MPS environment work *before* any fine-tuning investment. The `Detector` protocol defined here is what every later stage codes against, so a stub can be swapped in for tests.

**Files:**
- Create: `src/courtvision/detection.py`, `tests/test_detection.py`, `scripts/validate_v2.py`

**Interfaces:**
- Consumes: `Detection`, `Box`, `PLAYER`, `BALL`, `RIM` (Task 1); `extract_frames` (Task 2).
- Produces:
  - `Detector` — a `typing.Protocol` with `detect(self, image: np.ndarray) -> list[Detection]`
  - `COCO_CLASS_MAP: dict[int, str]` mapping COCO ids to our labels (`0 -> player`, `32 -> ball`)
  - `YoloDetector(weights: str, device: str, conf: float, class_map: dict[int, str])` implementing `Detector`
  - `boxes_from_result(result, class_map, conf) -> list[Detection]` — pure conversion from an ultralytics result to our types

- [ ] **Step 1: Write the failing test for the pure conversion function**

The model call itself isn't unit-tested (it needs weights and a GPU); the conversion from ultralytics output to our types is pure and is tested with a fake result object.

Create `tests/test_detection.py`:

```python
import numpy as np
import pytest

from courtvision.detection import COCO_CLASS_MAP, boxes_from_result
from courtvision.types import BALL, PLAYER


class FakeBoxes:
    """Mimics ultralytics `result.boxes`: parallel tensors as numpy arrays."""

    def __init__(self, xyxy, cls, conf):
        self.xyxy = np.array(xyxy, dtype=np.float32)
        self.cls = np.array(cls, dtype=np.float32)
        self.conf = np.array(conf, dtype=np.float32)

    def __len__(self):
        return len(self.cls)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_converts_person_to_player():
    result = FakeResult(FakeBoxes([[1, 2, 3, 4]], [0], [0.9]))
    detections = boxes_from_result(result, COCO_CLASS_MAP, conf=0.25)
    assert len(detections) == 1
    assert detections[0].label == PLAYER
    assert detections[0].conf == pytest.approx(0.9)
    assert detections[0].box.x1 == pytest.approx(1.0)
    assert detections[0].box.y2 == pytest.approx(4.0)


def test_converts_sports_ball_to_ball():
    result = FakeResult(FakeBoxes([[0, 0, 5, 5]], [32], [0.5]))
    detections = boxes_from_result(result, COCO_CLASS_MAP, conf=0.25)
    assert detections[0].label == BALL


def test_drops_classes_not_in_map():
    # COCO 15 is "cat" — irrelevant to basketball, must be discarded.
    result = FakeResult(FakeBoxes([[0, 0, 5, 5]], [15], [0.99]))
    assert boxes_from_result(result, COCO_CLASS_MAP, conf=0.25) == []


def test_drops_detections_below_confidence():
    result = FakeResult(FakeBoxes([[0, 0, 5, 5]], [0], [0.10]))
    assert boxes_from_result(result, COCO_CLASS_MAP, conf=0.25) == []


def test_handles_empty_result():
    result = FakeResult(FakeBoxes(np.empty((0, 4)), [], []))
    assert boxes_from_result(result, COCO_CLASS_MAP, conf=0.25) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_detection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'courtvision.detection'`

- [ ] **Step 3: Write `src/courtvision/detection.py`**

```python
"""Stages 2-3 — object detection.

Every downstream stage codes against the `Detector` protocol, never against
ultralytics directly, so tests can substitute a stub with no model weights.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from courtvision.types import BALL, PLAYER, RIM, Box, Detection

# Stock COCO ids we care about. There is no COCO class for a basketball rim,
# which is exactly why V3 fine-tuning exists.
COCO_CLASS_MAP: dict[int, str] = {0: PLAYER, 32: BALL}

# After fine-tuning we own the class order, so it is dense and starts at zero.
FINETUNED_CLASS_MAP: dict[int, str] = {0: PLAYER, 1: BALL, 2: RIM}


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


def boxes_from_result(
    result, class_map: dict[int, str], conf: float
) -> list[Detection]:
    """Convert one ultralytics result into our types, filtering by class and confidence."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    detections: list[Detection] = []
    for xyxy, class_id, score in zip(boxes.xyxy, boxes.cls, boxes.conf):
        label = class_map.get(int(class_id))
        if label is None or float(score) < conf:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        detections.append(Detection(Box(x1, y1, x2, y2), label, float(score)))
    return detections


class YoloDetector:
    """Ultralytics YOLO behind the `Detector` protocol."""

    def __init__(
        self,
        weights: str,
        device: str,
        conf: float,
        class_map: dict[int, str],
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._device = device
        self._conf = conf
        self._class_map = class_map

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            image, device=self._device, conf=self._conf, verbose=False
        )
        if not results:
            return []
        return boxes_from_result(results[0], self._class_map, self._conf)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_detection.py -v`
Expected: PASS, 5 passed.

- [ ] **Step 5: Write `scripts/validate_v2.py`**

```python
"""V2 — Stock detector sanity check (spec §6).

Runs off-the-shelf YOLO11n with COCO weights on 5 frames of a real clip, before
any basketball-specific fine-tuning. This is the baseline that V3 must beat, and
it confirms ultralytics + MPS actually work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from courtvision.config import Config
from courtvision.detection import COCO_CLASS_MAP, YoloDetector
from courtvision.device import resolve_device
from courtvision.extraction import extract_frames
from courtvision.types import PLAYER

CLIP = Path("data/raw_clips/sample.mp4")
OUT_DIR = Path("outputs/v2_detections")
N_FRAMES = 5
MIN_PLAYERS_PER_FRAME = 2


def main() -> int:
    if not CLIP.exists():
        print(f"V2 FAIL — no clip at {CLIP}")
        return 1

    config = Config()
    device = resolve_device()
    # yolo11n.pt downloads on first use (~6 MB) into the ultralytics cache.
    detector = YoloDetector("yolo11n.pt", device, config.detector_conf, COCO_CLASS_MAP)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts: list[int] = []
    for index, _, image in extract_frames(str(CLIP), config.target_fps):
        if index >= N_FRAMES:
            break
        detections = [d for d in detector.detect(image) if d.label == PLAYER]
        counts.append(len(detections))
        for det in detections:
            cv2.rectangle(
                image,
                (int(det.box.x1), int(det.box.y1)),
                (int(det.box.x2), int(det.box.y2)),
                (0, 255, 0),
                2,
            )
        cv2.imwrite(str(OUT_DIR / f"det_{index:03d}.png"), image)

    ok = bool(counts) and all(c >= MIN_PLAYERS_PER_FRAME for c in counts)
    verdict = "PASS" if ok else "FAIL"
    print(
        f"V2 {verdict} — device={device}; players per frame {counts} "
        f"(need >={MIN_PLAYERS_PER_FRAME} each); overlays in {OUT_DIR}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run V2 and inspect the overlays**

```bash
./.venv/bin/python scripts/validate_v2.py
```
Expected: `V2 PASS — device=mps; players per frame [...]`. The count check is a floor, not proof — **open the PNGs in `outputs/v2_detections/` and confirm boxes land on people**, which is the actual spec criterion. A stock COCO model will *not* find the rim and will be unreliable on the ball; that is expected and is what V3 fixes.

- [ ] **Step 7: Commit**

```bash
git add src/courtvision/detection.py tests/test_detection.py scripts/validate_v2.py
git commit -m "feat: detector protocol and stock YOLO baseline (V2)"
```

---

### Task 4: Detector fine-tuning (V3)

Proves the training loop and data pipeline work end to end on a small subset, before scaling. This is the first task needing downloaded data — check disk first.

**Files:**
- Create: `scripts/prepare_dataset.py`, `scripts/validate_v3.py`
- Modify: `src/courtvision/detection.py` (add `load_finetuned`)

**Interfaces:**
- Consumes: `YoloDetector`, `FINETUNED_CLASS_MAP` (Task 3).
- Produces: `load_finetuned(weights_path: str, device: str, conf: float) -> YoloDetector`; a fine-tuned checkpoint at `checkpoints/detector.pt`; a YOLO-format dataset at `data/labeled/detector/` with `data.yaml`.

- [ ] **Step 1: Check disk, then acquire a small labeled subset**

```bash
df -h . | tail -1
```
Need at least 2 GB free *after* the venv. If not, stop and report to the user.

Then obtain 50–200 labeled frames with `player`/`ball`/`rim` boxes. SpaceJam is the smallest starting point; a small BARD slice also works. **Check the project's current licence and access terms before downloading** — the spec flags that these change. Convert to YOLO format under `data/labeled/detector/` as `images/{train,val}/` + `labels/{train,val}/`, with a 80/20 split.

If no dataset can be obtained, **stop and tell the user** — V3 cannot be faked, and V4 onward depends on a detector that finds the ball. Do not silently continue with stock COCO weights.

- [ ] **Step 2: Write `scripts/prepare_dataset.py`**

```python
"""Write the ultralytics data.yaml for the fine-tuning subset.

Class order here MUST match courtvision.detection.FINETUNED_CLASS_MAP.
"""

from __future__ import annotations

import sys
from pathlib import Path

from courtvision.detection import FINETUNED_CLASS_MAP

ROOT = Path("data/labeled/detector")


def main() -> int:
    names = [FINETUNED_CLASS_MAP[i] for i in sorted(FINETUNED_CLASS_MAP)]
    missing = [
        str(ROOT / sub)
        for sub in ("images/train", "images/val", "labels/train", "labels/val")
        if not (ROOT / sub).is_dir()
    ]
    if missing:
        print(f"FAIL — missing dataset directories: {missing}")
        return 1

    yaml_path = ROOT / "data.yaml"
    yaml_path.write_text(
        f"path: {ROOT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    n_train = len(list((ROOT / "images/train").glob("*")))
    n_val = len(list((ROOT / "images/val").glob("*")))
    print(f"OK — wrote {yaml_path}; {n_train} train / {n_val} val images; names={names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run: `./.venv/bin/python scripts/prepare_dataset.py`
Expected: `OK — wrote data/labeled/detector/data.yaml; ... names=['player', 'ball', 'rim']`

- [ ] **Step 3: Add `load_finetuned` to `src/courtvision/detection.py`**

Append to the file:

```python
def load_finetuned(weights_path: str, device: str, conf: float) -> YoloDetector:
    """Load our fine-tuned player/ball/rim detector."""
    return YoloDetector(weights_path, device, conf, FINETUNED_CLASS_MAP)
```

- [ ] **Step 4: Write `scripts/validate_v3.py`**

```python
"""V3 — Fine-tuning loop sanity check (spec §6).

Fine-tunes YOLO on the small labeled subset and compares mAP50-95 on the held-out
val split against the stock COCO baseline. Passing proves the training loop and
data pipeline work; it does not prove the detector is good enough for production.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

from courtvision.device import resolve_device

DATA_YAML = Path("data/labeled/detector/data.yaml")
CHECKPOINT = Path("checkpoints/detector.pt")
EPOCHS = 25
IMG_SIZE = 640
BATCH = 4  # small — 16 GB unified memory is shared with the OS


def main() -> int:
    if not DATA_YAML.exists():
        print(f"V3 FAIL — no dataset at {DATA_YAML}; run scripts/prepare_dataset.py")
        return 1

    device = resolve_device()

    # Baseline: stock COCO weights evaluated on our val split. It knows nothing
    # about "rim", so this number is expected to be low - that is the point.
    baseline = YOLO("yolo11n.pt").val(
        data=str(DATA_YAML), device=device, imgsz=IMG_SIZE, verbose=False
    )
    baseline_map = float(baseline.box.map)

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=device,
        project="outputs/train",
        name="detector",
        exist_ok=True,
        verbose=False,
    )
    tuned = model.val(data=str(DATA_YAML), device=device, imgsz=IMG_SIZE, verbose=False)
    tuned_map = float(tuned.box.map)

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    best = Path("outputs/train/detector/weights/best.pt")
    if best.exists():
        shutil.copy(best, CHECKPOINT)

    ok = tuned_map > baseline_map and CHECKPOINT.exists()
    verdict = "PASS" if ok else "FAIL"
    print(
        f"V3 {verdict} — mAP50-95 baseline {baseline_map:.4f} -> fine-tuned "
        f"{tuned_map:.4f} on {EPOCHS} epochs; checkpoint {CHECKPOINT}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run V3**

```bash
./.venv/bin/python scripts/validate_v3.py
```
Expected: `V3 PASS — mAP50-95 baseline 0.0xxx -> fine-tuned 0.yyyy ...` with fine-tuned above baseline. On MPS with 50–200 images and 25 epochs this takes roughly 5–20 minutes.

Per spec §9.6: if this fails twice after a fix attempt, **stop and flag it** rather than working around it. The most common real causes are a class-order mismatch between `data.yaml` and `FINETUNED_CLASS_MAP`, or label files whose coordinates aren't normalized 0–1.

- [ ] **Step 6: Commit**

```bash
git add src/courtvision/detection.py scripts/prepare_dataset.py scripts/validate_v3.py
git commit -m "feat: detector fine-tuning on labeled subset (V3)"
```

---
### Task 5: Tracking (V4)

ByteTrack assigns persistent IDs to players. Only players are tracked — the ball is small, fast and frequently occluded, so tracking it produces noise; it passes through with `track_id = -1` and stage 5 uses its position only.

**Files:**
- Create: `src/courtvision/tracking.py`, `tests/test_tracking.py`, `scripts/validate_v4.py`

**Interfaces:**
- Consumes: `Detection`, `Track`, `Frame`, `PLAYER` (Task 1); `StubDetector`, `synthetic` fixture (Task 2).
- Produces:
  - `PlayerTracker()` with `update(detections: list[Detection]) -> list[Track]`
  - `count_id_switches(frames: Sequence[Frame], truth: Sequence[dict[int, Box]]) -> int` — pure; counts how often a ground-truth player's assigned `track_id` changes

- [ ] **Step 1: Write the failing tracking tests**

Create `tests/test_tracking.py`:

```python
from courtvision.tracking import PlayerTracker, count_id_switches
from courtvision.types import BALL, PLAYER, Box, Detection, Frame, Track
from tests.fixtures.synthetic import StubDetector


def test_ball_passes_through_untracked():
    tracker = PlayerTracker()
    tracks = tracker.update([Detection(Box(0, 0, 5, 5), BALL, 0.9)])
    assert len(tracks) == 1
    assert tracks[0].label == BALL
    assert tracks[0].track_id == -1


def test_players_receive_non_negative_ids():
    tracker = PlayerTracker()
    detections = [
        Detection(Box(10, 10, 40, 80), PLAYER, 0.9),
        Detection(Box(200, 10, 230, 80), PLAYER, 0.9),
    ]
    # ByteTrack needs a couple of frames before it promotes tentative tracks.
    for _ in range(5):
        tracks = tracker.update(detections)
    players = [t for t in tracks if t.label == PLAYER]
    assert len(players) == 2
    assert all(t.track_id >= 0 for t in players)
    assert len({t.track_id for t in players}) == 2


def test_ids_are_stable_on_the_synthetic_clip(synthetic):
    """Players move on non-crossing lanes, so a correct tracker never switches IDs."""
    stub = StubDetector(synthetic)
    tracker = PlayerTracker()
    frames = []
    for index in range(synthetic.n_frames):
        tracks = tracker.update(stub.detect_at(index))
        frames.append(Frame(index, index / synthetic.fps, tuple(tracks)))

    switches = count_id_switches(frames, synthetic.player_boxes)
    assert switches == 0, f"expected no ID switches on non-crossing lanes, got {switches}"


def test_count_id_switches_detects_a_swap():
    box_a, box_b = Box(0, 0, 10, 20), Box(100, 0, 110, 20)
    truth = [{0: box_a, 1: box_b}, {0: box_a, 1: box_b}]
    frames = [
        Frame(0, 0.0, (Track(7, box_a, PLAYER, 0.9), Track(8, box_b, PLAYER, 0.9))),
        # Player 0's box is now labelled with the other id — one switch.
        Frame(1, 0.1, (Track(8, box_a, PLAYER, 0.9), Track(7, box_b, PLAYER, 0.9))),
    ]
    assert count_id_switches(frames, truth) == 2


def test_count_id_switches_is_zero_when_stable():
    box_a = Box(0, 0, 10, 20)
    truth = [{0: box_a}, {0: box_a}]
    frames = [
        Frame(0, 0.0, (Track(7, box_a, PLAYER, 0.9),)),
        Frame(1, 0.1, (Track(7, box_a, PLAYER, 0.9),)),
    ]
    assert count_id_switches(frames, truth) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_tracking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'courtvision.tracking'`

- [ ] **Step 3: Write `src/courtvision/tracking.py`**

```python
"""Stage 3 — persistent player identities via ByteTrack.

Only players are tracked. The ball is small, fast and often occluded; running it
through a tracker yields flickering IDs that help nothing, since stage 5 needs
only its position. Ball and rim therefore pass through with track_id = -1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import supervision as sv

from courtvision.types import PLAYER, Box, Detection, Frame, Track

UNTRACKED = -1


class PlayerTracker:
    """ByteTrack behind a `Detection` list in / `Track` list out interface."""

    def __init__(self) -> None:
        self._tracker = sv.ByteTrack()

    def update(self, detections: list[Detection]) -> list[Track]:
        players = [d for d in detections if d.label == PLAYER]
        others = [d for d in detections if d.label != PLAYER]

        tracks: list[Track] = [
            Track(UNTRACKED, d.box, d.label, d.conf) for d in others
        ]

        if players:
            sv_detections = sv.Detections(
                xyxy=np.array(
                    [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in players],
                    dtype=np.float32,
                ),
                confidence=np.array([d.conf for d in players], dtype=np.float32),
                class_id=np.zeros(len(players), dtype=int),
            )
            tracked = self._tracker.update_with_detections(sv_detections)
            for xyxy, conf, track_id in zip(
                tracked.xyxy, tracked.confidence, tracked.tracker_id
            ):
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                tracks.append(
                    Track(int(track_id), Box(x1, y1, x2, y2), PLAYER, float(conf))
                )

        return tracks


def _iou(a: Box, b: Box) -> float:
    inter_x1, inter_y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    inter_x2, inter_y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def count_id_switches(
    frames: Sequence[Frame], truth: Sequence[dict[int, Box]]
) -> int:
    """Count how often a ground-truth player's assigned track_id changes.

    Each truth box is matched to the tracked box with the highest IoU. A switch is
    counted whenever a truth player's matched track_id differs from the previous
    frame in which it was matched at all.
    """
    last_id: dict[int, int] = {}
    switches = 0

    for frame, truth_boxes in zip(frames, truth):
        players = frame.players()
        for truth_index, truth_box in truth_boxes.items():
            if not players:
                continue
            best = max(players, key=lambda t: _iou(t.box, truth_box))
            if _iou(best.box, truth_box) <= 0.0:
                continue
            previous = last_id.get(truth_index)
            if previous is not None and previous != best.track_id:
                switches += 1
            last_id[truth_index] = best.track_id

    return switches
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_tracking.py -v`
Expected: PASS, 5 passed.

If `test_ids_are_stable_on_the_synthetic_clip` fails, the tracker — not the test — is wrong: the synthetic players never cross lanes, so any switch is a genuine defect.

- [ ] **Step 5: Write `scripts/validate_v4.py`**

```python
"""V4 — Tracking sanity check (spec §6).

Runs the fine-tuned detector + ByteTrack over a real 10-second clip and writes an
annotated video with track IDs drawn, for visual inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from courtvision.config import Config
from courtvision.detection import load_finetuned
from courtvision.device import resolve_device
from courtvision.extraction import extract_frames
from courtvision.tracking import PlayerTracker
from courtvision.types import PLAYER

CLIP = Path("data/raw_clips/sample.mp4")
CHECKPOINT = Path("checkpoints/detector.pt")
OUT_PATH = Path("outputs/v4_tracking.mp4")
MAX_SECONDS = 10


def main() -> int:
    if not CLIP.exists() or not CHECKPOINT.exists():
        print(f"V4 FAIL — need clip at {CLIP} and checkpoint at {CHECKPOINT}")
        return 1

    config = Config()
    detector = load_finetuned(str(CHECKPOINT), resolve_device(), config.detector_conf)
    tracker = PlayerTracker()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    seen_ids: set[int] = set()
    n_frames = 0

    for index, time_s, image in extract_frames(str(CLIP), config.target_fps):
        if time_s > MAX_SECONDS:
            break
        tracks = tracker.update(detector.detect(image))
        for track in tracks:
            if track.label != PLAYER:
                continue
            seen_ids.add(track.track_id)
            cv2.rectangle(
                image,
                (int(track.box.x1), int(track.box.y1)),
                (int(track.box.x2), int(track.box.y2)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image,
                f"#{track.track_id}",
                (int(track.box.x1), int(track.box.y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        if writer is None:
            height, width = image.shape[:2]
            writer = cv2.VideoWriter(
                str(OUT_PATH),
                cv2.VideoWriter_fourcc(*"mp4v"),
                config.target_fps,
                (width, height),
            )
        writer.write(image)
        n_frames += 1

    if writer is not None:
        writer.release()

    # 10 players on court; many more unique IDs than that means heavy switching.
    ok = n_frames > 0 and len(seen_ids) <= 20
    verdict = "PASS" if ok else "FAIL"
    print(
        f"V4 {verdict} — {n_frames} frames, {len(seen_ids)} unique track IDs "
        f"(want <=20); review {OUT_PATH} for ID stability when players cross"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run V4 and watch the video**

```bash
./.venv/bin/python scripts/validate_v4.py && open outputs/v4_tracking.mp4
```
The unique-ID count is a proxy; the spec's real criterion is visual. **Watch the video** and confirm IDs stay attached to players through crossings. Occasional switches during heavy occlusion are acceptable; constant renumbering is not.

- [ ] **Step 7: Commit**

```bash
git add src/courtvision/tracking.py tests/test_tracking.py scripts/validate_v4.py
git commit -m "feat: ByteTrack player tracking with ID-switch metric (V4)"
```

---

### Task 6: Team assignment (V5)

k-means with k=2 over jersey colors. Two traps this task must handle explicitly: k-means cluster labels are arbitrary between runs, and a single frame's crop can be wrong (occlusion, motion blur). Both are solved here — deterministic cluster ordering, plus per-track majority voting across frames.

**Files:**
- Create: `src/courtvision/team_assignment.py`, `tests/test_team_assignment.py`, `scripts/validate_v5.py`

**Interfaces:**
- Consumes: `Box`, `Track`, `Frame`, `TEAMS` (Task 1); `synthetic` fixture (Task 2).
- Produces:
  - `torso_crop(image: np.ndarray, box: Box) -> np.ndarray`
  - `mean_lab_color(crop: np.ndarray) -> np.ndarray` — shape `(3,)` float
  - `assign_teams(samples: list[tuple[int, np.ndarray]]) -> dict[int, str]` mapping `track_id -> "A" | "B"`
  - `collect_samples(images, frames) -> list[tuple[int, np.ndarray]]`

- [ ] **Step 1: Write the failing team-assignment tests**

Create `tests/test_team_assignment.py`:

```python
import numpy as np

from courtvision.team_assignment import (
    assign_teams,
    collect_samples,
    mean_lab_color,
    torso_crop,
)
from courtvision.types import PLAYER, Box, Frame, Track
from tests.fixtures.synthetic import StubDetector


def test_torso_crop_is_inside_the_box():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = torso_crop(image, Box(10, 10, 50, 90))
    assert crop.size > 0
    # Narrower and shorter than the full box: it targets the jersey, not limbs.
    assert crop.shape[1] < 40
    assert crop.shape[0] < 80


def test_torso_crop_of_degenerate_box_is_empty():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert torso_crop(image, Box(10, 10, 10, 10)).size == 0


def test_mean_lab_color_distinguishes_red_from_blue():
    red = np.full((10, 10, 3), (40, 40, 200), dtype=np.uint8)   # BGR
    blue = np.full((10, 10, 3), (200, 60, 40), dtype=np.uint8)
    assert not np.allclose(mean_lab_color(red), mean_lab_color(blue), atol=5.0)


def test_assign_teams_splits_two_color_groups():
    reds = [(i, np.array([50.0, 60.0, 40.0])) for i in (0, 1)]
    blues = [(i, np.array([50.0, -20.0, -40.0])) for i in (2, 3)]
    teams = assign_teams(reds + blues)
    assert set(teams) == {0, 1, 2, 3}
    # Which group is called "A" is arbitrary; the partition is what matters.
    assert teams[0] == teams[1]
    assert teams[2] == teams[3]
    assert teams[0] != teams[2]


def test_assign_teams_is_deterministic():
    samples = [
        (0, np.array([50.0, 60.0, 40.0])),
        (1, np.array([50.0, -20.0, -40.0])),
    ]
    assert assign_teams(samples) == assign_teams(list(reversed(samples)))


def test_assign_teams_uses_majority_vote_per_track():
    # Track 0 has three red samples and one stray blue one: it must come out red.
    samples = [
        (0, np.array([50.0, 60.0, 40.0])),
        (0, np.array([50.0, 62.0, 41.0])),
        (0, np.array([50.0, 58.0, 39.0])),
        (0, np.array([50.0, -20.0, -40.0])),
        (1, np.array([50.0, -21.0, -41.0])),
        (1, np.array([50.0, -19.0, -39.0])),
    ]
    teams = assign_teams(samples)
    assert teams[0] != teams[1]


def test_assign_teams_on_synthetic_clip_recovers_the_true_partition(synthetic):
    """Team A players must share a label, team B players must share the other."""
    import cv2

    from courtvision.extraction import extract_frames

    stub = StubDetector(synthetic)
    images, frames = [], []
    for index, time_s, image in extract_frames(synthetic.path, synthetic.fps):
        detections = stub.detect_at(index)
        # The stub's box order matches the truth's player index order.
        tracks = tuple(
            Track(player_index, det.box, det.label, det.conf)
            for player_index, det in enumerate(detections)
            if det.label == PLAYER
        )
        images.append(image)
        frames.append(Frame(index, time_s, tracks))

    teams = assign_teams(collect_samples(images, frames))
    truth = synthetic.teams
    group_a = {t for t, label in truth.items() if label == "A"}
    group_b = {t for t, label in truth.items() if label == "B"}
    assert len({teams[t] for t in group_a}) == 1
    assert len({teams[t] for t in group_b}) == 1
    assert teams[next(iter(group_a))] != teams[next(iter(group_b))]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_team_assignment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'courtvision.team_assignment'`

- [ ] **Step 3: Write `src/courtvision/team_assignment.py`**

```python
"""Stage 4 — assign each track to team "A" or "B" by jersey colour.

Two things make the naive version wrong, and both are handled here:

1. k-means cluster labels are arbitrary — cluster 0 on one run is cluster 1 on the
   next. Clusters are therefore ordered deterministically by their centre before
   being named, so repeated runs agree.
2. A single frame's crop can be garbage (occlusion, motion blur, a player leaving
   frame). Every track is voted on across all its frames rather than trusted once.

Which physical team ends up called "A" is still arbitrary, and nothing downstream
may depend on it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import cv2
import numpy as np
from sklearn.cluster import KMeans

from courtvision.types import PLAYER, Box, Frame

# Fraction of the box occupied by the torso: centre horizontally, upper-middle
# vertically. Avoids the head, shorts, legs and the court either side.
TORSO_X = (0.25, 0.75)
TORSO_Y = (0.20, 0.55)


def torso_crop(image: np.ndarray, box: Box) -> np.ndarray:
    """Crop the jersey region of a player box. Returns an empty array if degenerate."""
    height, width = image.shape[:2]
    x1 = int(round(box.x1 + box.width * TORSO_X[0]))
    x2 = int(round(box.x1 + box.width * TORSO_X[1]))
    y1 = int(round(box.y1 + box.height * TORSO_Y[0]))
    y2 = int(round(box.y1 + box.height * TORSO_Y[1]))

    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2]


def mean_lab_color(crop: np.ndarray) -> np.ndarray:
    """Mean colour of a BGR crop in CIELAB, where Euclidean distance tracks perception."""
    if crop.size == 0:
        return np.zeros(3, dtype=np.float64)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float64)


def collect_samples(
    images: Sequence[np.ndarray], frames: Sequence[Frame]
) -> list[tuple[int, np.ndarray]]:
    """One (track_id, mean_lab_colour) sample per player per frame."""
    samples: list[tuple[int, np.ndarray]] = []
    for image, frame in zip(images, frames):
        for track in frame.players():
            crop = torso_crop(image, track.box)
            if crop.size == 0:
                continue
            samples.append((track.track_id, mean_lab_color(crop)))
    return samples


def assign_teams(samples: list[tuple[int, np.ndarray]]) -> dict[int, str]:
    """Cluster jersey colours into two teams and vote a single label per track."""
    if not samples:
        return {}

    track_ids = [track_id for track_id, _ in samples]
    colors = np.vstack([color for _, color in samples])

    unique_tracks = sorted(set(track_ids))
    if len(unique_tracks) < 2:
        return {track_id: "A" for track_id in unique_tracks}

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(colors)

    # Deterministic naming: order the two centres by their coordinates so the
    # same input always yields the same A/B assignment.
    centers = kmeans.cluster_centers_
    order = sorted(range(2), key=lambda i: tuple(centers[i]))
    cluster_to_team = {order[0]: "A", order[1]: "B"}

    votes: dict[int, Counter] = {}
    for track_id, cluster in zip(track_ids, kmeans.labels_):
        votes.setdefault(track_id, Counter())[cluster_to_team[int(cluster)]] += 1

    return {
        track_id: counter.most_common(1)[0][0] for track_id, counter in votes.items()
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_team_assignment.py -v`
Expected: PASS, 7 passed.

- [ ] **Step 5: Write `scripts/validate_v5.py`**

```python
"""V5 — Team assignment sanity check (spec §6).

Clusters jersey colours over a real clip and writes a contact sheet of torso crops
grouped by assigned team, so the two clusters can be eyeballed against the real
jersey colours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from courtvision.config import Config
from courtvision.detection import load_finetuned
from courtvision.device import resolve_device
from courtvision.extraction import extract_frames
from courtvision.team_assignment import assign_teams, collect_samples, torso_crop
from courtvision.tracking import PlayerTracker
from courtvision.types import Frame

CLIP = Path("data/raw_clips/sample.mp4")
CHECKPOINT = Path("checkpoints/detector.pt")
OUT_DIR = Path("outputs/v5_teams")
CROP_SIZE = (64, 64)


def main() -> int:
    if not CLIP.exists() or not CHECKPOINT.exists():
        print(f"V5 FAIL — need clip at {CLIP} and checkpoint at {CHECKPOINT}")
        return 1

    config = Config()
    detector = load_finetuned(str(CHECKPOINT), resolve_device(), config.detector_conf)
    tracker = PlayerTracker()

    images, frames = [], []
    for index, time_s, image in extract_frames(str(CLIP), config.target_fps):
        tracks = tracker.update(detector.detect(image))
        images.append(image)
        frames.append(Frame(index, time_s, tuple(tracks)))

    teams = assign_teams(collect_samples(images, frames))
    if not teams:
        print("V5 FAIL — no player tracks to cluster")
        return 1

    # Contact sheet: one row per team, one column per sampled crop.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for team in ("A", "B"):
        crops = []
        for image, frame in zip(images, frames):
            for track in frame.players():
                if teams.get(track.track_id) != team:
                    continue
                crop = torso_crop(image, track.box)
                if crop.size:
                    crops.append(cv2.resize(crop, CROP_SIZE))
            if len(crops) >= 24:
                break
        if crops:
            cv2.imwrite(str(OUT_DIR / f"team_{team}.png"), np.hstack(crops))

    counts = {team: sum(1 for t in teams.values() if t == team) for team in ("A", "B")}
    # Both teams must actually be populated — a 1-vs-rest split means clustering failed.
    ok = counts["A"] >= 2 and counts["B"] >= 2
    verdict = "PASS" if ok else "FAIL"
    print(
        f"V5 {verdict} — {len(teams)} tracks split A={counts['A']} B={counts['B']}; "
        f"compare {OUT_DIR}/team_A.png and team_B.png against the real jerseys"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run V5 and compare the contact sheets**

```bash
./.venv/bin/python scripts/validate_v5.py && open outputs/v5_teams/
```
Expected: `V5 PASS — ... A=n B=m`. Open both PNGs: each should be visually dominated by one jersey colour. If they're mixed, the usual cause is referees, the crowd, or a background wall being detected as players — tighten `detector_conf` or the torso crop fractions.

- [ ] **Step 7: Commit**

```bash
git add src/courtvision/team_assignment.py tests/test_team_assignment.py scripts/validate_v5.py
git commit -m "feat: k-means team assignment with stable labels and per-track voting (V5)"
```

---
### Task 7: Possession heuristic (V6)

Pure logic, no model — the best-suited stage in the pipeline for strict TDD. Two ideas carry it: distance normalized by player height (so the threshold works whether a player is near or far from the camera), and temporal smoothing with hysteresis (so a single noisy frame can't flip possession).

**Files:**
- Create: `src/courtvision/possession.py`, `tests/test_possession.py`, `scripts/validate_v6.py`

**Interfaces:**
- Consumes: `Frame`, `Track`, `Box`, `Config` (Task 1); `synthetic` fixture (Task 2).
- Produces:
  - `normalized_distance(player: Track, ball: Track) -> float`
  - `raw_holder(frame: Frame, max_norm_dist: float) -> int | None`
  - `smooth_holders(raw: Sequence[int | None], min_hold_frames: int, max_gap_frames: int) -> list[int | None]`
  - `possession_timeline(frames: Sequence[Frame], config: Config) -> list[int | None]`

- [ ] **Step 1: Write the failing possession tests**

Create `tests/test_possession.py`:

```python
import pytest

from courtvision.config import Config
from courtvision.possession import (
    normalized_distance,
    possession_timeline,
    raw_holder,
    smooth_holders,
)
from courtvision.types import BALL, PLAYER, Box, Frame, Track


def player(track_id: int, x: float, y: float, height: float = 80.0) -> Track:
    return Track(track_id, Box(x, y, x + 30.0, y + height), PLAYER, 0.9)


def ball(x: float, y: float) -> Track:
    return Track(-1, Box(x - 5, y - 5, x + 5, y + 5), BALL, 0.9)


def test_normalized_distance_is_scale_invariant():
    """A player twice as far away has a half-size box; the ratio must not change."""
    near = normalized_distance(player(1, 0, 0, height=80.0), ball(15.0, 120.0))
    far = normalized_distance(player(1, 0, 0, height=40.0), ball(15.0, 60.0))
    assert near == pytest.approx(far)


def test_normalized_distance_is_zero_at_the_player_centre():
    subject = player(1, 0, 0)
    center_x, center_y = subject.box.center
    assert normalized_distance(subject, ball(center_x, center_y)) == pytest.approx(0.0)


def test_raw_holder_picks_the_nearest_player():
    near, far = player(1, 100, 100), player(2, 300, 100)
    center_x, center_y = near.box.center
    frame = Frame(0, 0.0, (near, far, ball(center_x, center_y)))
    assert raw_holder(frame, max_norm_dist=0.8) == 1


def test_raw_holder_returns_none_when_ball_is_far_from_everyone():
    frame = Frame(0, 0.0, (player(1, 100, 100), ball(600.0, 20.0)))
    assert raw_holder(frame, max_norm_dist=0.8) is None


def test_raw_holder_returns_none_without_a_ball():
    frame = Frame(0, 0.0, (player(1, 100, 100),))
    assert raw_holder(frame, max_norm_dist=0.8) is None


def test_raw_holder_returns_none_without_players():
    frame = Frame(0, 0.0, (ball(100.0, 100.0),))
    assert raw_holder(frame, max_norm_dist=0.8) is None


def test_smooth_requires_sustained_possession_before_switching():
    # Player 2 appears for a single frame — noise, not a real change of possession.
    raw = [1, 1, 1, 2, 1, 1, 1]
    assert smooth_holders(raw, min_hold_frames=3, max_gap_frames=5) == [1] * 7


def test_smooth_switches_after_sustained_possession():
    raw = [1, 1, 1, 2, 2, 2, 2]
    assert smooth_holders(raw, min_hold_frames=3, max_gap_frames=5) == [1, 1, 1, 2, 2, 2, 2]


def test_smooth_bridges_a_short_gap():
    # Ball briefly occluded; the holder should carry through.
    # Player 1 needs min_hold_frames of support first to become the holder at all.
    raw = [1, 1, 1, None, None, 1]
    assert smooth_holders(raw, min_hold_frames=3, max_gap_frames=5) == [1, 1, 1, 1, 1, 1]


def test_smooth_drops_the_holder_after_a_long_gap():
    raw = [1, 1, 1, None, None, None, None]
    result = smooth_holders(raw, min_hold_frames=3, max_gap_frames=2)
    assert result == [1, 1, 1, 1, 1, None, None]


def test_smooth_needs_support_to_establish_the_first_holder():
    raw = [1, None, None, None, None, None]
    result = smooth_holders(raw, min_hold_frames=3, max_gap_frames=0)
    assert result[0] is None


def test_smooth_handles_empty_input():
    assert smooth_holders([], min_hold_frames=3, max_gap_frames=5) == []


def test_possession_timeline_matches_synthetic_truth(synthetic):
    """Ground truth with in-flight gaps bridged by the previous holder."""
    from tests.fixtures.synthetic import StubDetector

    stub = StubDetector(synthetic)
    frames = []
    for index in range(synthetic.n_frames):
        detections = stub.detect_at(index)
        tracks = []
        player_index = 0
        for det in detections:
            if det.label == PLAYER:
                tracks.append(Track(player_index, det.box, det.label, det.conf))
                player_index += 1
            else:
                tracks.append(Track(-1, det.box, det.label, det.conf))
        frames.append(Frame(index, index / synthetic.fps, tuple(tracks)))

    timeline = possession_timeline(frames, Config())

    # Gaps shorter than max_gap_frames are bridged, so compare against truth with
    # each None replaced by the preceding holder.
    expected: list[int | None] = []
    previous: int | None = None
    for holder in synthetic.holder_by_frame:
        if holder is None:
            expected.append(previous)
        else:
            expected.append(holder)
            previous = holder

    matches = sum(1 for got, want in zip(timeline, expected) if got == want)
    accuracy = matches / len(expected)
    assert accuracy >= 0.9, f"possession accuracy {accuracy:.2f} on synthetic truth"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_possession.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'courtvision.possession'`

- [ ] **Step 3: Write `src/courtvision/possession.py`**

```python
"""Stage 5 — who has the ball, by proximity plus temporal smoothing.

This is a heuristic, not ground truth (spec §10). It is wrong on contested
rebounds, blocked shots and balls in flight. The design goal is "mostly right and
never flickering", because stage 7 turns these into discrete events and a single
bad frame there becomes a fabricated possession change in the commentary.

Two decisions do the work:

* Distance is divided by the player's box height, so one threshold works for
  players near and far from the camera. A raw pixel threshold would track
  perspective, not possession.
* Switching possession requires `min_hold_frames` of consecutive support
  (hysteresis), and a missing ball carries the previous holder for up to
  `max_gap_frames` before possession is dropped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from courtvision.config import Config
from courtvision.types import Frame, Track


def normalized_distance(player: Track, ball: Track) -> float:
    """Centre-to-centre distance in units of the player's box height."""
    px, py = player.box.center
    bx, by = ball.box.center
    return math.hypot(bx - px, by - py) / max(player.box.height, 1e-6)


def raw_holder(frame: Frame, max_norm_dist: float) -> int | None:
    """Nearest player to the ball, if close enough. No temporal context."""
    ball = frame.ball()
    players = frame.players()
    if ball is None or not players:
        return None

    nearest = min(players, key=lambda p: normalized_distance(p, ball))
    if normalized_distance(nearest, ball) > max_norm_dist:
        return None
    return nearest.track_id


def smooth_holders(
    raw: Sequence[int | None], min_hold_frames: int, max_gap_frames: int
) -> list[int | None]:
    """Apply hysteresis and gap-bridging to a per-frame raw holder sequence."""
    smoothed: list[int | None] = []
    current: int | None = None
    gap = 0

    for index, candidate in enumerate(raw):
        if candidate is None:
            # Ball missing: carry the current holder for a bounded number of frames.
            gap += 1
            if gap > max_gap_frames:
                current = None
        elif candidate == current:
            gap = 0
        else:
            # A different player claims the ball. Require sustained support so a
            # single noisy frame cannot flip possession.
            window = list(raw[index : index + min_hold_frames])
            if len(window) == min_hold_frames and all(c == candidate for c in window):
                current = candidate
                gap = 0
            else:
                gap = 0
        smoothed.append(current)

    return smoothed


def possession_timeline(
    frames: Sequence[Frame], config: Config
) -> list[int | None]:
    """Per-frame holder track_id (or None) for a whole clip."""
    raw = [raw_holder(frame, config.possession_max_norm_dist) for frame in frames]
    return smooth_holders(
        raw,
        min_hold_frames=config.possession_min_hold_frames,
        max_gap_frames=config.possession_max_gap_frames,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_possession.py -v`
Expected: PASS, 13 passed.

- [ ] **Step 5: Write `scripts/validate_v6.py`**

The spec's V6 criterion is 8/10 correct against moments identified by eye. That needs a small hand-written answer key.

```python
"""V6 — Possession heuristic sanity check (spec §6).

Scores the heuristic against a hand-built answer key of known "who has the ball"
moments. Create outputs/v6_answer_key.json first by watching the annotated V4
video and recording 10 timestamps with the track ID that visibly has the ball:

    [{"time_s": 1.4, "track_id": 3}, {"time_s": 2.8, "track_id": 7}, ...]

The key is judged against the SAME track IDs the tracker produced, so build it
from outputs/v4_tracking.mp4, not from jersey numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from courtvision.config import Config
from courtvision.detection import load_finetuned
from courtvision.device import resolve_device
from courtvision.extraction import extract_frames
from courtvision.possession import possession_timeline
from courtvision.tracking import PlayerTracker
from courtvision.types import Frame

CLIP = Path("data/raw_clips/sample.mp4")
CHECKPOINT = Path("checkpoints/detector.pt")
ANSWER_KEY = Path("outputs/v6_answer_key.json")
REQUIRED_CORRECT = 8


def main() -> int:
    if not ANSWER_KEY.exists():
        print(
            f"V6 FAIL — no answer key at {ANSWER_KEY}; watch outputs/v4_tracking.mp4 "
            "and record 10 moments as [{\"time_s\": float, \"track_id\": int}, ...]"
        )
        return 1
    if not CLIP.exists() or not CHECKPOINT.exists():
        print(f"V6 FAIL — need clip at {CLIP} and checkpoint at {CHECKPOINT}")
        return 1

    config = Config()
    detector = load_finetuned(str(CHECKPOINT), resolve_device(), config.detector_conf)
    tracker = PlayerTracker()

    frames: list[Frame] = []
    for index, time_s, image in extract_frames(str(CLIP), config.target_fps):
        frames.append(Frame(index, time_s, tuple(tracker.update(detector.detect(image)))))

    timeline = possession_timeline(frames, config)
    key = json.loads(ANSWER_KEY.read_text())

    correct = 0
    misses = []
    for entry in key:
        # Nearest sampled frame to the annotated moment.
        frame = min(frames, key=lambda f: abs(f.time_s - entry["time_s"]))
        predicted = timeline[frame.index]
        if predicted == entry["track_id"]:
            correct += 1
        else:
            misses.append(
                {"time_s": entry["time_s"], "want": entry["track_id"], "got": predicted}
            )

    ok = correct >= REQUIRED_CORRECT
    verdict = "PASS" if ok else "FAIL"
    print(f"V6 {verdict} — {correct}/{len(key)} correct (need >={REQUIRED_CORRECT})")
    for miss in misses:
        print(f"  miss at {miss['time_s']:.1f}s: wanted {miss['want']}, got {miss['got']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Build the answer key and run V6**

Watch `outputs/v4_tracking.mp4`, note 10 unambiguous moments, write `outputs/v6_answer_key.json`, then:

```bash
./.venv/bin/python scripts/validate_v6.py
```
Expected: `V6 PASS — 8/10 correct` or better. The printed misses matter as much as the score: per the spec, misses should be genuinely ambiguous moments (contested rebounds, passes in flight), not a player standing alone with the ball. If a miss is obvious, tune `possession_max_norm_dist` in `Config` — do not edit the answer key to match the output.

- [ ] **Step 7: Commit**

```bash
git add src/courtvision/possession.py tests/test_possession.py scripts/validate_v6.py
git commit -m "feat: possession heuristic with scale-invariant distance and hysteresis (V6)"
```

---

### Task 8: Action classification (V7)

Fine-tune a pretrained VideoMAE on 16-frame windows. The windowing logic is pure and tested; the model is gated by V7. A `StubActionClassifier` keeps later tasks unblocked if the classifier is still training.

**Files:**
- Create: `src/courtvision/action_classifier.py`, `tests/test_action_classifier.py`, `scripts/validate_v7.py`
- Modify: `tests/fixtures/synthetic.py` (add `StubActionClassifier`)

**Interfaces:**
- Consumes: `ActionWindow`, `ACTIONS`, `Config` (Task 1).
- Produces:
  - `plan_windows(n_frames: int, size: int, stride: int) -> list[tuple[int, int]]` — inclusive `(start, end)` index pairs
  - `ActionClassifier` protocol with `classify(self, clip: np.ndarray) -> tuple[str, float]`
  - `VideoMaeClassifier(weights_dir: str, device: str)` implementing it
  - `classify_windows(images, frames, classifier, config) -> list[ActionWindow]`
  - `StubActionClassifier(label: str = "dribble")` in the test fixtures

- [ ] **Step 1: Write the failing windowing tests**

Create `tests/test_action_classifier.py`:

```python
import numpy as np

from courtvision.action_classifier import classify_windows, plan_windows
from courtvision.config import Config
from courtvision.types import ACTIONS, Frame
from tests.fixtures.synthetic import StubActionClassifier


def test_plan_windows_covers_a_clip_at_stride():
    windows = plan_windows(n_frames=32, size=16, stride=8)
    assert windows == [(0, 15), (8, 23), (16, 31)]


def test_plan_windows_bounds_are_inclusive_and_sized():
    for start, end in plan_windows(n_frames=64, size=16, stride=8):
        assert end - start + 1 == 16


def test_plan_windows_returns_nothing_for_too_short_a_clip():
    assert plan_windows(n_frames=10, size=16, stride=8) == []


def test_plan_windows_handles_exact_single_window():
    assert plan_windows(n_frames=16, size=16, stride=8) == [(0, 15)]


def test_classify_windows_produces_labelled_windows():
    config = Config()
    n = 32
    images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(n)]
    frames = [Frame(i, i / config.target_fps, ()) for i in range(n)]

    windows = classify_windows(images, frames, StubActionClassifier("shot"), config)

    assert len(windows) == 3
    assert all(w.label == "shot" for w in windows)
    assert all(w.label in ACTIONS for w in windows)
    assert windows[0].start_index == 0
    assert windows[0].end_index == 15
    assert windows[0].start_time_s == 0.0
    assert windows[1].start_time_s > windows[0].start_time_s
```

- [ ] **Step 2: Add `StubActionClassifier` to `tests/fixtures/synthetic.py`**

Append to that file:

```python
class StubActionClassifier:
    """Always returns the same label. Lets stages 7 and 9 be tested without weights."""

    def __init__(self, label: str = "dribble", conf: float = 0.99) -> None:
        self._label = label
        self._conf = conf

    def classify(self, clip: np.ndarray) -> tuple[str, float]:
        return self._label, self._conf
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_action_classifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'courtvision.action_classifier'`

- [ ] **Step 4: Write `src/courtvision/action_classifier.py`**

```python
"""Stage 6 — classify short windows of play into the five spec actions.

Windowing is pure and tested. The model is a fine-tuned VideoMAE, which expects
exactly 16 frames at 224x224 — hence `Config.action_window_frames = 16`.
Per spec §4, nothing here is trained from scratch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from courtvision.config import Config
from courtvision.types import ACTIONS, ActionWindow, Frame

FRAME_SIZE = 224


def plan_windows(n_frames: int, size: int, stride: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs. Partial trailing windows are dropped."""
    if n_frames < size:
        return []
    return [
        (start, start + size - 1)
        for start in range(0, n_frames - size + 1, stride)
    ]


class ActionClassifier(Protocol):
    def classify(self, clip: np.ndarray) -> tuple[str, float]: ...


class VideoMaeClassifier:
    """Fine-tuned VideoMAE behind the `ActionClassifier` protocol."""

    def __init__(self, weights_dir: str, device: str) -> None:
        import torch
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        self._torch = torch
        self._processor = VideoMAEImageProcessor.from_pretrained(weights_dir)
        self._model = VideoMAEForVideoClassification.from_pretrained(weights_dir)
        self._model.to(device).eval()
        self._device = device

    def classify(self, clip: np.ndarray) -> tuple[str, float]:
        """clip: (n_frames, height, width, 3) uint8 RGB."""
        inputs = self._processor(list(clip), return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        probabilities = logits.softmax(dim=-1)[0]
        index = int(probabilities.argmax())
        return self._model.config.id2label[index], float(probabilities[index])


def classify_windows(
    images: Sequence[np.ndarray],
    frames: Sequence[Frame],
    classifier: ActionClassifier,
    config: Config,
) -> list[ActionWindow]:
    """Slide a window over the clip and classify each one."""
    import cv2

    windows: list[ActionWindow] = []
    for start, end in plan_windows(
        len(images), config.action_window_frames, config.action_stride_frames
    ):
        clip = np.stack(
            [
                cv2.cvtColor(
                    cv2.resize(images[i], (FRAME_SIZE, FRAME_SIZE)), cv2.COLOR_BGR2RGB
                )
                for i in range(start, end + 1)
            ]
        )
        label, conf = classifier.classify(clip)
        if label not in ACTIONS:
            label = "other"
        windows.append(
            ActionWindow(
                start_index=start,
                end_index=end,
                start_time_s=frames[start].time_s,
                end_time_s=frames[end].time_s,
                label=label,
                conf=conf,
            )
        )
    return windows
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_action_classifier.py -v`
Expected: PASS, 5 passed.

- [ ] **Step 6: Write `scripts/validate_v7.py`**

Expects a labeled clip subset at `data/labeled/actions/<action>/*.mp4`, one directory per label from `ACTIONS`.

```python
"""V7 — Action classifier sanity check (spec §6).

Fine-tunes VideoMAE on a small labeled clip subset and reports held-out accuracy.
With 5 classes, random chance is ~20%; the spec asks for well above that, not
barely above it, so the bar here is 40%.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

from courtvision.device import resolve_device
from courtvision.types import ACTIONS

DATA_DIR = Path("data/labeled/actions")
OUT_DIR = Path("checkpoints/action_classifier")
BASE_MODEL = "MCG-NJU/videomae-base"
N_FRAMES = 16
FRAME_SIZE = 224
EPOCHS = 8
VAL_FRACTION = 0.2
CHANCE = 1.0 / len(ACTIONS)
REQUIRED_ACCURACY = 0.40


def load_clip(path: Path) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(
            cv2.cvtColor(cv2.resize(image, (FRAME_SIZE, FRAME_SIZE)), cv2.COLOR_BGR2RGB)
        )
    capture.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    # Sample N_FRAMES evenly, repeating the last frame if the clip is short.
    indices = np.linspace(0, len(frames) - 1, N_FRAMES).round().astype(int)
    return np.stack([frames[i] for i in indices])


def main() -> int:
    import torch
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    if not DATA_DIR.is_dir():
        print(f"V7 FAIL — no labeled clips at {DATA_DIR}/<action>/*.mp4")
        return 1

    samples: list[tuple[Path, int]] = []
    for label_index, action in enumerate(ACTIONS):
        for clip_path in sorted((DATA_DIR / action).glob("*.mp4")):
            samples.append((clip_path, label_index))

    if len(samples) < 20:
        print(f"V7 FAIL — only {len(samples)} labeled clips; need at least 20")
        return 1

    random.Random(0).shuffle(samples)
    split = int(len(samples) * (1 - VAL_FRACTION))
    train, val = samples[:split], samples[split:]

    device = resolve_device()
    processor = VideoMAEImageProcessor.from_pretrained(BASE_MODEL)
    model = VideoMAEForVideoClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(ACTIONS),
        id2label={i: a for i, a in enumerate(ACTIONS)},
        label2id={a: i for i, a in enumerate(ACTIONS)},
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(EPOCHS):
        random.Random(epoch).shuffle(train)
        total_loss = 0.0
        for clip_path, label_index in train:
            inputs = processor(list(load_clip(clip_path)), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = torch.tensor([label_index], device=device)
            loss = model(**inputs, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss)
        print(f"  epoch {epoch + 1}/{EPOCHS} train loss {total_loss / len(train):.4f}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for clip_path, label_index in val:
            inputs = processor(list(load_clip(clip_path)), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            predicted = int(model(**inputs).logits.argmax(dim=-1))
            correct += int(predicted == label_index)

    accuracy = correct / len(val)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)

    ok = accuracy >= REQUIRED_ACCURACY
    verdict = "PASS" if ok else "FAIL"
    print(
        f"V7 {verdict} — held-out accuracy {accuracy:.3f} on {len(val)} clips "
        f"(chance {CHANCE:.3f}, required {REQUIRED_ACCURACY:.2f}); saved to {OUT_DIR}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: Run V7**

```bash
./.venv/bin/python scripts/validate_v7.py
```
Expected: `V7 PASS — held-out accuracy 0.xxx ...` above 0.40.

Spec §10 warns this stage is the most likely to plateau, and §9.6 says to stop and flag rather than work around it. If accuracy sits near chance after one honest fix attempt, **stop and report** — the likely answer is more labeled clips, not a different architecture. Note the disk cost: `videomae-base` is roughly 400 MB, and the saved fine-tune another ~400 MB. Check `df -h .` before running.

- [ ] **Step 8: Commit**

```bash
git add src/courtvision/action_classifier.py tests/ scripts/validate_v7.py
git commit -m "feat: VideoMAE action classification over sliding windows (V7)"
```

---
### Task 9: Event structuring

Pure logic that merges possession, teams and action windows into the discrete `Event` list. Spec §10 is blunt about why this matters: if these events are noisy, the LLM will confidently narrate the noise. This is the highest-leverage stage for correctness and gets thorough tests.

**Files:**
- Create: `src/courtvision/events.py`, `tests/test_events.py`

**Interfaces:**
- Consumes: `ActionWindow`, `Event` (Task 1); `possession_timeline` output (Task 7); `assign_teams` output (Task 6).
- Produces:
  - `dominant_holder(holders: Sequence[int | None], start: int, end: int) -> int | None`
  - `build_events(windows, holders, teams) -> list[Event]`

- [ ] **Step 1: Write the failing event tests**

Create `tests/test_events.py`:

```python
from courtvision.events import build_events, dominant_holder
from courtvision.types import ActionWindow


def window(start: int, end: int, label: str = "dribble") -> ActionWindow:
    return ActionWindow(start, end, start / 10.0, end / 10.0, label, 0.9)


def test_dominant_holder_picks_the_majority():
    assert dominant_holder([1, 1, 2, 1], 0, 3) == 1


def test_dominant_holder_ignores_none():
    assert dominant_holder([None, 3, 3, None], 0, 3) == 3


def test_dominant_holder_returns_none_when_all_none():
    assert dominant_holder([None, None], 0, 1) is None


def test_dominant_holder_respects_the_window_bounds():
    # Player 9 dominates overall but is outside the requested window.
    assert dominant_holder([1, 1, 9, 9, 9], 0, 1) == 1


def test_build_events_attaches_holder_and_team():
    events = build_events([window(0, 3, "shot")], [5, 5, 5, 5], {5: "A"})
    assert len(events) == 1
    assert events[0].action == "shot"
    assert events[0].track_id == 5
    assert events[0].team == "A"
    assert events[0].time_s == 0.0


def test_build_events_marks_the_first_event_as_no_possession_change():
    events = build_events([window(0, 3)], [5, 5, 5, 5], {5: "A"})
    assert events[0].possession_change is False


def test_build_events_flags_a_change_of_holder():
    windows = [window(0, 1), window(2, 3)]
    events = build_events(windows, [5, 5, 6, 6], {5: "A", 6: "B"})
    assert events[0].possession_change is False
    assert events[1].possession_change is True
    assert events[1].track_id == 6
    assert events[1].team == "B"


def test_build_events_does_not_flag_a_repeated_holder():
    windows = [window(0, 1), window(2, 3)]
    events = build_events(windows, [5, 5, 5, 5], {5: "A"})
    assert events[1].possession_change is False


def test_build_events_handles_a_window_with_no_holder():
    events = build_events([window(0, 1)], [None, None], {})
    assert events[0].track_id is None
    assert events[0].team is None
    assert events[0].possession_change is False


def test_build_events_handles_a_holder_with_no_team():
    events = build_events([window(0, 1)], [5, 5], {})
    assert events[0].track_id == 5
    assert events[0].team is None


def test_build_events_returns_events_in_time_order():
    windows = [window(4, 5), window(0, 1), window(2, 3)]
    events = build_events(windows, [1] * 6, {1: "A"})
    assert [e.time_s for e in events] == sorted(e.time_s for e in events)


def test_build_events_on_empty_input():
    assert build_events([], [], {}) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'courtvision.events'`

- [ ] **Step 3: Write `src/courtvision/events.py`**

```python
"""Stage 7 — turn per-frame signals into discrete events.

Spec §10: commentary quality depends entirely on this stage. If these events are
noisy, stage 8 will narrate the noise confidently. So the holder for a window is
the majority vote across its frames, not the value at some single instant.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from courtvision.types import ActionWindow, Event


def dominant_holder(
    holders: Sequence[int | None], start: int, end: int
) -> int | None:
    """Most frequent non-None holder across the inclusive frame range."""
    votes = Counter(h for h in holders[start : end + 1] if h is not None)
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def build_events(
    windows: Sequence[ActionWindow],
    holders: Sequence[int | None],
    teams: dict[int, str],
) -> list[Event]:
    """Merge action windows, possession and team labels into ordered events."""
    events: list[Event] = []
    previous_holder: int | None = None

    for window in sorted(windows, key=lambda w: w.start_time_s):
        holder = dominant_holder(holders, window.start_index, window.end_index)
        # A change is only meaningful between two known holders; going to or from
        # "nobody" is the ball being in flight, not a turnover.
        changed = (
            holder is not None
            and previous_holder is not None
            and holder != previous_holder
        )
        events.append(
            Event(
                time_s=window.start_time_s,
                track_id=holder,
                team=teams.get(holder) if holder is not None else None,
                action=window.label,
                possession_change=changed,
            )
        )
        if holder is not None:
            previous_holder = holder

    return events
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_events.py -v`
Expected: PASS, 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/courtvision/events.py tests/test_events.py
git commit -m "feat: event structuring from possession, teams and action windows"
```

---

### Task 10: Commentary generation (V8)

A LangGraph graph — `narrate` → `validate` → retry-or-finish — wrapping a Claude call. The validator is the important part: it mechanically rejects any line that names a player or team not present in the event it describes. Spec §10 warns the LLM will narrate noise confidently; this makes fabrication a caught error rather than a silent one.

**Requires credentials.** Set `ANTHROPIC_API_KEY`, or install the `ant` CLI and run `ant auth login`, before Step 7. The unit tests use a fake narrator and need no key.

**Files:**
- Create: `src/courtvision/commentary.py`, `tests/test_commentary.py`, `scripts/validate_v8.py`

**Interfaces:**
- Consumes: `Event`, `Config` (Task 1).
- Produces:
  - `CommentaryLine(BaseModel)` with `time_s: float`, `text: str`
  - `Commentary(BaseModel)` with `lines: list[CommentaryLine]`
  - `format_timestamp(seconds: float) -> str` → `"M:SS"`
  - `validate_commentary(events: Sequence[Event], lines: Sequence[CommentaryLine]) -> list[str]`
  - `Narrator` protocol with `narrate(self, events: Sequence[Event]) -> Commentary`
  - `AnthropicNarrator(config: Config)` implementing it
  - `generate_commentary(events, narrator, config) -> tuple[list[CommentaryLine], list[str]]`

- [ ] **Step 1: Write the failing commentary tests**

Create `tests/test_commentary.py`:

```python
import pytest

from courtvision.commentary import (
    Commentary,
    CommentaryLine,
    format_timestamp,
    generate_commentary,
    validate_commentary,
)
from courtvision.config import Config
from courtvision.types import Event


def event(time_s=0.0, track_id=7, team="A", action="shot", change=False) -> Event:
    return Event(time_s, track_id, team, action, change)


class FakeNarrator:
    """Returns canned responses in order, so the graph is testable without an API key."""

    def __init__(self, *responses: Commentary) -> None:
        self._responses = list(responses)
        self.calls = 0

    def narrate(self, events) -> Commentary:
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def test_format_timestamp():
    assert format_timestamp(0.0) == "0:00"
    assert format_timestamp(14.2) == "0:14"
    assert format_timestamp(75.0) == "1:15"
    assert format_timestamp(605.0) == "10:05"


def test_validate_accepts_a_faithful_line():
    events = [event(time_s=14.0, track_id=7, team="A")]
    lines = [CommentaryLine(time_s=14.0, text="Player 7 (Team A) rises for a jump shot.")]
    assert validate_commentary(events, lines) == []


def test_validate_rejects_a_fabricated_player():
    events = [event(track_id=7)]
    lines = [CommentaryLine(time_s=0.0, text="Player 3 drives to the basket.")]
    errors = validate_commentary(events, lines)
    assert len(errors) == 1
    assert "Player 3" in errors[0]


def test_validate_rejects_a_fabricated_team():
    events = [event(team="A")]
    lines = [CommentaryLine(time_s=0.0, text="Player 7 of Team B pulls up.")]
    assert validate_commentary(events, lines) != []


def test_validate_rejects_a_line_count_mismatch():
    events = [event(), event(time_s=1.0)]
    lines = [CommentaryLine(time_s=0.0, text="Player 7 shoots.")]
    assert any("count" in e.lower() for e in validate_commentary(events, lines))


def test_validate_rejects_a_drifting_timestamp():
    events = [event(time_s=14.0)]
    lines = [CommentaryLine(time_s=40.0, text="Player 7 shoots.")]
    assert any("timestamp" in e.lower() for e in validate_commentary(events, lines))


def test_validate_allows_a_line_naming_no_player():
    events = [Event(0.0, None, None, "other", False)]
    lines = [CommentaryLine(time_s=0.0, text="The ball is loose under the basket.")]
    assert validate_commentary(events, lines) == []


def test_generate_returns_lines_when_the_first_attempt_is_clean():
    events = [event()]
    good = Commentary(lines=[CommentaryLine(time_s=0.0, text="Player 7 (Team A) shoots.")])
    narrator = FakeNarrator(good)
    lines, errors = generate_commentary(events, narrator, Config())
    assert errors == []
    assert len(lines) == 1
    assert narrator.calls == 1


def test_generate_retries_once_on_a_fabrication_then_succeeds():
    events = [event()]
    bad = Commentary(lines=[CommentaryLine(time_s=0.0, text="Player 3 shoots.")])
    good = Commentary(lines=[CommentaryLine(time_s=0.0, text="Player 7 (Team A) shoots.")])
    narrator = FakeNarrator(bad, good)
    lines, errors = generate_commentary(events, narrator, Config())
    assert narrator.calls == 2
    assert errors == []
    assert "Player 7" in lines[0].text


def test_generate_gives_up_after_max_attempts_and_reports_errors():
    events = [event()]
    bad = Commentary(lines=[CommentaryLine(time_s=0.0, text="Player 3 shoots.")])
    narrator = FakeNarrator(bad)
    config = Config()
    lines, errors = generate_commentary(events, narrator, config)
    assert narrator.calls == config.llm_max_attempts
    assert errors != []


def test_generate_on_empty_events_makes_no_call():
    narrator = FakeNarrator(Commentary(lines=[]))
    lines, errors = generate_commentary([], narrator, Config())
    assert lines == []
    assert errors == []
    assert narrator.calls == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_commentary.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'courtvision.commentary'`

- [ ] **Step 3: Write `src/courtvision/commentary.py`**

```python
"""Stage 8 — natural-language play-by-play from structured events.

The LLM's job is narration, not decision-making: every fact it needs is already
computed upstream. That keeps hallucination risk low (spec §3), and
`validate_commentary` enforces it mechanically — any player or team named in a
line that isn't in the corresponding event is an error, and the graph retries.

LangGraph owns the control flow (narrate -> validate -> retry or finish); the
Anthropic SDK is called directly inside the narrate node.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from courtvision.config import Config
from courtvision.types import Event

PLAYER_MENTION = re.compile(r"\bPlayer\s+(\d+)\b", re.IGNORECASE)
TEAM_MENTION = re.compile(r"\bTeam\s+([AB])\b", re.IGNORECASE)
MAX_TIMESTAMP_DRIFT_S = 0.5

SYSTEM_PROMPT = """You are a basketball play-by-play commentator.

You will receive a JSON list of events that were computed by a computer vision
pipeline. Write exactly one line of commentary per event, in the same order.

Rules, which are absolute:
- Describe ONLY what is in the events. Never invent a player, team, action, score,
  foul, or game situation that is not present in the input.
- Refer to a player as "Player <track_id>" using the exact track_id from the event.
- Refer to a team as "Team A" or "Team B" using the exact team from the event.
- If an event has a null track_id, do not name any player in that line.
- Set each line's time_s to exactly the event's time_s.
- Vary the phrasing so it reads like live commentary, but never at the cost of accuracy.
"""


class CommentaryLine(BaseModel):
    time_s: float = Field(description="Timestamp in seconds, copied from the event")
    text: str = Field(description="One sentence of play-by-play commentary")


class Commentary(BaseModel):
    lines: list[CommentaryLine]


def format_timestamp(seconds: float) -> str:
    """Seconds to M:SS, for the human-readable log."""
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def validate_commentary(
    events: Sequence[Event], lines: Sequence[CommentaryLine]
) -> list[str]:
    """Return a list of fabrication/consistency errors. Empty means the output is faithful."""
    errors: list[str] = []
    if len(lines) != len(events):
        errors.append(
            f"line count {len(lines)} does not match event count {len(events)}"
        )
        return errors

    for index, (event, line) in enumerate(zip(events, lines)):
        if abs(line.time_s - event.time_s) > MAX_TIMESTAMP_DRIFT_S:
            errors.append(
                f"line {index}: timestamp {line.time_s} drifts from event {event.time_s}"
            )

        for mentioned in PLAYER_MENTION.findall(line.text):
            if event.track_id is None or int(mentioned) != event.track_id:
                errors.append(
                    f"line {index}: names Player {mentioned}, "
                    f"but the event's player is {event.track_id}"
                )

        for mentioned in TEAM_MENTION.findall(line.text):
            if event.team is None or mentioned.upper() != event.team:
                errors.append(
                    f"line {index}: names Team {mentioned.upper()}, "
                    f"but the event's team is {event.team}"
                )

    return errors


def events_to_payload(events: Sequence[Event]) -> str:
    return json.dumps(
        [
            {
                "time_s": round(e.time_s, 2),
                "track_id": e.track_id,
                "team": e.team,
                "action": e.action,
                "possession_change": e.possession_change,
            }
            for e in events
        ],
        indent=2,
    )


class Narrator(Protocol):
    def narrate(self, events: Sequence[Event]) -> Commentary: ...


class AnthropicNarrator:
    """Calls Claude to narrate a list of events."""

    def __init__(self, config: Config) -> None:
        import anthropic

        # Credentials resolve from ANTHROPIC_API_KEY or an `ant auth login` profile.
        self._client = anthropic.Anthropic()
        self._config = config

    def narrate(self, events: Sequence[Event]) -> Commentary:
        response = self._client.messages.parse(
            model=self._config.llm_model,
            max_tokens=self._config.llm_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": events_to_payload(events)}],
            output_format=Commentary,
        )
        # A refusal returns HTTP 200 with no usable content — check before reading.
        if response.stop_reason == "refusal":
            raise RuntimeError(f"model refused to narrate: {response.stop_details}")
        return response.parsed_output


class CommentaryState(TypedDict):
    events: list[Event]
    lines: list[CommentaryLine]
    errors: list[str]
    attempts: int


def build_graph(narrator: Narrator, config: Config):
    """narrate -> validate -> (retry if fabricated and attempts remain, else finish)."""

    def narrate(state: CommentaryState) -> dict:
        commentary = narrator.narrate(state["events"])
        return {"lines": list(commentary.lines), "attempts": state["attempts"] + 1}

    def validate(state: CommentaryState) -> dict:
        return {"errors": validate_commentary(state["events"], state["lines"])}

    def route(state: CommentaryState) -> str:
        if not state["errors"]:
            return "done"
        if state["attempts"] >= config.llm_max_attempts:
            return "done"
        return "retry"

    graph = StateGraph(CommentaryState)
    graph.add_node("narrate", narrate)
    graph.add_node("validate", validate)
    graph.add_edge(START, "narrate")
    graph.add_edge("narrate", "validate")
    graph.add_conditional_edges("validate", route, {"retry": "narrate", "done": END})
    return graph.compile()


def generate_commentary(
    events: Sequence[Event], narrator: Narrator, config: Config
) -> tuple[list[CommentaryLine], list[str]]:
    """Run the graph. Returns (lines, errors); non-empty errors mean it gave up."""
    if not events:
        return [], []

    final = build_graph(narrator, config).invoke(
        {"events": list(events), "lines": [], "errors": [], "attempts": 0}
    )
    return final["lines"], final["errors"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_commentary.py -v`
Expected: PASS, 11 passed. No API key needed — every test injects `FakeNarrator`.

- [ ] **Step 5: Write `scripts/validate_v8.py`**

```python
"""V8 — Commentary generation sanity check (spec §6).

Feeds hand-constructed events through the real LangGraph + Claude path and prints
the output for manual review, plus the mechanical fabrication check.
"""

from __future__ import annotations

import os
import sys

from courtvision.commentary import (
    AnthropicNarrator,
    format_timestamp,
    generate_commentary,
)
from courtvision.config import Config
from courtvision.types import Event

# Hand-built events covering the interesting cases: a possession change, a null
# holder (loose ball), and both teams.
EVENTS = [
    Event(0.0, 7, "A", "dribble", False),
    Event(2.0, 7, "A", "pass", False),
    Event(4.0, 3, "A", "shot", True),
    Event(6.5, None, None, "rebound", False),
    Event(8.0, 11, "B", "dribble", True),
    Event(10.5, 11, "B", "shot", False),
]


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "V8 note — ANTHROPIC_API_KEY not set; relying on an `ant auth login` "
            "profile. If this fails to authenticate, set the key and re-run."
        )

    config = Config()
    try:
        lines, errors = generate_commentary(EVENTS, AnthropicNarrator(config), config)
    except Exception as exc:  # noqa: BLE001 - surface the real cause to the operator
        print(f"V8 FAIL — narrator raised {type(exc).__name__}: {exc}")
        return 1

    print(f"--- commentary ({config.llm_model}) ---")
    for line in lines:
        print(f"{format_timestamp(line.time_s)} — {line.text}")
    print("--- end ---")

    ok = not errors and len(lines) == len(EVENTS)
    verdict = "PASS" if ok else "FAIL"
    print(f"V8 {verdict} — {len(lines)}/{len(EVENTS)} lines, {len(errors)} fabrication errors")
    for error in errors:
        print(f"  {error}")
    if ok:
        print("  Now read the lines above: do they describe these events, and read naturally?")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run V8**

```bash
./.venv/bin/python scripts/validate_v8.py
```
Expected: `V8 PASS — 6/6 lines, 0 fabrication errors`. The automated check catches invented players and teams; the spec also asks for a human read, so **read the printed lines** and confirm they describe these events and read like commentary. Note the loose-ball event at 6.5s must not name a player.

- [ ] **Step 7: Commit**

```bash
git add src/courtvision/commentary.py tests/test_commentary.py scripts/validate_v8.py
git commit -m "feat: LangGraph commentary generation with anti-fabrication validation (V8)"
```

---
### Task 11: Output renderer

Annotated video plus the JSON log. Drawing is hard to unit-test meaningfully, so the tests cover what is checkable: the log's schema, timestamp formatting, and that rendering produces a readable video with the right frame count.

**Files:**
- Create: `src/courtvision/render.py`, `tests/test_render.py`

**Interfaces:**
- Consumes: `Frame`, `Event`, `PLAYER` (Task 1); `CommentaryLine`, `format_timestamp` (Task 10).
- Produces:
  - `TEAM_DRAW_COLORS: dict[str, tuple[int, int, int]]`
  - `draw_frame(image, frame, teams, holder_id) -> np.ndarray`
  - `render_video(images, frames, teams, holders, out_path, fps) -> int` (returns frames written)
  - `build_log(events, lines) -> dict`
  - `write_log(events, lines, out_path) -> None`

- [ ] **Step 1: Write the failing render tests**

Create `tests/test_render.py`:

```python
import json

import cv2
import numpy as np

from courtvision.commentary import CommentaryLine
from courtvision.render import build_log, draw_frame, render_video, write_log
from courtvision.types import BALL, PLAYER, Box, Event, Frame, Track


def make_frame(index: int = 0) -> Frame:
    return Frame(
        index,
        index / 10.0,
        (
            Track(1, Box(10, 10, 40, 90), PLAYER, 0.9),
            Track(2, Box(100, 10, 130, 90), PLAYER, 0.9),
            Track(-1, Box(20, 40, 30, 50), BALL, 0.8),
        ),
    )


def test_draw_frame_returns_a_same_shaped_image():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    drawn = draw_frame(image, make_frame(), {1: "A", 2: "B"}, holder_id=1)
    assert drawn.shape == image.shape
    assert drawn.dtype == np.uint8


def test_draw_frame_does_not_mutate_the_input():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    draw_frame(image, make_frame(), {1: "A", 2: "B"}, holder_id=1)
    assert image.sum() == 0


def test_draw_frame_actually_draws_something():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    drawn = draw_frame(image, make_frame(), {1: "A", 2: "B"}, holder_id=1)
    assert drawn.sum() > 0


def test_draw_frame_tolerates_missing_team_labels():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    drawn = draw_frame(image, make_frame(), {}, holder_id=None)
    assert drawn.shape == image.shape


def test_render_video_writes_every_frame(tmp_path):
    images = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(12)]
    frames = [make_frame(i) for i in range(12)]
    out = tmp_path / "out.mp4"

    written = render_video(images, frames, {1: "A", 2: "B"}, [1] * 12, str(out), fps=10)

    assert written == 12
    assert out.exists() and out.stat().st_size > 0
    capture = cv2.VideoCapture(str(out))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    capture.release()


def test_build_log_pairs_events_with_commentary():
    events = [Event(14.0, 7, "A", "shot", True)]
    lines = [CommentaryLine(time_s=14.0, text="Player 7 (Team A) shoots.")]
    log = build_log(events, lines)

    assert len(log["events"]) == 1
    entry = log["events"][0]
    assert entry["timestamp"] == "0:14"
    assert entry["time_s"] == 14.0
    assert entry["track_id"] == 7
    assert entry["team"] == "A"
    assert entry["action"] == "shot"
    assert entry["possession_change"] is True
    assert entry["commentary"] == "Player 7 (Team A) shoots."


def test_build_log_handles_missing_commentary():
    log = build_log([Event(1.0, 7, "A", "shot", False)], [])
    assert log["events"][0]["commentary"] is None


def test_write_log_produces_valid_json(tmp_path):
    out = tmp_path / "commentary.json"
    write_log(
        [Event(14.0, 7, "A", "shot", True)],
        [CommentaryLine(time_s=14.0, text="Player 7 (Team A) shoots.")],
        str(out),
    )
    parsed = json.loads(out.read_text())
    assert parsed["events"][0]["commentary"] == "Player 7 (Team A) shoots."
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'courtvision.render'`

- [ ] **Step 3: Write `src/courtvision/render.py`**

```python
"""Stage 9 — annotated video and the commentary log."""

from __future__ import annotations

import json
from collections.abc import Sequence

import cv2
import numpy as np

from courtvision.commentary import CommentaryLine, format_timestamp
from courtvision.types import BALL, Event, Frame

# BGR. Deliberately not the jersey colours — these are overlay colours, and the
# A/B assignment is arbitrary anyway.
TEAM_DRAW_COLORS: dict[str, tuple[int, int, int]] = {
    "A": (60, 60, 240),
    "B": (240, 160, 60),
}
UNKNOWN_COLOR = (160, 160, 160)
HOLDER_COLOR = (0, 255, 255)
BALL_COLOR = (0, 200, 255)


def draw_frame(
    image: np.ndarray,
    frame: Frame,
    teams: dict[int, str],
    holder_id: int | None,
) -> np.ndarray:
    """Draw player boxes coloured by team, the ball, and a highlight on the holder."""
    canvas = image.copy()

    for track in frame.players():
        team = teams.get(track.track_id)
        color = TEAM_DRAW_COLORS.get(team, UNKNOWN_COLOR) if team else UNKNOWN_COLOR
        is_holder = holder_id is not None and track.track_id == holder_id
        cv2.rectangle(
            canvas,
            (int(track.box.x1), int(track.box.y1)),
            (int(track.box.x2), int(track.box.y2)),
            HOLDER_COLOR if is_holder else color,
            4 if is_holder else 2,
        )
        label = f"#{track.track_id}" + (f" {team}" if team else "")
        cv2.putText(
            canvas,
            label,
            (int(track.box.x1), max(12, int(track.box.y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            HOLDER_COLOR if is_holder else color,
            2,
        )

    ball = frame.ball()
    if ball is not None:
        cx, cy = ball.box.center
        cv2.circle(canvas, (int(cx), int(cy)), 8, BALL_COLOR, 2)

    cv2.putText(
        canvas,
        format_timestamp(frame.time_s),
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return canvas


def render_video(
    images: Sequence[np.ndarray],
    frames: Sequence[Frame],
    teams: dict[int, str],
    holders: Sequence[int | None],
    out_path: str,
    fps: int,
) -> int:
    """Write the annotated video. Returns the number of frames written."""
    if not images:
        return 0

    height, width = images[0].shape[:2]
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {out_path}")

    written = 0
    try:
        for index, (image, frame) in enumerate(zip(images, frames)):
            holder = holders[index] if index < len(holders) else None
            writer.write(draw_frame(image, frame, teams, holder))
            written += 1
    finally:
        writer.release()
    return written


def build_log(
    events: Sequence[Event], lines: Sequence[CommentaryLine]
) -> dict:
    """Pair each event with its commentary line, positionally."""
    return {
        "events": [
            {
                "timestamp": format_timestamp(event.time_s),
                "time_s": round(event.time_s, 2),
                "track_id": event.track_id,
                "team": event.team,
                "action": event.action,
                "possession_change": event.possession_change,
                "commentary": lines[index].text if index < len(lines) else None,
            }
            for index, event in enumerate(events)
        ]
    }


def write_log(
    events: Sequence[Event], lines: Sequence[CommentaryLine], out_path: str
) -> None:
    with open(out_path, "w") as handle:
        json.dump(build_log(events, lines), handle, indent=2)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_render.py -v`
Expected: PASS, 8 passed.

- [ ] **Step 5: Run the whole suite**

Run: `./.venv/bin/python -m pytest -v`
Expected: PASS, all ~70 tests. This is the last task before the end-to-end run, so everything must be green here.

- [ ] **Step 6: Commit**

```bash
git add src/courtvision/render.py tests/test_render.py
git commit -m "feat: annotated video renderer and JSON commentary log"
```

---

### Task 12: End-to-end pipeline (V9)

Per spec §9.3, this is written **only after V1–V8 all pass individually**. It wires the finished stages together and runs on a held-out clip that was not used in any fine-tuning.

**Files:**
- Create: `scripts/run_pipeline.py`, `scripts/validate_v9.py`
- Modify: `README.md` (tick the §6 checkboxes)

**Interfaces:**
- Consumes: every stage built in Tasks 1–11.
- Produces: `run_pipeline(clip_path: str, out_dir: str, config: Config) -> dict` returning a summary dict with keys `n_frames`, `n_tracks`, `n_events`, `n_lines`, `errors`, `video_path`, `log_path`.

- [ ] **Step 1: Confirm every prior gate has passed**

```bash
for n in 1 2 3 4 5 6 7 8; do ./.venv/bin/python scripts/validate_v$n.py >/dev/null 2>&1 && echo "V$n ok" || echo "V$n NOT PASSING"; done
```
Expected: all eight report `ok`. If any does not, **stop** — spec §9.3 forbids writing the end-to-end pipeline until V1–V8 pass.

- [ ] **Step 2: Write `scripts/run_pipeline.py`**

```python
"""Full end-to-end pipeline: clip in, annotated video + commentary log out.

Per spec §9.3 this exists only after V1-V8 pass individually. Each stage is still
its own module; this script only sequences them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from courtvision.action_classifier import VideoMaeClassifier, classify_windows
from courtvision.commentary import AnthropicNarrator, generate_commentary
from courtvision.config import Config
from courtvision.detection import load_finetuned
from courtvision.device import resolve_device
from courtvision.events import build_events
from courtvision.extraction import extract_frames
from courtvision.possession import possession_timeline
from courtvision.render import render_video, write_log
from courtvision.team_assignment import assign_teams, collect_samples
from courtvision.tracking import PlayerTracker
from courtvision.types import Frame

DETECTOR = Path("checkpoints/detector.pt")
ACTION_MODEL = Path("checkpoints/action_classifier")


def run_pipeline(clip_path: str, out_dir: str, config: Config) -> dict:
    device = resolve_device()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Stages 1-3: extract, detect, track.
    detector = load_finetuned(str(DETECTOR), device, config.detector_conf)
    tracker = PlayerTracker()
    images, frames = [], []
    for index, time_s, image in extract_frames(clip_path, config.target_fps):
        images.append(image)
        frames.append(Frame(index, time_s, tuple(tracker.update(detector.detect(image)))))
    print(f"  stages 1-3: {len(frames)} frames tracked")

    # Stage 4: team assignment.
    teams = assign_teams(collect_samples(images, frames))
    print(f"  stage 4: {len(teams)} tracks assigned to teams")

    # Stage 5: possession.
    holders = possession_timeline(frames, config)
    n_held = sum(1 for h in holders if h is not None)
    print(f"  stage 5: possession resolved on {n_held}/{len(holders)} frames")

    # Stage 6: action classification.
    classifier = VideoMaeClassifier(str(ACTION_MODEL), device)
    windows = classify_windows(images, frames, classifier, config)
    print(f"  stage 6: {len(windows)} action windows classified")

    # Stage 7: event structuring.
    events = build_events(windows, holders, teams)
    print(f"  stage 7: {len(events)} events")

    # Stage 8: commentary.
    lines, errors = generate_commentary(events, AnthropicNarrator(config), config)
    print(f"  stage 8: {len(lines)} commentary lines, {len(errors)} fabrication errors")

    # Stage 9: render.
    video_path = out / "annotated.mp4"
    log_path = out / "commentary.json"
    render_video(images, frames, teams, holders, str(video_path), config.target_fps)
    write_log(events, lines, str(log_path))
    print(f"  stage 9: wrote {video_path} and {log_path}")

    return {
        "n_frames": len(frames),
        "n_tracks": len(teams),
        "n_events": len(events),
        "n_lines": len(lines),
        "errors": errors,
        "video_path": str(video_path),
        "log_path": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CourtVision pipeline")
    parser.add_argument("clip", help="path to an input .mp4")
    parser.add_argument("--out", default="outputs/run", help="output directory")
    args = parser.parse_args()

    if not Path(args.clip).exists():
        print(f"no such clip: {args.clip}")
        return 1
    for path in (DETECTOR, ACTION_MODEL):
        if not path.exists():
            print(f"missing model: {path}; run the V3 and V7 validation scripts first")
            return 1

    summary = run_pipeline(args.clip, args.out, Config())
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Write `scripts/validate_v9.py`**

```python
"""V9 — End-to-end run (spec §6).

Runs the full pipeline on a HELD-OUT clip that was not used in any fine-tuning,
and checks it completes without crashing and produces both outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from courtvision.config import Config
from scripts.run_pipeline import run_pipeline

CLIP = Path("data/raw_clips/holdout.mp4")
OUT_DIR = Path("outputs/v9")


def main() -> int:
    if not CLIP.exists():
        print(
            f"V9 FAIL — no held-out clip at {CLIP}. It must NOT be a clip used in "
            "V3 or V7 fine-tuning, or this checks nothing."
        )
        return 1

    try:
        summary = run_pipeline(str(CLIP), str(OUT_DIR), Config())
    except Exception as exc:  # noqa: BLE001 - a crash here is the failure mode under test
        print(f"V9 FAIL — pipeline raised {type(exc).__name__}: {exc}")
        return 1

    video_ok = Path(summary["video_path"]).stat().st_size > 0
    log = json.loads(Path(summary["log_path"]).read_text())
    log_ok = len(log["events"]) == summary["n_events"] and summary["n_events"] > 0
    ok = video_ok and log_ok and not summary["errors"]

    verdict = "PASS" if ok else "FAIL"
    print(
        f"V9 {verdict} — {summary['n_frames']} frames, {summary['n_tracks']} tracks, "
        f"{summary['n_events']} events, {summary['n_lines']} commentary lines, "
        f"{len(summary['errors'])} fabrication errors"
    )
    if ok:
        print(
            f"  Now spot-check 5 random moments in {summary['video_path']} against "
            f"{summary['log_path']} — that comparison is the actual V9 criterion."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Add `scripts/__init__.py` so the validator can import the pipeline**

```bash
touch scripts/__init__.py
```

- [ ] **Step 5: Run V9 on a held-out clip**

Place a clip at `data/raw_clips/holdout.mp4` that was **not** used in V3 or V7 fine-tuning, then:

```bash
./.venv/bin/python -m scripts.validate_v9
```
Expected: `V9 PASS — ...`. Then do the spot-check the script prints: open the video, pick five random moments, and confirm the log's commentary roughly matches what's happening. Spec §10 is the guide to interpreting mismatches — if the commentary is wrong, check the structured events first, not the prompt.

- [ ] **Step 6: Mark spec §6 complete in `README.md`**

Change each `- [ ]` in §6 to `- [x]`. Per spec §9.7 this is the gate that unlocks v2 work — nothing from §7 starts until this edit is real and honest. Do not tick a box whose script does not currently pass.

- [ ] **Step 7: Run the full test suite one last time**

Run: `./.venv/bin/python -m pytest -v`
Expected: PASS, all tests green.

- [ ] **Step 8: Commit**

```bash
git add scripts/ README.md
git commit -m "feat: end-to-end pipeline and V9 validation; mark spec section 6 complete"
```

---

## What v1 does not answer

Deliberately out of scope, recorded so they aren't mistaken for oversights:

- **Real player identity.** v1 says "Player 7 (Team A)" where 7 is a tracker ID that resets between clips, not a jersey number. Jersey OCR is spec §7.3.
- **The CUDA kernel (spec §7.1) cannot be written on this machine.** It needs a rented NVIDIA GPU. The profiling that should precede it (per the spec's own advice) *can* be done here — wall-clock timing per stage on MPS will still show which stage dominates, and that result should drive what gets written later.
- **Possession during contested play.** The heuristic is proximity-based and will be wrong on rebounds and balls in flight (spec §10). V6's 8/10 bar accepts this.
- **Anything in spec §7.** Not until §6 is genuinely ticked.
