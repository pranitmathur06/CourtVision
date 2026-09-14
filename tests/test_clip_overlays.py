"""Which boxes are drawn, and in whose pixel space."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from clip_overlays import CODE, window  # noqa: E402


def _cache():
    return {
        537.0: [{"cls": "player", "conf": 0.9, "xyxy": [10.4, 20.6, 60.0, 180.0]},
                {"cls": "ball", "conf": 0.8, "xyxy": [100.0, 50.0, 115.0, 65.0]}],
        537.2: [{"cls": "handler", "conf": 0.7, "xyxy": [12.0, 21.0, 62.0, 181.0]}],
        537.4: [{"cls": "player", "conf": 0.1, "xyxy": [1.0, 2.0, 3.0, 4.0]}],
    }


def test_each_step_becomes_a_frame_of_boxes():
    got = window(_cache(), 537.0, 0.6)
    assert [f[0] for f in got] == [0.0, 0.2]      # 537.4 dropped: low confidence


def test_boxes_keep_source_pixels_and_are_rounded():
    got = window(_cache(), 537.0, 0.2)
    assert got[0][1][0] == ["p", 10, 21, 60, 180]


def test_the_handler_is_its_own_code():
    got = window(_cache(), 537.2, 0.2)
    assert got[0][1][0][0] == "h"


def test_low_confidence_boxes_are_not_drawn():
    # The pipeline would not have used them, so the overlay must not imply it did.
    assert window(_cache(), 537.4, 0.2) == []


def test_a_gap_in_the_cache_is_skipped_not_faked():
    got = window({537.0: [{"cls": "ball", "conf": 0.9, "xyxy": [1, 2, 3, 4]}]}, 537.0, 1.0)
    assert len(got) == 1 and got[0][0] == 0.0


def test_every_drawn_class_has_a_code():
    assert set(CODE) == {"player", "handler", "ball", "rim"}
