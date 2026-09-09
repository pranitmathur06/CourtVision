"""Where the painted lines are, so a registration can be checked against them.

The ORB consistency test proves a registration is STABLE. It cannot prove the
registration is right: an error that is systematic -- a constant offset, or the
wrong end of the floor -- agrees with itself perfectly across 0.2 s, and would
pass. Something has to pin the answer to the court itself.

The court does that, because it is painted. Warp a frame into court
coordinates with the registration under test and the painted lines either land
on the lines this module draws or they do not. No annotations, no feed, no
model of mine, and it works on any broadcast of any arena, since every NBA
court carries the same markings in the same places.
"""

from __future__ import annotations

import numpy as np

from .court import (BASKET, CORNER_THREE_X, COURT_WIDTH, FREE_THROW_LINE_Y,
                    HALF_COURT_LENGTH, LANE_WIDTH, THREE_POINT_RADIUS)

COURT_LENGTH = HALF_COURT_LENGTH * 2
CENTRE_CIRCLE_RADIUS = 6.0
FREE_THROW_CIRCLE_RADIUS = 6.0
RESTRICTED_AREA_RADIUS = 4.0
#: Where the three-point arc meets the straight corner section.
CORNER_THREE_Y = 14.0


def _arc(centre, radius, start, end, steps=64):
    angles = np.linspace(start, end, steps)
    return [(centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a))
            for a in angles]


def court_lines() -> list[list[tuple[float, float]]]:
    """Every painted line, as polylines in court feet.

    Both ends are drawn, because a registration that lands on the wrong basket
    matches the near end's markings against the far end's and must not score
    well for it.
    """
    width, length = COURT_WIDTH, COURT_LENGTH
    lines: list[list[tuple[float, float]]] = [
        [(0, 0), (width, 0), (width, length), (0, length), (0, 0)],   # boundary
        [(0, HALF_COURT_LENGTH), (width, HALF_COURT_LENGTH)],         # half-court
        _arc((width / 2, HALF_COURT_LENGTH), CENTRE_CIRCLE_RADIUS, 0, 2 * np.pi),
    ]
    lane_left = (width - LANE_WIDTH) / 2
    lane_right = (width + LANE_WIDTH) / 2
    for near in (True, False):
        def y(value):
            return value if near else length - value

        basket = (BASKET[0], y(BASKET[1]))
        lines.append([(lane_left, y(0)), (lane_left, y(FREE_THROW_LINE_Y)),
                      (lane_right, y(FREE_THROW_LINE_Y)), (lane_right, y(0))])
        lines.append(_arc((BASKET[0], y(FREE_THROW_LINE_Y)),
                          FREE_THROW_CIRCLE_RADIUS, 0, 2 * np.pi))
        lines.append(_arc(basket, RESTRICTED_AREA_RADIUS,
                          0 if near else np.pi, np.pi if near else 2 * np.pi))
        # Three-point line: straight in the corners, then the arc.
        span = np.arcsin((CORNER_THREE_Y - BASKET[1]) / THREE_POINT_RADIUS)
        start, end = span, np.pi - span
        arc = _arc(basket, THREE_POINT_RADIUS,
                   start if near else -start, end if near else -end)
        lines.append([(CORNER_THREE_X, y(0)), (CORNER_THREE_X, y(CORNER_THREE_Y))])
        lines.append([(width - CORNER_THREE_X, y(0)),
                      (width - CORNER_THREE_X, y(CORNER_THREE_Y))])
        lines.append(arc)
    return lines


def line_mask(scale: float = 10.0, thickness_ft: float = 0.5) -> np.ndarray:
    """A top-down image, `scale` pixels per foot, white where paint should be."""
    import cv2

    height = int(round(COURT_LENGTH * scale))
    width = int(round(COURT_WIDTH * scale))
    mask = np.zeros((height, width), dtype=np.uint8)
    for polyline in court_lines():
        points = np.array([[x * scale, y * scale] for x, y in polyline],
                          dtype=np.int32)
        cv2.polylines(mask, [points], False, 255,
                      max(1, int(round(thickness_ft * scale))))
    return mask
