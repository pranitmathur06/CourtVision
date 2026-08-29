"""Stage 9 — annotated video and the commentary log."""

from __future__ import annotations

import json
from collections.abc import Sequence

import cv2
import numpy as np

from courtvision.commentary import CommentaryLine, format_timestamp
from courtvision.types import Event, Frame

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
                "player_name": event.player_name,
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
