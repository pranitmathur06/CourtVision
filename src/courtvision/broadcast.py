"""Reading the SOURCE broadcast at the moments a clip's detections came from.

WHY THIS EXISTS. Every accuracy number in this project must be measured on the
broadcast, not on the 854x480 clips published for the page. Measured rather
than assumed: rebuilding Finals G7's floor mask with the setting its own
pipeline already uses scores 0.836 kept ball-carrier from the clips against
0.866 from the source. Three points of a real metric, lost to resolution, on
every number that took the cheap path.

THE MAPPING CANNOT BE COMPUTED, ONLY COPIED. `clip_detect_raw.py` seeks the
source with `CAP_PROP_POS_MSEC` to the clip's start and then reads forward, so
a clip's row `position` is the frame `position * step` after that seek.
`round(start_s * fps) + f` is wrong by up to 24 frames, because a POS_MSEC seek
lands on a decodable frame -- which is also where ffmpeg landed when it cut the
clip, so those two agree with each other and not with the arithmetic. Verified
frame by frame on Finals G7: the clip's frame 0 matches the source's read-index
0 after the seek, on every clip checked. Getting it wrong scored 0.400 against
the shipped 0.872, which is what a floor applied to the wrong moment looks like.

AND THE BOXES ARE ALREADY IN SOURCE PIXELS. Scaling them to the clip is right
only when the image came from the clip. Against a source-resolution frame it
puts every player's feet in the top-left corner, which reads as the mask having
got tighter rather than as an error: 0.551, with over-keeping at 0.998.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import cv2


def clip_starts(clip_index: str | Path) -> dict[str, float]:
    """{clip name: its start in the SOURCE video, in seconds}.

    A row with no clip is an event whose footage was never cut; the cutter
    records the attempt either way, and treating one as coverage is its own
    bug (see `eval_play_events.clip_for`).
    """
    index = json.loads(Path(clip_index).read_text())
    rows = index["clips"] if isinstance(index, dict) else index
    return {row["clip"]: float(row["start_s"])
            for row in rows if row.get("clip")}


class SourceReader:
    """One open handle on the broadcast, seeked per clip and read forward."""

    def __init__(self, video: str | Path):
        self._capture = cv2.VideoCapture(str(video))
        self.ok = self._capture.isOpened()

    def frames(self, start_s: float, wanted: dict[int, object],
               step: int) -> Iterator[tuple[object, object]]:
        """Yield (key, image) for each wanted ROW POSITION of one clip.

        `wanted` maps a row position to whatever the caller wants back with the
        image. Positions are converted to source frames as `position * step`
        after the seek, which is the pipeline's own arithmetic and not a
        reconstruction of it.
        """
        if not self.ok or not wanted:
            return
        self._capture.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
        by_frame = {position * step: key for position, key in wanted.items()}
        last = max(by_frame)
        for index in range(last + 1):
            if not self._capture.grab():
                return
            key = by_frame.get(index)
            if key is None:
                continue
            ok, image = self._capture.retrieve()
            if ok:
                yield key, image

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> "SourceReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
