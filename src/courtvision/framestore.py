"""Frames on disk behaving like a list, so clip length stops bounding memory.

`run_pipeline` held every decoded frame in a list. At 1280x720 that is 2.8 MB a
frame, so the 88-frame test clip cost 243 MB and a 19-minute run would have
needed 31.9 GB — the pipeline was structurally limited to short clips, and the
limit was invisible because every test clip was under ten seconds.

Every consumer (`collect_samples`, `classify_windows`, `render_video`) takes a
`Sequence[np.ndarray]`, so nothing needs rewriting: a sequence that reads from
disk satisfies all of them. Frames are stored as JPEG, which is far smaller than raw and keeps a long run
inside a couple of gigabytes.

JPEG is lossy, so this is not bit-identical to the in-memory path. The settings
were measured rather than assumed — see `_encode_params` — and
tests/test_framestore.py pins the resulting torso-colour error.
"""

from __future__ import annotations

import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

JPEG_QUALITY = 95
CACHE_FRAMES = 48


def _encode_params():
    """Quality 95 with 4:4:4 chroma.

    Team assignment clusters mean torso COLOUR, so chroma loss matters more
    here than luma detail. JPEG's default 4:2:0 subsampling costs 4.6/255 of
    error on a saturated jersey patch and raising quality barely helps —
    even quality 100 leaves 4.2. Turning subsampling off drops it to 1.0 for
    16% more bytes, which is the trade worth making.
    """
    import cv2

    params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    factor = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
    best = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", None)
    if factor is not None and best is not None:
        params += [int(factor), int(best)]
    return params


class FrameStore(Sequence):
    """An append-only, disk-backed sequence of frames.

    The read cache holds `CACHE_FRAMES` decoded frames. Access is dominated by
    16-frame classification windows and a single sequential render pass, so a
    small LRU turns nearly every read into a hit without holding the clip.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self._owned = directory is None
        self._dir = Path(directory) if directory else Path(
            tempfile.mkdtemp(prefix="courtvision-frames-"))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._count = 0
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def append(self, image: np.ndarray) -> None:
        import cv2

        path = self._dir / f"{self._count:08d}.jpg"
        ok = cv2.imwrite(str(path), image, _encode_params())
        if not ok:
            raise RuntimeError(f"could not write frame {self._count} to {path}")
        self._count += 1

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index):
        import cv2

        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._count))]
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(f"frame {index} of {self._count}")
        hit = self._cache.get(index)
        if hit is not None:
            self._cache.move_to_end(index)
            return hit
        image = cv2.imread(str(self._dir / f"{index:08d}.jpg"))
        if image is None:
            raise RuntimeError(f"frame {index} missing from {self._dir}")
        self._cache[index] = image
        if len(self._cache) > CACHE_FRAMES:
            self._cache.popitem(last=False)
        return image

    def nbytes_on_disk(self) -> int:
        return sum(p.stat().st_size for p in self._dir.glob("*.jpg"))

    def close(self) -> None:
        """Drop the cache, and the directory too if this store created it."""
        self._cache.clear()
        if self._owned and self._dir.exists():
            shutil.rmtree(self._dir, ignore_errors=True)

    def __enter__(self) -> "FrameStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
