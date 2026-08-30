import numpy as np
import pytest

from courtvision.framestore import CACHE_FRAMES, FrameStore


def _frame(seed: int, h: int = 48, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_reads_back_in_order_and_supports_negative_and_slice():
    with FrameStore() as store:
        originals = [_frame(i) for i in range(5)]
        for image in originals:
            store.append(image)
        assert len(store) == 5
        for i in range(5):
            assert store[i].shape == originals[i].shape
        assert len(store[1:4]) == 3
        assert np.array_equal(store[-1], store[4])


def test_out_of_range_raises_indexerror():
    with FrameStore() as store:
        store.append(_frame(0))
        with pytest.raises(IndexError):
            store[5]
        with pytest.raises(IndexError):
            store[-2]


def test_memory_does_not_grow_with_clip_length():
    """The whole point: holding the clip must not scale with its length.

    A list of 200 frames this size would retain all 200; the store retains at
    most CACHE_FRAMES regardless of how many were written.
    """
    with FrameStore() as store:
        for i in range(200):
            store.append(_frame(i))
        for i in range(200):
            store[i]
        assert len(store._cache) <= CACHE_FRAMES
        assert len(store) == 200


def test_jpeg_quality_is_high_enough_for_torso_colour():
    """Team assignment clusters mean torso colour, so error must stay small.

    Measured rather than assumed — this pins the quality setting.
    """
    with FrameStore() as store:
        flat = np.full((48, 64, 3), 137, dtype=np.uint8)
        flat[10:30, 10:30] = (40, 90, 200)
        store.append(flat)
        back = store[0]
        patch_in = flat[10:30, 10:30].reshape(-1, 3).mean(axis=0)
        patch_out = back[10:30, 10:30].reshape(-1, 3).mean(axis=0)
        # 4:2:0 subsampling costs 4.6 here and quality alone cannot fix it
        # (even quality 100 leaves 4.2). 4:4:4 brings it to ~1.0.
        assert np.abs(patch_in - patch_out).max() < 1.5


def test_cache_returns_equal_frames_on_repeat_reads():
    with FrameStore() as store:
        store.append(_frame(1))
        assert np.array_equal(store[0], store[0])


def test_close_removes_a_directory_it_created(tmp_path):
    store = FrameStore()
    store.append(_frame(0))
    directory = store._dir
    assert directory.exists()
    store.close()
    assert not directory.exists()

    given = tmp_path / "keep"
    other = FrameStore(given)
    other.append(_frame(0))
    other.close()
    assert given.exists(), "a caller-supplied directory must not be deleted"
