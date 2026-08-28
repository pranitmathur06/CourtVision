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
