"""The rule that chooses the refinement threshold -- tested before it chooses one.

The previous threshold was picked by code that was never checked in. This rule
is, and these tests pin the parts that make it honest: what it refuses to read,
and that a failure counts as a failure in every statistic.
"""

import json
import subprocess
import sys

SCRIPT = "scripts/select_refinement_threshold.py"


def _run(tmp_path, data):
    dump = tmp_path / "dump.json"
    dump.write_text(json.dumps(data))
    result = subprocess.run([sys.executable, SCRIPT, str(dump)],
                            capture_output=True, text=True)
    return result.returncode, result.stdout


def _meta(**overrides):
    meta = {"video": "data/raw_clips/fullgame.mp4", "commit": "abc1234",
            "dirty": False, "polarity": "bright", "min_peak_ratio": 0.0,
            "registered": 10, "frames": []}
    meta.update(overrides)
    return meta


def test_a_dump_without_provenance_is_refused(tmp_path):
    code, out = _run(tmp_path, [{"t": 1.0, "peak_ratio": 9, "refined_err": 0.1}])
    assert code == 1 and "provenance" in out


def test_the_unseen_arena_may_not_choose_a_threshold(tmp_path):
    code, out = _run(tmp_path, {"meta": _meta(video="data/games/FZAUuuuREg0.mp4"),
                                "records": []})
    assert code == 1 and "unseen" in out


def test_a_dump_gated_above_the_lowest_candidate_is_refused(tmp_path):
    """Frames and refits refused at 3 were never measured, so candidate 2 --
    which would accept some of them -- cannot be simulated on it."""
    code, out = _run(tmp_path, {"meta": _meta(min_peak_ratio=3.0), "records": []})
    assert code == 1 and "regenerate" in out


def test_a_dump_gated_at_the_lowest_candidate_is_accepted(tmp_path):
    """Refused refits and their ratios are recorded, so every candidate at or
    above the dump's own threshold is simulated exactly."""
    records = [{"t": 1.0, "peak_ratio": 12, "refined_err": 0.1, "refit_ratio": 12}]
    frames = [{"t": 1.0, "refined": True, "peak_ratio": 12}]
    code, out = _run(tmp_path, {"meta": _meta(min_peak_ratio=2.0, frames=frames),
                                "records": records})
    assert code == 0, out


def test_failures_count_everywhere_and_the_smallest_passing_threshold_wins(tmp_path):
    """Frame A is sharp (ratio 12); frame B is not (ratio 3).

    At 2 and 3 frame B's poor and lost lines keep the median above 0.3 ft.
    At 5 frame B is refused -- and one of A's measurements must now count as a
    failure too, because ITS refit's ratio (4) would be refused at 5 even
    though its frame passes. That failure has to reach the p90.
    """
    records = [
        {"t": 1.0, "peak_ratio": 12, "refined_err": 0.10, "refit_ratio": 12},
        {"t": 1.0, "peak_ratio": 12, "refined_err": 0.20, "refit_ratio": 12},
        {"t": 1.0, "peak_ratio": 12, "refined_err": 0.15, "refit_ratio": 4},
        {"t": 2.0, "peak_ratio": 3, "refined_err": 0.90, "refit_ratio": 3},
        {"t": 2.0, "peak_ratio": 3, "refined_err": 0.80, "refit_ratio": 3},
        {"t": 2.0, "peak_ratio": 3, "refined_err": None, "refit_ratio": 3},
        {"t": 2.0, "peak_ratio": 3, "refined_err": None, "refit_ratio": 3},
    ]
    frames = [{"t": 1.0, "refined": True, "peak_ratio": 12},
              {"t": 2.0, "refined": True, "peak_ratio": 3}]
    code, out = _run(tmp_path, {"meta": _meta(frames=frames), "records": records})
    assert code == 0, out
    assert "MIN_PEAK_RATIO = 5.0" in out, out
    chosen_row = next(line for line in out.splitlines() if "<- chosen" in line)
    assert "inf" in chosen_row, "the refused refit must appear in the tail"


def test_when_nothing_reaches_the_target_the_fallback_is_the_lowest_median(tmp_path):
    """The rule that actually chose 3.0 on OKC, now checkable: at 2 and 3 the
    median is 0.50; at 5, 7 and 10 it is 0.45; the lowest threshold with the
    lowest median is 5."""
    records = [
        {"t": 1.0, "peak_ratio": 12, "refined_err": 0.40, "refit_ratio": 12},
        {"t": 1.0, "peak_ratio": 12, "refined_err": 0.45, "refit_ratio": 12},
        {"t": 1.0, "peak_ratio": 12, "refined_err": 0.50, "refit_ratio": 12},
        {"t": 2.0, "peak_ratio": 3, "refined_err": 0.80, "refit_ratio": 3},
        {"t": 2.0, "peak_ratio": 3, "refined_err": 0.90, "refit_ratio": 3},
    ]
    frames = [{"t": 1.0, "refined": True, "peak_ratio": 12},
              {"t": 2.0, "refined": True, "peak_ratio": 3}]
    code, out = _run(tmp_path, {"meta": _meta(frames=frames), "records": records})
    assert code == 0, out
    assert "FALLBACK" in out and "MIN_PEAK_RATIO = 5.0" in out, out
