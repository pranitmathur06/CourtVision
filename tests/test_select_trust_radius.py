"""The trust-radius rule, on hand-built dumps where the answer is known."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import select_trust_radius as rule  # noqa: E402


def _record(near_err, far_err, refused=False):
    """40 samples at 1 ft from support with one error, 40 at 10 ft with another."""
    idx = list(range(80))
    record = {"base_idx": idx}
    if refused:
        record["dist"] = None
        return record
    record.update(family_idx=idx, dist=[1.0] * 40 + [10.0] * 40,
                  fit_idx=idx, fit_off=[near_err] * 40 + [far_err] * 40)
    return record


def test_error_is_measured_only_on_trusted_ground():
    records = [_record(0.1, 2.0)] * 5
    assert rule.score(records, [], 3.0)["p50"] == 0.1
    assert rule.score(records, [], 12.0)["p50"] > 0.5


def test_a_trusted_family_the_refit_loses_is_a_failure():
    lost = dict(_record(0.1, 0.1), fit_idx=[], fit_off=[])
    s = rule.score([lost] * 3 + [_record(0.1, 0.1)], [], 3.0)
    assert s["failed"] == 0.75 and s["p50"] == float("inf")


def test_a_refused_refit_counts_against_coverage_not_error():
    s = rule.score([_record(0.1, 0.1), _record(0, 0, refused=True)], [], 3.0)
    assert s["n"] == 1 and s["p50"] == 0.1
    assert abs(s["paint_coverage"] - 40 / 160) < 1e-9


def test_feet_on_refused_frames_are_untrusted():
    frames = [{"refined": True, "feet_n": 2, "feet_dist": [1.0, 20.0]},
              {"refined": False, "feet_n": 2}]
    assert rule.score([], frames, 3.0)["feet_coverage"] == 0.25


def test_the_largest_passing_radius_wins_and_none_is_a_fallback():
    table = {2.0: {"p50": 0.1}, 4.0: {"p50": 0.24}, 6.0: {"p50": 0.26}}
    assert rule.choose(table) == (4.0, False)
    assert rule.choose({2.0: {"p50": 0.4}, 4.0: {"p50": 0.5}}) == (2.0, True)


def test_only_the_calibration_game_may_choose(tmp_path):
    dump = tmp_path / "d.json"
    dump.write_text(json.dumps({"meta": {"video": "data/games/FZAUuuuREg0.mp4", "commit": "x",
                                         "polarity": "rule", "min_peak_ratio": 2.0},
                                "records": [_record(0.1, 0.1)]}))
    sys.argv = ["x", str(dump)]
    assert rule.main() == 1
    sys.argv = ["x", str(dump), "--report"]
    assert rule.main() == 0
