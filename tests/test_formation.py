"""Formation classification from court coordinates."""

import numpy as np
import pytest

from courtvision.formation import Formation, classify_formation, spacing


def test_horns_is_recognised():
    """Two bigs at the elbows, two shooters in the corners, ball handler up top."""
    offense = np.array([
        [17.0, 19.0],   # left elbow
        [33.0, 19.0],   # right elbow
        [3.0, 6.0],     # left corner
        [47.0, 6.0],    # right corner
        [25.0, 28.0],   # top of the key
    ])
    assert classify_formation(offense).name == "horns"


def test_horns_needs_both_elbows():
    """One elbow and two corners is not Horns, and must not be called it."""
    offense = np.array([
        [17.0, 19.0],
        [25.0, 33.0],   # second big up top instead of the far elbow
        [3.0, 6.0],
        [47.0, 6.0],
        [25.0, 28.0],
    ])
    assert classify_formation(offense).name != "horns"


def test_five_out():
    offense = np.array([
        [3.0, 8.0], [47.0, 8.0], [10.0, 30.0], [40.0, 30.0], [25.0, 34.0],
    ])
    result = classify_formation(offense)
    assert result.name == "five_out"


def test_post_up():
    offense = np.array([
        [21.0, 6.0],    # on the block
        [8.0, 28.0], [42.0, 28.0], [25.0, 34.0], [45.0, 10.0],
    ])
    assert classify_formation(offense).name == "post_up"


def test_isolation():
    offense = np.array([
        [40.0, 26.0],   # isolated on the right
        [5.0, 6.0], [8.0, 12.0], [6.0, 20.0], [12.0, 8.0],
    ])
    result = classify_formation(offense)
    assert result.name == "isolation"
    assert "ft from" in result.evidence


def test_declines_when_too_few_players_are_located():
    """A tracker loses players constantly; three is not a formation."""
    offense = np.array([[17.0, 19.0], [33.0, 19.0], [25.0, 28.0]])
    result = classify_formation(offense)
    assert result.name == "unknown"
    assert "only 3 players" in result.evidence


def test_nan_positions_are_dropped_not_propagated():
    """A player behind the horizon maps to NaN; it must not poison the result."""
    offense = np.array([
        [17.0, 19.0], [33.0, 19.0], [3.0, 6.0], [47.0, 6.0],
        [np.nan, np.nan],
    ])
    assert classify_formation(offense).name == "horns"


def test_unrecognised_alignment_says_so_rather_than_guessing():
    offense = np.array([
        [20.0, 24.0], [24.0, 26.0], [28.0, 25.0], [26.0, 22.0], [22.0, 28.0],
    ])
    result = classify_formation(offense)
    assert result.name == "unknown"


def test_spacing_reports_nearest_neighbours_not_spread():
    """Two tight pairs far apart are BAD spacing, though the spread is wide."""
    bunched = np.array([[10.0, 10.0], [11.0, 10.0], [40.0, 10.0], [41.0, 10.0]])
    mean_gap, tightest = spacing(bunched)
    assert tightest == pytest.approx(1.0)
    assert mean_gap == pytest.approx(1.0)

    even = np.array([[5.0, 5.0], [20.0, 5.0], [35.0, 5.0], [20.0, 30.0]])
    even_gap, _ = spacing(even)
    assert even_gap > mean_gap


def test_spacing_of_a_single_player_is_undefined_not_zero():
    mean_gap, tightest = spacing(np.array([[25.0, 25.0]]))
    assert np.isnan(mean_gap) and np.isnan(tightest)


def test_formation_str_is_readable():
    f = Formation("horns", "both elbows occupied", 12.0, 8.0)
    assert str(f) == "horns (both elbows occupied)"
