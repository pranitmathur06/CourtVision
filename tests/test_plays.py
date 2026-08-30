"""Screen actions detected from court geometry over time."""

import numpy as np
import pytest

from courtvision.plays import detect_off_ball_screens, detect_screens


def sequence(handler_path, screener_path, handler_id=1, screener_id=2,
             handler_by_frame=None):
    """Build (positions, handlers, times) from two trajectories."""
    n = len(handler_path)
    positions = [{handler_id: tuple(handler_path[i]),
                  screener_id: tuple(screener_path[i])} for i in range(n)]
    handlers = handler_by_frame or [handler_id] * n
    times = [i * 0.1 for i in range(n)]
    return positions, handlers, times


def test_pick_and_roll():
    """Screener arrives, then cuts to the rim while the handler uses the space."""
    handler = [(25, 30), (25, 28), (25, 26), (25, 25), (24, 24),
               (22, 23), (20, 22), (18, 21), (17, 20), (16, 19)]
    screener = [(25, 18), (25, 20), (25, 22), (25, 24), (25, 21),
                (25, 17), (25, 13), (25, 10), (25, 8), (25, 7)]
    plays = detect_screens(*sequence(handler, screener))
    assert [p.name for p in plays] == ["pick_and_roll"]
    assert plays[0].screener_id == 2 and plays[0].handler_id == 1
    assert "toward the rim" in plays[0].evidence


def test_pick_and_pop():
    """Same screen, but the screener steps out beyond the arc instead."""
    handler = [(25, 30), (25, 28), (25, 26), (25, 25), (24, 24),
               (22, 23), (20, 22), (18, 21), (17, 20), (16, 19)]
    screener = [(25, 18), (25, 20), (25, 22), (25, 24), (26, 27),
                (27, 30), (28, 33), (29, 35), (30, 36), (30, 37)]
    plays = detect_screens(*sequence(handler, screener))
    assert [p.name for p in plays] == ["pick_and_pop"]
    assert "beyond the arc" in plays[0].evidence


def test_dribble_handoff_is_not_called_a_pick_and_roll():
    """If the ball changes hands at contact it is a hand-off, whatever follows."""
    handler = [(25, 30), (25, 28), (25, 26), (25, 25), (24, 24),
               (22, 23), (20, 22), (18, 21), (17, 20), (16, 19)]
    screener = [(25, 18), (25, 20), (25, 22), (25, 24), (25, 21),
                (25, 17), (25, 13), (25, 10), (25, 8), (25, 7)]
    handlers = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]        # ball moves to the screener
    plays = detect_screens(*sequence(handler, screener, handler_by_frame=handlers))
    assert [p.name for p in plays] == ["dribble_handoff"]


def test_players_who_never_converge_are_not_a_screen():
    handler = [(10, 30)] * 10
    screener = [(40, 10)] * 10
    assert detect_screens(*sequence(handler, screener)) == []


def test_players_standing_together_all_along_are_not_a_screen():
    """A screen requires them to COME together, not merely to be adjacent."""
    handler = [(25, 25)] * 10
    screener = [(27, 25)] * 10
    assert detect_screens(*sequence(handler, screener)) == []


def test_a_screen_is_reported_once_not_every_frame():
    handler = [(25, 30), (25, 27), (25, 25), (25, 24), (25, 24),
               (25, 24), (24, 23), (23, 22), (22, 21), (21, 20)]
    screener = [(25, 18), (25, 21), (25, 23), (25, 24), (25, 22),
                (25, 18), (25, 14), (25, 11), (25, 9), (25, 7)]
    plays = detect_screens(*sequence(handler, screener))
    assert len(plays) == 1


def test_missing_tracks_do_not_crash_it():
    """Trackers lose players constantly; absence must be tolerated."""
    positions = [
        {1: (25, 30), 2: (25, 18)},
        {1: (25, 26)},                       # screener lost
        {1: (25, 24), 2: (25, 24)},
        {2: (25, 16)},                       # handler lost
        {1: (22, 22), 2: (25, 10)},
    ]
    handlers = [1, 1, 1, None, 1]
    times = [0.0, 0.1, 0.2, 0.3, 0.4]
    plays = detect_screens(positions, handlers, times)
    assert all(p.screener_id == 2 for p in plays)


def test_no_handler_means_no_ball_screen():
    handler = [(25, 30), (25, 26), (25, 24), (25, 24), (22, 22)]
    screener = [(25, 18), (25, 21), (25, 24), (25, 18), (25, 10)]
    positions, _, times = sequence(handler, screener)
    assert detect_screens(positions, [None] * 5, times) == []


def test_off_ball_screen_between_two_non_handlers():
    positions = [
        {1: (25, 30), 2: (5, 25), 3: (5, 8)},
        {1: (25, 29), 2: (5, 20), 3: (5, 12)},
        {1: (25, 28), 2: (5, 16), 3: (5, 15)},
    ]
    plays = detect_off_ball_screens(positions, [1, 1, 1], [0.0, 0.1, 0.2])
    assert [p.name for p in plays] == ["off_ball_screen"]
    assert {plays[0].screener_id, plays[0].handler_id} == {2, 3}


def test_mismatched_input_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        detect_screens([{1: (0, 0)}], [None, None], [0.0])


def test_play_str_is_readable():
    handler = [(25, 30), (25, 27), (25, 25), (25, 24), (24, 23),
               (22, 22), (20, 21), (18, 20), (17, 19), (16, 18)]
    screener = [(25, 18), (25, 21), (25, 23), (25, 24), (25, 21),
                (25, 17), (25, 13), (25, 10), (25, 8), (25, 7)]
    play = detect_screens(*sequence(handler, screener))[0]
    assert "2 screens for 1" in str(play)


def _frames(paths: dict[int, list], handler_by_frame):
    n = len(next(iter(paths.values())))
    positions = [{tid: tuple(path[i]) for tid, path in paths.items()
                  if path[i] is not None} for i in range(n)]
    return positions, handler_by_frame, [i * 0.1 for i in range(n)]


def test_spain_pick_and_roll():
    """Pick-and-roll, then a third player back-screens the roller."""
    from courtvision.plays import detect_sets

    # 1 handles, 2 screens then rolls to the rim, 3 back-screens 2 mid-roll.
    handler = [(25, 30), (25, 28), (25, 26), (25, 25), (24, 24),
               (22, 23), (20, 22), (18, 21), (17, 20), (16, 19)]
    screener = [(25, 18), (25, 20), (25, 22), (25, 24), (25, 21),
                (25, 17), (25, 13), (25, 10), (25, 8), (25, 7)]
    third = [(40, 8), (40, 9), (39, 10), (38, 11), (36, 12),
             (33, 13), (29, 13), (26, 12), (25, 11), (25, 9)]
    positions, handlers, times = _frames(
        {1: handler, 2: screener, 3: third}, [1] * 10)

    sets = detect_sets(positions, handlers, times)
    names = [p.name for p in sets]
    assert "spain_pick_and_roll" in names
    spain = next(p for p in sets if p.name == "spain_pick_and_roll")
    assert spain.screener_id == 2 and spain.handler_id == 1
    assert "back-screens the roller" in spain.evidence


def test_plain_pick_and_roll_is_not_called_spain():
    """Without a third player screening the roller it is just a pick-and-roll."""
    from courtvision.plays import detect_sets

    handler = [(25, 30), (25, 28), (25, 26), (25, 25), (24, 24),
               (22, 23), (20, 22), (18, 21), (17, 20), (16, 19)]
    screener = [(25, 18), (25, 20), (25, 22), (25, 24), (25, 21),
                (25, 17), (25, 13), (25, 10), (25, 8), (25, 7)]
    far = [(45, 5)] * 10
    positions, handlers, times = _frames({1: handler, 2: screener, 3: far}, [1] * 10)

    assert not [p for p in detect_sets(positions, handlers, times)
                if p.name == "spain_pick_and_roll"]


def test_double_drag_two_screeners_for_one_handler():
    from courtvision.plays import detect_sets

    handler = [(25, 34), (25, 32), (25, 30), (25, 29), (25, 28),
               (25, 27), (25, 26), (25, 25), (25, 24), (25, 23)]
    first = [(25, 20), (25, 24), (25, 29), (25, 28), (25, 18),
             (25, 14), (25, 11), (25, 9), (25, 8), (25, 8)]
    second = [(38, 20), (36, 21), (33, 23), (30, 24), (28, 25),
              (25, 26), (25, 24), (25, 18), (25, 13), (25, 9)]
    positions, handlers, times = _frames(
        {1: handler, 2: first, 3: second}, [1] * 10)

    names = [p.name for p in detect_sets(positions, handlers, times)]
    assert "double_drag" in names or "re_screen" in names


def test_sets_are_ordered_in_time():
    from courtvision.plays import detect_sets

    handler = [(25, 34), (25, 32), (25, 30), (25, 29), (25, 28),
               (25, 27), (25, 26), (25, 25), (25, 24), (25, 23)]
    first = [(25, 20), (25, 24), (25, 29), (25, 28), (25, 18),
             (25, 14), (25, 11), (25, 9), (25, 8), (25, 8)]
    second = [(38, 20), (36, 21), (33, 23), (30, 24), (28, 25),
              (25, 26), (25, 24), (25, 18), (25, 13), (25, 9)]
    positions, handlers, times = _frames(
        {1: handler, 2: first, 3: second}, [1] * 10)
    sets = detect_sets(positions, handlers, times)
    assert [p.time_s for p in sets] == sorted(p.time_s for p in sets)


def test_no_screens_means_no_sets():
    from courtvision.plays import detect_sets

    positions = [{1: (25, 30), 2: (5, 5), 3: (45, 5)} for _ in range(10)]
    assert detect_sets(positions, [1] * 10, [i * 0.1 for i in range(10)]) == []
