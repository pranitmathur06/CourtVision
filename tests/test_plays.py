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
    screener = [(25, 19), (25, 22), (25, 24), (25, 24), (25, 24),
                (25, 21), (25, 17), (25, 13), (25, 10), (25, 7)]
    plays = detect_screens(*sequence(handler, screener))
    assert [p.name for p in plays] == ["pick_and_roll"]
    assert plays[0].screener_id == 2 and plays[0].handler_id == 1
    assert "toward the rim" in plays[0].evidence


def test_pick_and_pop():
    """Same screen, but the screener steps out beyond the arc instead."""
    handler = [(25, 30), (25, 28), (25, 26), (25, 25), (24, 24),
               (22, 23), (20, 22), (18, 21), (17, 20), (16, 19)]
    screener = [(25, 19), (25, 22), (25, 24), (25, 24), (25, 24),
                (26, 27), (27, 30), (28, 33), (29, 35), (30, 37)]
    plays = detect_screens(*sequence(handler, screener))
    assert [p.name for p in plays] == ["pick_and_pop"]
    assert "beyond the arc" in plays[0].evidence


def test_dribble_handoff_is_not_called_a_pick_and_roll():
    """If the ball changes hands at contact it is a hand-off, whatever follows."""
    handler = [(25, 30), (25, 28), (25, 26), (25, 25), (24, 24),
               (22, 23), (20, 22), (18, 21), (17, 20), (16, 19)]
    screener = [(25, 19), (25, 22), (25, 24), (25, 24), (25, 24),
                (25, 21), (25, 17), (25, 13), (25, 10), (25, 7)]
    handlers = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]        # ball moves to the screener
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
    screener = [(25, 19), (25, 22), (25, 24), (25, 24), (25, 24),
                (25, 20), (25, 16), (25, 12), (25, 9), (25, 7)]
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
    # The screener (2) is planted; the cutter (3) runs to him.
    positions = [
        {1: (25, 30), 2: (5, 16), 3: (5, 4)},
        {1: (25, 29), 2: (5, 16), 3: (5, 9)},
        {1: (25, 28), 2: (5, 16), 3: (5, 14)},
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
    screener = [(25, 19), (25, 22), (25, 24), (25, 24), (25, 24),
                (25, 20), (25, 16), (25, 12), (25, 9), (25, 7)]
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
    screener = [(25, 19), (25, 22), (25, 24), (25, 24), (25, 24),
                (25, 21), (25, 17), (25, 13), (25, 10), (25, 7)]
    third = [(40, 8), (39, 10), (37, 11), (34, 12), (31, 12),
             (28, 12), (26, 12), (26, 12), (26, 12), (26, 12)]
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
    screener = [(25, 19), (25, 22), (25, 24), (25, 24), (25, 24),
                (25, 21), (25, 17), (25, 13), (25, 10), (25, 7)]
    far = [(45, 5)] * 10
    positions, handlers, times = _frames({1: handler, 2: screener, 3: far}, [1] * 10)

    assert not [p for p in detect_sets(positions, handlers, times)
                if p.name == "spain_pick_and_roll"]


def test_double_drag_two_screeners_for_one_handler():
    from courtvision.plays import detect_sets

    handler = [(25, 34), (25, 32), (25, 30), (25, 29), (25, 28),
               (25, 27), (25, 26), (25, 25), (25, 24), (25, 23)]
    first = [(25, 22), (25, 26), (25, 28), (25, 28), (25, 28),
             (25, 18), (25, 14), (25, 11), (25, 9), (25, 8)]
    second = [(38, 20), (35, 22), (31, 24), (28, 26), (25, 26),
              (25, 26), (25, 26), (25, 22), (25, 16), (25, 10)]
    positions, handlers, times = _frames(
        {1: handler, 2: first, 3: second}, [1] * 10)

    names = [p.name for p in detect_sets(positions, handlers, times)]
    assert "double_drag" in names or "re_screen" in names


def test_sets_are_ordered_in_time():
    from courtvision.plays import detect_sets

    handler = [(25, 34), (25, 32), (25, 30), (25, 29), (25, 28),
               (25, 27), (25, 26), (25, 25), (25, 24), (25, 23)]
    first = [(25, 22), (25, 26), (25, 28), (25, 28), (25, 28),
             (25, 18), (25, 14), (25, 11), (25, 9), (25, 8)]
    second = [(38, 20), (35, 22), (31, 24), (28, 26), (25, 26),
              (25, 26), (25, 26), (25, 22), (25, 16), (25, 10)]
    positions, handlers, times = _frames(
        {1: handler, 2: first, 3: second}, [1] * 10)
    sets = detect_sets(positions, handlers, times)
    assert [p.time_s for p in sets] == sorted(p.time_s for p in sets)


def test_no_screens_means_no_sets():
    from courtvision.plays import detect_sets

    positions = [{1: (25, 30), 2: (5, 5), 3: (45, 5)} for _ in range(10)]
    assert detect_sets(positions, [1] * 10, [i * 0.1 for i in range(10)]) == []


def _off_ball(cutter_path, screener_path):
    """Handler parked up top; screener meets the cutter, who then goes somewhere.

    Roles are decided by motion, not by track id: the screener is whichever of
    the two travels LESS after contact, so the stationary path below is always
    read as the screener and the moving one as the cutter. The pair must be at
    least
    SCREEN_SEPARATION_FT apart before contact or no screen is detected at all,
    and the cutter must still be moving AFTER contact or there is no direction
    to name.
    """
    handler = [(25.0, 30.0)] * 10
    positions = [{1: handler[i], 2: screener_path[i], 3: cutter_path[i]}
                 for i in range(10)]
    return positions, [1] * 10, [i * 0.1 for i in range(10)]


def test_back_screen_cutter_goes_to_the_rim():
    from courtvision.plays import detect_off_ball_screens

    screener = [(20.0, 24.0)] * 10
    cutter = [(8.0, 27.0), (12.0, 26.0), (17.0, 25.0), (21.0, 25.0),
              (21.0, 20.0), (20.0, 16.0), (19.0, 13.0), (18.0, 11.0),
              (18.0, 10.0), (18.0, 10.0)]
    plays = detect_off_ball_screens(*_off_ball(cutter, screener))
    assert [p.name for p in plays] == ["back_screen"], [str(p) for p in plays]
    assert "toward the rim" in plays[0].evidence


def test_flare_screen_cutter_goes_away_from_the_ball():
    from courtvision.plays import detect_off_ball_screens

    screener = [(20.0, 24.0)] * 10
    cutter = [(32.0, 28.0), (30.0, 27.0), (27.0, 26.0), (23.0, 25.0),
              (18.0, 25.0), (14.0, 25.0), (10.0, 25.0), (7.0, 25.0),
              (5.0, 25.0), (5.0, 25.0)]
    plays = detect_off_ball_screens(*_off_ball(cutter, screener))
    assert [p.name for p in plays] == ["flare_screen"], [str(p) for p in plays]
    assert "away from the ball" in plays[0].evidence


def test_pin_down_cutter_comes_up_toward_the_ball():
    from courtvision.plays import detect_off_ball_screens

    screener = [(18.0, 12.0)] * 10
    cutter = [(30.0, 8.0), (27.0, 9.0), (23.0, 10.0), (20.0, 11.0),
              (20.0, 16.0), (20.0, 21.0), (19.0, 25.0), (19.0, 27.0),
              (19.0, 28.0), (19.0, 28.0)]
    plays = detect_off_ball_screens(*_off_ball(cutter, screener))
    assert [p.name for p in plays] == ["pin_down"], [str(p) for p in plays]
    assert "up toward the ball" in plays[0].evidence


def test_cutter_that_does_not_commit_is_left_unnamed():
    """No direction, no claim — better than inventing one."""
    from courtvision.plays import detect_off_ball_screens

    screener = [(20.0, 24.0)] * 10
    cutter = [(32.0, 28.0), (30.0, 27.0), (27.0, 26.0)] + [(23.0, 25.0)] * 7
    plays = detect_off_ball_screens(*_off_ball(cutter, screener))
    assert [p.name for p in plays] == ["off_ball_screen"], [str(p) for p in plays]
    assert "did not commit" in plays[0].evidence


def test_horns_flare_is_a_shape_plus_a_direction():
    """The example I kept citing as needing labels. It does not.

    Horns is an arrangement visible in one frame; a flare is a direction the
    cutter takes. The set is their conjunction inside a window.
    """
    from courtvision.plays import detect_sets

    screener = [(20.0, 24.0)] * 10
    cutter = [(32.0, 28.0), (30.0, 27.0), (27.0, 26.0), (23.0, 25.0),
              (18.0, 25.0), (14.0, 25.0), (10.0, 25.0), (7.0, 25.0),
              (5.0, 25.0), (5.0, 25.0)]
    positions, handlers, times = _off_ball(cutter, screener)
    formations = ["horns"] + [None] * 9

    sets = detect_sets(positions, handlers, times, formations)
    assert [p.name for p in sets] == ["horns_flare"], [str(p) for p in sets]
    assert "out of horns" in sets[0].evidence


def test_a_flare_without_horns_is_not_a_horns_set():
    from courtvision.plays import detect_sets

    screener = [(20.0, 24.0)] * 10
    cutter = [(32.0, 28.0), (30.0, 27.0), (27.0, 26.0), (23.0, 25.0),
              (18.0, 25.0), (14.0, 25.0), (10.0, 25.0), (7.0, 25.0),
              (5.0, 25.0), (5.0, 25.0)]
    positions, handlers, times = _off_ball(cutter, screener)

    assert detect_sets(positions, handlers, times, ["isolation"] + [None] * 9) == []
    assert detect_sets(positions, handlers, times) == []      # no formations given


def test_horns_alignment_far_before_the_screen_is_not_the_same_possession():
    """Two seconds is the window; beyond it they are unrelated events."""
    from courtvision.plays import detect_sets

    screener = [(20.0, 24.0)] * 40
    cutter = ([(32.0, 28.0)] * 30 + [(30.0, 27.0), (27.0, 26.0), (23.0, 25.0),
              (18.0, 25.0), (14.0, 25.0), (10.0, 25.0), (7.0, 25.0),
              (5.0, 25.0), (5.0, 25.0), (5.0, 25.0)])
    handler = [(25.0, 30.0)] * 40
    positions = [{1: handler[i], 2: screener[i], 3: cutter[i]} for i in range(40)]
    times = [i * 0.1 for i in range(40)]
    formations = ["horns"] + [None] * 39

    assert detect_sets(positions, [1] * 40, times, formations) == []


def test_transition_is_the_ball_covering_ground_fast():
    from courtvision.plays import detect_transition

    # Handler drives from half court to the rim in 0.9 s.
    path = [(25.0, 40.0), (25.0, 34.0), (25.0, 28.0), (25.0, 22.0),
            (25.0, 16.0), (25.0, 11.0), (25.0, 8.0), (25.0, 7.0),
            (25.0, 6.0), (25.0, 6.0)]
    positions = [{1: p} for p in path]
    plays = detect_transition(positions, [1] * 10, [i * 0.1 for i in range(10)])
    assert [p.name for p in plays] == ["transition"], [str(p) for p in plays]
    assert "toward the rim" in plays[0].evidence


def test_a_half_court_walk_up_is_not_transition():
    """Same direction, far slower and far less ground — must not be named."""
    from courtvision.plays import detect_transition

    path = [(25.0, 30.0), (25.0, 29.5), (25.0, 29.0), (25.0, 28.5),
            (25.0, 28.0), (25.0, 27.5), (25.0, 27.0), (25.0, 26.5),
            (25.0, 26.0), (25.0, 25.5)]
    positions = [{1: p} for p in path]
    assert detect_transition(positions, [1] * 10,
                             [i * 0.1 for i in range(10)]) == []


def test_a_pass_does_not_end_a_break_but_a_turnover_does():
    """This test used to assert the opposite, and that was the bug.

    A break begins with an outlet pass and often ends with a different player
    finishing, so refusing to follow the ball across a handler change threw
    away the clearest examples of transition. What must end it is the other
    TEAM getting the ball, which needs the team split to see.
    """
    from courtvision.plays import detect_transition

    path = [(25.0, 40.0), (25.0, 34.0), (25.0, 28.0), (25.0, 22.0),
            (25.0, 16.0), (25.0, 11.0), (25.0, 8.0), (25.0, 7.0),
            (25.0, 6.0), (25.0, 6.0)]
    positions = [{1: p, 2: p} for p in path]
    handlers = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2]      # ball changes hands mid-run
    times = [i * 0.1 for i in range(10)]

    # 1 passes to team-mate 2: still one break.
    teammates = [{1, 2} for _ in range(10)]
    assert [p.name for p in detect_transition(positions, handlers, times,
                                              offense=teammates)] == ["transition"]

    # The same ball movement, but 2 is an opponent: a turnover, not a break.
    opponents = [{1} for _ in range(10)]
    assert detect_transition(positions, handlers, times,
                             offense=opponents) == []


def test_stagger_is_two_screeners_for_one_cutter():
    from courtvision.plays import detect_stagger

    handler = [(25.0, 30.0)] * 14
    # Cutter (4) is screened by 2, then by 3, in quick succession.
    cutter = [(34.0, 10.0), (30.0, 12.0), (26.0, 14.0), (22.0, 16.0),
              (20.0, 18.0), (18.0, 20.0), (16.0, 22.0), (14.0, 24.0),
              (12.0, 26.0), (10.0, 27.0), (9.0, 28.0), (8.0, 28.0),
              (8.0, 28.0), (8.0, 28.0)]
    first = [(21.0, 17.0)] * 14
    second = [(11.0, 27.0)] * 14
    positions = [{1: handler[i], 2: first[i], 3: second[i], 4: cutter[i]}
                 for i in range(14)]
    plays = detect_stagger(positions, [1] * 14, [i * 0.1 for i in range(14)])
    assert [p.name for p in plays] == ["stagger_screen"], [str(p) for p in plays]
    assert plays[0].handler_id == 4


def test_one_screener_twice_is_not_a_stagger():
    from courtvision.plays import detect_stagger

    handler = [(25.0, 30.0)] * 10
    cutter = [(34.0, 10.0), (30.0, 12.0), (26.0, 14.0), (22.0, 16.0)] + \
             [(20.0, 18.0)] * 6
    screener = [(21.0, 17.0)] * 10
    positions = [{1: handler[i], 2: screener[i], 4: cutter[i]} for i in range(10)]
    assert detect_stagger(positions, [1] * 10,
                          [i * 0.1 for i in range(10)]) == []


def test_roles_do_not_depend_on_track_id_order():
    """The screener is the one who stays; swapping ids must not swap the name.

    Taking the lower id as screener got this backwards half the time, and since
    every off-ball screen is named by the CUTTER's direction, that turned pin
    downs into back screens.
    """
    from courtvision.plays import detect_off_ball_screens

    stays = [(20.0, 24.0)] * 10
    runs = [(8.0, 27.0), (12.0, 26.0), (17.0, 25.0), (21.0, 25.0),
            (21.0, 20.0), (20.0, 16.0), (19.0, 13.0), (18.0, 11.0),
            (18.0, 10.0), (18.0, 10.0)]
    handler = [(25.0, 30.0)] * 10
    times = [i * 0.1 for i in range(10)]

    low_id_stays = [{1: handler[i], 2: stays[i], 3: runs[i]} for i in range(10)]
    low_id_runs = [{1: handler[i], 2: runs[i], 3: stays[i]} for i in range(10)]
    first = detect_off_ball_screens(low_id_stays, [1] * 10, times)
    second = detect_off_ball_screens(low_id_runs, [1] * 10, times)
    assert [p.name for p in first] == [p.name for p in second] == ["back_screen"]
    assert first[0].screener_id == 2 and first[0].handler_id == 3
    assert second[0].screener_id == 3 and second[0].handler_id == 2


def test_defenders_converging_are_not_an_off_ball_screen():
    """Without a team split, every pair of non-handlers is a candidate."""
    from courtvision.plays import detect_off_ball_screens

    handler = [(25.0, 30.0)] * 10
    # They start 12 ft apart and close to within touching distance.
    one = [(8.0 + 1.4 * i, 24.0) for i in range(10)]
    two = [(20.0, 24.0)] * 10
    positions = [{1: handler[i], 2: two[i], 3: one[i]} for i in range(10)]
    times = [i * 0.1 for i in range(10)]
    # Tracks 2 and 3 defend; only track 1 attacks.
    offense = [{1} for _ in range(10)]
    assert detect_off_ball_screens(positions, [1] * 10, times, offense) == []
    assert detect_off_ball_screens(positions, [1] * 10, times) != []


def test_transition_thresholds_are_arguments_not_only_constants():
    """They had to become arguments to be fitted honestly.

    The defaults were set to pass synthetic fixtures and fire on 6 of 1172 real
    possessions. Sweeping them needs them passable; the module constants stay
    as the defaults so existing callers are unaffected.
    """
    from courtvision.plays import detect_transition

    # A walk-up: 12 ft of ground in 0.9 s, nowhere near the default 25 ft.
    handler = [(25, 40), (25, 38), (25, 36), (25, 34), (25, 32),
               (25, 30), (25, 28), (25, 28), (25, 28), (25, 28)]
    positions = [{1: p} for p in handler]
    times = [i * 0.1 for i in range(10)]
    assert detect_transition(positions, [1] * 10, times) == []
    loose = detect_transition(positions, [1] * 10, times,
                              gain_ft=10.0, speed_ft_s=4.0, window_s=4.0)
    assert [p.name for p in loose] == ["transition"]
