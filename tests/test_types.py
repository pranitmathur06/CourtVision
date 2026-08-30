from courtvision.types import ACTIONS, BALL, PLAYER, Box, Frame, Track


def test_box_geometry():
    box = Box(10.0, 20.0, 30.0, 60.0)
    assert box.width == 20.0
    assert box.height == 40.0
    assert box.center == (20.0, 40.0)


def test_frame_players_excludes_ball():
    player = Track(1, Box(0, 0, 10, 20), PLAYER, 0.9)
    ball = Track(-1, Box(5, 5, 7, 7), BALL, 0.8)
    frame = Frame(0, 0.0, (player, ball))
    assert frame.players() == (player,)


def test_frame_ball_returns_highest_confidence_ball():
    low = Track(-1, Box(0, 0, 2, 2), BALL, 0.3)
    high = Track(-1, Box(9, 9, 11, 11), BALL, 0.7)
    frame = Frame(0, 0.0, (low, high))
    assert frame.ball() is high


def test_frame_ball_returns_none_when_absent():
    frame = Frame(0, 0.0, (Track(1, Box(0, 0, 10, 20), PLAYER, 0.9),))
    assert frame.ball() is None


def test_actions_cover_the_spec_labels_plus_block_steal_and_background():
    # The spec's five, plus `block` (SpaceJam), `steal` (BARD), and
    # `background` — ordinary play, which is not an action and never becomes an
    # Event, but which the classifier needs somewhere to put.
    assert ACTIONS == ("dribble", "pass", "shot", "rebound", "block", "steal",
                       "other", "background")
    for spec_label in ("dribble", "pass", "shot", "rebound", "other"):
        assert spec_label in ACTIONS


def test_background_is_last_so_it_never_shifts_another_class_label_index():
    """Label indices are positional; appending keeps every existing one stable."""
    from courtvision.types import BACKGROUND

    assert ACTIONS[-1] == BACKGROUND
    assert ACTIONS.index("dribble") == 0


def test_one_class_changing_size_does_not_move_another_class_split():
    """The bug this replaced silently invalidated cross-run comparisons.

    The old split concatenated all classes and shuffled globally, so changing
    rebound from 226 clips to 223 reshuffled every class ordered after it: only
    15 of 79 block validation clips survived, and block's apparent 23-point
    regression was partly a different set of clips.
    """
    from pathlib import Path

    from courtvision.types import stratified_split

    def clips(action, n):
        return [Path(f"data/labeled/actions/{action}/{i:07d}.mp4") for i in range(n)]

    before = {"block": clips("block", 400), "rebound": clips("rebound", 226)}
    after = {"block": clips("block", 400), "rebound": clips("rebound", 223)}

    _, val_before = stratified_split(before)
    _, val_after = stratified_split(after)
    block_before = {p for p, _ in val_before if p.parent.name == "block"}
    block_after = {p for p, _ in val_after if p.parent.name == "block"}
    assert block_before == block_after, "block's split must not depend on rebound"


def test_every_class_is_represented_in_validation():
    """Splitting per class makes the validation set stratified."""
    from pathlib import Path

    from courtvision.types import stratified_split

    per_class = {
        "rare": [Path(f"a/rare/{i}.mp4") for i in range(10)],
        "common": [Path(f"a/common/{i}.mp4") for i in range(800)],
    }
    _, val = stratified_split(per_class)
    labels = {label for _, label in val}
    assert len(labels) == 2, "a rare class must still appear in validation"
