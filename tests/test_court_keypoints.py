"""The recovered court-landmark schema, and registration from it.

The schema was not read off documentation -- the dataset ships none -- it was
recovered from the annotations and then checked against the dataset's own
mirror symmetry. These tests pin both the recovery's result and the checks that
would have caught it being wrong.
"""

import numpy as np
import pytest

from courtvision.court_keypoints import (COURT_LENGTH_FT, COURT_WIDTH_FT,
                                         FLIP_INDEX, KEYPOINTS, MIN_KEYPOINTS,
                                         fuse_registrations,
                                         homography_from_keypoints,
                                         registration_disagreement,
                                         symmetry_error)


def test_every_landmark_is_on_the_court():
    for index, (x, y) in KEYPOINTS.items():
        assert -0.01 <= x <= COURT_WIDTH_FT + 0.01, index
        assert -0.01 <= y <= COURT_LENGTH_FT + 0.01, index


def test_the_schema_satisfies_the_datasets_own_mirror_symmetry():
    """`flip_idx` is annotation metadata the schema was never fitted to.

    The first attempt at these coordinates -- bootstrapped from the painted-key
    registration -- failed this at 65 ft, which is how it was caught.
    """
    assert symmetry_error(KEYPOINTS) < 0.01


def test_landmarks_match_known_nba_geometry():
    """Spot-checks against dimensions that are facts about the sport."""
    assert KEYPOINTS[8] == (25.0, 5.25)          # the basket
    assert KEYPOINTS[22] == (25.0, 47.0)         # centre court
    assert KEYPOINTS[13] == (25.0, 19.0)         # free-throw line centre
    assert KEYPOINTS[3][0] == 17.0 and KEYPOINTS[4][0] == 33.0   # lane, 16 wide
    assert KEYPOINTS[0] == (0.0, 0.0) and KEYPOINTS[42] == (50.0, 94.0)


def test_a_perfect_view_recovers_the_court_exactly():
    pytest.importorskip("cv2")
    import cv2
    # A synthetic camera looking at the court.
    court = np.array([[0, 0], [50, 0], [50, 94], [0, 94]], dtype=np.float32)
    image = np.array([[100, 600], [900, 600], [780, 120], [220, 120]],
                     dtype=np.float32)
    court_to_image = cv2.getPerspectiveTransform(court, image)

    seen = {}
    for index, (x, y) in KEYPOINTS.items():
        p = np.array([x, y, 1.0]) @ court_to_image.T
        seen[index] = (p[0] / p[2], p[1] / p[2])

    matrix, inliers = homography_from_keypoints(seen)
    assert matrix is not None and inliers >= 30
    for index, (x, y) in KEYPOINTS.items():
        px, py = seen[index]
        q = np.array([px, py, 1.0]) @ matrix.T
        got = q[:2] / q[2]
        assert np.hypot(got[0] - x, got[1] - y) < 0.1, index


def test_occluding_the_paint_still_registers():
    """The failure that defeats the painted key: players standing in the lane.

    The key method needs one unoccluded quad, so this costs it the frame. Here
    it costs a handful of landmarks out of the ones spread across the floor.
    """
    pytest.importorskip("cv2")
    import cv2
    court = np.array([[0, 0], [50, 0], [50, 94], [0, 94]], dtype=np.float32)
    image = np.array([[100, 600], [900, 600], [780, 120], [220, 120]],
                     dtype=np.float32)
    court_to_image = cv2.getPerspectiveTransform(court, image)
    hidden = {3, 4, 12, 13, 14, 8, 17}          # the whole near lane and arc
    seen = {}
    for index, (x, y) in KEYPOINTS.items():
        if index in hidden:
            continue
        p = np.array([x, y, 1.0]) @ court_to_image.T
        seen[index] = (p[0] / p[2], p[1] / p[2])
    matrix, inliers = homography_from_keypoints(seen)
    assert matrix is not None and inliers >= 20


def test_too_few_landmarks_is_refused_not_guessed():
    pytest.importorskip("cv2")
    few = {i: (100.0 + 10 * n, 200.0 + 5 * n)
           for n, i in enumerate(list(KEYPOINTS)[:MIN_KEYPOINTS - 1])}
    matrix, inliers = homography_from_keypoints(few)
    assert matrix is None


def test_unknown_indices_are_ignored_rather_than_guessed():
    pytest.importorskip("cv2")
    import cv2
    court = np.array([[0, 0], [50, 0], [50, 94], [0, 94]], dtype=np.float32)
    image = np.array([[100, 600], [900, 600], [780, 120], [220, 120]],
                     dtype=np.float32)
    court_to_image = cv2.getPerspectiveTransform(court, image)
    seen = {}
    for index, (x, y) in KEYPOINTS.items():
        p = np.array([x, y, 1.0]) @ court_to_image.T
        seen[index] = (p[0] / p[2], p[1] / p[2])
    seen[2] = (5.0, 5.0)        # never annotated in 850 frames
    seen[47] = (9.0, 9.0)
    matrix, inliers = homography_from_keypoints(seen)
    assert matrix is not None
    assert inliers <= len(KEYPOINTS), "the unknown indices must not be used"


def test_flip_index_is_an_involution():
    """A mirror applied twice is the identity; this is what makes it a check."""
    for i, j in enumerate(FLIP_INDEX):
        assert FLIP_INDEX[j] == i


def test_the_flip_pairs_swap_ends_which_is_what_a_left_right_flip_does():
    """`fliplr` augmentation is only safe if `flip_idx` matches the geometry.

    A broadcast camera looks along the sideline, so image-x runs along the
    court's length: flipping the image left-right swaps the two baskets. The
    dataset's own pairs must therefore mirror about half-court -- 0 pairs with
    35, which the schema places at (0, 0) and (0, 94). If they mirrored about
    the centre line instead, every flipped training image would carry
    systematically wrong labels, silently.
    """
    from courtvision.court_keypoints import COURT_LENGTH_FT

    checked = 0
    for a, b in ((i, FLIP_INDEX[i]) for i in KEYPOINTS):
        if b not in KEYPOINTS or a >= b:
            continue
        (ax, ay), (bx, by) = KEYPOINTS[a], KEYPOINTS[b]
        assert abs(ax - bx) < 0.01, (a, b)                    # same side
        assert abs(ay + by - COURT_LENGTH_FT) < 0.01, (a, b)  # opposite end
        checked += 1
    assert checked >= 14


def _camera(pan):
    """A homography for a camera panned `pan` pixels along the court."""
    import cv2
    court = np.array([[0, 0], [50, 0], [50, 94], [0, 94]], dtype=np.float32)
    image = np.array([[100 - pan, 600], [900 - pan, 600],
                      [780 - pan, 120], [220 - pan, 120]], dtype=np.float32)
    return cv2.getPerspectiveTransform(image, court)      # image -> court


def test_fusion_averages_away_per_frame_noise():
    pytest.importorskip("cv2")
    import cv2
    rng = np.random.default_rng(0)
    probe = np.array([[300, 500], [600, 520], [450, 400], [700, 450],
                      [250, 430], [550, 560]], dtype=np.float32)
    pans = np.arange(9) * 4.0
    truth = _camera(pans[4])

    carries, matrices = [], []
    for n, pan in enumerate(pans):
        exact = _camera(pan)
        # Jitter each frame's registration the way a landmark detector does.
        wobble = np.eye(3)
        wobble[:2, 2] = rng.normal(0, 1.1, 2)
        matrices.append(wobble @ exact)
        if n < len(pans) - 1:
            shift = np.eye(3)
            shift[0, 2] = -(pans[n + 1] - pan)
            carries.append(shift)

    def error(matrix):
        got = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        want = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), truth).reshape(-1, 2)
        return float(np.median(np.hypot(*(got - want).T)))

    fused, used = fuse_registrations(matrices, carries, probe)
    assert used == 9
    assert error(fused) < error(matrices[4])


def test_one_registration_at_the_wrong_end_does_not_drag_the_answer():
    """The failure mode that matters: not a small error, but the other basket."""
    pytest.importorskip("cv2")
    import cv2
    probe = np.array([[300, 500], [600, 520], [450, 400], [700, 450],
                      [250, 430], [550, 560]], dtype=np.float32)
    matrices = [_camera(0.0) for _ in range(5)]
    flip = np.array([[1, 0, 0], [0, -1, 94.0], [0, 0, 1]])   # swap ends
    matrices[1] = flip @ matrices[1]
    carries = [np.eye(3) for _ in range(4)]

    fused, used = fuse_registrations(matrices, carries, probe)
    assert used == 5
    got = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), fused).reshape(-1, 2)
    want = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), _camera(0.0)).reshape(-1, 2)
    assert np.median(np.hypot(*(got - want).T)) < 0.5


def test_a_refused_orb_hop_truncates_the_window_rather_than_guessing():
    pytest.importorskip("cv2")
    matrices = [_camera(p * 4.0) for p in range(7)]
    carries = [np.eye(3)] * 6
    carries[4] = None                       # ORB refused between 4 and 5
    fused, used = fuse_registrations(matrices, carries, np.array(
        [[300, 500], [600, 520], [450, 400], [700, 450], [250, 430], [550, 560]],
        dtype=np.float32))
    assert used == 5, "frames past the break must be dropped, not chained"


def test_fusion_reports_when_no_fusion_happened():
    pytest.importorskip("cv2")
    probe = np.array([[300, 500], [600, 520], [450, 400], [700, 450],
                      [250, 430], [550, 560]], dtype=np.float32)
    matrices = [None, _camera(0.0), None]
    fused, used = fuse_registrations(matrices, [None, None], probe, centre=1)
    assert used == 1, "a caller must be able to tell a fused result from a bare one"


def test_perfect_registrations_disagree_by_nothing():
    pytest.importorskip("cv2")
    probe = np.array([[300, 500], [600, 520], [450, 400]], dtype=np.float32)
    pan = np.eye(3); pan[0, 2] = -17.0          # camera moved 17 px between frames
    a, b = _camera(0.0), _camera(17.0)
    assert registration_disagreement(a, b, pan, probe).max() < 1e-6


def test_a_registration_off_by_a_known_amount_is_reported_as_that_amount():
    """The measurement must be in feet and must not be self-cancelling."""
    pytest.importorskip("cv2")
    probe = np.array([[300, 500], [600, 520], [450, 400]], dtype=np.float32)
    pan = np.eye(3); pan[0, 2] = -17.0
    slip = np.array([[1, 0, 3.0], [0, 1, 0], [0, 0, 1]])    # 3 ft along the court
    got = registration_disagreement(_camera(0.0), slip @ _camera(17.0), pan, probe)
    assert np.allclose(got, 3.0, atol=1e-6)


def test_a_failed_refit_falls_back_to_the_centre_rather_than_losing_the_instant():
    """Fusion must never do worse than not fusing.

    If the carried estimates disagree too much to admit a homography, the
    caller should still get the registration it would have had anyway --
    reported as unfused, so the distinction stays visible.
    """
    pytest.importorskip("cv2")
    # Probe points that are collinear cannot determine a homography, so the
    # refit fails while the centre registration is perfectly good.
    probe = np.array([[100, 300], [200, 300], [300, 300], [400, 300],
                      [500, 300], [600, 300]], dtype=np.float32)
    matrices = [_camera(0.0) for _ in range(3)]
    fused, used = fuse_registrations(matrices, [np.eye(3), np.eye(3)], probe)
    assert fused is not None
    assert used in (1, 3)
    if used == 1:
        assert np.allclose(fused, matrices[1])


def test_a_lone_neighbour_registration_is_carried_not_reported_as_nothing():
    """used>=1 must mean a matrix was produced.

    The centre frame failing while a neighbour registered is common -- one
    blurred frame in a pan. Returning (None, 1) claimed a registration and
    supplied none, so a caller checking `used` got a null matrix.
    """
    pytest.importorskip("cv2")
    import cv2
    probe = np.array([[300, 500], [600, 520], [450, 400], [700, 450],
                      [250, 430], [550, 560]], dtype=np.float32)
    matrices = [_camera(0.0), None, None]        # only the far neighbour worked
    carries = [np.eye(3), np.eye(3)]
    fused, used = fuse_registrations(matrices, carries, probe, centre=1)
    assert used >= 1
    assert fused is not None, "used >= 1 must mean a matrix came back"
    got = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), fused).reshape(-1, 2)
    want = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), _camera(0.0)).reshape(-1, 2)
    assert np.median(np.hypot(*(got - want).T)) < 0.5


def test_a_mirrored_registration_is_refused():
    """Corrects a claim this project recorded: that geometry cannot see an
    end swap. The paint is symmetric, but a reflection reverses orientation."""
    pytest.importorskip("cv2")
    import cv2
    from courtvision.court_keypoints import COURT_ORIENTATION, orientation_sign

    court = np.array([[0, 0], [50, 0], [50, 94], [0, 94]], dtype=np.float32)
    image = np.array([[100, 600], [900, 600], [780, 120], [220, 120]],
                     dtype=np.float32)
    c2i = cv2.getPerspectiveTransform(court, image)
    seen = {}
    for index, (x, y) in KEYPOINTS.items():
        p = np.array([x, y, 1.0]) @ c2i.T
        seen[index] = (p[0] / p[2], p[1] / p[2])

    good, _ = homography_from_keypoints(seen)
    assert good is not None
    pts = np.array(list(seen.values()), dtype=np.float32)
    assert orientation_sign(good, pts) == COURT_ORIENTATION

    mirrored = np.array([[1, 0, 0], [0, -1, 94.0], [0, 0, 1]]) @ good
    assert orientation_sign(mirrored, pts) == -COURT_ORIENTATION


def test_the_180_degree_rotation_is_NOT_caught_and_that_is_expected():
    """End AND side swap preserves orientation -- the reverse-angle camera.

    Pinned so the limit stays stated: this one needs a temporal or feed cue,
    not geometry.
    """
    pytest.importorskip("cv2")
    import cv2
    from courtvision.court_keypoints import orientation_sign

    court = np.array([[0, 0], [50, 0], [50, 94], [0, 94]], dtype=np.float32)
    image = np.array([[100, 600], [900, 600], [780, 120], [220, 120]],
                     dtype=np.float32)
    c2i = cv2.getPerspectiveTransform(court, image)
    seen = {}
    for index, (x, y) in KEYPOINTS.items():
        p = np.array([x, y, 1.0]) @ c2i.T
        seen[index] = (p[0] / p[2], p[1] / p[2])
    good, _ = homography_from_keypoints(seen)
    pts = np.array(list(seen.values()), dtype=np.float32)
    rotated = np.array([[-1, 0, 50.0], [0, -1, 94.0], [0, 0, 1]]) @ good
    assert orientation_sign(rotated, pts) == orientation_sign(good, pts)
