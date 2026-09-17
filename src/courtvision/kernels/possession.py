"""Who has the ball: a fused, learned operator over players in one frame.

WHY THIS EXISTS. The detector's `handler` class names the right player 49.7% of
the time, measured on 157 held-out frames a person labelled by hand. It is stuck
there for a structural reason: its training labels were produced by a proximity
rule -- whoever is nearest the most confident ball -- and that rule's own ceiling,
measured with the ball position supplied BY HAND, is 58.7%. On a dribble a
defender's hands are often nearer the ball than the holder's. A model cannot beat
its teacher, and every variant of the rule (nearest box, box containing the ball,
nearest in body-heights, spanning it horizontally, each of them voted over a
1.2 s window) lands between 42% and 49%.

So this looks at evidence the rules never had:

  A BALL-EVIDENCE SWEEP IN PIXELS, per player, over a grid on his upper body.
  The labelling round showed a person finds the ball on 65% of the frames where
  the global ball detector's best candidate is under 0.35 -- the ball is there
  and the detector missed it. "Is there a ball in THIS player's hands" is a much
  easier question than "where is the ball in this 1280x720 frame".

  THE GEOMETRY, NORMALISED BY BODY HEIGHT, so it does not change meaning as
  players move up the floor and get smaller.

  WHERE TO LOOK, LEARNED. The sweep's region is four parameters -- the fractions
  of the box it covers -- and they are trained, not guessed, through a bilinear
  sample. The gradient of the loss with respect to "look a bit lower on the
  body" is a real spatial derivative of the image, and deriving it by hand is
  most of what makes this a custom op rather than a fast forward.

THREE IMPLEMENTATIONS, ONE SET OF NUMBERS, which is the pattern `torso_color.py`
established here:

  `possession_reference`  NumPy float64. The oracle. Slow and obviously correct.
  `possession_torch`      torch ops, runs on CPU/MPS/CUDA today. Because autograd
                          differentiates it, it is also the oracle for the
                          BACKWARD: the hand-written gradients are checked
                          against this, not against my own algebra twice.
  `PossessionFunction`    the fused CUDA path, falling back to torch without a GPU.

SMOOTH MAX, NOT MAX. Evidence over the grid is pooled with a log-sum-exp at
inverse temperature BETA rather than a hard maximum, so the derivative with
respect to the sampling region exists everywhere instead of being zero almost
everywhere and undefined on a measure-zero set.
"""

from __future__ import annotations

import numpy as np

#: Samples across the swept region, per side. 8x8 = 64 points per player.
GRID = 8
#: Inverse temperature of the smooth maximum over those samples.
BETA = 12.0
#: Features per player, in this order:
#:   0 dx        ball to box centre, x, in body heights
#:   1 dy        ditto, y
#:   2 d         their length
#:   3 inside    1 when the ball's centre is inside the box
#:   4 peak      smooth max of ball-likeness over the swept region
#:   5 mean      its mean over the same region
#:   6 found     the ball detector's confidence, 0 when it offered nothing
#:
#: Feature 6 is what lets the operator learn WHEN to trust the geometry. Scored
#: by whether the detector found a ball, the first version read: with a ball,
#: 50% against 56% for plain proximity; without one, 52% against 43% for the
#: shipped handler class. Two regimes, and the right rule differs between them.
#: Handing the model the confidence lets it learn that gate from the training
#: half, instead of me reading it off the evaluation half and calling it a
#: design -- which is the way this project has fooled itself before.
FEATURES = 7
#: Hidden units in the scoring head.
HIDDEN = 8
#: Guard against a black pixel dividing by zero.
EPS = 1e-6
#: Radius of the surround ring, as a fraction of the player's box height.
#: A basketball is 24 cm across and a player is about 190 cm, so the ball is
#: ~0.13 of a body height and the ring sits just outside it.
#:
#: WHY A SURROUND AT ALL. The first version scored a sample by orange-ness
#: alone, and orange-ness cannot tell a basketball from the floor it bounces
#: on: measured on a broadcast frame the hardwood reads 0.099 and the crowd
#: 0.037, so the sweep learned to point at whoever stood on the most visible
#: wood. It trained to 31.8% against a 49.7% baseline. A ball is orange
#: AGAINST A CONTRASTING BACKGROUND; wood is orange everywhere. Centre minus
#: surround is near zero on the floor and large on a ball.
SURROUND = 0.13


def default_parameters(seed: int = 0) -> dict[str, np.ndarray]:
    """Small random weights and a sensible starting region (the upper body).

    The region starts where `team_assignment.TORSO_X/Y` looks for a jersey,
    because that is the part of a player the camera sees most reliably, and
    training moves it from there.
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(FEATURES)
    return {
        "first": rng.normal(0.0, scale, size=(HIDDEN, FEATURES)),
        "first_bias": np.zeros(HIDDEN),
        "second": rng.normal(0.0, 1.0 / np.sqrt(HIDDEN), size=HIDDEN),
        "second_bias": np.zeros(()),
        "nobody": np.zeros(()),
        # x0, x1, y0, y1 as fractions of the player's box
        "region": np.array([0.15, 0.85, 0.10, 0.65]),
        # Per-feature centring and scaling, fitted on the training set rather
        # than trained. Without it the head cannot use the sweep at all: dx and
        # dy run to +/-2 while the centre-surround response is ~0.01, so under a
        # tanh the geometry saturates and the pixels contribute nothing. The
        # first version without this scored 40.8% against a 49.7% baseline.
        "feature_mean": np.zeros(FEATURES),
        "feature_scale": np.ones(FEATURES),
    }


def _chromaticity(patch: np.ndarray) -> np.ndarray:
    """How orange a BGR pixel is, in [-0.5, 1].

    (r - (g+b)/2) / (r+g+b). Scale-free, so it does not move with the arena's
    exposure the way a raw channel would, and a basketball sits near the top of
    the range while wood, jerseys and skin do not.
    """
    patch = patch.astype(np.float64)
    blue, green, red = patch[..., 0], patch[..., 1], patch[..., 2]
    total = red + green + blue + EPS
    return (red - 0.5 * (green + blue)) / total


def _sample_bilinear(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear read of `field` at float positions, clamped at the edges."""
    height, width = field.shape
    x = np.clip(x, 0.0, width - 1.0 - 1e-6)
    y = np.clip(y, 0.0, height - 1.0 - 1e-6)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = x - x0
    fy = y - y0
    return ((1 - fx) * (1 - fy) * field[y0, x0]
            + fx * (1 - fy) * field[y0, x1]
            + (1 - fx) * fy * field[y1, x0]
            + fx * fy * field[y1, x1])


def _ball_response(field: np.ndarray, x: np.ndarray, y: np.ndarray,
                   radius: float) -> np.ndarray:
    """Centre minus the mean of four points a ball-radius away."""
    centre = _sample_bilinear(field, x, y)
    ring = 0.25 * (_sample_bilinear(field, x - radius, y)
                   + _sample_bilinear(field, x + radius, y)
                   + _sample_bilinear(field, x, y - radius)
                   + _sample_bilinear(field, x, y + radius))
    return centre - ring


def _grid_positions(box: np.ndarray, region: np.ndarray, grid: int = GRID):
    """Where the sweep samples this player, in image pixels."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    steps = (np.arange(grid) + 0.5) / grid
    fx = region[0] + (region[1] - region[0]) * steps
    fy = region[2] + (region[3] - region[2]) * steps
    xs = x1 + width * fx
    ys = y1 + height * fy
    return np.meshgrid(xs, ys, indexing="xy")


def possession_reference(image: np.ndarray, boxes: np.ndarray, ball: np.ndarray,
                         parameters: dict[str, np.ndarray],
                         grid: int = GRID, beta: float = BETA):
    """Probability each player has the ball, plus 'nobody'. The oracle.

    `image` (H, W, 3) uint8 BGR. `boxes` (P, 4) float. `ball` (3,) = x, y, conf;
    a confidence of zero means the detector offered nothing and the geometric
    features are zeroed, leaving the pixel sweep to answer alone -- which is the
    case this operator exists for.

    Returns (probabilities (P+1,), features (P, FEATURES)).
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    ball = np.asarray(ball, dtype=np.float64).reshape(3)
    chroma = _chromaticity(image)
    players = len(boxes)
    features = np.zeros((players, FEATURES))
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        body = max(y2 - y1, EPS)
        if ball[2] > 0:
            centre_x, centre_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dx = (ball[0] - centre_x) / body
            dy = (ball[1] - centre_y) / body
            inside = 1.0 if (x1 <= ball[0] <= x2 and y1 <= ball[1] <= y2) else 0.0
            features[i, 0:4] = (dx, dy, np.hypot(dx, dy), inside)
        features[i, 6] = ball[2]
        xs, ys = _grid_positions(box, parameters["region"], grid)
        radius = SURROUND * max(y2 - y1, EPS)
        swept = _ball_response(chroma, xs.ravel(), ys.ravel(), radius)
        # smooth maximum: (1/beta) * log(mean(exp(beta * e)))
        shifted = beta * swept
        peak = (shifted.max() + np.log(np.exp(shifted - shifted.max()).mean())) / beta
        features[i, 4] = peak
        features[i, 5] = swept.mean()

    standard = ((features - parameters["feature_mean"])
                / np.maximum(parameters["feature_scale"], EPS))
    hidden = np.tanh(standard @ parameters["first"].T + parameters["first_bias"])
    scores = hidden @ parameters["second"] + parameters["second_bias"]
    logits = np.concatenate([scores, np.atleast_1d(parameters["nobody"])])
    logits = logits - logits.max()
    weights = np.exp(logits)
    return weights / weights.sum(), features


def possession_torch(image, boxes, ball, parameters, grid: int = GRID,
                     beta: float = BETA):
    """The same numbers in torch, differentiable, on whatever device it is given.

    This is the gradient oracle. The fused kernel's hand-written backward is
    checked against autograd through this function rather than against a second
    hand derivation, which would only prove I made the same mistake twice.
    """
    import torch

    if not torch.is_tensor(image):
        image = torch.as_tensor(np.ascontiguousarray(image))
    device = parameters["first"].device
    image = image.to(device=device, dtype=torch.float64 if
                     parameters["first"].dtype == torch.float64 else torch.float32)
    dtype = parameters["first"].dtype
    boxes = torch.as_tensor(boxes, dtype=dtype, device=device).reshape(-1, 4)
    ball = torch.as_tensor(ball, dtype=dtype, device=device).reshape(3)

    blue, green, red = image[..., 0], image[..., 1], image[..., 2]
    chroma = (red - 0.5 * (green + blue)) / (red + green + blue + EPS)
    height, width = chroma.shape

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    box_w = (x2 - x1).unsqueeze(1)
    box_h = (y2 - y1).unsqueeze(1)
    region = parameters["region"]
    steps = (torch.arange(grid, dtype=dtype, device=device) + 0.5) / grid
    fx = region[0] + (region[1] - region[0]) * steps          # (G,)
    fy = region[2] + (region[3] - region[2]) * steps
    xs = x1.unsqueeze(1) + box_w * fx.unsqueeze(0)            # (P, G)
    ys = y1.unsqueeze(1) + box_h * fy.unsqueeze(0)
    grid_x = xs.unsqueeze(1).expand(-1, grid, -1).reshape(len(boxes), -1)
    grid_y = ys.unsqueeze(2).expand(-1, -1, grid).reshape(len(boxes), -1)

    def read(px, py):
        px = px.clamp(0.0, width - 1.0 - 1e-6)
        py = py.clamp(0.0, height - 1.0 - 1e-6)
        ax = px.floor()
        ay = py.floor()
        rx = px - ax
        ry = py - ay
        axi = ax.long()
        ayi = ay.long()
        bxi = (axi + 1).clamp(max=width - 1)
        byi = (ayi + 1).clamp(max=height - 1)
        return ((1 - rx) * (1 - ry) * chroma[ayi, axi]
                + rx * (1 - ry) * chroma[ayi, bxi]
                + (1 - rx) * ry * chroma[byi, axi]
                + rx * ry * chroma[byi, bxi])

    radius = (SURROUND * (y2 - y1).clamp_min(EPS)).unsqueeze(1)
    swept = read(grid_x, grid_y) - 0.25 * (read(grid_x - radius, grid_y)
                                           + read(grid_x + radius, grid_y)
                                           + read(grid_x, grid_y - radius)
                                           + read(grid_x, grid_y + radius))

    peak = torch.logsumexp(beta * swept, dim=1)
    peak = (peak - torch.log(torch.tensor(float(swept.shape[1]), dtype=dtype,
                                          device=device))) / beta
    mean = swept.mean(dim=1)

    body = (y2 - y1).clamp_min(EPS)
    if float(ball[2]) > 0:
        centre_x = (x1 + x2) / 2.0
        centre_y = (y1 + y2) / 2.0
        dx = (ball[0] - centre_x) / body
        dy = (ball[1] - centre_y) / body
        inside = ((ball[0] >= x1) & (ball[0] <= x2)
                  & (ball[1] >= y1) & (ball[1] <= y2)).to(dtype)
        geometry = torch.stack([dx, dy, torch.hypot(dx, dy), inside], dim=1)
    else:
        geometry = torch.zeros((len(boxes), 4), dtype=dtype, device=device)

    found = ball[2].expand(len(boxes)).unsqueeze(1)
    features = torch.cat([geometry, peak.unsqueeze(1), mean.unsqueeze(1), found],
                         dim=1)
    standard = ((features - parameters["feature_mean"])
                / parameters["feature_scale"].clamp_min(EPS))
    hidden = torch.tanh(standard @ parameters["first"].T + parameters["first_bias"])
    scores = hidden @ parameters["second"] + parameters["second_bias"]
    logits = torch.cat([scores, parameters["nobody"].reshape(1)])
    return torch.softmax(logits, dim=0), features


# ---- the fused path -------------------------------------------------------

_FUSED_CACHE: list = []


def entry_point_declarations(source: str) -> list[str]:
    """The `std::vector<torch::Tensor> name(...)` signatures in the .cu, as
    declarations.

    Read from the definitions rather than copied, so the two cannot drift.
    """
    out = []
    cursor = 0
    needle = "std::vector<torch::Tensor> "
    while True:
        start = source.find(needle, cursor)
        if start < 0:
            return out
        stop = source.index(") {", start)
        out.append(source[start:stop + 1] + ";")
        cursor = stop


def load_fused_kernel():
    """Compile possession.cu on first use, or None when there is no CUDA.

    Three mechanics here are not obvious and each was paid for once. The first
    two came from `torso_color.py`: load_inline generates its own pybind module,
    so the .cu's own PYBIND11_MODULE block is a duplicate and has to go; and the
    generated glue calls the entry points without ever seeing them, so their
    declarations must be handed over as `cpp_sources`. With `cpp_sources=""`
    nvcc compiles the .cu perfectly and the build then dies in the glue with
    "not declared in this scope", which reads like a CUDA failure and is not.

    The third is why the declarations are READ OUT OF THE .cu rather than
    written here. Kept by hand they went stale: the feature standardisation
    added two tensors to both entry points and this copy did not follow, so the
    extension compiled, linked, and failed at import with

        undefined symbol: _Z19possession_backwardN2at6TensorE...

    -- a mangled-name mismatch, which is what a wrong declaration looks like
    from the outside. Taking them from the definitions makes the two impossible
    to disagree.
    """
    import torch

    if _FUSED_CACHE:
        return _FUSED_CACHE[0]
    if not torch.cuda.is_available():
        return None
    from pathlib import Path

    from torch.utils.cpp_extension import load_inline

    source = Path(__file__).with_suffix(".cu").read_text()
    marker = "PYBIND11_MODULE"
    if marker in source:
        source = source[:source.index(marker)].rstrip() + "\n"
    declarations = "\n".join(entry_point_declarations(source))
    module = load_inline(
        name="courtvision_possession",
        cpp_sources=declarations,
        cuda_sources=source,
        functions=["possession_forward", "possession_backward",
                   "temporal_forward", "kernel_occupancy"],
        verbose=False,
    )
    _FUSED_CACHE.append(module)
    return module


class PossessionFunction:
    """Autograd bridge for the fused kernel. Built lazily so torch is optional."""

    _built: list = []

    @classmethod
    def get(cls):
        import torch

        if cls._built:
            return cls._built[0]

        class _Fused(torch.autograd.Function):
            @staticmethod
            def forward(ctx, image, boxes, ball, first, first_bias, second,
                        second_bias, nobody, region, beta):
                module = load_fused_kernel()
                probs, features = module.possession_forward(
                    image, boxes, ball, first, first_bias, second,
                    second_bias, nobody, region, float(beta))
                ctx.save_for_backward(image, boxes, first, first_bias, second,
                                      region, features, probs)
                ctx.beta = float(beta)
                return probs

            @staticmethod
            def backward(ctx, grad_probs):
                module = load_fused_kernel()
                (image, boxes, first, first_bias, second, region, features,
                 probs) = ctx.saved_tensors
                grads = module.possession_backward(
                    image, boxes, first, first_bias, second, region, features,
                    probs, grad_probs.contiguous(), ctx.beta)
                return (None, None, None, grads[0], grads[1], grads[2],
                        grads[3].reshape(()), grads[4].reshape(()), grads[5], None)

        cls._built.append(_Fused)
        return _Fused


def possession(image, boxes, ball, parameters, beta: float = BETA):
    """Probabilities that each player has the ball, plus 'nobody'.

    Takes the fused kernel when the parameters are on a CUDA device and it
    compiles; otherwise the portable torch path, which autograd differentiates
    for itself. Both produce the same numbers -- that is what the tests check.
    """
    import torch

    if torch.cuda.is_available() and parameters["first"].is_cuda:
        image_t = torch.as_tensor(np.ascontiguousarray(image)).to(parameters["first"].device)
        boxes_t = torch.as_tensor(np.asarray(boxes, dtype=np.float32),
                                  device=parameters["first"].device)
        ball_t = torch.as_tensor(np.asarray(ball, dtype=np.float32),
                                 device=parameters["first"].device)
        return PossessionFunction.get().apply(
            image_t, boxes_t, ball_t, parameters["first"],
            parameters["first_bias"], parameters["second"],
            parameters["second_bias"], parameters["nobody"],
            parameters["region"], beta)
    probs, _ = possession_torch(image, boxes, ball, parameters, beta=beta)
    return probs


# ---- kernel 2: possession over time ---------------------------------------
#
# WHY A SECOND KERNEL. Kernel 1 answers each frame alone, and possession is not
# a per-frame quantity: a player holds the ball for seconds and hand-offs are
# rare. On the held-out frames the shipped detector draws no handler box at all
# on 37 of 157 -- automatic misses for a per-frame method, and frames where a
# neighbour 0.25 s away is perfectly clear.
#
# A hard vote over a 1.2 s window was already tried and did not help (44.6%
# against 44.6%). This is a different object: the per-frame evidence stays SOFT,
# the cost of changing hands is LEARNED rather than assumed, and the whole thing
# is differentiable so kernel 1 trains through it.
#
# THE MODEL. States are the tracks in the window plus 'nobody'. A path assigns
# one state per frame and scores
#
#     sum_t s[t, k_t]  +  STAY * #{t : k_t = k_{t+1}}
#
# with STAY >= 0 (softplus of the trained parameter) so that keeping the ball is
# never punished. The answer read out is the posterior marginal at the centre
# frame, which is the frame a person labelled.
#
# WHY THIS SHAPE IS WORTH FUSING. The transition matrix is STAY on the diagonal
# and zero everywhere else, so the log-sum-exp over predecessors collapses:
#
#     logsumexp_j(alpha[t-1, j] + M[j, k]) = log( exp(alpha[t-1, k]) * (e^S - 1)
#                                                 + sum_j exp(alpha[t-1, j]) )
#
# One reduction shared by every state, then O(1) per state. That makes the pass
# a sequential scan over time that is parallel over states with a shared-memory
# reduction per step -- the classic shape that does not decompose into library
# ops, and the reason this is a kernel rather than a matmul.
#
# THE BACKWARD, DERIVED BY HAND. The loss is -log of the centre posterior at the
# labelled track, and
#
#     -log gamma[c, y] = logZ - logZ_clamped
#
# where the clamped model is the same model with frame c pinned to y -- because
# alpha[c, y] + beta[c, y] is precisely the log sum over paths through (c, y).
# Both terms are log partition functions, so each one's gradient is an
# expectation, and the gradient of the loss is the difference of two:
#
#     dL/ds[t, k] = gamma_free[t, k] - gamma_clamped[t, k]
#     dL/dSTAY    = sum_t xi_free[t]  - sum_t xi_clamped[t]
#
# with xi[t] the posterior probability that frame t and t+1 share a state. So
# the whole backward is a second forward-backward over a clamped model, and no
# recursion has to be differentiated term by term. The fused kernel runs both
# scans in one launch; `temporal_torch` runs the free one and lets autograd
# derive the rest, which is what the hand-written version is checked against.

#: Log-scores below the best by more than this contribute nothing measurable and
#: are floored, which keeps exp() away from underflow in the clamped pass.
CLAMP_FLOOR = -1e30


def softplus(x):
    """log(1 + e^x), computed the way that does not overflow."""
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _scan(scores: np.ndarray, stay: float, reverse: bool = False) -> np.ndarray:
    """Forward (or backward) log-messages for the stay/switch transition.

    Forward:  alpha[0] = scores[0]
              alpha[t] = scores[t] + combine(alpha[t-1])
    Backward: beta[T-1] = 0
              beta[t]   = combine(scores[t+1] + beta[t+1])

    where combine(v)[k] = log( exp(v[k]) * (e^S - 1) + sum_j exp(v[j]) ), shifted
    by max(v) so nothing overflows. e^S - 1 is written expm1(S) because S is
    often small and the subtraction would lose every digit of it.
    """
    frames = len(scores)
    out = np.zeros_like(scores)
    boost = np.expm1(stay)
    if not reverse:
        out[0] = scores[0]
        for t in range(1, frames):
            previous = out[t - 1]
            shift = previous.max()
            weights = np.exp(previous - shift)
            out[t] = scores[t] + shift + np.log(weights * boost + weights.sum())
    else:
        for t in range(frames - 2, -1, -1):
            previous = scores[t + 1] + out[t + 1]
            shift = previous.max()
            weights = np.exp(previous - shift)
            out[t] = shift + np.log(weights * boost + weights.sum())
    return out


def _marginals(scores: np.ndarray, stay: float):
    """Posterior over states per frame, the pairwise stay mass, and logZ."""
    alpha = _scan(scores, stay, reverse=False)
    beta = _scan(scores, stay, reverse=True)
    total = alpha[-1]
    shift = total.max()
    log_z = shift + np.log(np.exp(total - shift).sum())
    gamma = np.exp(alpha + beta - log_z)
    # xi[t] = P(state_t = state_{t+1}), summed over which state it is
    xi = 0.0
    for t in range(len(scores) - 1):
        joint = alpha[t] + stay + scores[t + 1] + beta[t + 1] - log_z
        xi += float(np.exp(joint).sum())
    return gamma, xi, log_z


def temporal_reference(scores: np.ndarray, stay_raw: float, target: int = -1,
                       centre: int = -1):
    """Forward-backward over a window. The oracle.

    `scores` (T, S) log-scores per frame per state, the last state being
    'nobody'. `stay_raw` is the untransformed stay parameter; the bonus applied
    is softplus(stay_raw), which cannot be negative.

    Returns (posterior at the centre frame (S,), loss, gradients dict). When
    `target` is negative only the posterior is meaningful -- there is nothing to
    take a loss against.
    """
    scores = np.asarray(scores, dtype=np.float64)
    frames = len(scores)
    centre = frames // 2 if centre < 0 else centre
    stay = float(softplus(stay_raw))

    gamma, xi, log_z = _marginals(scores, stay)
    posterior = gamma[centre] / gamma[centre].sum()
    if target < 0:
        return posterior, 0.0, {"scores": np.zeros_like(scores), "stay_raw": 0.0}

    clamped_scores = scores.copy()
    keep = clamped_scores[centre, target]
    clamped_scores[centre] = CLAMP_FLOOR
    clamped_scores[centre, target] = keep
    clamped_gamma, clamped_xi, clamped_log_z = _marginals(clamped_scores, stay)

    loss = log_z - clamped_log_z
    # d(softplus)/dx = sigmoid(x)
    sigmoid = 1.0 / (1.0 + np.exp(-np.float64(stay_raw)))
    return posterior, loss, {
        "scores": gamma - clamped_gamma,
        "stay_raw": (xi - clamped_xi) * float(sigmoid),
    }


def frame_logits(image, boxes, ball, parameters, grid: int = GRID,
                 beta: float = BETA) -> np.ndarray:
    """Kernel 1's per-player scores for one frame, unnormalised. NumPy.

    The scan wants log-scores per frame, not probabilities: normalising each
    frame on its own and then combining would throw away how confident the
    frame was, which is the whole reason a frame with no visible ball should
    defer to its neighbours.
    """
    _, features = possession_reference(image, boxes, ball, parameters, grid, beta)
    standard = ((features - parameters["feature_mean"])
                / np.maximum(parameters["feature_scale"], EPS))
    hidden = np.tanh(standard @ parameters["first"].T + parameters["first_bias"])
    return hidden @ parameters["second"] + parameters["second_bias"]


def load_parameters(path) -> dict[str, np.ndarray]:
    """Trained weights from disk, as the arrays every path here expects."""
    import json

    raw = json.load(open(path))
    return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}


def temporal_torch(scores, stay_raw, centre: int = -1):
    """The same scan in torch, differentiable. The gradient oracle.

    Only the free pass is written here. Autograd derives the gradient of
    -log(posterior[target]) from it, and that is what the hand-derived
    free-minus-clamped expressions above are checked against -- rather than
    against a second hand derivation, which would only prove the same mistake
    twice.
    """
    import torch

    frames = scores.shape[0]
    centre = frames // 2 if centre < 0 else centre
    stay = torch.nn.functional.softplus(stay_raw)
    boost = torch.expm1(stay)

    forward = [scores[0]]
    for t in range(1, frames):
        previous = forward[-1]
        shift = previous.max().detach()
        weights = torch.exp(previous - shift)
        forward.append(scores[t] + shift
                       + torch.log(weights * boost + weights.sum()))
    backward = [None] * frames
    backward[frames - 1] = torch.zeros_like(scores[0])
    for t in range(frames - 2, -1, -1):
        previous = scores[t + 1] + backward[t + 1]
        shift = previous.max().detach()
        weights = torch.exp(previous - shift)
        backward[t] = shift + torch.log(weights * boost + weights.sum())

    log_z = torch.logsumexp(forward[-1], dim=0)
    centre_log = forward[centre] + backward[centre] - log_z
    return torch.softmax(centre_log, dim=0)


class TemporalFunction:
    """Autograd bridge for the fused scan. Built lazily so torch is optional.

    The kernel returns the loss AND its gradients from one launch, because the
    clamped pass it needs for the gradient is most of the work of the forward
    anyway. So `forward` keeps the gradients and `backward` only scales them by
    what came from above -- which is exact here, the loss being a scalar.
    """

    _built: list = []

    @classmethod
    def get(cls):
        import torch

        if cls._built:
            return cls._built[0]

        class _Fused(torch.autograd.Function):
            @staticmethod
            def forward(ctx, scores, stay_raw, target, centre):
                module = load_fused_kernel()
                posterior, loss, grad_scores, grad_stay = module.temporal_forward(
                    scores.contiguous(), stay_raw.reshape(1).contiguous(),
                    int(target), int(centre))
                ctx.save_for_backward(grad_scores, grad_stay)
                ctx.mark_non_differentiable(posterior)
                return loss.reshape(()), posterior

            @staticmethod
            def backward(ctx, grad_loss, _grad_posterior):
                grad_scores, grad_stay = ctx.saved_tensors
                return (grad_loss * grad_scores, grad_loss * grad_stay.reshape(()),
                        None, None)

        cls._built.append(_Fused)
        return _Fused


def temporal(scores, stay_raw, target: int = -1, centre: int = -1):
    """Loss and centre-frame posterior for one window.

    Takes the fused kernel when the scores are on a CUDA device and it compiles;
    otherwise the portable scan, which autograd differentiates for itself. Both
    produce the same numbers -- that is what the tests and the host harness
    check.
    """
    import torch

    frames = scores.shape[0]
    centre = frames // 2 if centre < 0 else centre
    if torch.cuda.is_available() and scores.is_cuda:
        return TemporalFunction.get().apply(scores, stay_raw, target, centre)
    posterior = temporal_torch(scores, stay_raw, centre)
    if target < 0:
        return torch.zeros((), dtype=scores.dtype, device=scores.device), posterior
    return -torch.log(posterior[target] + 1e-12), posterior
