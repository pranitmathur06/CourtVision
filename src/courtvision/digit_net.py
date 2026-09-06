"""A jersey-number reader trained on SVHN rather than on our own crops.

Every earlier reader here learned from the hand-labelled jersey crops
themselves -- about 130 digit instances, which is a training set two orders of
magnitude too small, and which forced leave-one-out contortions that leaked
near-duplicate frames into the pool. SVHN is 73k real-world digits under
exactly the conditions that break a jersey read: low resolution, motion blur,
odd fonts, both polarities, distractor digits crowding the sides.

Training elsewhere buys two things. The hand labels become a pure test set, and
the confidence score becomes meaningful -- the network is calibrated on 26k
held-out digits, so a floor chosen there transfers, instead of being read off
the answer sheet.

Measured on 130 hand-labelled broadcast crops, with the floor chosen by
two-fold cross-validation over shot groups so no crop is scored by a floor its
own half selected: 88% precision at 25% coverage. At the more conservative
floor picked on SVHN alone (0.90) it is 94% on 18 answers at 14% coverage.

Reading is deliberately partial. A jersey read that is wrong attributes a play
to the wrong player, which is the one error a coaching tool cannot make, so
most crops get no answer at all and the number comes from voting over a track.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Input size, matching SVHN's cropped-digit format.
GLYPH_PX = 32
# Chosen by cross-validation on held-out folds of the hand-labelled crops.
DEFAULT_FLOOR = 0.68
# A digit is roughly this fraction as wide as it is tall; a blob much wider is
# two digits touching, which binarisation merges into one component.
DIGIT_ASPECT = 0.62
WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "jersey_digits.pt"


def build_net():
    """The classifier. Small on purpose: 32x32 grayscale, ten classes."""
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, 32, 3, padding=1);   self.b1 = nn.BatchNorm2d(32)
            self.c2 = nn.Conv2d(32, 32, 3, padding=1);  self.b2 = nn.BatchNorm2d(32)
            self.c3 = nn.Conv2d(32, 64, 3, padding=1);  self.b3 = nn.BatchNorm2d(64)
            self.c4 = nn.Conv2d(64, 64, 3, padding=1);  self.b4 = nn.BatchNorm2d(64)
            self.c5 = nn.Conv2d(64, 128, 3, padding=1); self.b5 = nn.BatchNorm2d(128)
            self.fc = nn.Linear(128 * 4 * 4, 10)
            self.drop = nn.Dropout(0.3)

        def forward(self, x):
            import torch.nn.functional as F
            x = F.relu(self.b1(self.c1(x)))
            x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2)
            x = F.relu(self.b3(self.c3(x)))
            x = F.max_pool2d(F.relu(self.b4(self.c4(x))), 2)
            x = F.max_pool2d(F.relu(self.b5(self.c5(x))), 2)
            return self.fc(self.drop(x.flatten(1)))

    return Net()


def _square_patch(gray: np.ndarray, x: int, y: int, w: int, h: int):
    """One digit's grayscale box, padded square and standardised.

    Padding to a square rather than stretching matters: a stretched '1' becomes
    something between a '1' and a '0', and SVHN never shows the network that.
    The padding value is the region's median so the fill reads as background.
    """
    import cv2

    pad = int(0.15 * h)
    top, bottom = max(0, y - pad), min(gray.shape[0], y + h + pad)
    left, right = max(0, x - pad), min(gray.shape[1], x + w + pad)
    sub = gray[top:bottom, left:right]
    if sub.size == 0:
        return None
    side = max(sub.shape)
    canvas = np.full((side, side), float(np.median(sub)), np.float32)
    oy, ox = (side - sub.shape[0]) // 2, (side - sub.shape[1]) // 2
    canvas[oy:oy + sub.shape[0], ox:ox + sub.shape[1]] = sub
    out = cv2.resize(canvas, (GLYPH_PX, GLYPH_PX), interpolation=cv2.INTER_AREA)
    return (out - out.mean()) / (out.std() + 1e-6)


def digit_patches(crop: np.ndarray) -> list[np.ndarray]:
    """Up to two digit-shaped patches from a torso crop, left to right.

    Both threshold polarities are tried because a jersey is as often a light
    number on a dark kit as the reverse, and whichever finds more digits wins.
    A blob much wider than one digit is split in half rather than discarded:
    touching digits binarise as a single component, and dropping those was
    rejecting a third of all crops.
    """
    import cv2

    if crop is None or crop.size == 0:
        return []
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY) if big.ndim == 3 else big
    gray = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
    scaled = gray.astype(np.float32) / 255.0
    height, width = gray.shape
    best: list[tuple[np.ndarray, int]] = []
    for invert in (False, True):
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
        blobs = []
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if not 0.18 * height <= h <= 0.75 * height:
                continue
            if not 0.04 * width <= w <= 0.60 * width:
                continue
            if not 0.25 < w / max(h, 1) < 2.2:
                continue
            if area < 0.25 * w * h:
                continue
            cx, cy = centres[i]
            if not (0.15 * width < cx < 0.85 * width
                    and 0.15 * height < cy < 0.85 * height):
                continue
            blobs.append((x, y, w, h))
        if not blobs:
            continue
        blobs.sort(key=lambda b: -b[3])
        x0, y0, _, h0 = blobs[0]
        # Digits of one number share a height and a baseline; anything else in
        # the frame is a logo, a seam or a limb.
        group = sorted((b for b in blobs
                        if abs(b[3] - h0) < 0.30 * h0
                        and abs(b[1] - y0) < 0.40 * h0
                        and abs(b[0] - x0) < 2.8 * max(blobs[0][2], 1)),
                       key=lambda b: b[0])
        found = []
        for x, y, w, h in group:
            parts = min(2, max(1, int(round(w / max(h * DIGIT_ASPECT, 1)))))
            step = w // parts
            for p in range(parts):
                one = _square_patch(scaled, x + p * step, y,
                                    step if p < parts - 1 else w - p * step, h)
                if one is not None:
                    found.append((one, x + p * step))
        found = sorted(found, key=lambda f: f[1])[:2]
        if len(found) > len(best):
            best = found
    return [patch for patch, _ in best]


class DigitReader:
    """Reads a jersey number from a torso crop, or declines to."""

    def __init__(self, weights=WEIGHTS, floor: float = DEFAULT_FLOOR,
                 device: str | None = None):
        import torch

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.floor = floor
        self.net = build_net().to(device)
        self.net.load_state_dict(torch.load(str(weights), map_location=device))
        self.net.eval()

    def read(self, crop: np.ndarray) -> tuple[str, float] | None:
        """(number, confidence) when confident, otherwise None.

        Confidence is the WEAKEST digit's probability, not the mean: a number
        is only as trustworthy as its worst character, and averaging lets a
        certain '4' carry an unreadable second digit.
        """
        import torch
        import torch.nn.functional as F

        patches = digit_patches(crop)
        if not patches:
            return None
        batch = torch.from_numpy(np.stack(patches)[:, None].astype(np.float32))
        with torch.no_grad():
            probs = F.softmax(self.net(batch.to(self.device)), dim=1).cpu()
        confidence, predicted = probs.max(1)
        weakest = float(confidence.min())
        if weakest < self.floor:
            return None
        number = "".join(str(int(p)) for p in predicted)
        # A leading zero is worn (#00, #0) but "07" is not a number anyone wears.
        if len(number) == 2 and number[0] == "0" and number[1] != "0":
            return None
        return number, weakest
