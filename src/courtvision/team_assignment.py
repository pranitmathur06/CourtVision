"""Stage 4 — assign each track to team "A" or "B" by jersey colour.

Two things make the naive version wrong, and both are handled here:

1. k-means cluster labels are arbitrary — cluster 0 on one run is cluster 1 on the
   next. Clusters are therefore ordered deterministically by their centre before
   being named, so repeated runs agree.
2. A single frame's crop can be garbage (occlusion, motion blur, a player leaving
   frame). Every track is voted on across all its frames rather than trusted once.

Which physical team ends up called "A" is still arbitrary, and nothing downstream
may depend on it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import cv2
import numpy as np
from sklearn.cluster import KMeans

from courtvision.types import Box, Frame

# Fraction of the box occupied by the torso: centre horizontally, upper-middle
# vertically. Avoids the head, shorts, legs and the court either side.
TORSO_X = (0.25, 0.75)
TORSO_Y = (0.20, 0.55)


def torso_crop(image: np.ndarray, box: Box) -> np.ndarray:
    """Crop the jersey region of a player box. Returns an empty array if degenerate."""
    height, width = image.shape[:2]
    x1 = int(round(box.x1 + box.width * TORSO_X[0]))
    x2 = int(round(box.x1 + box.width * TORSO_X[1]))
    y1 = int(round(box.y1 + box.height * TORSO_Y[0]))
    y2 = int(round(box.y1 + box.height * TORSO_Y[1]))

    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2]


def mean_lab_color(crop: np.ndarray) -> np.ndarray:
    """Mean colour of a BGR crop in CIELAB, where Euclidean distance tracks perception."""
    if crop.size == 0:
        return np.zeros(3, dtype=np.float64)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float64)


def collect_samples(
    images: Sequence[np.ndarray], frames: Sequence[Frame]
) -> list[tuple[int, np.ndarray]]:
    """One (track_id, mean_lab_colour) sample per player per frame."""
    samples: list[tuple[int, np.ndarray]] = []
    for image, frame in zip(images, frames):
        for track in frame.players():
            crop = torso_crop(image, track.box)
            if crop.size == 0:
                continue
            samples.append((track.track_id, mean_lab_color(crop)))
    return samples


def assign_teams(samples: list[tuple[int, np.ndarray]]) -> dict[int, str]:
    """Cluster jersey colours into two teams and vote a single label per track."""
    if not samples:
        return {}

    track_ids = [track_id for track_id, _ in samples]
    colors = np.vstack([color for _, color in samples])

    unique_tracks = sorted(set(track_ids))
    if len(unique_tracks) < 2:
        return {track_id: "A" for track_id in unique_tracks}

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(colors)

    # Deterministic naming: order the two centres by their coordinates so the
    # same input always yields the same A/B assignment.
    centers = kmeans.cluster_centers_
    order = sorted(range(2), key=lambda i: tuple(centers[i]))
    cluster_to_team = {order[0]: "A", order[1]: "B"}

    votes: dict[int, Counter] = {}
    for track_id, cluster in zip(track_ids, kmeans.labels_):
        votes.setdefault(track_id, Counter())[cluster_to_team[int(cluster)]] += 1

    return {
        track_id: counter.most_common(1)[0][0] for track_id, counter in votes.items()
    }
