"""Where the vectors live, and the rule about who filters and who ranks.

D1 FILTERS, VECTORIZE RANKS -- NEVER THE REVERSE. A vector index can attach
scalar metadata and filter on equality, and it cannot express "this possession
involved Haliburton": involvement is array-valued and the player vocabulary is
about five hundred wide, so an `$in` list is either unbounded or wrong. A
relational store resolves the candidate set in one statement and the vector
index ranks inside it.

The consequence for this module is that `search` takes a candidate set, and a
store that cannot honour one says so rather than ignoring it. A retriever that
silently ranks the whole season when it was asked about one game returns
plausible rows from the wrong night.

THE LOCAL TWIN IS NOT A MOCK. `LocalStore` holds the vectors in a numpy array
and does exact cosine, so it is the ground truth an approximate index is checked
against -- and it is what lets the whole retrieval evaluation run with zero
cloud spend, which the plan requires before any Cloudflare work begins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Hit:
    """One retrieved card: what it is, how close, and where it came from."""
    card_id: str
    score: float
    fields: dict


@runtime_checkable
class VectorStore(Protocol):
    """Vectorize, pgvector and the local twin behind one shape.

    The same pattern `PlayerTracker` already uses: the thing that varies is
    hidden, the thing that must not vary -- that a candidate set is honoured --
    is in the signature.
    """

    def add(self, card_ids: Sequence[str], vectors, fields: Sequence[dict]) -> int:
        """Upsert. Returns how many were written."""

    def search(self, vector, limit: int = 10,
               candidates: Sequence[str] | None = None) -> list[Hit]:
        """Nearest by cosine, restricted to `candidates` when one is given."""

    def __len__(self) -> int:
        ...


class LocalStore:
    """Exact cosine over a numpy matrix. No index, no approximation, no network."""

    def __init__(self) -> None:
        import numpy as np

        self._np = np
        self._ids: list[str] = []
        self._at: dict[str, int] = {}
        self._fields: list[dict] = []
        self._matrix = None

    def add(self, card_ids: Sequence[str], vectors, fields: Sequence[dict]) -> int:
        np = self._np
        vectors = np.asarray(vectors, dtype=np.float32).reshape(len(card_ids), -1)
        # Normalised on the way in, so `search` is a matrix multiply and the
        # score it returns is a cosine rather than something proportional to it.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        fresh = [i for i, cid in enumerate(card_ids) if cid not in self._at]
        for i in fresh:
            self._at[card_ids[i]] = len(self._ids)
            self._ids.append(card_ids[i])
            self._fields.append(dict(fields[i]))
        block = vectors[fresh] if len(fresh) != len(card_ids) else vectors
        self._matrix = (block if self._matrix is None
                        else np.vstack([self._matrix, block]))
        return len(fresh)

    def search(self, vector, limit: int = 10,
               candidates: Sequence[str] | None = None) -> list[Hit]:
        np = self._np
        if self._matrix is None or not len(self._ids):
            return []
        query = np.asarray(vector, dtype=np.float32).ravel()
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        scores = self._matrix @ query
        if candidates is not None:
            allowed = np.zeros(len(self._ids), dtype=bool)
            for cid in candidates:
                i = self._at.get(cid)
                if i is not None:
                    allowed[i] = True
            # -inf rather than a mask, so a candidate set that matches nothing
            # returns nothing instead of quietly returning the whole season.
            scores = np.where(allowed, scores, -np.inf)
        order = np.argsort(-scores)[:limit]
        return [Hit(self._ids[i], float(scores[i]), self._fields[i])
                for i in order if np.isfinite(scores[i])]

    def __len__(self) -> int:
        return len(self._ids)
