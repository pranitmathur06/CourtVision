"""bge-base-en-v1.5, 768 dimensions, and the two details that silently cost recall.

WHY THIS MODEL. It is small enough to embed a season on a rented GPU for about
nothing -- 263,000 cards is minutes of work -- and it is one of the checkpoints
Workers AI serves, so the same weights can run at ingest locally and at query
time in the edge. That only helps if both sides agree, which is what
`parity` exists to check.

THE TWO DETAILS.

  IT IS ASYMMETRIC. A query has to carry the instruction prefix and a document
  must NOT. Embedding both sides the same way costs several points of recall and
  produces no error, no warning, and no obviously wrong output -- the neighbours
  are simply a bit worse, forever. `encode_query` and `encode_documents` are
  separate functions for that reason, and there is no argument that switches
  between them.

  IT POOLS ON CLS, NOT ON THE MEAN. Mean pooling is the common default and is
  what most code reaches for; on this checkpoint it is wrong and, again, it
  merely degrades. Both are implemented here so the test can show the
  difference rather than assert the right one on faith.

NOTHING HERE TOUCHES A NETWORK AT QUERY TIME. The model is loaded once from the
local HuggingFace cache. The whole retrieval evaluation runs offline, which the
plan requires before any cloud work starts.
"""

from __future__ import annotations

from functools import lru_cache

#: The checkpoint, pinned. A different revision is a different index.
MODEL = "BAAI/bge-base-en-v1.5"
#: bge's own instruction for the query side. Documents get nothing.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
#: What the model returns, and what any store has to be built for.
DIMENSIONS = 768


@lru_cache(maxsize=1)
def _loaded(name: str = MODEL):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokeniser = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    torch.set_grad_enabled(False)
    return tokeniser, model


def _encode(texts, pooling: str = "cls", batch: int = 32):
    import numpy as np
    import torch

    tokeniser, model = _loaded()
    out = []
    for start in range(0, len(texts), batch):
        chunk = list(texts[start:start + batch])
        encoded = tokeniser(chunk, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        if pooling == "cls":
            vectors = hidden[:, 0]
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        out.append(vectors.cpu().numpy().astype(np.float32))
    return np.vstack(out) if out else np.zeros((0, DIMENSIONS), dtype=np.float32)


def encode_documents(texts, pooling: str = "cls"):
    """Card text. NO instruction prefix -- that is the query side's."""
    return _encode(list(texts), pooling=pooling)


def encode_query(text: str, pooling: str = "cls"):
    """One question, with bge's instruction prefix attached."""
    return _encode([QUERY_PREFIX + text], pooling=pooling)[0]


def parity(texts, other) -> float:
    """Mean cosine between these vectors and another implementation's.

    The repo gate for "the same checkpoint on both sides". A mismatch in
    pooling, prefix or revision shows up here as a number well below 1 and
    nowhere else at all -- the index still builds, the search still returns ten
    rows, and they are quietly the wrong ten.
    """
    import numpy as np

    ours = encode_documents(texts)
    theirs = np.asarray(other, dtype=np.float32).reshape(len(texts), -1)
    theirs = theirs / np.maximum(np.linalg.norm(theirs, axis=1, keepdims=True), 1e-12)
    return float((ours * theirs).sum(axis=1).mean())
