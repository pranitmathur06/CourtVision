"""Retrieval over a season of basketball, with embeddings that earn their place.

The page this replaces answers by regular expression over a flat log, and it is
good at exactly the questions a regular expression is good at: a name, an action,
a quarter. It cannot answer "when did they switch the pick and roll" because no
row contains the word "switch", and it cannot answer a paraphrase because the
words are not there to match.

THE ARGUMENT THIS PACKAGE HAS TO WIN, stated before any of it was built:

  G1  the hybrid must not LOSE to the regular expression on counting questions.
      The page gets a scorer's 29 points right because it reads the whole log;
      any retrieval that answers from a top-k list gets that wrong, quietly.
  G2  the vector arm must WIN tactical and paraphrase questions by at least ten
      points of recall@10, or the embedding layer is redesigned rather than
      shipped. An embedding that only matches literal names is a slower regex.
  G3  vector-only must LOSE to the hybrid, or the structured filters are doing
      nothing and should be deleted.

And one thing that is not a gate but a property: `exact aggregation` --
`tests/test_exact_aggregation.py` asserts that the page's totals come from the
whole log and not from the retrieved rows, by replacing the retriever with one
that returns a single row and requiring the answer to be unchanged. Nothing in
here may make that test fail.
"""

from courtvision.retrieval.cards import Card, cards_from_stream, possessions
from courtvision.retrieval.store import LocalStore, VectorStore

__all__ = ["Card", "cards_from_stream", "possessions", "LocalStore", "VectorStore"]
