"""Spec for carryia/serve/retrieval.py -- the serve-side additions to the lifted
retrieval layer. The id-based adapters + RRF are already pinned by
test_eval_retrieval.py (same functions, re-exported there); this file covers only
what serve adds: rehydrating ranked tip_ids back to full docs.

TDD state: GREEN regression locks (deterministic, no index, no model). The one
model-dependent path -- build_hybrid_docs_retriever over real embeddings -- is a
SKIPPED live smoke (opt in: CARRYIA_LIVE=1), matching test_rag_helper.
"""

import os

import pytest

from carryia.serve import retrieval


def _corpus():
    return [
        {"tip_id": "a", "tip": "Ward the pit before dragon.", "rationale": "r", "creator_id": "c", "source_url": "u"},
        {"tip_id": "b", "tip": "Freeze when ahead.", "rationale": "r", "creator_id": "c", "source_url": "u"},
        {"tip_id": "c", "tip": "Roam after pushing.", "rationale": "r", "creator_id": "c", "source_url": "u"},
    ]


# --- GREEN: docs_retriever rehydrates ranked ids -> full docs, in order --------

def test_docs_retriever_maps_ids_to_docs_preserving_rank_order():
    corpus = _corpus()
    by_id = {d["tip_id"]: d for d in corpus}
    ranked = retrieval.docs_retriever(lambda q: ["c", "a"], by_id)
    out = ranked("any question")
    assert [d["tip_id"] for d in out] == ["c", "a"]   # order is the ranking's, not the corpus'
    assert out[0]["tip"] == "Roam after pushing."      # full doc, not just the id


def test_docs_retriever_skips_ids_absent_from_the_corpus():
    # A retriever can only ever return ids it saw at index time, but guard anyway:
    # a stray id must be skipped, not crash the answer.
    by_id = {d["tip_id"]: d for d in _corpus()}
    ranked = retrieval.docs_retriever(lambda q: ["a", "ghost", "b"], by_id)
    assert [d["tip_id"] for d in ranked("q")] == ["a", "b"]


# --- SKIPPED: the hybrid retriever over real embeddings (opt in) ---------------

@pytest.mark.skipif(not os.getenv("CARRYIA_LIVE"),
                    reason="builds real indexes (fastembed download); set CARRYIA_LIVE=1")
def test_build_hybrid_docs_retriever_returns_grounded_docs():
    retrieve = retrieval.build_hybrid_docs_retriever(_corpus(), k=2)
    out = retrieve("how do i get vision for dragon")
    assert out and all("tip" in d and "source_url" in d for d in out)
    assert len(out) <= 2
