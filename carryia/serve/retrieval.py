"""The retrieval layer -- query-time `question -> ranked results`, shared by the
app (serve) and the P0-5 eval so both exercise the SAME code (no drift between
what was scored and what ships).

Two output shapes over one ranking:
  - the id-based `Retriever` (`str -> list[tip_id]`) -- what the eval scores
    against the ground-truth answer key.
  - `docs_retriever`, which wraps an id-based `Retriever` to return the full doc
    dicts -- what the app grounds an answer on (`build_context` needs the fields,
    and a bare tip_id can't be cited).

The P0-5 winner is HYBRID: keyword (minsearch BM25) + vector (local fastembed)
fused by Reciprocal Rank Fusion. `hybrid > vector > keyword`, consistent on
hit-rate and MRR (Journal 2026-08-27). `rrf_fuse` / `hybrid_retriever` were lifted
here from `eval/eval_retrieval.py` unchanged; that module now re-exports them, so
the eval's committed numbers still describe this exact code.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from carryia.pipeline.ingest import build_text_index, build_vector_index, embed_tips

# A retriever is any callable: a question -> ranked `tip_id`s, best first. The k
# (top-k depth) is baked into the retriever at construction, so the scoring loop
# stays approach-agnostic -- it consumes whatever ranking the retriever returns.
Retriever = Callable[[str], list[str]]

DEFAULT_K = 5


# --- adapt a retrieval approach to the Retriever contract --------------------

def keyword_retriever(index, k: int = DEFAULT_K) -> Retriever:
    """Wrap a minsearch keyword `Index` (from `ingest.build_text_index`) as a
    `Retriever`: given a question, run `index.search(question, num_results=k)`
    and return the top-k `tip_id`s in rank order (best first).

    This is the adapter, not the index -- `vector_retriever` has the same
    signature over the vector index, and the scoring code can't tell them apart.
    """
    def retrieve(question: str) -> list[str]:
        return [doc["tip_id"] for doc in index.search(question, num_results=k)]

    return retrieve


def vector_retriever(
    index,
    embed: Callable[[list[str]], "np.ndarray"] = embed_tips,
    k: int = DEFAULT_K,
) -> Retriever:
    """Wrap a minsearch `VectorSearch` (from `ingest.build_vector_index`) as a
    `Retriever`: embed the question with `embed`, search the vector index, and
    return the top-k `tip_id`s in rank order. Same `str -> tip_id`s contract as
    `keyword_retriever`, so the scoring code can't tell them apart.

    `embed` is injectable so tests pass a deterministic fake (no model
    download); it defaults to `embed_tips` -- the *same* local fastembed model
    the index was built with, which is what keeps query and document vectors in
    one space (a mismatched embedder would score garbage).
    """
    def retrieve(question: str) -> list[str]:
        query_vector = embed([question])[0]
        return [doc["tip_id"] for doc in index.search(query_vector, num_results=k)]

    return retrieve


# --- fuse retrievers: hybrid via Reciprocal Rank Fusion ----------------------

def rrf_fuse(
    rankings: list[list[str]], k: int = DEFAULT_K, rrf_k: int = 60,
) -> list[str]:
    """Reciprocal Rank Fusion: merge several ranked `tip_id` lists into one
    top-k. Each list contributes `1 / (rrf_k + rank)` (1-based rank) to each
    tip's score; a tip absent from a list contributes nothing. Return the top-k
    tip_ids by fused score, best first.

    Rank-only by design -- it needs no comparable relevance scores, which is why
    it fuses minsearch's keyword and vector rankings (they expose ranks, not
    scores). A tip ranked decently in BOTH lists beats one ranked #1 in only one:
    RRF rewards agreement. `rrf_k=60` is the standard damping constant. Ties keep
    first-seen order (stable sort over insertion order), so the fusion is
    deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, tip_id in enumerate(ranking, start=1):
            scores[tip_id] = scores.get(tip_id, 0.0) + 1.0 / (rrf_k + rank)
    ranked = sorted(scores, key=lambda tip_id: scores[tip_id], reverse=True)
    return ranked[:k]


def hybrid_retriever(
    retrievers: list[Retriever], k: int = DEFAULT_K, rrf_k: int = 60,
) -> Retriever:
    """Fuse several `Retriever`s into one via `rrf_fuse` -- the true-hybrid
    approach (keyword + vector) that won the Part C comparison.

    Composes the existing adapters: pass e.g. `[keyword_retriever(...,  k=20),
    vector_retriever(..., k=20)]`. Build the sub-retrievers with a DEEPER k than
    the final `k` so fusion ranges over a wider candidate pool before truncating.
    Returns a `Retriever` the scoring loop can't tell apart from any other.
    """
    def retrieve(question: str) -> list[str]:
        rankings = [retrieve_one(question) for retrieve_one in retrievers]
        return rrf_fuse(rankings, k=k, rrf_k=rrf_k)

    return retrieve


# --- serve boundary: rehydrate ranked ids -> full docs -----------------------

def docs_retriever(
    retriever: Retriever, by_id: dict[str, dict],
) -> Callable[[str], list[dict]]:
    """Wrap an id-based `Retriever` to return full doc dicts instead of tip_ids.

    The eval scores on tip_ids; the app grounds on docs (`RAGBase.build_context`
    needs `tip`/`rationale`/`creator_id`/`source_url`). Same ranking either way --
    this just maps each ranked id back to its doc via `by_id`, preserving order.
    A tip_id missing from `by_id` is skipped rather than crashing the answer.
    """
    def retrieve(question: str) -> list[dict]:
        return [by_id[tip_id] for tip_id in retriever(question) if tip_id in by_id]

    return retrieve


def build_hybrid_docs_retriever(
    corpus: list[dict], k: int = DEFAULT_K, pool: int = 20,
) -> Callable[[str], list[dict]]:
    """The app's retriever: build both indexes from `corpus`, fuse keyword+vector
    by RRF, and return full docs (rehydrated) for grounding.

    The ranking is the SAME hybrid the P0-5 eval scored (`hybrid > vector`); only
    the output is rehydrated to docs. `pool` is the per-arm candidate depth fed to
    RRF (deeper than the final `k` so fusion has room to reward agreement)."""
    text_index = build_text_index(corpus)
    vector_index = build_vector_index(corpus)
    hybrid = hybrid_retriever(
        [keyword_retriever(text_index, k=pool), vector_retriever(vector_index, k=pool)],
        k=k,
    )
    by_id = {doc["tip_id"]: doc for doc in corpus}
    return docs_retriever(hybrid, by_id)
