"""P0-5 Part B -- the retrieval eval harness (keyword-first vertical slice).

The measurement rig for the graded retrieval comparison. It does NOT retrieve or
generate -- it SCORES. Given a `Retriever` (a question -> ranked `tip_id`s) and
the ground-truth `{question, seed_tip_id}` rows, it builds a per-question
relevance matrix, rolls it up to **hit-rate + MRR**, and dumps the misses that
feed the single-seed-vs-set-based decision.

**Approach-agnostic by design.** `Retriever` is any `str -> list[tip_id]`
callable, so the SAME harness scores keyword-only first (this slice) and
vector/hybrid later -- you just pass a different retriever. Keyword-first is the
cheapest, most deterministic start: `keyword_retriever` wraps ingest's
`minsearch.Index` with ZERO embedding model, so the whole loop (ground truth <->
retrieval <-> scoring) is proven before the vector dependency is taken on
(keyword-before-vector build order).

Spec: tests/test_eval_retrieval.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from carryia.paths import DATA
from carryia.pipeline.ingest import (  # reuse stage ⑤'s indexes -- no LLM, local embeddings
    build_text_index,
    build_vector_index,
    embed_tips,
    load_corpus,
)
from carryia.serve.retrieval import (  # the retrieval layer moved to serve; re-exported here
    DEFAULT_K,
    Retriever,
    hybrid_retriever,
    keyword_retriever,
    rrf_fuse,
    vector_retriever,
)

GROUND_TRUTH_PATH = DATA / "ground_truth.jsonl"


class EvalError(RuntimeError):
    """The ground truth could not be loaded or parsed."""


# --- load the ground truth ---------------------------------------------------

def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> list[dict]:
    """Read `ground_truth.jsonl` into `{question, seed_tip_id}` dicts (one per
    line). Mirrors `ingest.load_corpus`: deserialise only -- no validation here
    -- and fail loud on a malformed line so a corrupt answer key halts the eval
    instead of silently scoring against nothing.
    """
    rows: list[dict] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise EvalError(f"{path.name}:{lineno}: bad JSON -- {exc}") from exc
    if not rows:
        raise EvalError(f"{path.name}: no rows")
    return rows


# --- the retrievers (`keyword_retriever`, `vector_retriever`, `rrf_fuse`,
# `hybrid_retriever`) now live in `carryia.serve.retrieval` and are imported above
# so the app ships the exact code this harness scores. They are re-exported here
# (`eval_retrieval.keyword_retriever`, ...) so the specs below and their tests
# reference them unchanged.


# --- widen the answer key: set-based gold ------------------------------------

def build_gold_sets(
    documents: list[dict],
    embed: Callable[[list[str]], "np.ndarray"] = embed_tips,
    threshold: float = 0.87,
) -> dict[str, set[str]]:
    """Build the set-based answer key: map each `tip_id` to the set of `tip_id`s
    that count as an equally-correct answer for a question seeded from it.

    The corpus is dense with near-duplicates by design (no-cull; warding alone ->
    132 tips), so single-seed gold under-credits retrieval -- it demands the one
    arbitrary seeded id when a near-dupe is just as correct. This widens the key:
    each seed's gold set is the seed itself plus every other tip whose `tip`
    embeds within `threshold` cosine of it.

    Definition -- per-seed neighbourhood, NOT transitive clusters: a tip joins a
    seed's set iff cosine(seed, tip) >= threshold, decided per seed. Do NOT union
    these into connected components -- A~B and B~C would drag A and C into one
    blob, loosening the sets until every approach saturates to ~1.0 and the
    comparison goes flat (the benchmark-saturation trap). Tighter is safer; raise
    `threshold` if the sets come out large.

    `embed` is the SAME local fastembed model the vector index uses (default
    `embed_tips`, no API key), so document and gold vectors share one space; it's
    injectable so tests pass a deterministic fake. Suggested body: embed each
    `doc["tip"]`, L2-normalise, take the pairwise cosine matrix (709x709 is
    cheap), and threshold each row.

    Returns `{tip_id -> set of tip_ids}`, one entry per document, each set
    containing at least the tip itself (reflexive).
    """
    tip_ids = [doc["tip_id"] for doc in documents]
    vectors = np.asarray(embed([doc["tip"] for doc in documents]), dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms == 0.0, 1.0, norms)  # L2-normalise; guard zeros
    sims = unit @ unit.T  # pairwise cosine -- 709x709 is cheap
    gold: dict[str, set[str]] = {}
    for i, tip_id in enumerate(tip_ids):
        members = {tip_ids[j] for j in np.nonzero(sims[i] >= threshold)[0]}
        members.add(tip_id)  # reflexive even if float error nudges self-cos below 1
        gold[tip_id] = members
    return gold


# --- score an approach against the ground truth ------------------------------

def relevance_matrix(
    retriever: Retriever,
    ground_truth: list[dict],
    gold: dict[str, set[str]] | None = None,
) -> list[list[bool]]:
    """For each ground-truth row, run `retriever(question)` and return a boolean
    row: True at each rank holding a *correct* tip, else False.

    `gold` selects the answer key -- the flip this function was built to absorb:
      - `None` -> single-seed gold: the row's `seed_tip_id` is the sole correct
        answer (the default, unchanged behaviour).
      - a `{seed_tip_id -> set of acceptable tip_ids}` map (from
        `build_gold_sets`) -> set-based gold: any tip in the seed's near-duplicate
        set counts as a hit.

    One row per question; a row's length is however many results the retriever
    returned. `hit_rate`/`mrr` consume this matrix unchanged -- they don't care
    how a True was decided, which is exactly what lets the answer key widen here
    without touching the metrics.
    """
    matrix: list[list[bool]] = []
    for row in ground_truth:
        if gold is None:
            gold_set = {row["seed_tip_id"]}
        else:
            gold_set = gold[row["seed_tip_id"]]
        ranked = retriever(row["question"])
        matrix.append([tip_id in gold_set for tip_id in ranked])
    return matrix


def hit_rate(relevance: list[list[bool]]) -> float:
    """Fraction of questions with the gold anywhere in top-k (any True in the
    row). The share of questions where retrieval put a correct answer in front
    of the user at all.
    """
    if not relevance:
        return 0.0
    return sum(1 for row in relevance if any(row)) / len(relevance)


def mrr(relevance: list[list[bool]]) -> float:
    """Mean reciprocal rank: for each question take 1/(1-based rank of the first
    True), 0 if the row has no True, then average over all questions. Rewards
    putting the correct tip HIGH, not just somewhere in top-k.
    """
    if not relevance:
        return 0.0
    total = 0.0
    for row in relevance:
        for rank, hit in enumerate(row, start=1):
            if hit:
                total += 1.0 / rank
                break
    return total / len(relevance)


def find_misses(
    retriever: Retriever, ground_truth: list[dict],
) -> list[dict]:
    """The ground-truth rows whose `seed_tip_id` was NOT in the retriever's
    top-k. This is the inspect-loop input: eyeball
    these to decide whether the misses are near-dupes of the seed (=> dupe
    artifact => escalate to set-based gold) or genuinely irrelevant tips (=>
    retrieval is actually bad => fix retrieval, not the gold).
    """
    return [
        row for row in ground_truth
        if row["seed_tip_id"] not in retriever(row["question"])
    ]


# --- CLI: score the keyword approach, print the metric row -------------------

def main(argv: list[str] | None = None) -> int:
    """Load corpus + ground truth, build the keyword index, score it, and print
    the metric row -- the keyword vertical slice end-to-end. The graded table
    grows a row per approach as vector/hybrid land. Returns a process exit code.
    """
    try:
        ground_truth = load_ground_truth()
    except EvalError as exc:
        print(f"eval failed: {exc}", file=sys.stderr)
        return 1
    documents = load_corpus()
    approaches: dict[str, Retriever] = {
        "keyword": keyword_retriever(build_text_index(documents)),
        "vector": vector_retriever(build_vector_index(documents)),
    }
    print(f"{'approach':<12}{'hit-rate':>10}{'MRR':>8}")
    for name, retriever in approaches.items():
        relevance = relevance_matrix(retriever, ground_truth)
        print(f"{name:<12}{hit_rate(relevance):>10.3f}{mrr(relevance):>8.3f}")
    print(f"(scored {len(ground_truth)} questions @ k={DEFAULT_K})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
