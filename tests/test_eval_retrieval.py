"""Spec for carryia/eval/eval_retrieval.py -- P0-5 Part B, the retrieval eval harness.

TDD state on first run (the house three-state, as in test_ingest.py):
  - GREEN: `load_ground_truth` I/O + `main()`'s plumbing (the scorers
    monkeypatched out) are implemented, so these pass now and lock the wiring.
  - RED: `keyword_retriever`, `relevance_matrix`, `hit_rate`, `mrr`, and
    `find_misses` are NotImplementedError stubs, so their contract tests fail
    until you write each body. That failing test IS the spec you code to.
  - (No SKIPPED: the keyword slice needs no model download -- that is the whole
    point of going keyword-first.)

The metric tests (`hit_rate`, `mrr`) run on hand-built relevance matrices -- pure,
no index. `relevance_matrix`/`find_misses` run on a fake dict retriever -- also no
index. Only the `keyword_retriever` and end-to-end tests touch real minsearch,
over the 3-record `corpus.sample.jsonl` fixture (same fixture ingest scores).
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from carryia.eval import eval_retrieval
from carryia.pipeline import ingest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures" / "corpus.sample.jsonl"


@pytest.fixture
def docs():
    return ingest.load_corpus(SAMPLE)


def _fixed_retriever(rankings: dict):
    """A fake `Retriever`: maps a question string to a preset list of tip_ids,
    so the scoring/miss logic is tested without any index."""
    return lambda question: rankings[question]


def _fake_embed(texts):
    """Deterministic content-derived vectors -- no model download (mirrors
    test_ingest._fake_embed). Identical text embeds identically, so an exact-tip
    query lands on its own document at cosine 1.0."""
    def vec(text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()[:16]
        return np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    return np.array([vec(t) for t in texts], dtype=np.float32)


def _vec_embed(mapping):
    """A fake embedder that maps each tip text to a preset vector, so a test
    controls the exact cosine between tips (unlike `_fake_embed`'s hash noise).
    Order-preserving, so it works for any subset the code embeds."""
    return lambda texts: np.array([mapping[t] for t in texts], dtype=np.float32)


# --- GREEN: load_ground_truth ------------------------------------------------

def test_load_ground_truth_reads_every_row(tmp_path):
    p = tmp_path / "gt.jsonl"
    p.write_text(
        '{"question": "why do i keep dying on recall?", "seed_tip_id": "tip-a"}\n'
        '{"question": "when do i give the wave?", "seed_tip_id": "tip-b"}\n'
    )
    rows = eval_retrieval.load_ground_truth(p)
    assert len(rows) == 2
    assert rows[0]["seed_tip_id"] == "tip-a"
    assert rows[1]["question"] == "when do i give the wave?"


def test_load_ground_truth_skips_blank_lines(tmp_path):
    p = tmp_path / "gt.jsonl"
    p.write_text('{"question": "q", "seed_tip_id": "tip-a"}\n\n\n')
    assert len(eval_retrieval.load_ground_truth(p)) == 1


def test_load_ground_truth_rejects_bad_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"question": "q", "seed_tip_id": "tip-a"}\nnot json\n')
    with pytest.raises(eval_retrieval.EvalError):
        eval_retrieval.load_ground_truth(p)


def test_load_ground_truth_rejects_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n")
    with pytest.raises(eval_retrieval.EvalError):
        eval_retrieval.load_ground_truth(p)


# --- GREEN: main() plumbing (scorers stubbed) --------------------------------

def test_main_prints_metric_table_and_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(eval_retrieval, "load_ground_truth",
                        lambda: [{"question": "q", "seed_tip_id": "g"}])
    monkeypatch.setattr(eval_retrieval, "load_corpus",
                        lambda: [{"tip": "t", "tip_id": "g"}])
    monkeypatch.setattr(eval_retrieval, "build_text_index", lambda d: "INDEX")
    monkeypatch.setattr(eval_retrieval, "build_vector_index", lambda d: "VINDEX")
    monkeypatch.setattr(eval_retrieval, "keyword_retriever",
                        lambda index: (lambda q: ["g"]))
    monkeypatch.setattr(eval_retrieval, "vector_retriever",
                        lambda index: (lambda q: ["g"]))
    monkeypatch.setattr(eval_retrieval, "relevance_matrix",
                        lambda retriever, gt: [[True]])
    monkeypatch.setattr(eval_retrieval, "hit_rate", lambda rel: 1.0)
    monkeypatch.setattr(eval_retrieval, "mrr", lambda rel: 1.0)
    assert eval_retrieval.main() == 0
    out = capsys.readouterr().out
    assert "keyword" in out
    assert "vector" in out
    assert "1.000" in out


def test_main_reports_failure_on_bad_ground_truth(monkeypatch, capsys):
    def boom():
        raise eval_retrieval.EvalError("nope")
    monkeypatch.setattr(eval_retrieval, "load_ground_truth", boom)
    assert eval_retrieval.main() == 1
    assert "eval failed" in capsys.readouterr().err


# --- RED: hit_rate + mrr (pure metrics over hand-built matrices) --------------

def test_hit_rate_is_fraction_of_rows_with_a_hit():
    rel = [[False, True, False], [False, False, False], [True]]
    assert eval_retrieval.hit_rate(rel) == pytest.approx(2 / 3)


def test_hit_rate_is_zero_when_nothing_hits():
    assert eval_retrieval.hit_rate([[False, False], [False]]) == 0.0


def test_mrr_averages_reciprocal_rank_of_first_hit():
    # first-hit ranks: 1, 2, none -> (1/1 + 1/2 + 0) / 3
    rel = [[True], [False, True], [False, False, False]]
    assert eval_retrieval.mrr(rel) == pytest.approx((1 + 0.5 + 0) / 3)


def test_mrr_credits_only_the_first_hit_in_a_row():
    # two Trues in one row -> counts the earlier rank (1/2), not both
    assert eval_retrieval.mrr([[False, True, True]]) == pytest.approx(0.5)


# --- RED: relevance_matrix (fake retriever, no index) -------------------------

def test_relevance_matrix_marks_the_seed_rank():
    retrieve = _fixed_retriever({
        "q1": ["x", "gold", "y"],   # seed at rank 2
        "q2": ["a", "b"],           # seed absent
    })
    gt = [
        {"question": "q1", "seed_tip_id": "gold"},
        {"question": "q2", "seed_tip_id": "gold"},
    ]
    assert eval_retrieval.relevance_matrix(retrieve, gt) == [
        [False, True, False],
        [False, False],
    ]


# --- RED: find_misses (fake retriever, no index) ------------------------------

def test_find_misses_returns_only_unhit_rows():
    retrieve = _fixed_retriever({
        "hit": ["gold", "x"],
        "miss": ["x", "y"],
    })
    gt = [
        {"question": "hit", "seed_tip_id": "gold"},
        {"question": "miss", "seed_tip_id": "gold"},
    ]
    misses = eval_retrieval.find_misses(retrieve, gt)
    assert [m["question"] for m in misses] == ["miss"]


# --- RED: keyword_retriever (real minsearch over the sample corpus) -----------

def test_keyword_retriever_ranks_tip_ids_best_first(docs):
    retrieve = eval_retrieval.keyword_retriever(ingest.build_text_index(docs), k=3)
    ranked = retrieve("control ward in the river brush before I recall")
    assert ranked[0] == "tip-796fa522de2d"
    assert all(isinstance(t, str) for t in ranked)


def test_keyword_retriever_respects_k(docs):
    retrieve = eval_retrieval.keyword_retriever(ingest.build_text_index(docs), k=2)
    assert len(retrieve("ward")) <= 2


# --- vector_retriever (real minsearch VectorSearch, fake embedder) ------------

def test_vector_retriever_finds_the_matching_tip(docs):
    index = ingest.build_vector_index(docs, embed=_fake_embed)
    retrieve = eval_retrieval.vector_retriever(index, embed=_fake_embed, k=1)
    ranked = retrieve(docs[2]["tip"])   # query = a tip's exact text -> cosine 1.0
    assert ranked[0] == docs[2]["tip_id"]


def test_vector_retriever_embeds_the_query(docs):
    index = ingest.build_vector_index(docs, embed=_fake_embed)
    seen = {}

    def spy(texts):
        seen["texts"] = list(texts)
        return _fake_embed(texts)

    eval_retrieval.vector_retriever(index, embed=spy, k=2)("some question")
    assert seen["texts"] == ["some question"]


# --- end-to-end keyword slice (retriever -> matrix -> metrics) ----------------

def test_end_to_end_keyword_slice(docs):
    retrieve = eval_retrieval.keyword_retriever(ingest.build_text_index(docs), k=3)
    gt = [
        {"question": "control ward in the river brush before I recall",
         "seed_tip_id": "tip-796fa522de2d"},
        {"question": "Nautilus dredge line on cooldown, should I walk up to trade",
         "seed_tip_id": "tip-3b1d4c999cf8"},
    ]
    rel = eval_retrieval.relevance_matrix(retrieve, gt)
    assert eval_retrieval.hit_rate(rel) == 1.0
    assert eval_retrieval.mrr(rel) > 0.0
    assert eval_retrieval.find_misses(retrieve, gt) == []


# --- RED: build_gold_sets (fake embedder, controlled cosine) ------------------

@pytest.fixture
def near_dupe_docs():
    """Three tips with hand-chosen unit vectors: a1 & a2 are near-dupes
    (cosine 0.99), b is orthogonal to both (cosine 0). Returns (docs, embed)."""
    docs = [
        {"tip": "ward the river bush before you recall", "tip_id": "a1"},
        {"tip": "place a ward in river brush before recalling", "tip_id": "a2"},
        {"tip": "take Warding Totem as your start trinket", "tip_id": "b"},
    ]
    embed = _vec_embed({
        docs[0]["tip"]: [1.0, 0.0],
        docs[1]["tip"]: [0.99, float(np.sqrt(1 - 0.99 ** 2))],  # ~0.99 cos to a1
        docs[2]["tip"]: [0.0, 1.0],                             # orthogonal to a1
    })
    return docs, embed


def test_build_gold_sets_groups_near_dupes_and_is_reflexive(near_dupe_docs):
    docs, embed = near_dupe_docs
    gold = eval_retrieval.build_gold_sets(docs, embed=embed, threshold=0.9)
    assert "a1" in gold["a1"]           # reflexive: a tip is in its own set
    assert gold["a1"] == {"a1", "a2"}   # the 0.99 near-dupe is grouped in
    assert gold["b"] == {"b"}           # the orthogonal tip stays alone


def test_build_gold_sets_threshold_tightens(near_dupe_docs):
    docs, embed = near_dupe_docs
    # raise the bar above 0.99 -> a1 no longer accepts a2; sets shrink to singletons
    gold = eval_retrieval.build_gold_sets(docs, embed=embed, threshold=0.995)
    assert gold["a1"] == {"a1"}
    assert gold["a2"] == {"a2"}


# --- RED: relevance_matrix with set-based gold --------------------------------

def test_relevance_matrix_accepts_any_gold_set_member():
    # the seed itself is NOT retrieved, but a sibling in its gold set is -> hit
    retrieve = _fixed_retriever({"q": ["x", "sibling", "y"]})
    gt = [{"question": "q", "seed_tip_id": "seed"}]
    gold = {"seed": {"seed", "sibling"}}
    assert eval_retrieval.relevance_matrix(retrieve, gt, gold=gold) == [
        [False, True, False],
    ]


# --- rrf_fuse + hybrid_retriever (rank-only fusion, no index) ------------------

def test_rrf_fuse_rewards_agreement_across_lists():
    # A is #1 in one list only; B is #2 and #1 -> B outranks A (rewards agreement)
    kw = ["A", "B", "C"]   # C appears only here
    vec = ["B", "D", "A"]  # D appears only here
    assert eval_retrieval.rrf_fuse([kw, vec], k=4, rrf_k=60) == ["B", "A", "D", "C"]


def test_rrf_fuse_truncates_to_k():
    kw = ["A", "B", "C"]
    vec = ["B", "D", "A"]
    assert eval_retrieval.rrf_fuse([kw, vec], k=2, rrf_k=60) == ["B", "A"]


def test_hybrid_retriever_fuses_its_sub_retrievers():
    retr1 = _fixed_retriever({"q": ["A", "B", "C"]})
    retr2 = _fixed_retriever({"q": ["B", "D", "A"]})
    hybrid = eval_retrieval.hybrid_retriever([retr1, retr2], k=4)
    assert hybrid("q") == ["B", "A", "D", "C"]
