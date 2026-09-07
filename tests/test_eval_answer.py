"""Spec for carryia/eval/eval_answer.py -- P0-6 pairwise answer eval.

TDD state (the house three-state, as in test_eval_retrieval.py):
  - GREEN: every function is implemented, so these lock the wiring. The judge core
    (`compare`, `make_judge`) runs on fake judges / a mocked `llm_structured_retry`
    -- no key, no network. `load_held_set` / `tally` / `run_pairwise` / `main` are
    pure or fully monkeypatched.
  - SKIPPED: `test_live_smoke` runs the real pipeline (generate both variants + judge)
    over the sample corpus. Off by default; set CARRYIA_LIVE=1 to run it.

The load-bearing test is `compare`: it must judge both orders and collapse pure
position bias to a tie, or the whole eval is just measuring which answer came first.
"""

import os
from pathlib import Path

import pytest

from carryia.eval import eval_answer
from carryia.eval.eval_answer import PairwiseVerdict

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures" / "corpus.sample.jsonl"


# --- GREEN: load_held_set (deterministic sampling) ---------------------------

def test_load_held_set_is_deterministic_for_a_seed(monkeypatch):
    rows = [{"question": f"q{i}", "seed_tip_id": f"t{i}"} for i in range(100)]
    monkeypatch.setattr(eval_answer, "load_ground_truth", lambda: rows)
    a = eval_answer.load_held_set(10, seed=0)
    b = eval_answer.load_held_set(10, seed=0)
    assert a == b                       # same seed -> same sample
    assert len(a) == 10
    assert all(r in rows for r in a)    # a real subset


def test_load_held_set_returns_all_when_n_exceeds_corpus(monkeypatch):
    rows = [{"question": "q1"}, {"question": "q2"}, {"question": "q3"}]
    monkeypatch.setattr(eval_answer, "load_ground_truth", lambda: rows)
    assert eval_answer.load_held_set(10) == rows


# --- GREEN: tally ------------------------------------------------------------

def test_tally_counts_wins_and_ties():
    assert eval_answer.tally(["BLUNT", "WHY", "WHY", "tie"]) == {
        "BLUNT": 1, "WHY": 2, "tie": 1}


def test_tally_of_nothing_is_empty():
    assert eval_answer.tally([]) == {}


# --- GREEN: compare (the debias core -- fake judges, no API) ------------------

def test_compare_credits_a_consistent_winner():
    # order-insensitive judge that genuinely prefers the text "GOOD"
    pref = lambda q, a, b: "A" if "GOOD" in a else "B"
    assert eval_answer.compare(pref, "q", "GOOD", "meh") == "A"
    assert eval_answer.compare(pref, "q", "meh", "GOOD") == "B"


def test_compare_judges_both_orders():
    calls = []
    def spy(q, a, b):
        calls.append((a, b))
        return "A"
    eval_answer.compare(spy, "q", "AA", "BB")
    assert calls == [("AA", "BB"), ("BB", "AA")]


def test_compare_collapses_pure_position_bias_to_tie():
    # a judge that always favours whatever is shown first -> no real winner
    always_first = lambda q, a, b: "A"
    assert eval_answer.compare(always_first, "q", "x", "y") == "tie"


def test_compare_returns_tie_when_the_judge_ties():
    always_tie = lambda q, a, b: "tie"
    assert eval_answer.compare(always_tie, "q", "x", "y") == "tie"


# --- GREEN: make_judge (contract over a mocked llm_structured_retry) ----------

def test_make_judge_formats_the_prompt_and_returns_the_winner(monkeypatch):
    seen = {}
    def fake_lsr(client, instructions, prompt, output_type, model):
        seen.update(client=client, instructions=instructions, prompt=prompt,
                    output_type=output_type, model=model)
        return PairwiseVerdict(reasoning="because", winner="B"), None
    monkeypatch.setattr(eval_answer, "llm_structured_retry", fake_lsr)

    judge = eval_answer.make_judge("CLIENT", instructions="RULES", model="m")
    out = judge("why do i die?", "answer A text", "answer B text")

    assert out == "B"
    assert seen["client"] == "CLIENT"
    assert seen["instructions"] == "RULES"
    assert seen["model"] == "m"
    assert seen["output_type"] is PairwiseVerdict
    for fragment in ("why do i die?", "answer A text", "answer B text"):
        assert fragment in seen["prompt"]


# --- GREEN: run_pairwise (wiring -- compare monkeypatched) --------------------

def test_run_pairwise_maps_outcomes_to_labels(monkeypatch):
    monkeypatch.setattr(eval_answer, "compare",
                        lambda judge, q, a, b: {"q1": "A", "q2": "B", "q3": "tie"}[q])
    held = [{"question": "q1"}, {"question": "q2"}, {"question": "q3"}]
    winners = eval_answer.run_pairwise(
        held, gen_a=lambda q: "a", gen_b=lambda q: "b",
        judge=None, label_a="BLUNT", label_b="WHY")
    assert winners == ["BLUNT", "WHY", "tie"]


def test_run_pairwise_generates_both_answers_per_question(monkeypatch):
    monkeypatch.setattr(eval_answer, "compare", lambda judge, q, a, b: "A")
    seen_a, seen_b = [], []
    eval_answer.run_pairwise(
        [{"question": "q1"}, {"question": "q2"}],
        gen_a=lambda q: seen_a.append(q) or "a",
        gen_b=lambda q: seen_b.append(q) or "b",
        judge=None, label_a="BLUNT", label_b="WHY")
    assert seen_a == ["q1", "q2"]
    assert seen_b == ["q1", "q2"]


# --- GREEN: main() plumbing (everything live monkeypatched) -------------------

def test_main_prints_the_tally_and_exits_zero(monkeypatch, capsys):
    from carryia.pipeline import ingest
    from carryia.serve import llm_backend
    from carryia.serve import rag_helper

    monkeypatch.setattr(eval_answer, "load_held_set", lambda: [{"question": "q"}])
    monkeypatch.setattr(ingest, "load_corpus", lambda: [{"tip": "t"}])
    monkeypatch.setattr(ingest, "build_vector_index", lambda docs: "INDEX")
    monkeypatch.setattr(llm_backend, "make_client", lambda: "CLIENT")
    monkeypatch.setattr(llm_backend, "model_id", lambda: "MODEL")
    monkeypatch.setattr(rag_helper, "RAGBase",
                        lambda **kw: type("C", (), {"rag": lambda self, q: "ans"})())
    monkeypatch.setattr(eval_answer, "make_judge",
                        lambda client, model=None: (lambda q, a, b: "A"))
    monkeypatch.setattr(eval_answer, "run_pairwise",
                        lambda held, ga, gb, judge, la, lb: ["BLUNT"])
    monkeypatch.setattr(eval_answer, "tally", lambda w: {"BLUNT": 1})

    assert eval_answer.main() == 0
    out = capsys.readouterr().out
    assert "BLUNT" in out and "WHY" in out and "tie" in out


# --- SKIPPED: live smoke (opt in: CARRYIA_LIVE=1) ----------------------------

@pytest.mark.skipif(not os.getenv("CARRYIA_LIVE"),
                    reason="live API smoke; set CARRYIA_LIVE=1 (needs a key)")
def test_live_smoke():
    from carryia.pipeline import ingest
    from carryia.serve.llm_backend import make_client, model_id
    from carryia.serve.rag_helper import COACH_BLUNT, COACH_WHY, RAGBase

    docs = ingest.load_corpus(SAMPLE)
    index = ingest.build_vector_index(docs)
    client = make_client()    # CARRYIA_LLM_BACKEND selects anthropic vs bedrock
    model = model_id()
    blunt = RAGBase(index=index, llm_client=client, instructions=COACH_BLUNT, model=model)
    why = RAGBase(index=index, llm_client=client, instructions=COACH_WHY, model=model)
    judge = eval_answer.make_judge(client, model=model)

    q = "i keep getting caught out with no vision, what should i do"
    winner = eval_answer.compare(judge, q, blunt.rag(q), why.rag(q))
    assert winner in ("A", "B", "tie")
