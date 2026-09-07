"""Spec for carryia/serve/monitoring.py -- the P0-8 instrumented answer path.

GREEN: RAGWithMetrics runs on fakes -- a fake retriever (no index/embed), a fake
Anthropic client that returns text + usage (no key), an injected judge (no 2nd model
call), and a recorder store (no DB). We pin that one ask produces one telemetry row
with the right fields, that timing covers only the answer call, and -- the load-bearing
contract -- that a monitoring failure never costs the player their answer.
"""

from __future__ import annotations

import pytest

from carryia.serve import monitoring
from carryia.serve.db import ConversationRecord
from carryia.serve.monitoring import RAGWithMetrics, RelevanceVerdict, make_relevance_judge


# --- fakes -------------------------------------------------------------------

class _Usage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _AnswerMessages:
    """messages.create -> a response with .content[0].text and .usage (the answer call)."""

    def __init__(self, reply, usage):
        self.reply, self.usage = reply, usage
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = type("Block", (), {"text": self.reply})()
        return type("Resp", (), {"content": [block], "usage": self.usage})()


class _AnswerClient:
    def __init__(self, reply="Ward earlier. (mobalytics)", usage=None):
        self.messages = _AnswerMessages(reply, usage or _Usage(1000, 100))


class _ParseMessages:
    """messages.parse -> a response with .parsed_output and .usage (the judge call)."""

    def __init__(self, verdict, usage):
        self.verdict, self.usage = verdict, usage
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return type("Resp", (), {"parsed_output": self.verdict, "usage": self.usage})()


class _ParseClient:
    def __init__(self, verdict, usage=None):
        self.messages = _ParseMessages(verdict, usage or _Usage(300, 20))


def _docs():
    return [{"tip": "Ward the river bush.", "rationale": "Denies dives.",
             "creator_id": "mobalytics", "source_url": "https://mobalytics.gg/a"}]


def _fake_retriever(query):
    return _docs()


def _ok_judge(usage=None):
    """An injected judge that returns a fixed verdict + usage, calling no model."""
    calls = []

    def judge(question, answer):
        calls.append((question, answer))
        return RelevanceVerdict(explanation="answers it", relevance="RELEVANT"), \
            (usage or _Usage(300, 20))

    judge.calls = calls
    return judge


# --- cost ---------------------------------------------------------------------

def test_cost_usd_prices_input_at_1_and_output_at_5_per_million():
    assert monitoring.cost_usd(_Usage(1_000_000, 1_000_000)) == pytest.approx(6.0)


def test_cost_usd_matches_the_eval_calc_price():
    # serve/ defines pricing locally (no import from eval/); pin the two together so
    # they can't drift.
    from carryia.eval.evaluation_utils import calc_price

    usage = _Usage(1234, 567)
    assert monitoring.cost_usd(usage) == pytest.approx(calc_price(usage)["total_cost"])


# --- the relevance judge ------------------------------------------------------

def test_make_relevance_judge_formats_prompt_and_returns_verdict_and_usage():
    client = _ParseClient(RelevanceVerdict(explanation="ok", relevance="RELEVANT"),
                          _Usage(42, 7))
    judge = make_relevance_judge(client, model="claude-haiku-4-5")
    verdict, usage = judge("why do i lose lane", "ward earlier")

    assert verdict.relevance == "RELEVANT"
    assert (usage.input_tokens, usage.output_tokens) == (42, 7)
    sent = client.messages.calls[0]
    assert sent["output_format"] is RelevanceVerdict
    content = sent["messages"][0]["content"]
    assert "why do i lose lane" in content and "ward earlier" in content


def test_relevance_judge_coerces_an_unknown_label():
    client = _ParseClient(RelevanceVerdict(explanation="?", relevance="TOTALLY_RAD"))
    verdict, _ = make_relevance_judge(client, model="m")("q", "a")
    assert verdict.relevance == "PARTLY_RELEVANT"   # never a stray label into a NOT-NULL col


# --- RAGWithMetrics: one ask -> one recorded row ------------------------------

def test_rag_records_and_stores_one_conversation(monkeypatch):
    monkeypatch.setenv("CARRYIA_LLM_BACKEND", "anthropic")
    stored = []
    judge = _ok_judge(usage=_Usage(300, 20))
    rag = RAGWithMetrics(
        index=None, llm_client=_AnswerClient(reply="Ward earlier. (mobalytics)",
                                             usage=_Usage(1000, 100)),
        retriever=_fake_retriever, model="claude-haiku-4-5",
        store=stored.append, judge=judge,
    )
    out = rag.rag("why do i keep losing lane", personal_context="KP weak")

    assert out == "Ward earlier. (mobalytics)"
    assert len(stored) == 1
    rec = stored[0]
    assert isinstance(rec, ConversationRecord)
    assert rec.question == "why do i keep losing lane"
    assert rec.answer == "Ward earlier. (mobalytics)"
    assert rec.model == "claude-haiku-4-5"
    assert rec.backend == "anthropic"
    assert (rec.prompt_tokens, rec.completion_tokens, rec.total_tokens) == (1000, 100, 1100)
    assert rec.cost_usd == pytest.approx(1000 / 1e6 * 1 + 100 / 1e6 * 5)
    assert rec.relevance == "RELEVANT"
    assert (rec.eval_prompt_tokens, rec.eval_total_tokens) == (300, 320)
    assert rec.response_time_ms >= 0.0
    assert rag.last_conversation_id == rec.id
    assert judge.calls == [("why do i keep losing lane", "Ward earlier. (mobalytics)")]


def test_rag_without_a_store_still_answers_and_builds_the_record():
    rag = RAGWithMetrics(index=None, llm_client=_AnswerClient(), retriever=_fake_retriever,
                         model="m", store=None, judge=_ok_judge())
    out = rag.rag("q")
    assert out  # answered
    assert rag.last_record is not None and rag.last_conversation_id is not None


# --- best-effort: monitoring must not break the answer ------------------------

def test_a_judge_failure_does_not_cost_the_answer():
    def boom_judge(question, answer):
        raise RuntimeError("judge rate-limited")

    stored = []
    rag = RAGWithMetrics(index=None, llm_client=_AnswerClient(reply="here is your answer"),
                         retriever=_fake_retriever, model="m",
                         store=stored.append, judge=boom_judge)
    out = rag.rag("q")

    assert out == "here is your answer"     # answer survives
    assert stored == []                     # nothing persisted
    assert rag.last_conversation_id is None  # no row id to attach feedback to


def test_a_store_failure_does_not_cost_the_answer():
    def boom_store(rec):
        raise RuntimeError("db down")

    rag = RAGWithMetrics(index=None, llm_client=_AnswerClient(reply="answer text"),
                         retriever=_fake_retriever, model="m",
                         store=boom_store, judge=_ok_judge())
    assert rag.rag("q") == "answer text"
    assert rag.last_conversation_id is None
