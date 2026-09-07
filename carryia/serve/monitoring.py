"""P0-8 monitoring -- the instrumented answer path.

`RAGWithMetrics(RAGBase)` wraps a normal ask with the telemetry the Grafana
dashboard charts: it times the answer call, reads token usage,
derives cost, runs a reference-free relevance judge on the answer, and persists one
`ConversationRecord`. The app swaps `RAGBase` -> `RAGWithMetrics` when Postgres is
reachable; nothing else in the answer path changes.

Two things make this cost a **second LLM call per ask** (answer + judge), exactly as
the course's monitored RAG does. The judge is reference-free -- it rates the answer
against the question alone (no gold answer exists for coaching), distinct from P0-6's
*pairwise* judge, which compares two answers.

Seams for testing without a key or a DB:
  - `store`  = callable(ConversationRecord) -> None. The app passes `db.save_conversation`
               bound to its connection; a test passes a recorder. None = don't persist.
  - `judge`  = callable(question, answer) -> (RelevanceVerdict, usage). Built from the
               client by default; a test injects a fake so no model is called.
Self-contained on purpose: it imports nothing from `carryia.eval`, which the runtime
image does not ship.
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from carryia.serve.db import ConversationRecord
from carryia.serve.llm_backend import backend
from carryia.serve.rag_helper import RAGBase

# Haiku 4.5 pricing, per million tokens -- matches eval/evaluation_utils.calc_price
# (kept here so serve/ carries no dependency on eval/). A test pins the two together.
_PRICE_IN_PER_M = 1.00
_PRICE_OUT_PER_M = 5.00


def cost_usd(usage) -> float:
    """Dollar cost of one call from its token usage (input $1 / output $5 per M)."""
    return (usage.input_tokens / 1_000_000) * _PRICE_IN_PER_M + (
        usage.output_tokens / 1_000_000
    ) * _PRICE_OUT_PER_M


# --- the relevance judge (reference-free, single answer) ----------------------

class RelevanceVerdict(BaseModel):
    """Structured judge output -- reasoning first so the model thinks before it labels."""

    explanation: str = Field(description="Brief reason for the relevance label.")
    relevance: str = Field(
        description="One of RELEVANT, PARTLY_RELEVANT, NON_RELEVANT."
    )


RELEVANCE_INSTRUCTIONS = """
You are evaluating whether a League of Legends coaching answer actually addresses the
player's question. You are given the QUESTION and the ANSWER only -- there is no
reference answer to compare against; judge the answer on its own.

Classify how well the answer responds to the question:
- RELEVANT: directly and usefully answers what was asked.
- PARTLY_RELEVANT: touches the topic but is generic, incomplete, or partly off-point.
- NON_RELEVANT: does not address the question, or declines to answer it.

Judge only relevance to the question -- not writing quality or length.
""".strip()

RELEVANCE_PROMPT = """
QUESTION:
{question}

ANSWER:
{answer}
""".strip()

# What we accept back from the judge; anything else is coerced so a bad label never
# reaches the NOT-NULL column.
_RELEVANCE_LABELS = {"RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"}

# A judge turns (question, answer) into a verdict + the usage of its own call.
RelevanceJudge = Callable[[str, str], "tuple[RelevanceVerdict, object]"]


def make_relevance_judge(client, model: str) -> RelevanceJudge:
    """Adapt an Anthropic-compatible client into a `RelevanceJudge` using structured
    output (`messages.parse`, the same call eval/evaluation_utils uses; works on both
    the direct and Bedrock clients). Returns the verdict and the call's usage."""

    def judge(question: str, answer: str):
        response = client.messages.parse(
            model=model,
            system=RELEVANCE_INSTRUCTIONS,
            messages=[{"role": "user",
                       "content": RELEVANCE_PROMPT.format(question=question, answer=answer)}],
            output_format=RelevanceVerdict,
            max_tokens=1024,
        )
        verdict = response.parsed_output
        if verdict.relevance not in _RELEVANCE_LABELS:
            verdict.relevance = "PARTLY_RELEVANT"   # never let a stray label break the insert
        return verdict, response.usage

    return judge


# --- the instrumented RAG ----------------------------------------------------

class RAGWithMetrics(RAGBase):

    def __init__(self, *args, store=None, judge: RelevanceJudge | None = None,
                 judge_model: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.store = store          # callable(ConversationRecord)->None; None = don't persist
        self.judge = judge or make_relevance_judge(self.llm_client, judge_model or self.model)
        self.last_conversation_id: str | None = None
        self.last_record: ConversationRecord | None = None
        self._last_answer_usage = None

    def llm(self, prompt, system=None):
        """Override to keep the answer call's usage (the base drops it)."""
        response = self._complete(prompt, system)
        self._last_answer_usage = response.usage
        return response.content[0].text

    def rag(self, query, personal_context=None, history=None):
        """Answer as the base does, but time the answer call, judge its relevance, and
        persist one telemetry row. Timing covers ONLY the answer call (player-facing)
        -- retrieval and the judge call are excluded.

        `history` (P0-7 conversation) reaches the answer prompt + system for chat memory;
        the recorded `question` is the player's original wording.

        Telemetry is best-effort: the answer is produced first, then judged/persisted
        inside a guard, so a judge or DB failure records nothing but never costs the
        player their answer (monitoring must not break the feature it monitors)."""
        self.last_conversation_id = None
        self.last_record = None

        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results, personal_context, history)

        t0 = time.perf_counter()
        answer = self.llm(prompt, system=self.system_prompt(history))
        response_time_ms = (time.perf_counter() - t0) * 1000.0
        answer_usage = self._last_answer_usage

        try:
            verdict, eval_usage = self.judge(query, answer)
            record = ConversationRecord(
                id=str(uuid4()),
                question=query,
                answer=answer,
                model=self.model,
                backend=backend(),
                response_time_ms=response_time_ms,
                prompt_tokens=answer_usage.input_tokens,
                completion_tokens=answer_usage.output_tokens,
                total_tokens=answer_usage.input_tokens + answer_usage.output_tokens,
                cost_usd=cost_usd(answer_usage),
                relevance=verdict.relevance,
                relevance_explanation=verdict.explanation,
                eval_prompt_tokens=eval_usage.input_tokens,
                eval_completion_tokens=eval_usage.output_tokens,
                eval_total_tokens=eval_usage.input_tokens + eval_usage.output_tokens,
                eval_cost_usd=cost_usd(eval_usage),
            )
            self.last_record = record
            if self.store is not None:
                self.store(record)
            # set only after a successful store, so a persist failure never hands the
            # app a conversation id with no row behind it (feedback FKs to this id)
            self.last_conversation_id = record.id
        except Exception:
            print("monitoring: failed to judge/persist conversation",
                  file=sys.stderr, flush=True)
            traceback.print_exc()

        return answer
