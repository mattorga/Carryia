"""Spec for carryia/serve/rag_helper.py -- P0-6 answer generation (the RAG generator).

TDD state (the house three-state, as in test_eval_retrieval.py):
  - GREEN: every method is implemented, so these lock the wiring now. The RAG core
    (`search`, `build_context`) runs on hand-built fakes -- no index, no model, no
    API. The I/O boundary (`llm`) runs on a fake Anthropic client that records the
    call, so we assert the CONTRACT (model / system / messages / max_tokens) without
    a key or a network hop.
  - SKIPPED: `test_live_smoke` calls real Haiku over the sample corpus. Off by
    default (it costs a key + a model download); set CARRYIA_LIVE=1 to run it.

The two prompt variants are the P0-6 experiment: the tests pin that `instructions`
is the only knob that reaches the model, so the comparison stays fair.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from carryia.pipeline import ingest
from carryia.serve import rag_helper

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures" / "corpus.sample.jsonl"


# --- fakes: no index, no model, no network -----------------------------------

class _FakeIndex:
    """Stand-in for minsearch.VectorSearch: records how it was queried and returns
    preset docs (truncated to num_results, like the real top-k)."""

    def __init__(self, docs):
        self._docs = docs
        self.calls = []

    def search(self, query_vector, num_results):
        self.calls.append((query_vector, num_results))
        return self._docs[:num_results]


class _FakeMessages:
    """Records each create() call and returns a response shaped like the Anthropic
    SDK's: `.content[0].text`."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = type("Block", (), {"text": self.reply})()
        return type("Resp", (), {"content": [block]})()


class _FakeClient:
    def __init__(self, reply="Ward the river bush before you recall. (mobalytics)"):
        self.messages = _FakeMessages(reply)


def _docs(n=2):
    base = [
        {"tip": "Ward the river bush before you recall.",
         "rationale": "Vision denies the enemy jungler a free dive.",
         "creator_id": "mobalytics", "source_url": "https://mobalytics.gg/a"},
        {"tip": "Freeze the wave outside your turret when ahead.",
         "rationale": "A freeze starves the enemy of CS and XP.",
         "creator_id": "skillcapped", "source_url": "https://skill-capped.com/b"},
    ]
    return base[:n]


# --- GREEN: build_context (pure -- docs -> grounded, citable block) -----------

def test_build_context_renders_tip_rationale_and_source():
    ctx = rag_helper.RAGBase(index=None, llm_client=None).build_context(_docs(2))
    assert "Ward the river bush before you recall." in ctx
    assert "Vision denies the enemy jungler a free dive." in ctx
    assert "mobalytics" in ctx
    assert "https://mobalytics.gg/a" in ctx
    assert "Freeze the wave outside your turret when ahead." in ctx


def test_build_context_has_one_entry_per_doc():
    ctx = rag_helper.RAGBase(index=None, llm_client=None).build_context(_docs(2))
    assert ctx.count("Tip:") == 2


def test_build_context_of_nothing_is_empty():
    assert rag_helper.RAGBase(index=None, llm_client=None).build_context([]) == ""


# --- GREEN: search (fake index + spy embed -- no model) -----------------------

def test_search_embeds_the_query_and_returns_full_docs():
    docs = _docs(2)
    index = _FakeIndex(docs)
    seen = {}

    def spy_embed(texts):
        seen["texts"] = list(texts)
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    rag = rag_helper.RAGBase(index=index, llm_client=None, embed=spy_embed,
                             num_results=2)
    out = rag.search("why do i keep getting dived on recall")

    assert seen["texts"] == ["why do i keep getting dived on recall"]   # embeds query
    assert out == docs                                                  # full docs
    assert index.calls[0][1] == 2                                       # num_results


def test_search_respects_num_results():
    index = _FakeIndex(_docs(2))
    rag = rag_helper.RAGBase(index=index, llm_client=None,
                             embed=lambda t: np.zeros((1, 3), dtype=np.float32),
                             num_results=1)
    assert len(rag.search("q")) == 1


def test_search_delegates_to_retriever_when_set():
    # The app ships a docs-returning hybrid retriever (serve.retrieval); when one is
    # passed, search returns its docs and never touches the vector index/embed path.
    docs = _docs(2)
    seen = {}

    def fake_retriever(query):
        seen["query"] = query
        return docs

    def boom_embed(texts):  # must not be called on the retriever path
        raise AssertionError("embed should not run when a retriever is set")

    rag = rag_helper.RAGBase(index=None, llm_client=None, embed=boom_embed,
                             retriever=fake_retriever)
    out = rag.search("why do i keep losing lane")

    assert out == docs
    assert seen["query"] == "why do i keep losing lane"


# --- GREEN: build_prompt (question + context into the template) ---------------

def test_build_prompt_injects_question_and_context():
    rag = rag_helper.RAGBase(index=None, llm_client=None)
    prompt = rag.build_prompt("why blind trinket?", _docs(1))
    assert "why blind trinket?" in prompt
    assert "Ward the river bush before you recall." in prompt


# --- GREEN: personal-context grounding (P0-7) is an additive, optional slot ---

def test_build_prompt_without_personal_context_is_unchanged():
    # The no-personal prompt must stay byte-identical to the corpus-only P0-6 run,
    # so adding personal grounding never rewrites that committed eval.
    rag = rag_helper.RAGBase(index=None, llm_client=None)
    assert rag.build_prompt("q", _docs(1)) == rag.build_prompt("q", _docs(1), None)
    prompt = rag.build_prompt("q", _docs(1))
    assert "recent-form pattern" not in prompt


def test_build_prompt_injects_the_personal_block_when_given():
    rag = rag_helper.RAGBase(index=None, llm_client=None)
    block = "Kill participation — WEAK: 46% (cohort avg 53%)"
    prompt = rag.build_prompt("why do i fall behind", _docs(1), personal_context=block)
    assert block in prompt
    # positioned between the question and the coaching notes.
    assert prompt.index("why do i fall behind") < prompt.index(block) < prompt.index("Tip:")


def test_rag_threads_personal_context_to_the_model():
    index = _FakeIndex(_docs(1))
    client = _FakeClient(reply="ok")
    rag = rag_helper.RAGBase(index=index, llm_client=client,
                             embed=lambda t: np.zeros((1, 3), dtype=np.float32), num_results=1)
    rag.rag("q", personal_context="Late ward activity — WEAK: 8.1 per game")
    sent = client.messages.calls[0]["messages"][0]["content"]
    assert "Late ward activity — WEAK: 8.1 per game" in sent


# --- GREEN: conversation memory (P0-7) is additive + eval-safe ----------------

_HISTORY = [
    {"role": "user", "content": "why do i keep losing lane?"},
    {"role": "assistant", "content": "Ward the tri-bush earlier. (mobalytics)"},
]


def test_build_prompt_without_history_is_byte_identical():
    # No history -> the {history} slot collapses, so the prompt matches the corpus-only
    # P0-6 run exactly; conversation memory never rewrites that committed eval.
    rag = rag_helper.RAGBase(index=None, llm_client=None)
    assert rag.build_prompt("q", _docs(1)) == rag.build_prompt("q", _docs(1), history=None)
    assert "Conversation so far" not in rag.build_prompt("q", _docs(1))


def test_build_prompt_prepends_the_transcript_when_history_is_given():
    rag = rag_helper.RAGBase(index=None, llm_client=None)
    prompt = rag.build_prompt("why?", _docs(1), history=_HISTORY)
    assert "Conversation so far" in prompt
    assert "why do i keep losing lane?" in prompt          # the prior turn is in the transcript
    assert "Ward the tri-bush earlier. (mobalytics)" in prompt
    # transcript sits above the current question + the notes
    assert prompt.index("Conversation so far") < prompt.index("The player asked") < prompt.index("Tip:")


def test_system_prompt_is_pristine_without_history_and_augmented_with_it():
    rag = rag_helper.RAGBase(index=None, llm_client=None, instructions="SYSTEM")
    assert rag.system_prompt(None) == "SYSTEM"                       # eval + first turn: untouched
    augmented = rag.system_prompt(_HISTORY)
    assert augmented.startswith("SYSTEM")
    assert rag_helper.CONVERSATION_ADDENDUM in augmented             # conversation-aware only in chat


def test_rag_with_history_sends_the_addendum_and_transcript_to_the_answer_call():
    index = _FakeIndex(_docs(1))
    client = _FakeClient(reply="ok")
    rag = rag_helper.RAGBase(index=index, llm_client=client, instructions="SYSTEM",
                             embed=lambda t: np.zeros((1, 3), dtype=np.float32), num_results=1)
    rag.rag("why?", history=_HISTORY)
    # retrieval runs on the raw turn (no rewrite); the single call is the answer.
    answer_call = client.messages.calls[-1]
    assert rag_helper.CONVERSATION_ADDENDUM in answer_call["system"]
    assert "Conversation so far" in answer_call["messages"][0]["content"]


# --- GREEN: llm (contract test over a fake Anthropic client) ------------------

def test_llm_calls_messages_create_with_the_right_shape():
    client = _FakeClient(reply="Do X, then Y. (mobalytics)")
    rag = rag_helper.RAGBase(index=None, llm_client=client, instructions="SYSTEM",
                             model="claude-haiku-4-5", max_tokens=512)
    out = rag.llm("PROMPT TEXT")

    assert out == "Do X, then Y. (mobalytics)"
    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["system"] == "SYSTEM"            # system prompt, not a user turn
    assert call["max_tokens"] == 512
    assert call["messages"] == [{"role": "user", "content": "PROMPT TEXT"}]


# --- GREEN: the variant seam is `instructions`, and only that -----------------

def test_the_two_prompt_variants_differ():
    assert rag_helper.COACH_BLUNT != rag_helper.COACH_WHY


def test_the_chosen_variant_is_what_reaches_the_model():
    client = _FakeClient()
    rag_helper.RAGBase(index=None, llm_client=client,
                       instructions=rag_helper.COACH_BLUNT).llm("p")
    assert client.messages.calls[0]["system"] == rag_helper.COACH_BLUNT


# --- GREEN: rag end-to-end (fakes wired -- search -> prompt -> llm) ------------

def test_rag_threads_retrieved_context_into_the_call():
    index = _FakeIndex(_docs(1))
    client = _FakeClient(reply="Ward, because vision. (mobalytics)")
    rag = rag_helper.RAGBase(index=index, llm_client=client,
                             embed=lambda t: np.zeros((1, 3), dtype=np.float32),
                             num_results=1)
    out = rag.rag("how do i stop getting dived")

    assert out == "Ward, because vision. (mobalytics)"
    sent = client.messages.calls[0]["messages"][0]["content"]
    assert "Ward the river bush before you recall." in sent   # retrieved tip is grounded
    assert "how do i stop getting dived" in sent               # question is present


# --- SKIPPED: live smoke against real Haiku (opt in: CARRYIA_LIVE=1) -----------

@pytest.mark.skipif(not os.getenv("CARRYIA_LIVE"),
                    reason="live API smoke; set CARRYIA_LIVE=1 (needs a key)")
def test_live_smoke():
    from carryia.serve.llm_backend import make_client, model_id

    docs = ingest.load_corpus(SAMPLE)
    index = ingest.build_vector_index(docs)          # real fastembed
    rag = rag_helper.RAGBase(index=index, llm_client=make_client(), model=model_id())
    answer = rag.rag("i keep getting caught out with no vision, what do i do")

    assert isinstance(answer, str) and answer.strip()
