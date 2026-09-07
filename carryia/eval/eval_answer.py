"""P0-6 answer eval -- pairwise LLM-as-judge over the two prompt variants.

The graded P0-6 deliverable: compare BLUNT vs WHY and document
the winner. Structural twin of eval_retrieval.py -- a scorer over a pluggable seam --
except the scorer is a Haiku *judge* rating answers, not hit-rate/MRR over rankings.

Why pairwise (not the course's reference-based judge): coaching questions have no
single correct answer, so there's nothing to score against. Pairwise only asks "which
of these two is better?", which needs no reference and is what LLM judges do reliably.

Position bias is handled: `compare` judges each pair in BOTH orders and only credits a
win when the verdict survives the swap -- so the winner isn't an artifact of which
answer was shown first.

Two seams keep the judge testable without a key (mirrors eval_retrieval's Retriever):
  Judge      = (question, first_answer, second_answer) -> "A" | "B" | "tie"
  Generator  = question -> answer            (a variant's RAGBase.rag)

`make_judge` (the structured judge call) and `compare` (the both-orders debias +
reconciliation) are the judge core; the rest is wiring. Nothing pins this yet -- there
is no test_eval_answer.py (contrast test_eval_retrieval.py), so a GREEN metrics +
fake-judge suite is the next step.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from typing import Callable, Literal

from pydantic import BaseModel, Field

from carryia.eval.eval_retrieval import load_ground_truth      # reuse the P0-5 answer key
from carryia.eval.evaluation_utils import llm_structured_retry  # course structured-output helper

# A Judge sees the question and two answers IN ORDER and says which is better; "A" =
# the first answer, "B" = the second, "tie" = about equal. Order-relative on purpose
# -- `compare` is what cancels the order.
Judge = Callable[[str, str, str], str]
# A Generator turns a question into one variant's answer (a RAGBase.rag, bound to
# COACH_BLUNT or COACH_WHY).
Generator = Callable[[str], str]

DEFAULT_N = 709           # full ground-truth census (all rows, no sampling); ~2,836 calls
DEFAULT_SEED = 0          # deterministic sample, so the eval is reproducible
JUDGE_MODEL = "claude-haiku-4-5"


class PairwiseVerdict(BaseModel):
    """Structured judge output -- reasoning first so the model thinks before it picks
    (as in the course's AnswerEvaluation)."""
    reasoning: str = Field(description="Brief reasoning for the choice.")
    winner: Literal["A", "B", "tie"] = Field(
        description="'A' if Answer A is the better coaching answer, 'B' if Answer B, "
                    "'tie' if they are about equal."
    )


# The criteria -- reference-free, judgeable from the answers alone (groundedness is
# already enforced at generation, so it's left off the judge's checklist).
JUDGE_INSTRUCTIONS = """
You are comparing two coaching answers to a League of Legends support / bot-lane
player's question. Both were written from the same coaching notes.

Pick the answer that would help the player improve more. Weigh:
- Specific and actionable: concrete moves the player can execute, not vague advice.
- Clear and direct: gets to the point, no padding.
- Useful for reviewing a game they just played.

Do not favour an answer for being longer. If they are about equal, say "tie".
""".strip()

JUDGE_PROMPT = """
Player's question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}
""".strip()


# --- load a held set ---------------------------------------------------------

def load_held_set(n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> list[dict]:
    """Deterministically sample `n` ground-truth rows to judge over. Seeded so the
    eval is reproducible; returns all rows if `n` exceeds the corpus."""
    rows = load_ground_truth()
    if n >= len(rows):
        return rows
    return random.Random(seed).sample(rows, n)


# --- the judge core ----------------------------------------------------------

def make_judge(client, instructions: str = JUDGE_INSTRUCTIONS,
               model: str = JUDGE_MODEL) -> Judge:
    """Adapt an Anthropic client into a `Judge`: format `JUDGE_PROMPT`, ask for a
    structured `PairwiseVerdict`, hand back its `winner`. The adapter, like
    eval_retrieval.keyword_retriever -- `compare` can't tell a real client from a
    fake."""
    def judge(question: str, answer_a: str, answer_b: str) -> str:
        prompt = JUDGE_PROMPT.format(
            question=question, answer_a=answer_a, answer_b=answer_b)
        verdict, _usage = llm_structured_retry(
            client, instructions, prompt, PairwiseVerdict, model=model)
        return verdict.winner

    return judge


def compare(judge: Judge, question: str, answer_a: str, answer_b: str) -> str:
    """One debiased pairwise decision, returning "A" (answer_a wins), "B" (answer_b
    wins), or "tie". Judge both orders and credit a win only if it survives the swap:
    the second call's labels are flipped (its "A" means answer_b), so normalise it
    back before comparing. Disagreement across orders -- position bias -- is a tie."""
    normalise = {"A": "B", "B": "A", "tie": "tie"}
    first = judge(question, answer_a, answer_b)             # answer_a shown first
    second = normalise[judge(question, answer_b, answer_a)]  # answer_b shown first
    if first == second and first in ("A", "B"):
        return first
    return "tie"


# --- run + tally (GREEN) ------------------------------------------------------

def run_pairwise(held_set: list[dict], gen_a: Generator, gen_b: Generator,
                 judge: Judge, label_a: str, label_b: str) -> list[str]:
    """For every held-out question, generate both variants' answers and compare them,
    returning the winner per question as a human label (`label_a`, `label_b`, or
    "tie"). Generation and judging are injected, so this wiring is what the GREEN test
    pins (with `compare` monkeypatched)."""
    winners: list[str] = []
    running = Counter()
    total = len(held_set)
    for i, row in enumerate(held_set, 1):
        question = row["question"]
        answer_a = gen_a(question)
        answer_b = gen_b(question)
        outcome = compare(judge, question, answer_a, answer_b)
        winner = {"A": label_a, "B": label_b}.get(outcome, "tie")
        winners.append(winner)
        running[winner] += 1
        # progress to stderr so an unattended run is observable + recoverable from the
        # log if it dies mid-batch (stdout stays the final tally)
        print(f"[{i}/{total}] {label_a}={running[label_a]} "
              f"{label_b}={running[label_b]} tie={running['tie']}",
              file=sys.stderr, flush=True)
    return winners


def tally(winners: list[str]) -> dict[str, int]:
    """Count wins per label (and ties). The documented P0-6 result is whichever label
    leads this tally."""
    return dict(Counter(winners))


# --- CLI: run the real comparison, print the tally ---------------------------

def main(argv: list[str] | None = None) -> int:
    """Build both coaches over the real corpus, judge them head-to-head on the held
    set, and print the tally -- P0-6 end to end. Needs a key (generation + judging are
    live Haiku). Returns a process exit code."""
    from carryia.pipeline.ingest import build_vector_index, load_corpus
    from carryia.serve.llm_backend import make_client, model_id
    from carryia.serve.rag_helper import COACH_BLUNT, COACH_WHY, RAGBase

    held = load_held_set()
    documents = load_corpus()
    index = build_vector_index(documents)      # local fastembed; no key
    client = make_client()                     # anthropic or bedrock, per CARRYIA_LLM_BACKEND
    model = model_id()                         # matching model id for that backend

    blunt = RAGBase(index=index, llm_client=client, instructions=COACH_BLUNT, model=model)
    why = RAGBase(index=index, llm_client=client, instructions=COACH_WHY, model=model)
    judge = make_judge(client, model=model)

    winners = run_pairwise(held, blunt.rag, why.rag, judge, "BLUNT", "WHY")
    counts = tally(winners)

    print(f"pairwise over {len(held)} questions (both orders each):")
    for label in ("BLUNT", "WHY", "tie"):
        print(f"  {label:<6}{counts.get(label, 0):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
