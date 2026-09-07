"""P0-6 answer generation -- the grounded, cited coaching answer.

Adapted from the LLM-Zoomcamp RAGBase (framing reused, code rewritten -- CLAUDE.md).
Runs on the Claude API (Haiku 4.5): the caller passes an
Anthropic client, and reviewers bring their own ANTHROPIC_API_KEY. Retrieval is the
vector approach that won P0-5 -- the class holds an `ingest.VectorSearch` index and
embeds the query with the SAME local fastembed model the index was built with (a
mismatched embedder scores garbage). The answer is grounded strictly in the retrieved
tips and cites their source.

P0-6 compares TWO PROMPTS on a held set: the swappable part is `instructions` --
build the class with COACH_BLUNT or COACH_WHY, keep retrieval fixed, and judge which
one gives a support main better advice. `instructions` is the ONLY thing that differs
between the two variants; everything else is held constant so the comparison is fair.

`search` + `build_context` are the RAG core (query the vector index; assemble the
grounded, citable context); the rest is wiring, pinned by `test_rag_helper.py`.

**Personal grounding (P0-7):** `rag`/`build_prompt` take an optional `personal_context`
-- the Stat-Line Producer's block (`personal.producer.subject_context_block`) placing
this player against the Silver cohort. When absent, the prompt collapses to the
corpus-only form the P0-6 eval was scored on, so that run stays reproducible.
"""

from __future__ import annotations

from carryia.pipeline.ingest import embed_tips  # the SAME local model the vector index was built with


# --- the two prompt variants P0-6 compares -----------------------------------
# Starting drafts -- TUNE THESE. The wording is the experiment: the only thing that
# differs is whether the coach explains WHY. Hold everything else identical (persona,
# grounding guard, citation rule) so the comparison isolates that one change.

COACH_BLUNT = '''
You are a League of Legends coach for support / bot-lane players, reviewing a game.

Answer the player's question with 2-3 concrete, specific actions. Be direct and
imperative -- no preamble, no encouragement, no filler.

Use ONLY the coaching notes provided below. If they do not cover the question, say
"I don't have a note on that" rather than guessing. Cite the source of each action
in parentheses, e.g. (mobalytics).
'''.strip()

COACH_WHY = '''
You are a League of Legends coach for support / bot-lane players, reviewing a game.

Answer the player's question with concrete, specific actions, and for each one add the
short reason it works, so the player learns the principle -- not just the instruction.

Use ONLY the coaching notes provided below. If they do not cover the question, say
"I don't have a note on that" rather than guessing. Cite the source of each action
in parentheses, e.g. (mobalytics).
'''.strip()

# The optional personal-plane grounding: the Stat-Line Producer's block, inserted
# between the question and the notes when the caller passes one (P0-7). Empty for the
# corpus-only P0-6 eval, so that prompt stays byte-identical to the committed run.
PERSONAL_TEMPLATE = '''
This player's recent-form pattern (ground your advice in what THIS player actually does):
{personal}
'''.strip()

PROMPT_TEMPLATE = '''
{history}The player asked:
{question}
{personal}
Coaching notes (use ONLY these):
{context}
'''.strip()

# The conversation transcript, prepended to the user turn so the answer model has
# memory of the chat (P0-7 conversational). Empty for the eval + the first turn, so the
# prompt stays byte-identical to the committed P0-6 run when there's no history.
HISTORY_TEMPLATE = '''
Conversation so far:
{history}

'''.lstrip("\n")

# Appended to the system prompt ONLY when there's history -- so the frozen COACH_* eval
# variants reach the model byte-for-byte on the corpus-only P0-6 run, and the coach only
# becomes conversation-aware in the live chat. This is the single, scoped exception to
# "use only the notes": it lets the coach answer questions ABOUT the chat (what was said,
# which message this is) from the transcript, while coaching ADVICE still comes only from
# the cited notes.
CONVERSATION_ADDENDUM = '''
This is an ongoing conversation; the user turn includes the conversation so far. Use it to
stay coherent and to resolve follow-ups. You MAY answer questions about the conversation
itself (what was said earlier, which message this is) from that transcript -- that is the
one exception to using only the notes. Coaching ADVICE still comes only from the coaching
notes, and you still cite each note.
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=COACH_WHY,
        prompt_template=PROMPT_TEMPLATE,
        embed=embed_tips,
        model='claude-haiku-4-5',
        num_results=5,
        max_tokens=1024,
        retriever=None,
    ):
        self.index = index              # ingest.VectorSearch (P0-5 winner); unused when retriever is set
        self.llm_client = llm_client    # anthropic.Anthropic
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.embed = embed
        self.model = model
        self.num_results = num_results
        self.max_tokens = max_tokens
        self.retriever = retriever      # optional docs-returning retriever (serve.retrieval); overrides the index path

    def search(self, query):
        """Return the top FULL doc dicts to ground on (build_context needs the fields,
        not tip_ids). Two paths, one contract:
          - `retriever` set -> delegate to it (the app ships the hybrid that won P0-5:
            `serve.retrieval.build_hybrid_docs_retriever`, ranked ids rehydrated to docs).
          - else -> the vector path: embed `query` and search the vector index directly
            (what the P0-6 eval ran on; one shared embedding space keeps vectors comparable)."""
        if self.retriever is not None:
            return self.retriever(query)
        query_vector = self.embed([query])[0]
        return self.index.search(query_vector, num_results=self.num_results)

    def build_context(self, search_results):
        """Render the retrieved docs into the grounded context block: one entry per
        doc carrying its `tip`, its `rationale`, and a source tag (`creator_id` +
        `source_url`) the model can cite. This is the whole of what the answer is
        allowed to stand on -- a field that isn't here can't be cited."""
        entries = []
        for doc in search_results:
            entries.append(
                f"- Tip: {doc['tip']}\n"
                f"  Why: {doc['rationale']}\n"
                f"  Source: {doc['creator_id']} ({doc['source_url']})"
            )
        return '\n\n'.join(entries)

    def build_prompt(self, query, search_results, personal_context=None, history=None):
        """Assemble the user turn: (optional transcript) + question + (optional personal
        block) + notes.

        `personal_context` is the Stat-Line Producer's block (P0-7). `history` is the
        prior chat turns (P0-7 conversation), prepended so the answer model has memory of
        the chat. When BOTH are absent the `{history}`/`{personal}` slots collapse to
        nothing, so the prompt is byte-identical to the corpus-only P0-6 run -- adding
        either grounding never rewrites that eval."""
        context = self.build_context(search_results)
        personal = ""
        if personal_context:
            personal = "\n" + PERSONAL_TEMPLATE.format(personal=personal_context) + "\n"
        transcript = ""
        if history:
            from carryia.serve.rewrite import render_transcript  # lazy: eval path import-light
            transcript = HISTORY_TEMPLATE.format(history=render_transcript(history))
        return self.prompt_template.format(
            question=query, context=context, personal=personal, history=transcript)

    def system_prompt(self, history=None):
        """The system prompt for one answer: the frozen variant `instructions`, plus the
        conversational addendum ONLY when there's history. No history (the eval, the first
        turn) -> pristine COACH_* reaches the model, so P0-6 stays byte-identical; a live
        follow-up -> the coach is allowed to answer about the conversation too."""
        if not history:
            return self.instructions
        return self.instructions + "\n\n" + CONVERSATION_ADDENDUM

    def _complete(self, prompt, system=None):
        """The raw model call -- returns the full response so a subclass can read
        `.usage` (P0-8 monitoring). `llm` keeps returning just the text, so this split
        is invisible to the base contract. `system` defaults to the frozen `instructions`
        so the eval + tests are unchanged; `rag` passes the conversation-aware system."""
        return self.llm_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.instructions if system is None else system,
            messages=[{'role': 'user', 'content': prompt}],
        )

    def llm(self, prompt, system=None):
        return self._complete(prompt, system).content[0].text

    def rag(self, query, personal_context=None, history=None):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results, personal_context, history)
        return self.llm(prompt, system=self.system_prompt(history))
