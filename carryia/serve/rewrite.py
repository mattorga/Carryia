"""Conversation transcript rendering for the coach chat (P0-7).

The multi-turn coach prepends the conversation so far to the answer prompt so the model
has memory of the chat (`rag_helper.build_prompt` + the conversation addendum). This is
the one place that shapes the prior turns into the plain transcript the model reads.

(Query rewriting -- condensing a follow-up like "why?" into a standalone retrieval query
-- was tried and removed 2026-09-06; retrieval runs on the raw turn. See the Journal.)
"""

from __future__ import annotations


def render_transcript(history) -> str:
    """The prior turns as a plain transcript the model can read. Reads only
    `role`/`content`, so the chat's message dicts pass through as-is (extra keys like
    `conv_id` are ignored)."""
    lines = []
    for turn in history:
        who = "Player" if turn.get("role") == "user" else "Coach"
        lines.append(f"{who}: {turn.get('content', '')}")
    return "\n".join(lines)
