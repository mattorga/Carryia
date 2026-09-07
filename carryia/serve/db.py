"""P0-8 monitoring -- the Postgres write layer.

Grafana charts the monitoring tables; this module is the app's
only writer into them. It holds no schema DDL -- Postgres creates the tables from
`docker/init.sql` on first boot (compose-mounted). Here we only INSERT.

Two seams keep it testable without a live database:
  - `connect()` opens a real psycopg2 connection from the environment (lazy import,
    so importing this module -- e.g. in a test -- needs no driver and no server).
  - `save_conversation` / `save_feedback` take the connection as an argument, so a
    fake connection can record the SQL + params. The app opens one real connection
    (cached) and passes it in.

The conversation INSERT is generated from `ConversationRecord`'s fields, so the
column list, the placeholders, and the values tuple can never fall out of step with
each other (a schema-drift test checks them against `docker/init.sql`).
"""

from __future__ import annotations

import os
from dataclasses import astuple, dataclass, fields


@dataclass
class ConversationRecord:
    """One answered question's telemetry -- the row written to `conversations`.

    Field order and names mirror `docker/init.sql` (minus `created_at`, which the DB
    defaults to now()). Values come from `monitoring.RAGWithMetrics.rag()`."""

    id: str                        # app-generated uuid; the feedback FK target
    question: str
    answer: str
    model: str                     # answer model id
    backend: str                   # anthropic | bedrock
    response_time_ms: float        # answer call only (player-facing)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float                # answer call cost
    relevance: str                 # RELEVANT | PARTLY_RELEVANT | NON_RELEVANT
    relevance_explanation: str
    eval_prompt_tokens: int
    eval_completion_tokens: int
    eval_total_tokens: int
    eval_cost_usd: float           # relevance-judge call cost


_CONV_COLUMNS = [f.name for f in fields(ConversationRecord)]
_CONV_INSERT = (
    f"INSERT INTO conversations ({', '.join(_CONV_COLUMNS)}) "
    f"VALUES ({', '.join(['%s'] * len(_CONV_COLUMNS))})"
)
_FEEDBACK_INSERT = (
    "INSERT INTO feedback (conversation_id, rating) VALUES (%s, %s)"
)


def dsn() -> str:
    """The Postgres connection string, from the environment. `DATABASE_URL` wins if
    set (compose passes it); otherwise assemble it from the POSTGRES_* parts, with
    localhost defaults for a developer running Postgres outside compose."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    return (
        f"host={os.environ.get('POSTGRES_HOST', 'localhost')} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ.get('POSTGRES_DB', 'carryia')} "
        f"user={os.environ.get('POSTGRES_USER', 'carryia')} "
        f"password={os.environ.get('POSTGRES_PASSWORD', 'carryia')}"
    )


def connect():
    """Open a real Postgres connection. psycopg2 is imported here, not at module top,
    so serve code (and its tests) import without the driver present."""
    import psycopg2

    return psycopg2.connect(dsn())


def save_conversation(conn, record: ConversationRecord) -> None:
    """Insert one conversation row. `astuple` follows the dataclass field order, which
    is exactly `_CONV_COLUMNS`, so values line up with columns by construction."""
    with conn.cursor() as cur:
        cur.execute(_CONV_INSERT, astuple(record))
    conn.commit()


def save_feedback(conn, conversation_id: str, rating: int) -> None:
    """Insert one thumbs rating (+1 / -1) against the conversation it rates."""
    with conn.cursor() as cur:
        cur.execute(_FEEDBACK_INSERT, (conversation_id, rating))
    conn.commit()
