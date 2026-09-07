"""Spec for carryia/serve/db.py -- the P0-8 monitoring write layer.

GREEN throughout: the inserts run against a fake DB-API connection that records the
SQL + params (no psycopg2, no server), so we pin the write CONTRACT. One structural
test guards the record against docker/init.sql so the app's columns can't drift from
the schema Postgres actually creates.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

from carryia.serve import db
from carryia.serve.db import ConversationRecord

ROOT = Path(__file__).resolve().parent.parent
INIT_SQL = ROOT / "docker" / "init.sql"


# --- a fake DB-API connection ------------------------------------------------

class _FakeCursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self.log.append((sql, params))


class _FakeConn:
    def __init__(self):
        self.executed = []
        self.commits = 0

    def cursor(self):
        return _FakeCursor(self.executed)

    def commit(self):
        self.commits += 1


def _record(**over):
    base = dict(
        id="conv-1", question="why do i lose lane", answer="Ward earlier. (mobalytics)",
        model="claude-haiku-4-5", backend="anthropic", response_time_ms=812.5,
        prompt_tokens=1200, completion_tokens=140, total_tokens=1340, cost_usd=0.0019,
        relevance="RELEVANT", relevance_explanation="Directly answers the lane question.",
        eval_prompt_tokens=300, eval_completion_tokens=20, eval_total_tokens=320,
        eval_cost_usd=0.0004,
    )
    base.update(over)
    return ConversationRecord(**base)


# --- save_conversation -------------------------------------------------------

def test_save_conversation_inserts_all_columns_in_field_order():
    conn = _FakeConn()
    rec = _record()
    db.save_conversation(conn, rec)

    assert len(conn.executed) == 1
    sql, params = conn.executed[0]
    assert sql.startswith("INSERT INTO conversations")
    # one placeholder + one column per dataclass field, in the same order
    names = [f.name for f in fields(ConversationRecord)]
    assert sql.count("%s") == len(names)
    assert params == tuple(getattr(rec, n) for n in names)
    assert conn.commits == 1


def test_save_feedback_inserts_conversation_id_and_rating():
    conn = _FakeConn()
    db.save_feedback(conn, "conv-1", -1)

    sql, params = conn.executed[0]
    assert sql.startswith("INSERT INTO feedback")
    assert params == ("conv-1", -1)
    assert conn.commits == 1


# --- schema-drift guard: record columns must exist in init.sql ---------------

def _columns_of(table: str, sql_text: str) -> set[str]:
    """Pull column names from a `CREATE TABLE <table> ( ... )` block: the first token
    of each definition line that isn't a comment or a table-level constraint."""
    block = re.search(rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\);",
                      sql_text, re.DOTALL).group(1)
    cols = set()
    for line in block.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("--"):
            continue
        first = line.split()[0]
        if first.upper() in {"PRIMARY", "FOREIGN", "CONSTRAINT", "UNIQUE", "CHECK"}:
            continue
        cols.add(first)
    return cols


def test_conversation_record_fields_all_exist_in_the_schema():
    schema_cols = _columns_of("conversations", INIT_SQL.read_text())
    for name in (f.name for f in fields(ConversationRecord)):
        assert name in schema_cols, f"{name} missing from conversations schema"


def test_feedback_insert_targets_real_schema_columns():
    schema_cols = _columns_of("feedback", INIT_SQL.read_text())
    assert {"conversation_id", "rating"} <= schema_cols


# --- dsn ---------------------------------------------------------------------

def test_dsn_prefers_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/carryia")
    assert db.dsn() == "postgresql://u:p@h:5432/carryia"


def test_dsn_assembles_from_parts_when_no_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_HOST", "db")
    monkeypatch.setenv("POSTGRES_DB", "carryia")
    out = db.dsn()
    assert "host=db" in out and "dbname=carryia" in out
