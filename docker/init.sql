-- Carryia P0-8 monitoring schema.
--
-- Postgres runs every .sql in /docker-entrypoint-initdb.d/ once, on the FIRST boot
-- of an empty data volume (docker-compose mounts this file there). It is the single
-- source of truth for the two monitoring tables; the app only INSERTs, never DDLs.
--
-- Two tables, one row per answered question + its thumbs:
--   conversations -- one row per ask: the question, the grounded answer, and the
--                    telemetry the Grafana dashboard charts (timing, tokens, cost,
--                    and a reference-free LLM relevance judgement of the answer).
--   feedback      -- one row per thumbs click, FK'd to the conversation it rates.
--
-- The 8 course monitoring metrics are all SQL over these two tables:
--   total LLM calls        count(*) from conversations
--   avg response time      avg(response_time_ms)
--   total cost             sum(cost_usd + eval_cost_usd)
--   avg tokens             avg(total_tokens)
--   cost over time         sum(...) grouped by time
--   response time o/t      avg(response_time_ms) grouped by time
--   relevance              count grouped by relevance
--   thumbs up/down         count grouped by rating (feedback)

CREATE TABLE IF NOT EXISTS conversations (
    id                    TEXT PRIMARY KEY,           -- app-generated uuid; the feedback FK target
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    question              TEXT NOT NULL,
    answer                TEXT NOT NULL,
    model                 TEXT NOT NULL,              -- the answer model id
    backend               TEXT NOT NULL,              -- anthropic | bedrock (llm_backend.backend())

    -- answer call (player-facing): timing, usage, derived cost
    response_time_ms      DOUBLE PRECISION NOT NULL,
    prompt_tokens         INTEGER NOT NULL,
    completion_tokens     INTEGER NOT NULL,
    total_tokens          INTEGER NOT NULL,
    cost_usd              DOUBLE PRECISION NOT NULL,

    -- relevance judge (a 2nd LLM call per ask): its verdict +
    -- its own usage/cost, split from the answer call so response_time/cost stay
    -- player-facing.
    relevance             TEXT NOT NULL,              -- RELEVANT | PARTLY_RELEVANT | NON_RELEVANT
    relevance_explanation TEXT NOT NULL,
    eval_prompt_tokens    INTEGER NOT NULL,
    eval_completion_tokens INTEGER NOT NULL,
    eval_total_tokens     INTEGER NOT NULL,
    eval_cost_usd         DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id              SERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    rating          INTEGER NOT NULL,                 -- +1 thumbs up, -1 thumbs down
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations (created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_conversation_id ON feedback (conversation_id);
