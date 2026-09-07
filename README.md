# Carryia

**A post-game coach for League of Legends support mains.** Ask a question about your
recent ranked games and get a specific, actionable answer — grounded in *your own*
stats and cited to a reputable coaching source, not a generic guide or a model's
unsourced opinion.

*LLM Zoomcamp 2026 capstone (DataTalks.Club).*

---

## The problem

Support and bot-lane players who want to climb rarely get feedback on *their own*
games. Generic guides don't reference what they actually did, and human coaching is
expensive and scarce — so a player is left guessing which of a thousand contradictory
opinions to trust.

Carryia answers a player's question by combining two things a generic bot can't:

- **What you actually do** — a committed snapshot of the subject's recent support
  games, summarised and placed against a cohort of same-rank peers.
- **What good looks like** — a curated corpus of coaching tips distilled from
  reputable creators, each carrying a verbatim source quote and citation.

Every recommendation is grounded strictly in retrieved tips and cited to its source,
so the advice is specific to how you play *and* defensibly authoritative.

## How it works

Two data planes feed one RAG core:

| Plane | Holds | Source |
|-------|-------|--------|
| **Personal** | facts about the player (deaths, wards, objectives, vision, KP per phase) | committed match snapshot + a committed Silver-cohort benchmark |
| **Knowledge** | cited coaching authority (what to do) | 709-tip corpus distilled from 7 creators |

At query time: the snapshot is summarised into a personal-context block (the player
placed against the cohort by percentile), the question is grounded in it, the **hybrid
retriever** (keyword + vector, RRF-fused — the eval winner) retrieves the most relevant
tips, and the LLM composes a grounded, cited answer over them. Each answered question is
timed, cost-tracked, relevance-judged, and written to Postgres for a **Grafana
dashboard** (see [Monitoring](#monitoring)). See the interactive diagrams under
[`docs/`](docs/) (`architecture-system.html`, `architecture-coach.html`).

---

## Quickstart — run the app

You need one thing: an LLM API key (a cheap model, Haiku 4.5, is pinned). Copy the
env template and set it:

```bash
cp .env.example .env
#   then edit .env and set ANTHROPIC_API_KEY (get one at console.anthropic.com)
```

### Option A — Docker (one command, recommended)

**Prerequisite:** Docker + Docker Compose. Then:

```bash
docker compose up
```

That builds and starts the whole stack — the **app**, a **Postgres** monitoring store
(schema auto-created on first boot), and **Grafana** (dashboard auto-provisioned):

- App — <http://localhost:8501>
- Grafana — <http://localhost:3000> (admin / admin)

First boot downloads the local embedding model once (~30s) and builds the index.
After a schema change, recreate the DB volume with `docker compose down -v`.

### Option B — local Python

**Prerequisite:** Python 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
streamlit run carryia/serve/app.py
```

This runs the app without the monitoring stack — it answers normally, monitoring is
just off (no Postgres required). To enable monitoring locally, point `DATABASE_URL` at
your own Postgres (see [`.env.example`](.env.example)).

> Using [`uv`](https://docs.astral.sh/uv/)? Replace the pip steps with
> `uv pip install -r requirements.txt && uv pip install -e .`.

Either way the app opens to a **"Your recent form"** panel (your stats vs the cohort,
with percentiles), a question box, and cited answers with a 👍/👎 rating on each. It
reads only committed data and spends your key solely on the LLM at query time — no Riot
key needed.

## Configuration

All configuration is via `.env` (copied from [`.env.example`](.env.example)). The
answer generator runs on one of two backends, chosen by `CARRYIA_LLM_BACKEND`:

| Backend | Set | Auth |
|---------|-----|------|
| `anthropic` *(default)* | reviewers use this | `ANTHROPIC_API_KEY` |
| `bedrock` | Amazon Bedrock, billed to AWS | `AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION` with Haiku 4.5 access |

The default is the direct Anthropic API so a reviewer runs entirely on their own key.

## Monitoring

Under `docker compose up`, every answered question is instrumented and written to
Postgres, and **Grafana** (<http://localhost:3000>, admin / admin) charts it on an
auto-provisioned dashboard. Two tables back it — `conversations` (one row per ask) and
`feedback` (one row per 👍/👎) — and the dashboard covers the full course metric set
across 8 panels:

- total answers, avg response time, total cost, avg tokens (at-a-glance stats)
- cost and response time **over time**
- **answer relevance** — a reference-free LLM-as-judge rates each answer against the
  question (a 2nd LLM call per ask), charted RELEVANT / PARTLY / NON
- **user feedback** — the thumbs breakdown

Monitoring is best-effort: if the judge or DB is unavailable the answer is still
returned, and a bare `streamlit run` (Option B) simply runs with monitoring off.

## What's in the box (committed data)

The app runs against committed snapshots — no live API calls at query time:

- `data/corpus.jsonl` — **709** coaching tips (7 creators), each with a verbatim
  `source_excerpt` and citation.
- `data/snapshot/` — the subject's **10** recent support games (`guuji#miko`, SEA).
- `data/benchmark/cohort.json` — a **121-player Silver support** cohort (aggregate
  distributions only, no player identifiers) the subject is ranked against.
- `data/ground_truth.jsonl` — 709 question→tip pairs for the retrieval eval.

## Evaluation

Both graded evaluations are committed as runnable notebooks under `carryia/eval/`:

- **Retrieval (P0-5)** — `eval.ipynb` compares keyword, vector, and hybrid retrieval
  over the ground-truth set (hit-rate / MRR). **Vector beats keyword; hybrid edges
  vector** — the chosen approach is justified by the comparison.
- **Answer generation (P0-6)** — `eval_answer.py` runs a pairwise LLM-as-judge over
  two prompt variants across all 709 questions. **The explanatory ("WHY") coach wins,
  p < 0.001** — the shipped variant.

## Tests

```bash
pytest
```

## Rebuilding the pipeline (builder-side, optional)

Reviewers don't need this — the corpus, snapshot, and cohort are committed. The
offline pipeline that produced them (all under `carryia/pipeline/` and
`carryia/personal/`, run with the builder's keys):

```bash
python -m carryia.pipeline.validate_corpus   # ④ gate the corpus
python -m carryia.pipeline.ingest            # ⑤ build the retrieval indexes (keyless smoke)
```

---

*Status: the two-plane data pipeline, retrieval + answer evals (P0-5/6), the interface
(P0-7), monitoring (P0-8), and containerization (P0-9) are built; the whole stack comes
up with one `docker compose up` (P0-10).*
