# Carryia

**A post-game coach for League of Legends support mains.** It takes a question about a
player's recent ranked games and returns a specific, actionable answer that is grounded in
*their own* stats and cited to a reputable coaching source, not a generic guide or a
model's unsourced opinion.

The name is a pun that comes from the defending Worlds Champion T1 *Keria*.

---

## Contents
### Run the app
- [Quickstart](#quickstart)
- [Configuration](#configuration)

### Project details
- [The problem](#problem-statement)
- [Personal stats by phase](#personal-stats-by-phase) 
- [How it works](#how-it-works)
- [Monitoring](#monitoring)
- [Evaluation](#evaluation)
- [Sources](#sources)

## Quickstart

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

### Option B — local Python

**Prerequisite:** Python 3.12.
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
streamlit run carryia/serve/app.py

uv pip install -r requirements.txt && uv pip install -e . # if using uv
```

This runs the app without the monitoring stack — it answers normally, monitoring is
just off (no Postgres required). To enable monitoring locally, point `DATABASE_URL` at
your own Postgres (see [`.env.example`](.env.example)).

It reads only committed data and spends your key solely on the LLM at query time — no
Riot key needed.

## Configuration

All configuration is via `.env` (copied from [`.env.example`](.env.example)). The
answer generator runs on one of two backends, chosen by `CARRYIA_LLM_BACKEND`:

| Backend | Set | Auth |
|---------|-----|------|
| `anthropic` *(default)* | reviewers use this | `ANTHROPIC_API_KEY` |
| `bedrock` | Amazon Bedrock, billed to AWS | `AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION` with Haiku 4.5 access |

---

## Problem Statement

Support and bot-lane players who want to climb rarely get useful feedback:

- **No feedback on their own games** — they're left guessing which of a thousand
  contradictory opinions to trust.
- **Generic guides don't reference what they actually did.**
- **Human coaching is expensive and scarce.**

Carryia answers a player's question by combining two things a generic bot can't:

- **What the player actually does** — a committed snapshot of their recent support
  games, summarised **by lane phase** (laning / mid / late) rather than as whole-game
  averages, and placed against a cohort of same-rank peers.
- **What good looks like** — a curated corpus of coaching tips distilled from
  reputable creators, each carrying a verbatim source quote and citation.

> Every recommendation is grounded strictly in retrieved tips and cited to its source,
so the advice is specific to how the player plays *and* defensibly authoritative.

---

### Personal stats by phase

| Phase | Game-time window |
|-------|------------------|
| `laning` | before 14:00 |
| `mid` | 14:00 – 25:00 |
| `late` | 25:00 onward |

What's phased vs. whole-game (every value is a **per-game mean**):

| Scope | Metrics |
|-------|---------|
| **Per phase** — mean count per game, in each of laning / mid / late | deaths · ward activity (placed + cleared) · objective participation (epics + towers + grubs) |
| **Whole-game** — one per-game mean each | KDA · vision score/min · kill participation · effective heal+shield · team damage % · enemy immobilizations · ward takedowns |

---

## How it works

Two data planes feed one RAG core:

| Plane | Holds | Source |
|-------|-------|--------|
| **Personal** | the player's stats — most **split by lane phase** (see [below](#personal-stats-by-phase)), the rest whole-game | committed match snapshot + a committed Silver-cohort benchmark |
| **Knowledge** | cited coaching authority (what to do) | 709-tip corpus distilled from 7 creators |

At query time:

1. The snapshot is summarised into a **personal-context block** — the player placed
   against the cohort by percentile, **per lane phase**.
2. The question is **grounded** in that block.
3. The **hybrid retriever** (keyword + vector, RRF-fused — the eval winner) retrieves
   the most relevant tips.
4. The **LLM composes** a grounded, cited answer over them.
5. Each answered question is timed, cost-tracked, relevance-judged, and written to
   Postgres for a **Grafana dashboard** (see [Monitoring](#monitoring)).

See the interactive diagrams under [`docs/`](docs/) (`architecture-system.html`,
`architecture-coach.html`).

## Monitoring

Under `docker compose up`, every answered question is instrumented into Postgres and
charted by **Grafana** (<http://localhost:3000>, admin / admin) on an auto-provisioned
dashboard. Two tables back it:

| Table | One row per |
|-------|-------------|
| `conversations` | ask |
| `feedback` | 👍/👎 |

The dashboard covers the full course metric set across 8 panels:

- total answers, avg response time, total cost, avg tokens (at-a-glance stats)
- cost and response time **over time**
- **answer relevance** — LLM-as-judge categorizes the answer into RELEVANT / PARTLY / NON
- **user feedback** — the thumbs breakdown

Monitoring is best-effort: if the judge or DB is unavailable the answer is still
returned, and a bare `streamlit run` (Option B) simply runs with monitoring off.

## Evaluation

Both graded evaluations are committed as runnable notebooks under `carryia/eval/`:

- **Retrieval (P0-5)** — `eval.ipynb` compares keyword, vector, and hybrid retrieval
  over the ground-truth set (hit-rate / MRR). **Vector beats keyword; hybrid edges
  vector** — the chosen approach is justified by the comparison.
- **Answer generation (P0-6)** — `eval_answer.py` runs a pairwise LLM-as-judge over
  two prompt variants across all 709 questions. **The explanatory ("WHY") coach wins,
  p < 0.001** — the shipped variant.

## Sources

These are the sources I have used for the corpus.

| Creator | Medium | Tips | Sample source |
|---------|--------|-----:|---------------|
| Mobalytics | written · mobalytics.gg | 347 | [Warding guide](https://mobalytics.gg/lol/guides/warding-guide) |
| Skill Capped | video · YouTube | 138 | [video](https://www.youtube.com/watch?v=HKqhMssvwMI) |
| eiensiei | written · MOBAFire | 108 | [Support guide](https://www.mobafire.com/league-of-legends/build/13-3-eiensieis-guide-to-support-598073) |
| Shodesu | video · YouTube | 75 | [video](https://www.youtube.com/watch?v=FytX3ECnvzE) |
| Three Minute LoL | video · YouTube | 20 | [video](https://www.youtube.com/watch?v=PsD5QNyBkoY) |
| xPetu | video · YouTube | 17 | [video](https://www.youtube.com/watch?v=76P7yuf_50Y) |
| Herold NA | video · YouTube | 4 | [video](https://www.youtube.com/watch?v=-hA68UuaT9g) |

Total: **709** tips across 7 creators.