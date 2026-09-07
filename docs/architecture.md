# Carryia — Whole System

The system-altitude view. For the runtime coach's internals (how a question
becomes an answer), see `architecture.md` / `architecture.html` — the **Coaching
App** node here expands into that 10-step diagram.

Open the interactive version: `open architecture-system.html`. Toggle **Run-time**
to hide everything a reviewer doesn't touch.

---

## The one idea: subsystems connect through committed artifacts

There are four subsystems, and they **do not call each other live**. They hand
off through three files committed to git:

```
  PRODUCERS            ARTIFACTS (hub)          CONSUMERS
  (offline, my key)    (committed to git)
  ─────────────────    ───────────────────      ─────────────────────
  Riot Ingest      →   Match Snapshot       →   Coaching App (runtime)
  Benchmark Crawl  →   Rank Benchmark       →   Coaching App (runtime)
  Corpus Author    →   Tip Corpus           →   Coaching App + Retrieval Eval
```

Two shared dependencies cut across the middle:

- **The committed artifacts** — the data hub. Producers write them once; the
  runtime and the evaluators read them.
- **The hosted LLM** — one OpenAI-compatible endpoint, used in three places:
  authoring (drafting tips, my key), serving (synthesis, reviewer's key), and
  answer eval (LLM-as-judge).

This hand-off-through-files design is what makes the project **reproducible**: a
reviewer runs only the runtime slice, reading committed data, and their key is
spent only on the LLM at query time. Nothing they run calls Riot, crawls, or
generates the corpus.

---

## The four subsystems

### 1. Data production (offline · orange)
Run by me, once, with my keys. Output committed.

| Job | Produces | Notes |
|-----|----------|-------|
| **Riot Ingest** | Match Snapshot | match-v5, regional routing, rate limits; personal plane. |
| **Benchmark Crawl** | Rank Benchmark | games 1–2 tiers above the subject, by role/champion. |
| **Corpus Author** | Tip Corpus | transcript prep + windowed LLM drafting → validate → commit. |

### 2. Runtime coach (serving · sky)
The **Coaching App** — router, stat-line producer, weakness ranker, retrieval,
synthesis. Reads all three artifacts; calls the LLM with the reviewer's key.
Detailed in `architecture.html`.

### 3. Evaluation (offline · magenta) — graded
| Subsystem | Rubric | What it does |
|-----------|--------|--------------|
| **Retrieval Eval** | P0-5 | ≥2 retrieval approaches over ground-truth queries; NDCG / MRR / hit against *sets* of acceptable tips. Points are for comparing approaches, not a threshold. |
| **Answer Eval** | P0-6 | Runs the app over an eval set; LLM-as-judge scores answers against persona criteria. |

### 4. Packaging & deploy (orange)
`docker-compose up` brings up everything (2-point containerization bar); the app
also deploys to Streamlit Community Cloud (+2, cheapest bonus).

---

## Lifecycle flows (the interactive tabs)

1. **Build the data** — Riot Ingest → Snapshot; Benchmark Crawl → Benchmark;
   Author ↔ LLM → Corpus. Produces the three committed artifacts.
2. **Serve a review** — Reviewer → App → {Snapshot, Benchmark, Corpus} → LLM →
   Reviewer. (App expands to the runtime diagram.)
3. **Evaluate** — Retrieval Eval reads Corpus; Answer Eval drives the App and
   scores via the LLM judge.
4. **Package & deploy** — compose brings the App up; deploy to the cloud.

## Build-time vs Run-time (the mode toggle)

- **Build-time** — the whole system: producers, artifacts, runtime, eval, deploy.
- **Run-time** — hides everything a reviewer doesn't touch, leaving the
  reproducible slice: Reviewer, App, Snapshot, Benchmark, Corpus, LLM.

The gap between the two modes *is* the reproducibility boundary.
