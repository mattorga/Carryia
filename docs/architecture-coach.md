# Carryia — Runtime Coach

A support/bot-lane League of Legends **post-game coach**. Code judges the game
against a rank benchmark; retrieval pulls vetted coaching tips; the LLM only
phrases and grounds the result. Delivered as a Streamlit Q&A app.

Interactive version: open `architecture.html` in a browser
(`open architecture.html` on macOS). Click a flow tab, then step through with
→ / ← or Space. This file is the text mirror.

---

## The core idea

Two piles of data, joined by deterministic judgment:

1. **The player's game** — a committed snapshot of the subject's recent matches.
2. **A coaching corpus** — a curated, cited collection of support tips.

The app derives *what went wrong* from the game with **code** (not the LLM),
retrieves tips about those specific weaknesses, and has the LLM write the review
— constrained to the observed facts and the retrieved tips. The LLM diagnoses
and phrases; it never invents the mistake or the cause from nothing.

**Deaths are the exception that shaped the design.** A raw stat ("died at 4:12")
hides *why*. So the producer emits the death **with its timeline context**
(where, when, by whom) rather than a hardcoded cause, and the LLM diagnoses
against retrieved tips — instead of the old approach of asserting "likely
unwarded jungle path" in a rule.

---

## Components (nodes)

| Node | Role (color) | What it is |
|------|------|------------|
| **Streamlit UI** | user · mint | Post-game Q&A front end. |
| **Query Router** | app logic · sky | Classifies the question (broad vs targeted) and orchestrates. |
| **Role Profile** | seam · amber | **SEAM ①** — declarative config: which stats matter for support, weights, direction, extraction. Adding ADC = a second profile. |
| **Stat-Line Producer** | seam · amber | **SEAM ②** — turns match data into one normalized stat line. "Several games" vs "per-match" lives here, behind one interface. |
| **Weakness Ranker** | app logic · sky | Generic core: scores the stat line against the benchmark, ranks the gaps. Role- and mode-blind. |
| **Match Snapshot** | committed data · violet | The subject's N games + timeline death events. Read-only, committed. Produced offline by the Riot match ingest (`carryia/personal/`). |
| **Rank Benchmark** | committed data · violet | Reference distribution **1–2 tiers above** the subject, by role/champion. |
| **Tip Corpus** | committed data · violet | The coaching tips. Hybrid search (BM25 + local vector). Read-only at serving; append-only during authoring. |
| **Hosted LLM** | compute · magenta | OpenAI-compatible endpoint. Cheap model + reviewer's key at serving; my key when authoring. |
| **Corpus Author** | offline · orange | `author_corpus.py`. Offline, one-time, my key. Output committed. Reviewers never run it. |

### The two seams (why the shape is this way)

Everything role- and mode-specific lives in **exactly two places** — the Role
Profile and the Stat-Line Producer — and the core (ranker, retrieval, synthesis)
never branches on role or mode. Hold that line and:

- **Add ADC** = a new role profile. Nothing in the core changes.
- **Add per-match analysis** = a second producer emitting the same stat-line
  shape. Nothing in the core changes.

Thin seams, not a framework: the profile is a dataclass, the producer boundary
is one function signature.

---

## Benchmark choice: +1–2 tiers, not Challenger

The benchmark's job is to **discriminate** between the player's stats so the
ranker has something to rank. Against Challenger, every stat of a lower-ranked
player saturates ("all red") and the ranking loses its signal. Against 1–2 ranks
up, some stats sit close and a few genuinely lag — that gap structure is what
"worst stat" needs, and it's a reachable coaching target. Because the subject is
fixed at a known rank, only the bands just above them need collecting.

---

## Modes

Orthogonal to the flows. Toggle with the `O` key or the mode buttons.

| | **Serving** (runtime) | **Authoring** (offline, one-time) |
|---|---|---|
| Who runs it | The reviewer | Me |
| LLM key | Reviewer's, cheap model | My key |
| Corpus | Read-only, hybrid index | Append to `corpus.jsonl` |
| Corpus Author node | Hidden | Visible |
| Generation cost to reviewer | Only at query time | n/a |

The split exists so **ingest is deterministic and keyless** for the embeddings,
and the only thing a reviewer's key touches is the LLM at query time.

---

## Flows

### 1. Broad review — "Where can I improve?"

Open question → the app finds the weaknesses.

1. **User → Router** — `POST /ask` with an open question, `last_n` = several games.
2. **Router → Producer** — classified as broad; ranking will run.
3. **Profile → Producer** — load the support stat profile (SEAM ①).
4. **Producer → Snapshot** — read N committed games + timeline death events.
5. **Producer → Ranker** — emit one normalized stat line, averaged over N games
   (so one bad game isn't mistaken for a habit).
6. **Ranker → Benchmark** — compare vs the +1–2 tier support reference.
7. **Ranker → Corpus** — rank the gaps; one hybrid query per weakness.
8. **Corpus → LLM** — synthesize, boxed to the observed facts + retrieved tips.
9. **LLM → Router** — grounded review draft, each claim cited to a tip excerpt.
10. **Router → User** — render answer + clickable sources.

### 2. Targeted question — "Fix my vision"

Same pipeline, with the ranking step **skipped** because the player named the
target. Router sets `target = vision`; the ranker scores that single stat vs the
benchmark (giving the "18 vs 34" grounding number); retrieval and synthesis are
identical to Broad. *Specific is just Broad with one step turned off.*

### 3. Reproducible ingest (the reviewer's path)

Reads only committed data; **no generation call**.

1. **Router → Corpus** — read `corpus.jsonl`, build hybrid index with local
   fastembed + BM25. Deterministic, keyless, no LLM.
2. **Router → Snapshot** — load the committed match snapshot (no Riot key needed).
3. **Router → Benchmark** — load the committed rank benchmark.

Everything needed to answer ships in the repo; the reviewer's key is spent only
on the LLM at query time.

### 4. Corpus authoring (offline, authoring mode only)

How the corpus is built — run by me, once, and committed.

1. **Author → LLM** — walk a cleaned transcript in windows; LLM drafts tip
   records against the schema. The only place generation touches the corpus.
2. **LLM → Author** — records with tip, rationale, tags, and **character offsets**
   into the transcript (not model-written quotes).
3. **Author → Corpus** — slice the excerpt from the offsets and **assert it is a
   literal substring** of the source (no fabricated quotes), then append + commit.

---

## What this fixes vs the prior approach

- **No hardcoded causes.** Code states what happened (with context); the LLM
  diagnoses against vetted tips.
- **Role-aware from the start.** Support is measured on vision denial, wards
  cleared, CC, heal/shield, dragon participation, lane state — not ADC stats
  (cs/min, damage) that mislead for a support.
- **Extensible without a rewrite** via the two seams.
- **Reproducible for reviewers** — committed data, local embeddings, cheap
  pinned model, key used only at query time.
