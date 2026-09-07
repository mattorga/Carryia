"""Stage ⑤ Ingest -- build the retrieval indexes from the frozen corpus.

The reproducibility boundary (PLAN Step 5): ZERO LLM calls, reviewer-runnable,
local embeddings only. One `load_corpus()` pass feeds BOTH retrieval approaches
Part B compares, reusing the LLM-Zoomcamp `minsearch`
toolkit (framing reused, code rewritten -- CLAUDE.md):

  - `build_text_index`   -> keyword / BM25   (`minsearch.Index`)
  - `build_vector_index` -> vector           (`minsearch.VectorSearch` over local
                                              `fastembed` embeddings)

Field mapping: the searchable text is the distilled
`tip` ALONE -- not `source_excerpt`. `KEYWORD_FIELDS` are the filterable facets;
`tip_id` rides along so the P0-5 eval can score hits.

Indexes are built in-memory at startup (709 records is cheap) -- nothing is
persisted, so a reviewer rebuilds the world from `data/corpus.jsonl` with no key.
`fastembed` is imported lazily inside `embed_tips` so the harness (`load_corpus`
+ `main`'s plumbing) stays importable and testable without the vector model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from minsearch import Index, VectorSearch

from carryia.paths import DATA

CORPUS_PATH = DATA / "corpus.jsonl"

# Field mapping: `tip` is the SOLE retrieval surface.
TEXT_FIELDS: list[str] = ["tip"]
KEYWORD_FIELDS: list[str] = [
    "scope", "role", "phase", "champion", "source_id", "creator_id", "tip_id",
]
# Local embedding model -- fastembed, 384-dim, no API key.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class IngestError(RuntimeError):
    """The corpus could not be loaded or parsed."""


# --- load the frozen corpus --------------------------------------------------

def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    """Read `corpus.jsonl` into a list of record dicts (one per line).

    Plain plumbing -- no schema validation here (stage ④ already gated the file);
    this only deserialises. Raises `IngestError` on a malformed line so a corrupt
    corpus fails loud at ingest rather than silently retrieving nothing.
    """
    records: list[dict] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise IngestError(f"{path.name}:{lineno}: bad JSON -- {exc}") from exc
    if not records:
        raise IngestError(f"{path.name}: no records")
    return records


# --- build the two indexes ---------------------------------------------------

def build_text_index(documents: list[dict]) -> Index:
    """Fit a `minsearch.Index` for the keyword/BM25 approach.

    Searchable text is `tip` alone -- so `text_fields` is
    exactly `TEXT_FIELDS`, no `source_excerpt`; `KEYWORD_FIELDS` are the
    filterable facets (and carry `tip_id` for scoring).
    """
    index = Index(text_fields=TEXT_FIELDS, keyword_fields=KEYWORD_FIELDS)
    index.fit(documents)
    return index


def embed_tips(texts: list[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    """Embed tip strings to a `(len(texts), dim)` array -- the one place
    `fastembed` is touched, imported lazily so the rest of the module needs no
    vector model. One vector per input text, in order."""
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=model_name)
    return np.array(list(model.embed(texts)))


def build_vector_index(
    documents: list[dict],
    embed: Callable[[list[str]], np.ndarray] = embed_tips,
) -> VectorSearch:
    """Fit a `minsearch.VectorSearch` for the vector approach: embed each
    document's `tip` via `embed`, then fit against the documents. `embed` is
    injectable so tests pass a deterministic fake (no model download); it
    defaults to `embed_tips` in production."""
    vectors = embed([doc["tip"] for doc in documents])
    index = VectorSearch(keyword_fields=KEYWORD_FIELDS)
    index.fit(vectors, documents)
    return index


# --- CLI smoke: rebuild both indexes, no key ---------------------------------

def main(argv: list[str] | None = None) -> int:
    """Build both indexes from the committed corpus and report -- the ⑤
    reviewer-runnable smoke check (`docker compose` runs this before Streamlit).
    Returns a process exit code."""
    try:
        documents = load_corpus()
    except IngestError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1
    build_text_index(documents)
    build_vector_index(documents)
    print(f"ingested {len(documents)} tips -> keyword + vector indexes (no key)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
