"""Spec for carryia/pipeline/ingest.py -- stage ⑤, the retrieval-index builder.

TDD state on first run (the house three-state, as in test_clean.py):
  - GREEN: the harness -- `load_corpus` and `main()`'s plumbing (the builders
    monkeypatched out) -- is implemented, so these pass now and lock it.
  - RED: `build_text_index`, `build_vector_index`, and `embed_tips` are
    NotImplementedError stubs, so their contract tests fail until you write each
    body. That failing test IS the spec you code to.
  - SKIPPED: the real `fastembed` round-trip (`test_embed_tips_real_model`) is
    skipped by default -- it downloads the pinned model, so it's opt-in
    (`RUN_FASTEMBED=1 pytest ...`) and documents the embedding contract without
    paying the download on every run.

Field mapping: the searchable text is `tip` ALONE.
`test_text_index_searches_tip_only` pins that -- flip it and the graded Part B
comparison changes. The vector tests inject a fake embedder so no model is
downloaded; the one real call lives behind the skip above.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from carryia.pipeline import ingest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures" / "corpus.sample.jsonl"


@pytest.fixture
def docs():
    return ingest.load_corpus(SAMPLE)


def _fake_embed(texts):
    """Deterministic, content-derived vectors -- no model download. Each text
    maps to a stable 16-D point from its own hash (position-independent), so an
    identical text embeds identically and an exact-match query (cosine 1.0) lands
    on its own document."""
    def vec(text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()[:16]
        return np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    return np.array([vec(t) for t in texts], dtype=np.float32)


# --- GREEN: load_corpus ------------------------------------------------------

def test_load_corpus_reads_every_record():
    records = ingest.load_corpus(SAMPLE)
    assert len(records) == 3
    assert {r["tip_id"] for r in records} == {
        "tip-796fa522de2d", "tip-4db6e10a4e1f", "tip-3b1d4c999cf8",
    }
    assert records[0]["tip"].startswith("Drop a control ward")


def test_load_corpus_skips_blank_lines(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(SAMPLE.read_text() + "\n\n")
    assert len(ingest.load_corpus(p)) == 3


def test_load_corpus_rejects_bad_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"tip_id": "ok"}\nnot valid json\n')
    with pytest.raises(ingest.IngestError):
        ingest.load_corpus(p)


def test_load_corpus_rejects_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n")
    with pytest.raises(ingest.IngestError):
        ingest.load_corpus(p)


# --- GREEN: main() plumbing (builders stubbed) -------------------------------

def test_main_builds_both_indexes_and_exits_zero(monkeypatch, capsys):
    calls = {}
    monkeypatch.setattr(ingest, "load_corpus", lambda: ["d1", "d2"])
    monkeypatch.setattr(ingest, "build_text_index",
                        lambda d: calls.__setitem__("text", d))
    monkeypatch.setattr(ingest, "build_vector_index",
                        lambda d: calls.__setitem__("vector", d))
    assert ingest.main() == 0
    assert calls["text"] == ["d1", "d2"]
    assert calls["vector"] == ["d1", "d2"]
    assert "2 tips" in capsys.readouterr().out


def test_main_reports_failure_on_bad_corpus(monkeypatch, capsys):
    def boom():
        raise ingest.IngestError("nope")
    monkeypatch.setattr(ingest, "load_corpus", boom)
    assert ingest.main() == 1
    assert "ingest failed" in capsys.readouterr().err


# --- RED: build_text_index (keyword / BM25) ----------------------------------

def test_text_index_searches_tip_only(docs):
    # The decision: `tip` is the sole retrieval surface.
    index = ingest.build_text_index(docs)
    assert index.text_fields == ["tip"]
    assert "source_excerpt" not in index.text_fields


def test_text_index_finds_a_tip_by_its_words(docs):
    index = ingest.build_text_index(docs)
    results = index.search("control ward before recall", num_results=3)
    assert results
    assert results[0]["tip_id"] == "tip-796fa522de2d"


def test_text_index_keeps_tip_id_for_scoring(docs):
    index = ingest.build_text_index(docs)
    hit = index.search("Nautilus dredge line cooldown", num_results=1)[0]
    assert "tip_id" in hit


# --- RED: build_vector_index (vector, fake embedder) -------------------------

def test_vector_index_finds_the_matching_tip(docs):
    index = ingest.build_vector_index(docs, embed=_fake_embed)
    query = _fake_embed([docs[2]["tip"]])[0]
    hit = index.search(query, num_results=1)[0]
    assert hit["tip_id"] == docs[2]["tip_id"]


def test_vector_index_embeds_the_tip_field(docs):
    seen = {}

    def spy(texts):
        seen["texts"] = texts
        return _fake_embed(texts)

    ingest.build_vector_index(docs, embed=spy)
    assert seen["texts"] == [d["tip"] for d in docs]


# --- SKIPPED: the real embedding round-trip (opt-in; downloads the model) -----

@pytest.mark.skipif(os.environ.get("RUN_FASTEMBED") != "1",
                    reason="downloads the pinned fastembed model; opt-in")
def test_embed_tips_real_model(docs):
    vectors = ingest.embed_tips([d["tip"] for d in docs])
    assert len(vectors) == len(docs)
    assert len(vectors[0]) == 384  # BAAI/bge-small-en-v1.5
