"""Spec for carryia/schema.py -- the corpus record contract + the tip_id freeze point.

TDD state on first run:
  - `Scope`, `Medium`, `CorpusRecord`, and `derive_tip_id` are all implemented,
    so every test here is GREEN immediately -- they are regression locks. The
    ones that matter most guard `derive_tip_id`: it is the pipeline's FREEZE
    POINT (ground_truth.jsonl references tip_ids), so a future refactor that
    quietly changes an id -- most insidiously by reaching for the salted built-in
    `hash()` -- must trip a test here, not rot ground truth in silence.
"""

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from datetime import date
from pathlib import Path

import pytest

from carryia.schema import CorpusRecord, Medium, Scope, _normalise_tip, derive_tip_id
from carryia.phases import Phase

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures" / "corpus.sample.jsonl"


# --- enums serialise straight to their strings ------------------------------

def test_scope_and_medium_are_their_lowercase_strings():
    assert (Scope.FUNDAMENTAL, Scope.SUPPORT) == ("fundamental", "support")
    assert (Medium.VIDEO, Medium.WRITTEN) == ("video", "written")


def test_enums_serialise_straight_to_json():
    assert json.dumps({"scope": Scope.SUPPORT, "medium": Medium.WRITTEN}) == (
        '{"scope": "support", "medium": "written"}'
    )


# --- CorpusRecord shape (locks the one-home schema) ----------------

def test_corpus_record_has_the_fifteen_schema_fields_in_order():
    names = [f.name for f in fields(CorpusRecord)]
    assert names == [
        "tip_id", "tip", "rationale", "scope", "role", "champion", "phase",
        "source_id", "creator_id", "medium", "source_url", "source_excerpt",
        "timestamp", "section", "retrieved_at",
    ]


def test_corpus_record_is_frozen():
    rec = CorpusRecord(
        tip_id="x", tip="t", rationale="r", scope=Scope.SUPPORT, role="support",
        champion=None, phase=Phase.ALL, source_id="s", creator_id="c",
        medium=Medium.WRITTEN, source_url="u", source_excerpt="e",
        timestamp=None, section="sec", retrieved_at=date(2026, 8, 8),
    )
    with pytest.raises(FrozenInstanceError):
        rec.tip = "mutated"


# --- derive_tip_id: the freeze point ----------------------------------------

def test_tip_id_has_the_expected_shape():
    tid = derive_tip_id("some tip", "some_source")
    assert tid.startswith("tip-")
    assert len(tid) == len("tip-") + 12
    int(tid.removeprefix("tip-"), 16)  # the suffix is hex


def test_tip_id_is_idempotent():
    args = ("Ward before you recall.", "mobalytics_warding-guide")
    assert derive_tip_id(*args) == derive_tip_id(*args)


def test_same_tip_from_different_sources_gets_different_ids():
    # no-cull keeps corroborating records distinct; P0-5 scores against a SET.
    a = derive_tip_id("Ward before you recall.", "mobalytics_warding-guide")
    b = derive_tip_id("Ward before you recall.", "skill-capped_why-you-suck-at-support")
    assert a != b


@pytest.mark.parametrize(
    "variant",
    [
        "Ward before you recall.",
        "  Ward   before  you recall.  ",   # collapsed whitespace
        "ward before you recall.",           # case
        "WARD BEFORE YOU RECALL.",
    ],
)
def test_cosmetic_rewording_keeps_the_same_id(variant):
    canonical = derive_tip_id("Ward before you recall.", "src")
    assert derive_tip_id(variant, "src") == canonical


def test_delimiter_blocks_the_concatenation_collision():
    # ("ab","c") vs ("a","bc") must not hash to the same basis.
    assert derive_tip_id("ab", "c") != derive_tip_id("a", "bc")


def test_normalise_tip_collapses_whitespace_and_case():
    assert _normalise_tip("  Foo   Bar\tBaz\n") == "foo bar baz"


def _derive_in_subprocess(tip: str, source_id: str, seed: str) -> str:
    """Run derive_tip_id in a fresh interpreter under a chosen PYTHONHASHSEED."""
    code = (
        "from carryia.schema import derive_tip_id; "
        f"print(derive_tip_id({tip!r}, {source_id!r}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )
    return out.stdout.strip()


def test_tip_id_is_stable_across_processes_and_hash_seeds():
    # THE guard against a salted-hash regression: different PYTHONHASHSEEDs, in
    # separate interpreters, must still agree -- and agree with the in-process id.
    in_proc = derive_tip_id("Ward before you recall.", "src")
    seed_1 = _derive_in_subprocess("Ward before you recall.", "src", "1")
    seed_999 = _derive_in_subprocess("Ward before you recall.", "src", "999")
    assert in_proc == seed_1 == seed_999


# --- the sample fixture is valid + self-consistent --------------------------

def _sample_rows():
    return [json.loads(l) for l in SAMPLE.read_text().splitlines() if l.strip()]


def test_sample_fixture_rows_match_the_schema_shape():
    schema_keys = {f.name for f in fields(CorpusRecord)}
    for i, row in enumerate(_sample_rows()):
        assert set(row) == schema_keys, f"row {i} key mismatch"
        Scope(row["scope"]); Medium(row["medium"]); Phase(row["phase"])
        assert (row["timestamp"] is None) ^ (row["section"] is None)
        assert (row["medium"] == "video") == (row["timestamp"] is not None)


def test_sample_fixture_tip_ids_are_self_consistent():
    # Each committed tip_id equals what derive_tip_id would produce -- so a
    # hand-edit that drifts an id from its content is caught.
    for row in _sample_rows():
        assert row["tip_id"] == derive_tip_id(row["tip"], row["source_id"])
