"""Spec for carryia/pipeline/clean.py -- stage ②, the transcript/source cleaner.

TDD state on first run (the house three-state, as in test_validate_corpus.py):
  - GREEN: the harness -- parse_raw, and main()'s write/aggregate/exit-code
    plumbing (with the transforms monkeypatched) -- is implemented, so these
    pass now and lock it against regression.
  - RED: strip_artifacts, apply_repairs, and verify_diff (the guard) are
    NotImplementedError stubs, so their contract tests -- and the golden
    end-to-end + guard tests that lean on them -- fail until you write each body.
    That failing test IS the spec you code to.

Deterministic ②: the cleaner strips caption artifacts
and applies a per-source, front-matter-declared repair map -- nothing an LLM
touches. The guard proves no token is ever silently dropped or invented.
"""

import json
from pathlib import Path

import pytest

from carryia.pipeline import clean

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "fixtures" / "golden_raw.md"


# --- GREEN: parse_raw -------------------------------------------------------

def test_parse_raw_splits_front_matter_and_body():
    front, front_text, body = clean.parse_raw(GOLDEN.read_text())
    assert front["source_id"] == "fixture--golden-clip-01"
    assert front["medium"] == "video"
    assert front_text.startswith("---\n") and front_text.rstrip().endswith("---")
    assert body.lstrip().startswith("[0] ")


def test_parse_raw_reads_the_per_source_repairs_list():
    front, _, _ = clean.parse_raw(GOLDEN.read_text())
    assert ["blitz crank", "Blitzcrank"] in front["repairs"]
    assert ["lucían", "Lucian"] in front["repairs"]


def test_parse_raw_keeps_colons_inside_a_value():
    # A URL value has its own colons; only the first colon splits key from value.
    front, _, _ = clean.parse_raw(GOLDEN.read_text())
    assert front["source_url"].startswith("https://")


def test_parse_raw_rejects_a_file_with_no_front_matter():
    with pytest.raises(clean.CleanError):
        clean.parse_raw("[0] no front matter here\n")


# --- GREEN: main() writes, aggregates, exits (transforms stubbed out) --------

def _identity_transforms(monkeypatch):
    """Make the two cleaners no-ops and the guard a pass, so main()'s plumbing can
    be exercised without the (still-unwritten) transform bodies."""
    monkeypatch.setattr(clean, "strip_artifacts", lambda text: (text, []))
    monkeypatch.setattr(clean, "apply_repairs", lambda text, repairs: (text, []))
    monkeypatch.setattr(clean, "verify_diff", lambda before, after, changes: [])


def test_main_writes_clean_md_and_repairs_json_named_by_source_id(monkeypatch, tmp_path, capsys):
    _identity_transforms(monkeypatch)
    assert clean.main([str(GOLDEN), "--out-dir", str(tmp_path)]) == 0
    # Named by source_id (what ④ looks up), NOT the raw filename stem.
    assert (tmp_path / "fixture--golden-clip-01.md").exists()
    assert (tmp_path / "fixture--golden-clip-01.repairs.json").exists()
    assert "golden_raw.md" not in {p.name for p in tmp_path.glob("*.md")}
    assert "0 failed" in capsys.readouterr().out


def test_main_clean_output_preserves_front_matter_and_anchors(monkeypatch, tmp_path):
    _identity_transforms(monkeypatch)
    clean.main([str(GOLDEN), "--out-dir", str(tmp_path)])
    out = (tmp_path / "fixture--golden-clip-01.md").read_text()
    assert out.startswith("---\n")
    assert "\n[0] " in out and "\n[30] " in out   # every video anchor survives


def test_main_returns_1_when_a_source_fails_the_guard(monkeypatch, tmp_path, capsys):
    _identity_transforms(monkeypatch)
    monkeypatch.setattr(clean, "verify_diff", lambda *a, **k: ["token 'bot' vanished"])
    assert clean.main([str(GOLDEN), "--out-dir", str(tmp_path)]) == 1
    assert "FAIL" in capsys.readouterr().err
    assert not list(tmp_path.glob("*.md"))        # a guard failure writes nothing


# --- RED: strip_artifacts ---------------------------------------------------

def test_strip_artifacts_removes_a_caption_tag_and_reports_it():
    text, removed = clean.strip_artifacts("[music] okay so lets talk about bot lane")
    assert text == "okay so lets talk about bot lane"   # tag gone, no dangling space
    assert removed == ["[music]"]


def test_strip_artifacts_is_a_no_op_on_clean_text():
    text, removed = clean.strip_artifacts("you just give the wave")
    assert text == "you just give the wave"
    assert removed == []


# --- RED: apply_repairs -----------------------------------------------------

def test_apply_repairs_fixes_the_declared_proper_nouns():
    repairs = [("blitz crank", "Blitzcrank"), ("lucían", "Lucian")]
    text, applied = clean.apply_repairs("path up with blitz crank and lucían", repairs)
    assert text == "path up with Blitzcrank and Lucian"
    assert ("blitz crank", "Blitzcrank") in applied
    assert ("lucían", "Lucian") in applied


def test_apply_repairs_logs_one_entry_per_occurrence():
    text, applied = clean.apply_repairs("blitz crank then blitz crank", [("blitz crank", "Blitzcrank")])
    assert text == "Blitzcrank then Blitzcrank"
    assert applied.count(("blitz crank", "Blitzcrank")) == 2


def test_apply_repairs_that_match_nothing_log_nothing():
    text, applied = clean.apply_repairs("no proper nouns here", [("failites", "Faelights")])
    assert text == "no proper nouns here"
    assert applied == []


# --- RED: verify_diff (the guard) -------------------------------------------

def test_verify_diff_passes_when_a_removal_is_logged():
    changes = [{"kind": "artifact", "line": 0, "before": "[music]", "after": ""}]
    assert clean.verify_diff("[0] [music] give the wave", "[0] give the wave", changes) == []


def test_verify_diff_passes_when_a_repair_is_logged():
    changes = [{"kind": "repair", "line": 0, "before": "blitz crank", "after": "Blitzcrank"}]
    assert clean.verify_diff("[0] with blitz crank", "[0] with Blitzcrank", changes) == []


def test_verify_diff_ignores_punctuation_fused_to_a_repaired_word():
    # 'failites.' -> 'Faelights.' is a clean word swap; the trailing period is its
    # own token and must not read as an unlogged change (regression: whitespace
    # tokenising fused the period on and tripped the guard on real data).
    changes = [{"kind": "repair", "line": 0, "before": "failites", "after": "Faelights"}]
    assert clean.verify_diff("[0] using failites.", "[0] using Faelights.", changes) == []
    changes = [{"kind": "repair", "line": 0, "before": "pryo", "after": "prio"}]
    assert clean.verify_diff("[0] get pryo?", "[0] get prio?", changes) == []


def test_verify_diff_flags_an_unlogged_deletion():
    assert clean.verify_diff("[0] give the wave", "[0] give wave", []) != []


def test_verify_diff_flags_an_unlogged_insertion():
    assert clean.verify_diff("[0] give wave", "[0] give the wave", []) != []


# --- RED: golden end-to-end + guard halt (PLAN Step 2 done-when) -------------

def test_clean_source_cleans_the_golden_fixture():
    clean_text, log = clean.clean_source(GOLDEN.read_text())
    assert "[music]" not in clean_text                 # artifact stripped
    assert "path up with Blitzcrank and Lucian" in clean_text
    assert "[0] okay so lets talk about bot lane" in clean_text   # anchor + tidy text
    assert log["source_id"] == "fixture--golden-clip-01"
    assert len(log["changes"]) == 3                    # 1 artifact + 2 repairs


def test_clean_source_is_deterministic():
    a, _ = clean.clean_source(GOLDEN.read_text())
    b, _ = clean.clean_source(GOLDEN.read_text())
    assert a == b


def test_clean_source_halts_when_a_transform_deletes_without_logging(monkeypatch):
    # A saboteur strip that eats a token but reports nothing -- the guard must catch it.
    monkeypatch.setattr(clean, "strip_artifacts", lambda text: (text.replace(" bot", "", 1), []))
    monkeypatch.setattr(clean, "apply_repairs", lambda text, repairs: (text, []))
    with pytest.raises(clean.CleanError):
        clean.clean_source(GOLDEN.read_text())
