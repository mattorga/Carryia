"""Spec for carryia/pipeline/validate_corpus.py -- stage ④, the corpus gate.

TDD state on first run (the house three-state, as in test_phases.py):
  - GREEN: the harness -- load_corpus + main()'s aggregate/exit-code plumbing --
    is implemented, so those tests pass now and lock it against regression.
  - RED: every gate_* body is a NotImplementedError stub, so its contract test
    fails until you write the gate. That failing test IS the spec you code to.
  - SKIPPED: the gate_coverage all-phase test, and the quote/anchor tests that
    need stage-② clean output -- each waits on a decision or a dependency that is
    deliberately yours. Un-skip them as you get there.
"""

import json
from pathlib import Path

import pytest

from carryia.pipeline import validate_corpus as vc

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures" / "corpus.sample.jsonl"


def _sample_rows():
    return [json.loads(l) for l in SAMPLE.read_text().splitlines() if l.strip()]


# --- GREEN: the loader ------------------------------------------------------

def test_load_corpus_parses_the_sample_fixture():
    rows = vc.load_corpus(SAMPLE)
    assert len(rows) == 3
    assert all(isinstance(r, dict) for r in rows)


def test_load_corpus_skips_blank_lines(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"a": 1}\n\n  \n{"b": 2}\n')
    assert vc.load_corpus(p) == [{"a": 1}, {"b": 2}]


def test_load_corpus_is_fatal_on_unparseable_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"a": 1}\n{not json}\n')
    with pytest.raises(SystemExit):
        vc.load_corpus(p)


# --- GREEN: main() aggregation + exit codes (gates stubbed out) --------------

def _pass_all(monkeypatch):
    for name in ("gate_schema", "gate_count", "gate_creator_share",
                 "gate_coverage", "gate_quote", "gate_anchor"):
        monkeypatch.setattr(vc, name, lambda *a, **k: [])


def test_main_returns_0_when_every_gate_passes(monkeypatch, capsys):
    _pass_all(monkeypatch)
    assert vc.main([str(SAMPLE)]) == 0
    assert "0 gate(s) failed" in capsys.readouterr().out


def test_main_returns_1_when_any_gate_reports_a_violation(monkeypatch, capsys):
    _pass_all(monkeypatch)
    monkeypatch.setattr(vc, "gate_creator_share", lambda *a, **k: ["skill-capped at 71%"])
    assert vc.main([str(SAMPLE)]) == 1
    out = capsys.readouterr().out
    assert "FAIL creator_share" in out and "skill-capped at 71%" in out


# --- RED: gate contracts (fail until you implement each gate body) -----------

def test_gate_schema_accepts_the_valid_sample():
    assert vc.gate_schema(_sample_rows()) == []


def test_gate_schema_flags_an_out_of_domain_enum():
    rows = _sample_rows()
    rows[0]["scope"] = "not-a-scope"
    assert vc.gate_schema(rows) != []


def test_gate_schema_flags_a_tip_id_that_drifted_from_its_content():
    rows = _sample_rows()
    rows[0]["tip_id"] = "tip-000000000000"      # no longer matches derive_tip_id
    assert vc.gate_schema(rows) != []


def test_gate_count_flags_a_short_corpus():
    assert vc.gate_count(_sample_rows()) != []   # 3 < MIN_RECORDS


def test_gate_creator_share_flags_a_dominant_creator():
    rows = [{"creator_id": "a"} for _ in range(9)] + [{"creator_id": "b"}]
    assert vc.gate_creator_share(rows) != []     # a == 90% > 60%


def test_gate_creator_share_passes_a_balanced_corpus():
    rows = [{"creator_id": c} for c in ("a", "a", "b", "b", "c", "c")]
    assert vc.gate_creator_share(rows) == []


# --- coverage: the all-phase decision, resolved to (a) 2026-08-08 ------------

def test_gate_coverage_all_phase_tip_fills_every_temporal_cell():
    # Decision (a): an ALL-tagged tip counts toward every temporal phase's cell
    # for its scope -- mirroring matches(), where ALL answers every phase query.
    rows = [{"scope": "support", "phase": "all"} for _ in range(vc.MIN_PER_CELL)]
    violations = vc.gate_coverage(rows)
    assert not any("support" in v for v in violations)   # 10 ALL tips fill all 3 support cells
    assert any("fundamental" in v for v in violations)   # fundamental cells still empty -> flagged


def test_gate_coverage_temporal_tip_fills_only_its_own_cell():
    rows = [{"scope": "support", "phase": "laning"} for _ in range(vc.MIN_PER_CELL)]
    violations = vc.gate_coverage(rows)
    assert not any("(support, laning)" in v for v in violations)  # its own cell is satisfied
    assert any("(support, mid)" in v for v in violations)         # mid/late are NOT filled by it


# --- quote / anchor: against a stage-② clean file (tmp fixtures) -------------

def test_gate_quote_requires_a_literal_substring(tmp_path):
    (tmp_path / "src.md").write_text("preamble\n[420] you just give the wave\ntail\n")
    ok = [{"tip_id": "t1", "source_id": "src", "source_excerpt": "give the wave"}]
    bad = [{"tip_id": "t2", "source_id": "src", "source_excerpt": "a phrase never present"}]
    missing = [{"tip_id": "t3", "source_id": "absent", "source_excerpt": "x"}]
    assert vc.gate_quote(ok, tmp_path) == []
    assert vc.gate_quote(bad, tmp_path) != []
    assert vc.gate_quote(missing, tmp_path) != []       # clean source not found is a violation


def test_gate_anchor_resolves_video_and_written_anchors(tmp_path):
    (tmp_path / "vid.md").write_text("---\n---\n\n[0] intro\n[420] give the wave\n")
    (tmp_path / "art.md").write_text("---\n---\n\n## How to Ward During Laning\n\nbody\n")
    v_ok = [{"tip_id": "v1", "source_id": "vid", "medium": "video", "timestamp": 420, "section": None}]
    v_bad = [{"tip_id": "v2", "source_id": "vid", "medium": "video", "timestamp": 999, "section": None}]
    a_ok = [{"tip_id": "a1", "source_id": "art", "medium": "written", "timestamp": None, "section": "how-to-ward-during-laning"}]
    a_bad = [{"tip_id": "a2", "source_id": "art", "medium": "written", "timestamp": None, "section": "not-a-heading"}]
    assert vc.gate_anchor(v_ok, tmp_path) == []
    assert vc.gate_anchor(v_bad, tmp_path) != []
    assert vc.gate_anchor(a_ok, tmp_path) == []
    assert vc.gate_anchor(a_bad, tmp_path) != []
