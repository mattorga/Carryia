"""Spec for carryia/pipeline/author_corpus.py -- stage ③, THE CLERK.

author_corpus.py is deterministic assembly, not generation (the coaching judgment
lives upstream in the distill step), so this whole spec is
GREEN from the start -- regression locks on the mechanical contract:

  - the QUOTE GUARD drops any draft whose excerpt isn't a literal substring;
  - the ANCHOR is snapped from where the excerpt lands (video [NN] / written
    heading), derived the same way ④'s gate_anchor checks it;
  - the assembled corpus PASSES stage ④'s schema / quote / anchor gates -- ③ is
    built against ④ as its executable target (gate-before-producer).
"""

import json
from pathlib import Path

import pytest

from carryia.pipeline import author_corpus as ac
from carryia.pipeline import clean
from carryia.pipeline import validate_corpus as vc
from carryia.schema import Medium, derive_tip_id

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "tests" / "fixtures" / "clean"
VIDEO_ID = "fixture--clean-video"
WRITTEN_ID = "fixture--clean-written"


def _body(source_id: str) -> tuple[dict, str]:
    front, _, body = clean.parse_raw((CLEAN_DIR / f"{source_id}.md").read_text())
    return front, body


def _draft(**over) -> dict:
    d = {"tip": "tip text", "rationale": "why", "scope": "support",
         "phase": "laning", "champion": None, "excerpt": ""}
    d.update(over)
    return d


# --- anchor derivation ------------------------------------------------------

def test_video_timestamp_finds_the_caption_line():
    _, body = _body(VIDEO_ID)
    assert ac.video_timestamp(body, "you just give the wave and reset") == 10
    assert ac.video_timestamp(body, "help your jungler take grubs") == 20


def test_video_timestamp_none_when_excerpt_crosses_a_caption_boundary():
    _, body = _body(VIDEO_ID)
    # A substring of the file, but it straddles [10] -> [20], so no single caption
    # line contains it -> None -> the record is dropped.
    spanning = "and reset\n[20] roam mid"
    assert spanning in body
    assert ac.video_timestamp(body, spanning) is None


def test_written_section_uses_nearest_preceding_heading():
    _, body = _body(WRITTEN_ID)
    section = ac.written_section(body, "it is your only engage and your only way out")
    assert section == vc._slug("Trading Patterns") == "trading-patterns"
    section2 = ac.written_section(body, "place a control ward in the river brush")
    assert section2 == "vision-control"


def test_written_section_none_before_any_heading():
    assert ac.written_section("just prose about wards, no headings here", "wards") is None


# --- record_from_draft ------------------------------------------------------

def test_record_from_draft_video_happy_path():
    front, body = _body(VIDEO_ID)
    excerpt = "when the enemy has flash and exhaust up you just give the wave"
    rec, reason = ac.record_from_draft(_draft(tip="Give the wave under a live double-summoner threat.",
                                              excerpt=excerpt), VIDEO_ID, front, body)
    assert reason is None
    assert rec.medium == Medium.VIDEO
    assert rec.timestamp == 10 and rec.section is None
    assert rec.source_excerpt == excerpt
    assert rec.role == "support"
    assert rec.tip_id == derive_tip_id(rec.tip, VIDEO_ID)   # id derived from content, not the model


def test_record_from_draft_written_happy_path():
    front, body = _body(WRITTEN_ID)
    excerpt = "it is your only engage and your only way out"
    rec, reason = ac.record_from_draft(_draft(medium="written", excerpt=excerpt,
                                              tip="Don't step up with your hook down."), WRITTEN_ID, front, body)
    assert reason is None
    assert rec.medium == Medium.WRITTEN
    assert rec.section == "trading-patterns" and rec.timestamp is None


def test_record_from_draft_drops_a_non_substring_excerpt():
    front, body = _body(VIDEO_ID)
    rec, reason = ac.record_from_draft(_draft(excerpt="this paraphrase is not in the transcript"),
                                       VIDEO_ID, front, body)
    assert rec is None and "literal substring" in reason


def test_record_from_draft_drops_an_out_of_domain_enum():
    front, body = _body(VIDEO_ID)
    rec, reason = ac.record_from_draft(_draft(scope="jungle", excerpt="give the wave and reset"),
                                       VIDEO_ID, front, body)
    assert rec is None and "enum" in reason


def test_record_from_draft_drops_a_malformed_draft():
    front, body = _body(VIDEO_ID)
    bad = _draft(excerpt="give the wave and reset")
    del bad["rationale"]
    rec, reason = ac.record_from_draft(bad, VIDEO_ID, front, body)
    assert rec is None and "rationale" in reason


def test_record_from_draft_drops_a_video_excerpt_that_spans_captions():
    front, body = _body(VIDEO_ID)
    rec, reason = ac.record_from_draft(_draft(excerpt="and reset\n[20] roam mid"), VIDEO_ID, front, body)
    assert rec is None and "single [NN] caption line" in reason


# --- assemble_source --------------------------------------------------------

def test_assemble_source_missing_clean_file_raises():
    with pytest.raises(ac.AuthorError):
        ac.assemble_source("does-not-exist", [_draft()], CLEAN_DIR)


def test_assemble_source_splits_records_from_drops():
    drafts = [_draft(excerpt="give the wave and reset"),          # kept
              _draft(excerpt="not in the transcript at all")]     # dropped
    records, drops = ac.assemble_source(VIDEO_ID, drafts, CLEAN_DIR)
    assert len(records) == 1 and len(drops) == 1
    assert drops[0][0] == 1   # the second draft's index


# --- main() end-to-end ------------------------------------------------------

def _write_drafts(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "drafts.json"
    p.write_text(json.dumps(entries), encoding="utf-8")
    return p


def _good_entries() -> list[dict]:
    return [
        {"source_id": VIDEO_ID, "drafts": [
            _draft(tip="Give the wave under a live double-summoner threat.", phase="laning",
                   excerpt="when the enemy has flash and exhaust up you just give the wave"),
            _draft(tip="Roam mid off a pushed lane to help take grubs.", phase="mid",
                   excerpt="roam mid when your lane is pushed so you can help your jungler take grubs"),
        ]},
        {"source_id": WRITTEN_ID, "drafts": [
            _draft(tip="Don't step up with your hook down.", scope="support", phase="all",
                   excerpt="it is your only engage and your only way out"),
        ]},
    ]


def test_main_writes_corpus_and_output_passes_stage4_gates(tmp_path, capsys):
    drafts = _write_drafts(tmp_path, _good_entries())
    out = tmp_path / "corpus.jsonl"
    assert ac.main([str(drafts), "--clean-dir", str(CLEAN_DIR), "--out", str(out)]) == 0

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 3
    # ③ is built against ④: the assembled corpus must clear ④'s mechanical gates.
    assert vc.gate_schema(rows) == []
    assert vc.gate_quote(rows, CLEAN_DIR) == []
    assert vc.gate_anchor(rows, CLEAN_DIR) == []


def test_main_is_idempotent(tmp_path):
    drafts = _write_drafts(tmp_path, _good_entries())
    out = tmp_path / "corpus.jsonl"
    ac.main([str(drafts), "--clean-dir", str(CLEAN_DIR), "--out", str(out)])
    first = out.read_bytes()
    ac.main([str(drafts), "--clean-dir", str(CLEAN_DIR), "--out", str(out)])
    assert out.read_bytes() == first


def test_main_merges_identical_tip_ids(tmp_path):
    dupe = _draft(tip="Give the wave under a live double-summoner threat.",
                  excerpt="you just give the wave and reset")
    entries = [{"source_id": VIDEO_ID, "drafts": [dupe, dict(dupe)]}]   # same tip + source, twice
    drafts = _write_drafts(tmp_path, entries)
    out = tmp_path / "corpus.jsonl"
    ac.main([str(drafts), "--clean-dir", str(CLEAN_DIR), "--out", str(out)])
    rows = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 1


def test_main_returns_1_when_a_source_is_missing(tmp_path, capsys):
    entries = [{"source_id": "does-not-exist", "drafts": [_draft(excerpt="x")]}]
    drafts = _write_drafts(tmp_path, entries)
    out = tmp_path / "corpus.jsonl"
    assert ac.main([str(drafts), "--clean-dir", str(CLEAN_DIR), "--out", str(out)]) == 1
    assert "FAIL" in capsys.readouterr().err
