#!/usr/bin/env python3
"""Stage ③ -- assemble the coaching corpus from distill tip drafts.

THE CLERK, not the writer. The coaching *judgment* -- what is a tip, its scope /
phase / champion, and which verbatim span to quote -- is produced upstream by the
distill step. This script does only
the deterministic assembly they must never touch, so no provenance field is ever
model-authored:

  - QUOTE GUARD (the ③ stage gate): verify each draft's `excerpt` is a LITERAL
    substring of the clean source it cites, and DROP the draft if it is not. A
    paraphrase is a lost record, never a fabricated quote.
  - ANCHOR: snap it from where the excerpt lands -- a video `timestamp` = the
    `[NN]` caption line the quote sits in; a written `section` = the nearest
    preceding `## heading` slug, derived with validate_corpus._slug so ④'s
    gate_anchor keeps resolving (the validator is the contract).
  - SOURCE FIELDS: creator_id / medium / source_url / retrieved_at come from the
    clean file's front-matter; role = "support" (v1 constant); tip_id is derived
    from the tip's content (the freeze point) -- never from the model.

Input: one or more draft JSON files (the distill step's output) -- each a
list of {"source_id": ..., "drafts": [{tip, rationale, scope, phase, champion,
excerpt}, ...]}. Output: data/corpus.jsonl, REBUILT from the drafts every run
(idempotent), then handed to stage ④ (validate_corpus.py) for the full gate set.

    python -m carryia.pipeline.author_corpus data/drafts/*.json --out data/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

from carryia.paths import DATA
from carryia.pipeline import clean  # parse_raw -- reuse ②'s front-matter parser
from carryia.pipeline import validate_corpus as vc  # _slug -- the section contract's home
from carryia.schema import CorpusRecord, Medium, Phase, Scope, derive_tip_id

DEFAULT_OUT = DATA / "corpus.jsonl"
DEFAULT_CLEAN_DIR = DATA / "clean"
ROLE_V1 = "support"  # v1 is support-only; role is constant, set here not by the model

DRAFT_FIELDS = {"tip", "rationale", "scope", "phase", "champion", "excerpt"}
FRONT_FIELDS = ("creator_id", "medium", "source_url", "retrieved_at")

_CAPTION = re.compile(r"^\[(\d+)\] (.*)$", re.MULTILINE)
_HEADING = re.compile(r"^## (.+?)\s*$", re.MULTILINE)


class AuthorError(Exception):
    """A per-source failure that skips the source and fails the run -- a missing
    clean file, or front-matter that won't parse / lacks a needed field. Distinct
    from a per-draft DROP, which is expected attrition and never fails the run."""


# --- anchor derivation ------------------------------------------------------

def video_timestamp(body: str, excerpt: str) -> int | None:
    """The `[NN]` seconds of the caption line the excerpt sits in, or None if it
    is not wholly within one line. The distill step requires a video quote to stay
    inside a single caption block, so a cross-line excerpt -> None -> drop."""
    for m in _CAPTION.finditer(body):
        if excerpt in m.group(2):
            return int(m.group(1))
    return None


def written_section(body: str, excerpt: str) -> str | None:
    """The slug of the nearest `## heading` preceding the excerpt, or None if none
    precedes it. Slugged with validate_corpus._slug so it matches exactly what ④'s
    gate_anchor looks up -- ③ derives the anchor the validator will check."""
    idx = body.find(excerpt)
    if idx < 0:
        return None
    heading = None
    for m in _HEADING.finditer(body):
        if m.start() < idx:
            heading = m.group(1)
        else:
            break
    return vc._slug(heading) if heading is not None else None


# --- assembly ---------------------------------------------------------------

def record_from_draft(
    draft: dict, source_id: str, front: dict, body: str
) -> tuple[CorpusRecord | None, str | None]:
    """Turn one draft + its clean source into a CorpusRecord, or return (None,
    reason) to DROP it. Drops (expected attrition): a malformed draft, an out-of-
    domain enum, an excerpt that isn't a literal substring (the quote guard), or an
    anchor that won't resolve. The excerpt and both anchors are computed here from
    `body`; only judgment fields are taken from the draft."""
    missing = DRAFT_FIELDS - draft.keys()
    if missing:
        return None, f"draft missing field(s) {sorted(missing)}"

    excerpt = draft["excerpt"]
    if excerpt not in body:
        return None, "excerpt is not a literal substring of the clean source"

    try:
        scope = Scope(draft["scope"])
        phase = Phase(draft["phase"])
        medium = Medium(front["medium"])
    except ValueError as e:
        return None, f"invalid enum value ({e})"

    if medium == Medium.VIDEO:
        timestamp, section = video_timestamp(body, excerpt), None
        if timestamp is None:
            return None, "excerpt does not sit within a single [NN] caption line"
    else:
        timestamp, section = None, written_section(body, excerpt)
        if section is None:
            return None, "excerpt has no preceding ## heading to anchor a section"

    tip = draft["tip"]
    record = CorpusRecord(
        tip_id=derive_tip_id(tip, source_id),
        tip=tip,
        rationale=draft["rationale"],
        scope=scope,
        role=ROLE_V1,
        champion=draft["champion"] or None,
        phase=phase,
        source_id=source_id,
        creator_id=front["creator_id"],
        medium=medium,
        source_url=front["source_url"],
        source_excerpt=excerpt,
        timestamp=timestamp,
        section=section,
        retrieved_at=front["retrieved_at"],
    )
    return record, None


def assemble_source(
    source_id: str, drafts: list[dict], clean_dir: Path
) -> tuple[list[CorpusRecord], list[tuple[int, str]]]:
    """Assemble every draft for one source into records. Returns (records, drops)
    where drops is a list of (draft_index, reason). Raises AuthorError if the clean
    source is missing or its front-matter is unusable -- those are wiring faults,
    not the per-draft attrition that DROP covers."""
    path = clean_dir / f"{source_id}.md"
    if not path.exists():
        raise AuthorError(f"clean source {source_id}.md not found in {clean_dir}")
    try:
        front, _, body = clean.parse_raw(path.read_text(encoding="utf-8"))
    except clean.CleanError as e:
        raise AuthorError(f"{source_id}.md front-matter: {e}")
    for key in FRONT_FIELDS:
        if key not in front:
            raise AuthorError(f"{source_id}.md front-matter missing {key!r}")

    records: list[CorpusRecord] = []
    drops: list[tuple[int, str]] = []
    for i, draft in enumerate(drafts):
        record, reason = record_from_draft(draft, source_id, front, body)
        if record is None:
            drops.append((i, reason))
        else:
            records.append(record)
    return records, drops


def load_drafts(paths: list[Path]) -> list[tuple[str, list[dict]]]:
    """Read distill draft files into (source_id, drafts) pairs, in file then
    in-file order. Each file is a JSON list of {"source_id", "drafts"} objects.
    A broken file is fatal (SystemExit) -- there's nothing to assemble from it."""
    out: list[tuple[str, list[dict]]] = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{p}: unparseable JSON ({e})")
        for obj in data:
            try:
                out.append((obj["source_id"], obj["drafts"]))
            except (TypeError, KeyError):
                raise SystemExit(f"{p}: each entry needs 'source_id' and 'drafts'")
    return out


def dedup(records: list[CorpusRecord]) -> tuple[list[CorpusRecord], int]:
    """Drop exact tip_id collisions (same tip + same source), keeping the first --
    identical records add nothing and would double-count in ④'s coverage grid.
    Cross-source corroborations differ in source_id, so they keep distinct ids and
    survive: no-cull governs those, not this."""
    seen: set[str] = set()
    out: list[CorpusRecord] = []
    dupes = 0
    for rec in records:
        if rec.tip_id in seen:
            dupes += 1
            continue
        seen.add(rec.tip_id)
        out.append(rec)
    return out, dupes


# --- cli --------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Assemble the coaching corpus from distill drafts (stage ③).")
    ap.add_argument("drafts", type=Path, nargs="+", help="distill draft JSON files")
    ap.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR,
                    help=f"dir of stage-② clean sources (default: {DEFAULT_CLEAN_DIR})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"corpus output path (default: {DEFAULT_OUT})")
    args = ap.parse_args(argv)

    pairs = load_drafts(args.drafts)

    all_records: list[CorpusRecord] = []
    total_drops = 0
    failed = 0
    for source_id, drafts in pairs:
        try:
            records, drops = assemble_source(source_id, drafts, args.clean_dir)
        except AuthorError as e:
            failed += 1
            print(f"FAIL {source_id}: {e}", file=sys.stderr)
            continue
        total_drops += len(drops)
        for i, reason in drops:
            print(f"  drop {source_id}[{i}]: {reason}", file=sys.stderr)
        print(f"ok   {source_id}: {len(records)} record(s), {len(drops)} dropped")
        all_records.extend(records)

    deduped, dupes = dedup(all_records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in deduped:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    print(f"\n{len(deduped)} records -> {args.out}  "
          f"({total_drops} dropped, {dupes} duplicate id(s) merged, {failed} source(s) failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
