#!/usr/bin/env python3
"""Stage ④ -- validate the committed corpus. THE GATE before ground truth + ingest.

Reads corpus.jsonl, runs every gate, prints a per-gate report, and exits nonzero
if any gate fails (so the Makefile / CI halts). Gate-before-producer (Decisions
Log 2026-08-06): this validator is written and made to pass on
tests/fixtures/corpus.sample.jsonl *before* author_corpus.py (③) exists, so the
producer has an executable target.

Gates:
    schema · quote · anchor · count (>=120) · creator-share (<=60%) ·
    coverage (>=10 per scope/phase)

Spec: tests/test_validate_corpus.py.

    python -m carryia.pipeline.validate_corpus data/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import fields
from pathlib import Path

from carryia.paths import DATA
from carryia.phases import TEMPORAL, Phase
from carryia.schema import CorpusRecord, Medium, Scope, derive_tip_id

MIN_RECORDS = 120
# No MAX_RECORDS: distilling all 13 clean sources produced 709 records, 0 dropped,
# and the two STRUCTURAL gates (creator-share, coverage) pass at that size. The old
# 200 ceiling guarded against UNDER-fill; reality is rich over-fill, so the cap was
# fighting its own premise. MIN_RECORDS stays as the
# real under-fill guard; no-cull (2026-08-06) keeps near-duplicates.
MAX_CREATOR_SHARE = 0.60
MIN_PER_CELL = 10

Record = dict  # a parsed jsonl row; gate_schema is what proves it well-formed


def load_corpus(path: Path) -> list[Record]:
    """Parse corpus.jsonl into raw dict rows. Deliberately does NOT validate --
    that's gate_schema's job, so a malformed record becomes a readable gate
    violation rather than a loader traceback. Only genuinely unparseable JSON
    (a broken line) is fatal here, because then there's no record to report on."""
    rows: list[Record] = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{n}: unparseable JSON line: {e}")
    return rows


# --- gates ------------------------------------------------------------------
# Each gate takes the records (plus any extra it needs) and returns a list of
# human-readable violation strings. Empty list == pass. The uniform shape lets
# main() run them in a loop and count failures.


def gate_schema(records: list[Record]) -> list[str]:
    """Every row is a well-formed CorpusRecord: exactly the 15 fields, enums in
    domain (scope/medium/phase), exactly one of timestamp/section non-null keyed
    on medium (video -> timestamp, written -> section), and tip_id ==
    derive_tip_id(tip, source_id) so a hand-edited id can't drift from content.
    """
    expected = {f.name for f in fields(CorpusRecord)}
    out: list[str] = []
    for i, r in enumerate(records):
        keys = set(r)
        if keys != expected:
            miss, extra = sorted(expected - keys), sorted(keys - expected)
            out.append(f"record {i}: field mismatch"
                       + (f", missing {miss}" if miss else "")
                       + (f", unexpected {extra}" if extra else ""))
            continue  # remaining checks assume the fields exist
        tid = r["tip_id"]
        for enum, field in ((Scope, "scope"), (Medium, "medium"), (Phase, "phase")):
            try:
                enum(r[field])
            except ValueError:
                out.append(f"record {i} ({tid}): {field}={r[field]!r} not a valid {enum.__name__}")
        has_ts, has_sec = r["timestamp"] is not None, r["section"] is not None
        if has_ts == has_sec:
            out.append(f"record {i} ({tid}): exactly one of timestamp/section must be set")
        elif r["medium"] == Medium.VIDEO and not has_ts:
            out.append(f"record {i} ({tid}): video record must carry timestamp, not section")
        elif r["medium"] == Medium.WRITTEN and not has_sec:
            out.append(f"record {i} ({tid}): written record must carry section, not timestamp")
        derived = derive_tip_id(r["tip"], r["source_id"])
        if tid != derived:
            out.append(f"record {i}: tip_id {tid!r} != derived {derived!r} (drifted from content)")
    return out


def gate_count(records: list[Record]) -> list[str]:
    """len(records) >= MIN_RECORDS. Under-fill guard only -- no upper bound
."""
    n = len(records)
    if n < MIN_RECORDS:
        return [f"{n} records < minimum {MIN_RECORDS}"]
    return []


def gate_creator_share(records: list[Record]) -> list[str]:
    """No single creator_id exceeds MAX_CREATOR_SHARE (60%) of records -- the
    primary structural guardrail once community single-author sources are admitted
. Keys on creator_id (the opinion), not source_id."""
    n = len(records)
    if not n:
        return []
    out = []
    for creator, c in Counter(r["creator_id"] for r in records).most_common():
        if c / n > MAX_CREATOR_SHARE:
            out.append(f"creator {creator!r} is {c / n:.0%} of the corpus ({c}/{n}) > {MAX_CREATOR_SHARE:.0%}")
    return out


def gate_coverage(records: list[Record]) -> list[str]:
    """>= MIN_PER_CELL (10) records per (scope, temporal-phase) cell.

    ALL-PHASE DECISION (a), 2026-08-08: an ALL-tagged tip counts toward EVERY
    temporal phase's quota for its scope -- mirroring phases.matches(), where an
    ALL tip is retrievable in every phase query. So the grid is scope x TEMPORAL
    (2 x 3 = 6 cells); ALL is never its own cell, it distributes into the three.
    """
    grid = {(s, p): 0 for s in Scope for p in TEMPORAL}
    for r in records:
        try:
            scope, phase = Scope(r["scope"]), Phase(r["phase"])
        except (KeyError, ValueError):
            continue  # malformed record -- gate_schema reports it
        for p in (TEMPORAL if phase == Phase.ALL else (phase,)):
            grid[(scope, p)] += 1
    return [f"coverage ({s}, {p}): {c} < {MIN_PER_CELL}"
            for (s, p), c in sorted(grid.items()) if c < MIN_PER_CELL]


def _load_clean(clean_dir: Path, source_id: str, cache: dict[str, str | None]) -> str | None:
    """Read (and memoise) a stage-② clean source; None if it isn't there yet."""
    if source_id not in cache:
        path = clean_dir / f"{source_id}.md"
        cache[source_id] = path.read_text() if path.exists() else None
    return cache[source_id]


def gate_quote(records: list[Record], clean_dir: Path) -> list[str]:
    """Every source_excerpt is a LITERAL substring of the clean source it cites
    (clean_dir/<source_id>.md). The whole provenance / anti-hallucination story
    rests on this gate."""
    out: list[str] = []
    cache: dict[str, str | None] = {}
    for i, r in enumerate(records):
        text = _load_clean(clean_dir, r["source_id"], cache)
        if text is None:
            out.append(f"record {i} ({r['tip_id']}): clean source {r['source_id']}.md not found in {clean_dir}")
        elif r["source_excerpt"] not in text:
            out.append(f"record {i} ({r['tip_id']}): source_excerpt is not a literal substring of {r['source_id']}.md")
    return out


def _slug(heading: str) -> str:
    """Slugify a `## Heading` into the `section` anchor a written record cites.
    Defining it in the gate makes the validator the contract: stage ③ must derive
    `section` the same way. Lowercase; non-alphanumeric runs -> one hyphen; trimmed."""
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def gate_anchor(records: list[Record], clean_dir: Path) -> list[str]:
    """Every record's anchor resolves in its clean source: a video timestamp lands
    on a real `[<sec>]` caption line, a written section matches a `## heading` slug."""
    out: list[str] = []
    cache: dict[str, str | None] = {}
    headings: dict[str, set[str]] = {}
    for i, r in enumerate(records):
        sid = r["source_id"]
        text = _load_clean(clean_dir, sid, cache)
        if text is None:
            out.append(f"record {i} ({r['tip_id']}): clean source {sid}.md not found in {clean_dir}")
            continue
        if r["medium"] == Medium.VIDEO:
            if not re.search(rf"^\[{r['timestamp']}\] ", text, re.MULTILINE):
                out.append(f"record {i} ({r['tip_id']}): timestamp [{r['timestamp']}] is not an anchor in {sid}.md")
        elif r["medium"] == Medium.WRITTEN:
            if sid not in headings:
                headings[sid] = {_slug(h) for h in re.findall(r"^## (.+?)\s*$", text, re.MULTILINE)}
            if r["section"] not in headings[sid]:
                out.append(f"record {i} ({r['tip_id']}): section {r['section']!r} matches no ## heading slug in {sid}.md")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate the coaching corpus (stage ④).")
    ap.add_argument("corpus", type=Path, help="path to corpus.jsonl")
    ap.add_argument("--clean-dir", type=Path, default=DATA / "clean",
                    help="dir of stage-② clean sources (for the quote/anchor gates)")
    args = ap.parse_args(argv)

    records = load_corpus(args.corpus)

    # Ordered so the cheap, whole-corpus gates report before the per-source ones.
    results: dict[str, list[str]] = {
        "schema": gate_schema(records),
        "count": gate_count(records),
        "creator_share": gate_creator_share(records),
        "coverage": gate_coverage(records),
        "quote": gate_quote(records, args.clean_dir),
        "anchor": gate_anchor(records, args.clean_dir),
    }

    failed = 0
    for name, violations in results.items():
        if violations:
            failed += 1
            print(f"FAIL {name}: {len(violations)} violation(s)")
            for v in violations[:20]:
                print(f"       - {v}")
            if len(violations) > 20:
                print(f"       ... and {len(violations) - 20} more")
        else:
            print(f"ok   {name}")

    print(f"\n{len(records)} records checked; {failed} gate(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
