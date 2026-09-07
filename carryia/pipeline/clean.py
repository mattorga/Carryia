#!/usr/bin/env python3
"""Stage ② -- clean a raw source into a quotable, anchor-preserving .md.

Reads data/raw/<file>.md, rewrites the BODY TEXT only, and writes two committed
artifacts named by the source's identity: data/clean/<source_id>.md (the clean
source ③ distils from and ④ quotes against) and data/clean/<source_id>.repairs.json
(every change, for human review + the diff guard). The clean file is named by
`source_id`, NOT the raw filename, because that is exactly what stage ④'s
gate_quote / gate_anchor look up (clean_dir/<source_id>.md).

DETERMINISTIC, NOT AN LLM REWRITE. ② does exactly two
things, both auditable:
  - strip caption artifacts -- bracketed non-anchor tags like [music], [applause];
  - apply this source's PER-SOURCE repair map -- mis-transcribed proper nouns
    (`failites`->Faelights, `blitz crank`->Blitzcrank), declared in the raw file's
    `repairs:` front-matter, so a fix is scoped to the source it was verified in.
Why not an LLM: `source_excerpt` is later sliced as a LITERAL substring of this
file and cited as verbatim -- an LLM rewrite would make every quote the model's
paraphrase, and the token diff guard below can't catch an equal-length word swap.
Naturalness of the prose is left to the answer-time LLM; ②'s only jobs are
quotable excerpts and clean BM25 vocabulary.

THE ANCHOR INVARIANT ④ depends on: a `[<sec>] ` video anchor and a `## heading`
line pass through UNTOUCHED -- only the text between anchors is ever cleaned, so
gate_anchor keeps resolving and a heading's `_slug` never drifts.

THE DIFF GUARD: after cleaning, every token that left or entered the body must be
accounted for by a logged change. An unlogged deletion halts the run (nonzero
exit), so a transform can never silently drop or invent content -- the log is the
human-reviewable record and the guard proves it complete.

Spec: tests/test_clean.py.

    python -m carryia.pipeline.clean data/raw/eY8jrQEn7k8.md
    python -m carryia.pipeline.clean data/raw/*.md --out-dir data/clean
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from carryia.paths import DATA

DEFAULT_OUT_DIR = DATA / "clean"

# A `[<sec>] ` video anchor: the prefix is preserved verbatim, only the trailing
# text is cleaned. `## heading` lines (written anchors) are passed through whole.
VIDEO_ANCHOR = re.compile(r"^(\[\d+\] )(.*)$")

# A caption artifact: a bracketed non-anchor tag ([music], [Applause], [crowd
# cheering]). The negative lookahead spares a `[30]`-style anchor even if one ever
# reached here, so this can never eat a timestamp.
_ARTIFACT = re.compile(r"\[(?!\d+\])[^\]]*\]")

# The guard's token: a word run OR a single punctuation mark, so trailing
# punctuation floats free ("failites." -> ["failites", "."]). Whitespace-splitting
# instead would fuse the period on, and a word-only repair (`failites`->`Faelights`)
# would read as an unlogged change of the period-bearing token. Words carry unicode
# letters, so "lucían" stays one token.
_TOKEN = re.compile(r"\w+|[^\w\s]")


def _tokens(text: str) -> list[str]:
    """Tokenise for the diff guard: words and lone punctuation, no whitespace."""
    return _TOKEN.findall(text)

Change = dict  # {"kind": "artifact"|"repair", "line": int, "before": str, "after": str}


class CleanError(Exception):
    """Anything that should surface as a one-line FAIL, not a traceback --
    a missing front-matter block, or a diff-guard violation that stops the write."""


# --- parse ------------------------------------------------------------------

def parse_raw(text: str) -> tuple[dict, str, str]:
    """Split a raw source into (front_matter, front_text, body).

    front_matter -- the leading `---` block parsed to a dict. Each value is a JSON
        scalar (the convention fetch_transcript.py writes with `json.dumps`), so
        `json.loads` round-trips it; the key ends at the first colon, so a value's
        own colons (URLs, titles) stay inside its quotes. Used for `source_id` and
        the per-source `repairs` list.
    front_text -- that `---...---` block VERBATIM (plus the blank line after it),
        re-emitted unchanged so identity/provenance can't drift through cleaning.
    body -- everything after the front-matter block; the only text ② rewrites.
    """
    m = re.match(r"(?s)\A(---\n.*?\n---\n)(.*)\Z", text)
    if not m:
        raise CleanError("missing '---' front-matter block")
    front_text, body = m.group(1), m.group(2)
    front: dict = {}
    for line in front_text.splitlines():
        if line == "---" or not line.strip():
            continue
        key, sep, val = line.partition(":")
        if not sep:
            continue
        try:
            front[key.strip()] = json.loads(val.strip())
        except json.JSONDecodeError as e:
            raise CleanError(f"front-matter {key.strip()!r}: not a JSON value ({e})")
    return front, front_text, body


# --- transforms (YOUR BODIES) -----------------------------------------------
# Each takes a line's TEXT (never its anchor) and returns the cleaned text plus a
# record of what it changed, so clean_body can log every edit for the guard.

def strip_artifacts(text: str) -> tuple[str, list[str]]:
    """Remove caption artifacts -- bracketed non-anchor tags like `[music]`,
    `[applause]`, `[laughter]` -- from a line's text.

    Return (cleaned_text, removed) where `removed` lists each artifact literal you
    took out (e.g. `["[music]"]`), one entry per occurrence, for the log + guard.
    Collapse any whitespace the removal leaves behind so no double or dangling
    space remains. The `[<sec>]` anchor never reaches here (clean_body splits it
    off first), so your pattern only needs to spare nothing -- match bracket tags.
    """
    removed = _ARTIFACT.findall(text)
    if not removed:
        return text, []
    # Blank out each tag, then fold the whitespace it left so no double/edge space
    # survives. Whitespace runs aren't tokens, so this can't disturb the guard.
    cleaned = " ".join(_ARTIFACT.sub(" ", text).split())
    return cleaned, removed


def apply_repairs(text: str, repairs: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
    """Apply this source's per-source proper-noun repairs (`before` -> `after`) to
    a line's text.

    `repairs` is the raw file's declared list, scoped to this source. Return
    (cleaned_text, applied) where `applied` lists the (before, after) pair for each
    occurrence actually replaced -- so three `blitz crank`s in a line log three
    entries, and a repair that matched nothing logs none.
    """
    applied: list[tuple[str, str]] = []
    for before, after in repairs:
        n = text.count(before)          # non-overlapping, same basis as str.replace
        if n:
            text = text.replace(before, after)
            applied += [(before, after)] * n
    return text, applied


def verify_diff(before: str, after: str, changes: list[Change]) -> list[str]:
    """The diff guard. Prove that every token which left or entered the body is
    accounted for by a logged change -- nothing silently deleted or invented.

    Tokenise with `_tokens` (words + lone punctuation, so a repaired word next to
    a period reconciles cleanly). The identity to check, as multisets of tokens:

        Counter(before) - (sum of change['before'] tokens)
                        + (sum of change['after'] tokens)  ==  Counter(after)

    (Anchors like `[10]` sit in both `before` and `after`, so they cancel; a
    word-for-word repair swaps its tokens; an artifact removal has `after == ""`.)
    Return [] when it holds, else a list of human-readable violations naming the
    unaccounted tokens. A non-empty return makes clean_source refuse to write.
    """
    # Replay every logged change onto `before`; `subtract`/`update` keep exact
    # counts (unlike `-`, which clamps at zero), so over-removal stays visible.
    expected = Counter(_tokens(before))
    for ch in changes:
        expected.subtract(_tokens(ch["before"]))
        expected.update(_tokens(ch["after"]))

    disc = expected.copy()
    disc.subtract(Counter(_tokens(after)))   # >0: vanished unlogged; <0: appeared unlogged
    dropped = {t: n for t, n in disc.items() if n > 0}
    added = {t: -n for t, n in disc.items() if n < 0}

    def _fmt(counts: dict[str, int]) -> str:
        return ", ".join(f"{t!r}×{n}" if n > 1 else repr(t) for t, n in sorted(counts.items()))

    violations: list[str] = []
    if dropped:
        violations.append(f"unlogged deletion of {_fmt(dropped)}")
    if added:
        violations.append(f"unlogged insertion of {_fmt(added)}")
    return violations


# --- clean (harness) --------------------------------------------------------

def clean_body(body: str, repairs: list[tuple[str, str]]) -> tuple[str, list[Change]]:
    """Walk the body line by line, cleaning only text between anchors.

    For a `[<sec>] ` line the anchor prefix is held aside and only its text is
    passed to the transforms; a `## heading` line (a written anchor) is emitted
    untouched; every other line is cleaned whole. Returns the rebuilt body and the
    flat, line-numbered change log the guard and the sidecar consume.
    """
    out_lines: list[str] = []
    changes: list[Change] = []
    for i, line in enumerate(body.splitlines()):
        if line.startswith("## "):
            out_lines.append(line)
            continue
        m = VIDEO_ANCHOR.match(line)
        anchor, text = (m.group(1), m.group(2)) if m else ("", line)

        text, removed = strip_artifacts(text)
        text, repaired = apply_repairs(text, repairs)

        changes += [{"kind": "artifact", "line": i, "before": a, "after": ""} for a in removed]
        changes += [{"kind": "repair", "line": i, "before": b, "after": a} for b, a in repaired]
        out_lines.append(anchor + text)

    trailing = "\n" if body.endswith("\n") else ""
    return "\n".join(out_lines) + trailing, changes


def clean_source(text: str) -> tuple[str, dict]:
    """Clean one raw source's text into (clean_file_text, repair_log).

    Runs the transforms over the body, then the diff guard; a guard violation
    raises CleanError so nothing gets written. The front-matter is re-emitted
    verbatim. The log carries `source_id` (which names the output files) and the
    full change list.
    """
    front, front_text, body = parse_raw(text)
    repairs = [tuple(pair) for pair in front.get("repairs", [])]
    cleaned_body, changes = clean_body(body, repairs)

    violations = verify_diff(body, cleaned_body, changes)
    if violations:
        raise CleanError("diff guard: " + "; ".join(violations))

    source_id = front.get("source_id")
    if not source_id:
        raise CleanError("front-matter has no source_id to name the output by")
    return front_text + cleaned_body, {"source_id": source_id, "changes": changes}


# --- cli --------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Clean raw sources into data/clean (stage ②).")
    ap.add_argument("raw", type=Path, nargs="+", help="one or more data/raw/*.md files")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"output directory (default: {DEFAULT_OUT_DIR})")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for path in args.raw:
        try:
            clean_text, log = clean_source(path.read_text(encoding="utf-8"))
        except CleanError as e:
            failed += 1
            print(f"FAIL {path.name}: {e}", file=sys.stderr)
            continue
        stem = log["source_id"]
        (args.out_dir / f"{stem}.md").write_text(clean_text, encoding="utf-8")
        (args.out_dir / f"{stem}.repairs.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        kinds = Counter(c["kind"] for c in log["changes"])
        summary = ", ".join(f"{n} {k}" for k, n in kinds.items()) or "no changes"
        print(f"ok   {path.name} -> {stem}.md  ({summary})")

    print(f"\n{len(args.raw)} source(s) processed; {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
