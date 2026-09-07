"""Corpus record schema -- the one *code* home for a coaching-tip record.

Each line in `data/corpus.jsonl` deserialises to one `CorpusRecord`.

This module is the **contract**; `carryia/pipeline/validate_corpus.py` (stage 4) is what
**enforces** it. So the type here stays a plain shape -- the gates (schema,
quote, anchor, count, creator%, coverage) live in the validator, per
gate-before-producer. `phase` is imported from
`phases.py` so the phase vocabulary keeps its single home across both planes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from carryia.phases import Phase

__all__ = ["Scope", "Medium", "CorpusRecord", "derive_tip_id"]


class Scope(StrEnum):
    """A tip's coaching scope. StrEnum -> serialises straight to the `scope`
    string in corpus.jsonl (`Scope.SUPPORT == "support"`).

    SUPPORT (not "botlane"): in League usage "bot laner" means the ADC, so a
    support-inclusive `botlane` scope reads as its opposite. v1's subject is a
    support main; ADC, if added, becomes a peer value.
    """

    FUNDAMENTAL = "fundamental"
    SUPPORT = "support"


class Medium(StrEnum):
    """The source's medium. Keys which of `timestamp` / `section` is populated --
    a video record carries `timestamp`, a written one carries `section`; stage 4
    enforces exactly one being non-null."""

    VIDEO = "video"
    WRITTEN = "written"


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    """One distilled coaching tip -- the retrieval unit.

    Frozen because a corpus record is a committed artifact: once written it is
    read, embedded, and cited, never mutated in place. Field-level rules and the
    cross-field invariant (exactly one of `timestamp` / `section`, keyed on
    `medium`) are enforced by the validate stage, not here.
    """

    tip_id: str            # content-derived (see derive_tip_id) -- stable across re-distill
    tip: str               # the distilled coaching tip -- THE retrieval unit
    rationale: str         # why the tip holds
    scope: Scope           # fundamental | support
    role: str              # e.g. "support"
    champion: str | None   # null when champion-agnostic
    phase: Phase           # laning | mid | late | all  (one home: phases.py)
    source_id: str         # the document it came from
    creator_id: str        # the opinion -- "no creator >60%" gate keys on this
    medium: Medium         # video | written
    source_url: str        # link to the source
    source_excerpt: str    # verbatim substring of the clean source -- never model-generated
    timestamp: int | None  # video only: integer seconds, floored to the interval
    section: str | None    # written only: heading slug
    retrieved_at: date     # when the source was pulled


def _normalise_tip(tip: str) -> str:
    """Fold trivial formatting differences so a cosmetic re-word of the same tip
    keeps its id: collapse every whitespace run to a single space, strip, and
    lower-case. Conservative on purpose -- punctuation is preserved, so two
    genuinely different tips can't collide on formatting alone."""
    return " ".join(tip.split()).lower()


def derive_tip_id(tip: str, source_id: str) -> str:
    """Return a **content-derived**, stable id for a tip -- the pipeline's FREEZE POINT.

    `ground_truth.jsonl` references `tip_id`s and is generated once, *after*
    validation passes; re-running distillation (③) must reproduce the exact same
    ids or that ground truth silently rots. So this is a pure, deterministic
    function of the tip's content -- no salt, no clock, no `random`.

    Three choices are baked in (mechanics, revisable on contact with code):

      1. **Inputs = `tip` + `source_id`.** Hashing the tip text alone would
         collapse the same tip from two documents into one id -- but no-cull
         keeps those as separate corroborating
         records, and P0-5 scores a hit against a *set* of acceptable ids. Keying
         on the document (`source_id`) keeps genuine corroborations distinct while
         staying idempotent for the same tip in the same source. (Not `creator_id`:
         one creator's several docs can each corroborate the same tip.)
      2. **Normalise the tip** (`_normalise_tip`) so cosmetic re-wording doesn't
         churn the id; `source_id` is a controlled slug, so it's only stripped.
      3. **sha256, first 12 hex chars.** Deterministic across processes (unlike the
         salted built-in `hash()`), readable in ground truth, and collision-safe
         far past 200 records (birthday bound ~16.7M).
    """
    # \x1f (unit separator) can't occur in tip/source_id, so ("ab","c") and
    # ("a","bc") can't hash to the same basis.
    basis = f"{_normalise_tip(tip)}\x1f{source_id.strip()}"
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()
    return f"tip-{digest[:12]}"
