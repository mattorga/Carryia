"""Game-phase vocabulary — the single home for phase, shared by both planes.

One closed enum + one set of boundaries, imported everywhere phase is spoken:
distillation (stage 3) classifies tips into it, validation (stage 4) groups
coverage by it, and the runtime coach maps a timeline event's game-time through
it. Keeping it in one module is the whole point -- rise-to-challenger carried
three uncoordinated phase vocabularies that never joined; a single imported enum makes that class of drift impossible.

Two kinds of member:
  - TEMPORAL (laning / mid / late) -- clock windows. `phase_at` emits only these.
  - ALL -- phase-agnostic (whole game). A corpus-side wildcard: never produced by
    the mapper, but matched by every phase query at retrieval time.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["Phase", "TEMPORAL", "LANING_ENDS_S", "MID_ENDS_S", "phase_at", "matches"]


class Phase(StrEnum):
    """A tip's phase of applicability. StrEnum so it serialises straight to the
    `phase` string in corpus.jsonl (`Phase.LANING == "laning"`)."""

    LANING = "laning"
    MID = "mid"
    LATE = "late"
    ALL = "all"  # non-temporal wildcard -- never a `phase_at` output


# The temporal members, in order -- the only values the time->phase mapper emits.
TEMPORAL: tuple[Phase, ...] = (Phase.LANING, Phase.MID, Phase.LATE)

# Boundaries in seconds of game-time. Compared with
# a strict `<`, so 14:00 exactly is already `mid` -- matching "laning <14:00".
LANING_ENDS_S = 14 * 60  # 840  -- bot-lane 2v2 typically breaks when T1 falls
MID_ENDS_S = 25 * 60     # 1500


def phase_at(seconds: int) -> Phase:
    """Map a timeline event's game-time (seconds) to its temporal phase.

    Emits only TEMPORAL members -- `ALL` is a corpus tag, never a game moment.
    This is the personal-plane half of the contract: a death at 522s -> LANING,
    so the coach can retrieve phase-appropriate tips for it.
    """
    if seconds < 0:
        raise ValueError(f"game-time cannot be negative: {seconds}")
    if seconds < LANING_ENDS_S:
        return Phase.LANING
    if seconds < MID_ENDS_S:
        return Phase.MID
    return Phase.LATE


def matches(tip_phase: Phase, query_phase: Phase) -> bool:
    """Does a corpus tip's phase satisfy a phase-scoped retrieval query?

    This is where the temporal/`ALL` split becomes behaviour. Settled rule:

      - A query for `ALL` is un-phased ("don't filter by phase") and matches
        every tip -- restricting it to only `ALL`-tagged tips would be useless.
      - A temporal query matches a tip tagged with that same phase, or with
        `ALL` -- whole-game advice ("watch the minimap") must surface in every
        phase, or it silently never comes back.
      - Strict on the temporal members: no bleed between adjacent phases, so the
        contract stays crisp. If recall ever needs widening, do it at the
        retriever, not by blurring what a phase means.
    """
    if query_phase == Phase.ALL:
        return True
    return tip_phase in (query_phase, Phase.ALL)
