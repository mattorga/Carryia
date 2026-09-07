"""Per-player metric computation over GameRecords -- the shared primitive.

Turns a player's list of all-phase games into the metric summary that BOTH the
cohort benchmark (aggregated across players into quartiles, `cohort.py`) and the
future Stat-Line Producer (the subject, placed against those quartiles) read.
"One extractor, two callers" extends here into "one summariser, two callers".

**Phase is derived here, at read time, via phases.py.** The event streams on a
GameRecord carry only `game_time_s` (snapshot.py freezes no phase, so a moved
boundary can't rot the snapshot); this module is the single place a personal-plane
game-time becomes a phase.

Metric families:
  - **PHASED** (laning / mid / late): `deaths`, `ward_activity` (placed + cleared),
    `objective_participation` (epics + towers + grubs) -- each a per-game mean count
    per phase.
  - **WHOLE-GAME**: the `challenges.*` KPIs -- each a per-game mean.

Every value is a per-GAME mean, so when the cohort aggregates across players a
20-game player doesn't outweigh a 6-game one -- percentile stability scales with
distinct *players*, not games (the 09-01 sizing point). The cohort keeps only
all-phase games (>= 25:00), so the `late` denominator is never under-filled (the
08-31 "scope the late denominator to games that reached late" concern, dissolved
by construction)."""

from __future__ import annotations

from statistics import mean
from typing import Iterable

from carryia.phases import TEMPORAL, phase_at
from carryia.personal.snapshot import GameRecord

__all__ = ["PHASED_METRICS", "WHOLE_GAME_METRICS", "player_metrics"]

# The three phased families and the whole-game KPIs. These names are the contract
# the cohort artifact and the Producer key on -- one home for them.
PHASED_METRICS = ("deaths", "ward_activity", "objective_participation")
WHOLE_GAME_METRICS = (
    "vision_score_per_min",
    "kill_participation",
    "effective_heal_shield",
    "team_damage_pct",
    "enemy_immobilizations",
    "ward_takedowns",
    "kda",
)


def _phase_histogram(times: Iterable[int]) -> dict[str, int]:
    """Count event game-times into the three temporal phases (laning/mid/late)."""
    counts = {p.value: 0 for p in TEMPORAL}
    for t in times:
        counts[phase_at(t).value] += 1
    return counts


def player_metrics(games: list[GameRecord]) -> dict:
    """One player's games -> `{games, phased, whole_game}`, all per-game means.

    `phased[metric][phase]` is the mean count of that event-family in that phase
    per game; `whole_game[kpi]` is the mean of that KPI over the games."""
    n = len(games)
    if n == 0:
        raise ValueError("player_metrics needs at least one game")

    totals = {m: {p.value: 0 for p in TEMPORAL} for m in PHASED_METRICS}
    for g in games:
        for phase, c in _phase_histogram(e.game_time_s for e in g.deaths_ctx).items():
            totals["deaths"][phase] += c
        for phase, c in _phase_histogram(e.game_time_s for e in g.ward_events).items():
            totals["ward_activity"][phase] += c
        for phase, c in _phase_histogram(e.game_time_s for e in g.objective_events).items():
            totals["objective_participation"][phase] += c

    phased = {
        m: {p.value: totals[m][p.value] / n for p in TEMPORAL} for m in PHASED_METRICS
    }
    whole_game = {
        "vision_score_per_min": mean(g.vision_score_per_min for g in games),
        "kill_participation": mean(g.kill_participation for g in games),
        "effective_heal_shield": mean(g.effective_heal_shield for g in games),
        "team_damage_pct": mean(g.team_damage_pct for g in games),
        "enemy_immobilizations": mean(g.enemy_immobilizations for g in games),
        "ward_takedowns": mean(g.ward_takedowns for g in games),
        "kda": mean(g.kda for g in games),
    }
    return {"games": n, "phased": phased, "whole_game": whole_game}
