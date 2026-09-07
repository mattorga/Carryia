"""The Stat-Line Producer -- places the subject against the cohort (SEAM ②).

Turns the subject's own games into the **personal-context block**: the single
paragraph the coach grounds every answer in. It
reads `metrics.player_metrics` for the subject, then buckets each judged metric
against the committed cohort's quartiles (`data/benchmark/cohort.json`, ~120 Silver
support players) -- "one summariser, two callers" closing: the cohort aggregated
`player_metrics` across players into quartiles; here the same summariser runs on
one player and is *placed* against them.

**What gets judged:**
  - PHASED (all three, per phase): `deaths`, `ward_activity`, `objective_participation`.
  - WHOLE-GAME, bucketed: `vision_score_per_min`, `kill_participation`.
  - WHOLE-GAME, descriptive only: heal+shield, team-damage%, CC, ward-takedowns,
    KDA -- champion-confounded or redundant, so they are reported raw, never bucketed
    (bucketing a Soraka's heals against the cohort would mislead, not coach).

A bucket is **positional** (below q1 / between / above q3 -- percentile standing,
09-01), but coaching needs valence, so each bucket is turned into a **standing**
(weak / typical / strong) through the metric's direction: for `deaths` more is
worse; for every other judged metric more is better. The rendered block leads with
the weaknesses -- that is the "recurring mistake across these games" the pattern-
over-N interface is built to surface (242).
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

from carryia.paths import DATA
from carryia.personal import ingest
from carryia.personal.metrics import PHASED_METRICS, player_metrics
from carryia.phases import TEMPORAL

__all__ = [
    "BUCKETED_WHOLE_GAME",
    "DESCRIPTIVE_WHOLE_GAME",
    "Placement",
    "place_metrics",
    "load_cohort",
    "panel_rows",
    "render_context_block",
    "subject_context_block",
]

COHORT_PATH = DATA / "benchmark" / "cohort.json"

# The two whole-game KPIs that ARE judged against the cohort (09-01); the rest are
# descriptive-only. Phased families (all three) are always judged.
BUCKETED_WHOLE_GAME = ("vision_score_per_min", "kill_participation")
DESCRIPTIVE_WHOLE_GAME = (
    "effective_heal_shield",
    "team_damage_pct",
    "enemy_immobilizations",
    "ward_takedowns",
    "kda",
)

# Metrics where a LOWER value is the better one. Everything else judged -> higher is
# better. This is the only per-metric valence knob; it turns a positional bucket into
# a coaching standing.
LOWER_IS_BETTER = frozenset({"deaths"})

# Human labels for the block. Phased metrics get a "Laning/Mid/Late" prefix at render.
_LABELS = {
    "deaths": "deaths",
    "ward_activity": "ward activity",
    "objective_participation": "objective participation",
    "vision_score_per_min": "vision score per minute",
    "kill_participation": "kill participation",
    "effective_heal_shield": "effective heal+shield",
    "team_damage_pct": "team damage %",
    "enemy_immobilizations": "crowd control",
    "ward_takedowns": "ward takedowns",
    "kda": "KDA",
}
# Metrics stored as 0..1 fractions -> shown as percentages in the block.
_AS_PERCENT = frozenset({"kill_participation", "team_damage_pct"})

# Weakest first: the coach should open on what to fix.
_STANDING_ORDER = {"weak": 0, "typical": 1, "strong": 2}


@dataclass(frozen=True)
class Placement:
    """One judged metric, the subject placed against the cohort quartiles."""

    metric: str            # a PHASED_METRICS or BUCKETED_WHOLE_GAME key
    phase: str | None      # "laning"|"mid"|"late" for phased; None for whole-game
    value: float           # the subject's per-game mean
    q1: float
    median: float
    q3: float
    cohort_mean: float
    bucket: str            # "below" | "average" | "above" -- positional (by quartile)
    standing: str          # "weak" | "typical" | "strong" -- bucket through direction
    delta: float           # value - cohort_mean (the 09-01 mean-delta)
    percentile: float | None  # empirical rank of value in the cohort distribution (0..100)


def _bucket(value: float, q1: float, q3: float) -> str:
    """Positional quartile bucket: below q1 / between / above q3 (09-01)."""
    if value < q1:
        return "below"
    if value > q3:
        return "above"
    return "average"


def _percentile(value: float, values: list[float]) -> float | None:
    """Empirical percentile of `value` in the cohort distribution: the share of
    cohort players at or below it (0..100). Needs the full distribution (added to
    `cohort.json`'s `values`); returns None when it isn't present (quartiles alone
    can't yield an honest percentile). Raw and undirected -- for `deaths` a higher
    percentile means MORE deaths; the UI shows it beside the value, not as good/bad."""
    if not values:
        return None
    return round(100 * bisect.bisect_right(values, value) / len(values), 1)


def _standing(bucket: str, metric: str) -> str:
    """Turn a positional bucket into a coaching standing via the metric's direction."""
    if bucket == "average":
        return "typical"
    higher = bucket == "above"
    good = higher != (metric in LOWER_IS_BETTER)  # above & higher-better -> good
    return "strong" if good else "weak"


def _place(metric: str, phase: str | None, value: float, ref: dict) -> Placement:
    bucket = _bucket(value, ref["q1"], ref["q3"])
    return Placement(
        metric=metric,
        phase=phase,
        value=value,
        q1=ref["q1"],
        median=ref["median"],
        q3=ref["q3"],
        cohort_mean=ref["mean"],
        bucket=bucket,
        standing=_standing(bucket, metric),
        delta=value - ref["mean"],
        percentile=_percentile(value, ref.get("values", [])),
    )


def place_metrics(subject: dict, cohort: dict) -> list[Placement]:
    """Subject `player_metrics` output + cohort artifact -> the judged placements.

    Phased metrics (all three, per temporal phase) then the two bucketed whole-game
    KPIs. Pure -- no I/O; the wiring lives in `subject_context_block`."""
    cm = cohort["metrics"]
    placements: list[Placement] = []
    for metric in PHASED_METRICS:
        for phase in TEMPORAL:
            value = subject["phased"][metric][phase.value]
            placements.append(_place(metric, phase.value, value, cm[metric][phase.value]))
    for metric in BUCKETED_WHOLE_GAME:
        placements.append(_place(metric, None, subject["whole_game"][metric], cm[metric]))
    return placements


def load_cohort(path: Path = COHORT_PATH) -> dict:
    """Read the committed cohort benchmark (aggregate-only, no puuids)."""
    return json.loads(path.read_text())


def _label(p: Placement) -> str:
    base = _LABELS[p.metric]
    return f"{p.phase.capitalize()} {base}" if p.phase else base[:1].upper() + base[1:]


def _fmt(metric: str, value: float) -> str:
    if metric in _AS_PERCENT:
        return f"{value * 100:.0f}%"
    return f"{value:.1f}"


def render_context_block(
    placements: list[Placement], n_games: int, subject_wg: dict, cohort_size: int | None = None
) -> str:
    """Render the placements (weakest first) + the descriptive stats into the
    personal-context block that grounds the coach's answer."""
    ranked = sorted(placements, key=lambda p: (_STANDING_ORDER[p.standing], p.metric, p.phase or ""))
    cohort = f"a {cohort_size}-player Silver support cohort" if cohort_size else "a Silver support cohort"
    lines = [
        f"Player: Silver support main. Pattern across {n_games} recent ranked games, "
        f"placed against {cohort} (weakest areas first).",
        "",
        "Standing by metric:",
    ]
    for p in ranked:
        rng = f"{_fmt(p.metric, p.q1)}–{_fmt(p.metric, p.q3)}"
        per_game = " per game" if p.phase else ""  # phased metrics are per-game counts; KPIs are not
        lines.append(
            f"- {_label(p)} — {p.standing.upper()}: {_fmt(p.metric, p.value)}{per_game} "
            f"(cohort avg {_fmt(p.metric, p.cohort_mean)}, typical range {rng})"
        )
    descriptive = " · ".join(
        f"{_LABELS[m]} {_fmt(m, subject_wg[m])}" for m in DESCRIPTIVE_WHOLE_GAME
    )
    lines += ["", f"Descriptive (champion-dependent, not ranked): {descriptive}"]
    return "\n".join(lines)


def panel_rows(subject: dict, cohort: dict) -> tuple[list[dict], list[dict]]:
    """UI-ready rows for the app's "You vs others" panel (P0-7).

    Returns ``(judged, descriptive)``. Judged rows place the subject against the
    cohort -- the subject value, the cohort median, the typical (q1–q3) range, and
    the empirical `percentile` -- in `place_metrics` order (grouped by family/phase).
    Descriptive rows (the champion-confounded KPIs, 09-01) carry the value alone, no
    benchmark. Formatting (percentages, per-game counts) is applied here so the app
    only renders; no weak/strong label -- the value beside the cohort speaks."""
    judged = [
        {
            "metric": _label(p),
            "you": _fmt(p.metric, p.value),
            "cohort median": _fmt(p.metric, p.median),
            "typical range": f"{_fmt(p.metric, p.q1)}–{_fmt(p.metric, p.q3)}",
            "percentile": p.percentile,
        }
        for p in place_metrics(subject, cohort)
    ]
    wg = subject["whole_game"]
    descriptive = [
        {"metric": _LABELS[m], "you": _fmt(m, wg[m])} for m in DESCRIPTIVE_WHOLE_GAME
    ]
    return judged, descriptive


def subject_context_block(
    snapshot_path: Path = ingest.SNAPSHOT_DIR / "games.jsonl",
    cohort_path: Path = COHORT_PATH,
) -> str:
    """The one call the app makes: committed snapshot + cohort -> the block string.

    Wires `load_snapshot` -> `player_metrics` -> `place_metrics` -> render. No Riot
    calls, no LLM -- deterministic over committed data (P0-10)."""
    games = ingest.load_snapshot(snapshot_path)
    subject = player_metrics(games)
    cohort = load_cohort(cohort_path)
    placements = place_metrics(subject, cohort)
    cohort_size = cohort.get("meta", {}).get("player_count")
    return render_context_block(placements, subject["games"], subject["whole_game"], cohort_size)
