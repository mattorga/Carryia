"""Spec for carryia/personal/producer.py -- the Stat-Line Producer (SEAM ②).

All GREEN regression locks: the bucketing rules are settled decisions (2026-09-01
metric spec, 242 personal-context block), so these pin the contract the interface
will ground on -- they don't await an undecided fork.

What's pinned:
  - which metrics are judged: the three phased families + vision/min + kill-participation
    ONLY (the other whole-game KPIs are descriptive, never bucketed);
  - the positional bucket (below q1 / between / above q3) and its edges;
  - the valence flip that turns a bucket into a standing (deaths lower-is-better,
    everything else higher-is-better);
  - the mean-delta, the weakest-first render order, and the snapshot round-trip
    the Producer reads through.

Run:  .venv/bin/pytest tests/test_producer.py -v
"""

from datetime import date

import pytest

from carryia.personal import ingest, producer
from carryia.personal.metrics import PHASED_METRICS, player_metrics
from carryia.personal.snapshot import DeathEvent, GameRecord, ObjectiveEvent, WardEvent


# --- fixtures: a hand-built cohort + subject, no files -------------------------

def _ref(q1, median, q3, mean, n=125, values=None):
    # `values` is the anonymous per-player distribution (real artifact carries it);
    # default to the quartile points so a percentile can still be computed in tests.
    return {"q1": q1, "median": median, "q3": q3, "mean": mean, "n": n,
            "values": sorted(values if values is not None else [q1, median, q3])}


def _phased_ref():
    # same triple for all three phases; the values matter, not their spread here.
    return {p: _ref(2.0, 2.5, 3.0, 2.5) for p in ("laning", "mid", "late")}


def _cohort():
    return {
        "meta": {"tier": "SILVER", "player_count": 125},
        "metrics": {
            "deaths": _phased_ref(),
            "ward_activity": _phased_ref(),
            "objective_participation": _phased_ref(),
            "vision_score_per_min": _ref(2.3, 2.6, 2.8, 2.6),
            "kill_participation": _ref(0.49, 0.53, 0.55, 0.53),
            # descriptive refs exist in the real artifact but must be ignored:
            "kda": _ref(2.0, 3.0, 4.0, 3.0),
        },
    }


def _subject(*, deaths_val=2.5, vspm=2.6, kp=0.53):
    phased = {m: {p: 2.5 for p in ("laning", "mid", "late")} for m in PHASED_METRICS}
    phased["deaths"] = {p: deaths_val for p in ("laning", "mid", "late")}
    return {
        "games": 10,
        "phased": phased,
        "whole_game": {
            "vision_score_per_min": vspm,
            "kill_participation": kp,
            "effective_heal_shield": 3825.0,
            "team_damage_pct": 0.15,
            "enemy_immobilizations": 31.7,
            "ward_takedowns": 9.5,
            "kda": 2.9,
        },
    }


# --- bucket: positional, by quartile -----------------------------------------

@pytest.mark.parametrize("value, expected", [
    (1.9, "below"),   # < q1
    (2.0, "average"), # == q1 edge
    (2.5, "average"),
    (3.0, "average"), # == q3 edge
    (3.1, "above"),   # > q3
])
def test_bucket_is_positional_with_inclusive_quartile_edges(value, expected):
    assert producer._bucket(value, 2.0, 3.0) == expected


# --- standing: bucket through the metric's direction --------------------------

def test_deaths_more_is_weak_fewer_is_strong():
    # deaths is lower-is-better: above cohort -> weak, below -> strong.
    assert producer._standing("above", "deaths") == "weak"
    assert producer._standing("below", "deaths") == "strong"


def test_higher_is_better_metric_flips_the_other_way():
    assert producer._standing("above", "ward_activity") == "strong"
    assert producer._standing("below", "ward_activity") == "weak"


def test_average_bucket_is_typical_regardless_of_direction():
    assert producer._standing("average", "deaths") == "typical"
    assert producer._standing("average", "vision_score_per_min") == "typical"


# --- place_metrics: which metrics are judged, and the delta -------------------

def test_places_three_phased_families_times_three_phases_plus_two_kpis():
    placements = producer.place_metrics(_subject(), _cohort())
    assert len(placements) == 3 * 3 + 2
    judged = {(p.metric, p.phase) for p in placements}
    assert ("vision_score_per_min", None) in judged
    assert ("kill_participation", None) in judged


def test_descriptive_whole_game_kpis_are_never_placed():
    placements = producer.place_metrics(_subject(), _cohort())
    assert not any(p.metric in producer.DESCRIPTIVE_WHOLE_GAME for p in placements)


def test_delta_is_value_minus_cohort_mean():
    placements = producer.place_metrics(_subject(deaths_val=3.5), _cohort())
    d = next(p for p in placements if p.metric == "deaths" and p.phase == "laning")
    assert d.delta == pytest.approx(3.5 - 2.5)
    assert d.standing == "weak"  # 3.5 > q3 (3.0), deaths lower-is-better


def test_a_strong_death_line_reads_strong():
    placements = producer.place_metrics(_subject(deaths_val=1.5), _cohort())
    d = next(p for p in placements if p.metric == "deaths" and p.phase == "mid")
    assert (d.bucket, d.standing) == ("below", "strong")


def test_placement_carries_empirical_percentile():
    # value 4 sits at or above 4 of the 5 cohort values -> 80th percentile (raw, undirected).
    cohort = _cohort()
    cohort["metrics"]["kill_participation"] = _ref(
        0.2, 0.4, 0.6, 0.4, values=[0.1, 0.2, 0.4, 0.4, 0.9]
    )
    subject = _subject(kp=0.4)
    kp = next(p for p in producer.place_metrics(subject, cohort) if p.metric == "kill_participation")
    assert kp.percentile == 80.0


def test_percentile_is_none_without_a_distribution():
    # quartiles-only ref (no `values`) can't yield an honest percentile.
    cohort = _cohort()
    cohort["metrics"]["kill_participation"] = {"q1": 0.2, "median": 0.4, "q3": 0.6, "mean": 0.4, "n": 5}
    kp = next(p for p in producer.place_metrics(_subject(kp=0.4), cohort) if p.metric == "kill_participation")
    assert kp.percentile is None


# --- panel_rows: the app's "You vs others" table ------------------------------

def test_panel_rows_split_judged_and_descriptive():
    judged, descriptive = producer.panel_rows(_subject(kp=0.4), _cohort())
    assert len(judged) == 3 * 3 + 2          # phased families + vision/min + kill-participation
    assert len(descriptive) == len(producer.DESCRIPTIVE_WHOLE_GAME)
    # every judged row carries the You-vs-others fields incl. a numeric percentile.
    row = next(r for r in judged if "kill participation" in r["metric"].lower())
    assert set(row) == {"metric", "you", "cohort median", "typical range", "percentile"}
    assert row["you"] == "40%" and isinstance(row["percentile"], float)
    # descriptive rows show the value alone -- no benchmark, no percentile.
    assert set(descriptive[0]) == {"metric", "you"}


# --- render: weakest first, descriptives present, no bucket on them -----------

def test_block_lists_weaknesses_before_strengths():
    block = producer.render_context_block(
        producer.place_metrics(_subject(deaths_val=3.5, vspm=2.9), _cohort()),
        n_games=10,
        subject_wg=_subject()["whole_game"],
    )
    # deaths 3.5 is WEAK, vision 2.9 is STRONG -> the weak line comes first.
    assert block.index("WEAK") < block.index("STRONG")


def test_block_reports_descriptive_stats_without_a_standing():
    block = producer.render_context_block(
        producer.place_metrics(_subject(), _cohort()),
        n_games=10,
        subject_wg=_subject()["whole_game"],
    )
    assert "Descriptive" in block
    assert "KDA 2.9" in block


# --- snapshot round-trip: the reader the Producer runs metrics over -----------

def _game():
    return GameRecord(
        match_id="SG2_1", game_creation=date(2026, 8, 29), game_duration_s=1800,
        champion="Lux", queue_id=420, win=True, kills=1, deaths=2, assists=5, kda=3.0,
        vision_score_per_min=2.4, kill_participation=0.46, effective_heal_shield=3825,
        team_damage_pct=0.15, dragon_takedowns=1, enemy_immobilizations=31, ward_takedowns=9,
        deaths_ctx=(DeathEvent(game_time_s=500, x=1, y=2),),
        ward_events=(WardEvent(game_time_s=400, action="placed"),),
        objective_events=(ObjectiveEvent(game_time_s=1600, kind="dragon"),),
    )


def test_dict_to_record_inverts_record_to_dict(tmp_path):
    original = _game()
    line = ingest._record_to_dict(original)
    assert ingest._dict_to_record(line) == original


def test_load_snapshot_reads_jsonl_into_records(tmp_path):
    import json
    path = tmp_path / "games.jsonl"
    path.write_text(json.dumps(ingest._record_to_dict(_game())) + "\n")
    games = ingest.load_snapshot(path)
    assert len(games) == 1 and games[0].match_id == "SG2_1"


def test_producer_runs_over_a_loaded_snapshot(tmp_path):
    # end-to-end on hand-built data: load -> metrics -> place -> render, no files of ours.
    import json
    path = tmp_path / "games.jsonl"
    path.write_text(json.dumps(ingest._record_to_dict(_game())) + "\n")
    subject = player_metrics(ingest.load_snapshot(path))
    block = producer.render_context_block(
        producer.place_metrics(subject, _cohort()), subject["games"], subject["whole_game"]
    )
    assert "Player: Silver support main" in block
