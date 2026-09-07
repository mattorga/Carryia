"""Spec for carryia/personal/metrics.py -- per-player metric computation.

All GREEN regression locks. Pins the two contracts the cohort artifact and the
future Stat-Line Producer both depend on: (1) an event is phased at read time from
its `game_time_s` via phases.py (laning <840, mid <1500, late >=1500), and (2)
every value is a per-GAME mean, so a high-game player doesn't outweigh a low-game
one when the cohort aggregates across players.

Run:  .venv/bin/pytest tests/test_metrics.py -v
"""

from datetime import date

import pytest

from carryia.personal.metrics import PHASED_METRICS, WHOLE_GAME_METRICS, player_metrics
from carryia.personal.snapshot import DeathEvent, GameRecord, ObjectiveEvent, WardEvent


def _game(*, deaths=(), wards=(), objectives=(), vspm=2.0, kp=0.5, heal=1000,
          tdmg=0.1, cc=10, wt=5, kda=3.0, dur=1800):
    """A GameRecord with the given event times; `wards` is a list of (time, action)."""
    return GameRecord(
        match_id="M", game_creation=date(2026, 1, 1), game_duration_s=dur, champion="Lux",
        queue_id=420, win=True, kills=1, deaths=len(deaths), assists=5, kda=kda,
        vision_score_per_min=vspm, kill_participation=kp, effective_heal_shield=heal,
        team_damage_pct=tdmg, dragon_takedowns=1, enemy_immobilizations=cc, ward_takedowns=wt,
        deaths_ctx=tuple(DeathEvent(game_time_s=t, x=0, y=0) for t in deaths),
        ward_events=tuple(WardEvent(game_time_s=t, action=a) for t, a in wards),
        objective_events=tuple(ObjectiveEvent(game_time_s=t, kind="dragon") for t in objectives),
    )


def test_deaths_are_phased_by_game_time():
    # 500 -> laning (<840), 1000 -> mid (<1500), 2000 -> late (>=1500).
    g = _game(deaths=[500, 1000, 2000])
    phased = player_metrics([g])["phased"]["deaths"]
    assert phased == {"laning": 1.0, "mid": 1.0, "late": 1.0}


def test_ward_activity_counts_placed_and_cleared_together():
    g = _game(wards=[(400, "placed"), (900, "cleared"), (1600, "placed")])
    phased = player_metrics([g])["phased"]["ward_activity"]
    assert phased == {"laning": 1.0, "mid": 1.0, "late": 1.0}


def test_objective_participation_phased():
    g = _game(objectives=[820, 1499, 1500])  # laning, mid, late (1500 is the late edge)
    phased = player_metrics([g])["phased"]["objective_participation"]
    assert phased == {"laning": 1.0, "mid": 1.0, "late": 1.0}


def test_phased_values_are_per_game_means():
    # game1 has 2 laning deaths, game2 has 0 -> mean 1.0 per game.
    games = [_game(deaths=[300, 400]), _game(deaths=[])]
    assert player_metrics(games)["phased"]["deaths"]["laning"] == 1.0


def test_whole_game_kpis_are_means():
    games = [_game(vspm=2.0, kp=0.4), _game(vspm=4.0, kp=0.6)]
    wg = player_metrics(games)["whole_game"]
    assert wg["vision_score_per_min"] == 3.0
    assert wg["kill_participation"] == pytest.approx(0.5)


def test_structure_has_all_metric_families():
    m = player_metrics([_game()])
    assert set(m["phased"]) == set(PHASED_METRICS)
    assert set(m["whole_game"]) == set(WHOLE_GAME_METRICS)
    assert m["games"] == 1


def test_empty_games_raises():
    with pytest.raises(ValueError):
        player_metrics([])
