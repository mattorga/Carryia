"""Spec for carryia/personal/cohort.py -- the cohort-crawl pilot.

Scope: the pure, deterministic logic -- seeding routing, the puuid fallback, the
per-player collect loop (over-fetch, the all-phase floor, the support-main
screen), and the 5->N projection math. The Riot HTTP layer is a hand-rolled fake
client (house zero-dependency style, as in test_riot_client.py); nothing here
touches the network or disk.

TDD state: all GREEN -- cohort.py is implemented, so these are regression locks on
its settled behaviour (the measured ratios `project` scales must stay honest, and
the collect loop must pay the heavy timeline call only for a kept game).

Run:  .venv/bin/pytest tests/test_cohort.py -v
"""

import json

import pytest

from carryia.personal import cohort
from carryia.personal.cohort import (
    PlayerDiag,
    aggregate_cohort,
    collect_player_games,
    project,
    seed_players,
)
from carryia.personal.ingest import IngestError
from carryia.personal.metrics import PHASED_METRICS, WHOLE_GAME_METRICS
from carryia.personal.riot_client import PlatformRoute, RegionalRoute

PUUID = "PUUID-SEED"


# --- fake client -------------------------------------------------------------

def _detail(match_id, position, duration_s, *, puuid=PUUID, pid=2):
    """Minimal Match-V5 detail: enough for `_subject`, `_duration_s`, and
    `extract_game`. `gameEndTimestamp` present => gameDuration is already seconds."""
    return {
        "metadata": {"matchId": match_id},
        "info": {
            "gameCreation": 1_700_000_000_000,
            "gameDuration": duration_s,
            "gameEndTimestamp": 1_700_000_000_000 + duration_s * 1000,
            "queueId": 420,
            "participants": [
                {
                    "puuid": puuid,
                    "participantId": pid,
                    "teamPosition": position,
                    "championName": "Lux",
                    "win": True,
                    "kills": 1, "deaths": 2, "assists": 11,
                    "challenges": {"kda": 6.0, "visionScorePerMinute": 2.4},
                },
                {"puuid": "PUUID-OTHER", "participantId": 7, "championName": "Jinx"},
            ],
        },
    }


class FakeClient:
    """Serves canned Riot responses from a per-puuid list of (position, duration_s)
    game specs, and records how many of each call it answered."""

    def __init__(self, games_by_puuid, *, entries=None, summoner_puuid=None):
        self._games = games_by_puuid           # puuid -> [(position, duration_s), ...]
        self._entries = entries or []          # for seed_players
        self._summoner_puuid = summoner_puuid   # for summoner_by_id fallback
        self.last_app_rate_limit = "100:120"
        self.last_app_rate_limit_count = "5:120"
        self.calls = {"match_ids": 0, "match": 0, "timeline": 0, "summoner": 0, "seed": 0}

    def match_ids_by_puuid(self, puuid, *, queue=420, count=60, **_):
        self.calls["match_ids"] += 1
        return [f"{puuid}_M{i}" for i in range(len(self._games[puuid]))][:count]

    def match(self, match_id):
        self.calls["match"] += 1
        puuid, idx = match_id.rsplit("_M", 1)
        pos, dur = self._games[puuid][int(idx)]
        return _detail(match_id, pos, dur, puuid=puuid)

    def match_timeline(self, match_id):
        self.calls["timeline"] += 1
        return {"info": {"frames": [{"events": []}]}}

    def apex_league(self, tier, *, queue="RANKED_SOLO_5x5"):
        self.calls["seed"] += 1
        return {"entries": self._entries}

    def entries_by_queue(self, tier, division, *, queue="RANKED_SOLO_5x5", page=1):
        self.calls["seed"] += 1
        return self._entries if page == 1 else []  # one page, then empty (iter_seeds stops)

    def summoner_by_id(self, summoner_id):
        self.calls["summoner"] += 1
        return {"puuid": self._summoner_puuid}


@pytest.fixture(autouse=True)
def _no_disk(monkeypatch):
    """The collect loop persists raw payloads; stub it so tests never write."""
    monkeypatch.setattr(cohort, "_persist_raw", lambda *a, **k: None)


# --- seeding: apex vs paged band ---------------------------------------------

def test_seed_apex_tier_uses_apex_endpoint():
    c = FakeClient({}, entries=[{"puuid": "A"}, {"puuid": "B"}])
    assert seed_players(c, "CHALLENGER", None) == [{"puuid": "A"}, {"puuid": "B"}]
    assert c.calls["seed"] == 1


def test_seed_nonapex_tier_uses_band_endpoint():
    c = FakeClient({}, entries=[{"summonerId": "s1"}])
    assert seed_players(c, "EMERALD", "II") == [{"summonerId": "s1"}]


def test_seed_nonapex_without_division_raises():
    c = FakeClient({}, entries=[])
    with pytest.raises(IngestError):
        seed_players(c, "EMERALD", None)


# --- puuid resolution: direct vs summoner fallback ---------------------------

def test_entry_puuid_prefers_direct_puuid():
    c = FakeClient({})
    diag = PlayerDiag(puuid="")
    assert cohort._entry_puuid(c, {"puuid": "DIRECT", "summonerId": "s1"}, diag) == "DIRECT"
    assert c.calls["summoner"] == 0 and diag.summoner_calls == 0


def test_entry_puuid_falls_back_to_summoner_and_counts_it():
    c = FakeClient({}, summoner_puuid="RESOLVED")
    diag = PlayerDiag(puuid="")
    assert cohort._entry_puuid(c, {"summonerId": "s1"}, diag) == "RESOLVED"
    assert c.calls["summoner"] == 1 and diag.summoner_calls == 1


def test_entry_puuid_none_when_no_ids():
    assert cohort._entry_puuid(FakeClient({}), {}, PlayerDiag(puuid="")) is None


# --- collect: over-fetch, all-phase floor, timeline-only-on-keep --------------

def test_collect_filters_utility_and_all_phase():
    # 7 UTILITY games; only the 5 with duration >= 1500s (all-phase) are kept.
    games = [("UTILITY", 1800), ("UTILITY", 1600), ("UTILITY", 2000),
             ("UTILITY", 1500), ("UTILITY", 1700), ("UTILITY", 1000), ("UTILITY", 1200)]
    c = FakeClient({PUUID: games})
    recs, diag = collect_player_games(c, PUUID, games_per_player=10, scan_cap=60)
    assert len(recs) == 5                 # the 5 all-phase games
    assert diag.utility_games == 7        # all 7 were UTILITY
    assert diag.all_phase_kept == 5
    assert diag.details_pulled == 7       # a detail per scanned game
    assert diag.timelines_pulled == 5     # heavy call ONLY for kept games
    assert diag.qualified


def test_collect_stops_at_games_per_player():
    games = [("UTILITY", 1800)] * 12
    c = FakeClient({PUUID: games})
    recs, diag = collect_player_games(c, PUUID, games_per_player=10, scan_cap=60)
    assert len(recs) == 10
    assert diag.details_pulled == 10      # stops before scanning the 11th
    assert diag.timelines_pulled == 10


def test_collect_screens_out_non_support_main():
    # First 15 scanned are all non-UTILITY -> screen bails at SCREEN_AFTER.
    games = [("MIDDLE", 1800)] * 20
    c = FakeClient({PUUID: games})
    recs, diag = collect_player_games(c, PUUID, games_per_player=10, scan_cap=60)
    assert recs == []
    assert not diag.qualified
    assert "not support-main" in diag.note
    assert diag.details_pulled == cohort.SCREEN_AFTER  # bailed early, didn't scan all 20


def test_screen_bail_pays_no_timelines():
    # The ZAekOrmo fix: all-phase support games found DURING the screen window are
    # discarded with zero timeline cost when the verdict fails. 5 UTILITY all-phase
    # + 10 MIDDLE in the first 15 -> 5/15 == 0.33 < 0.40 -> bail at 15.
    games = [("UTILITY", 1800)] * 5 + [("MIDDLE", 1800)] * 12
    c = FakeClient({PUUID: games})
    recs, diag = collect_player_games(c, PUUID, games_per_player=10, scan_cap=60)
    assert recs == [] and not diag.qualified
    assert "not support-main" in diag.note
    assert diag.details_pulled == cohort.SCREEN_AFTER  # bailed at the mark, not later
    assert diag.timelines_pulled == 0                  # nothing extracted despite 5 candidates


def test_screen_decides_once_and_commits():
    # Passes at the 15-game mark (8/15 UTILITY), then trailing non-UTILITY games
    # drag the overall fraction under 0.40 -- but the verdict is NOT re-run, so the
    # player stays qualified (no mid-scan bail, no discard).
    games = [("UTILITY", 1800)] * 8 + [("MIDDLE", 1800)] * 20  # 8/28 overall = 0.29
    c = FakeClient({PUUID: games})
    recs, diag = collect_player_games(c, PUUID, games_per_player=10, scan_cap=60)
    assert diag.qualified                 # passed at 15 (8/15=0.53); overall 0.29 never re-checked
    assert diag.all_phase_kept == 8
    assert diag.timelines_pulled == 8
    assert diag.details_pulled == 28      # kept scanning for more candidates, never bailed


def test_collect_rejects_below_min_all_phase():
    # 4 all-phase UTILITY games (< MIN_QUALIFY_GAMES) -> dropped, zero timelines.
    games = [("UTILITY", 1800)] * 4
    c = FakeClient({PUUID: games})
    recs, diag = collect_player_games(c, PUUID, games_per_player=10, scan_cap=60)
    assert recs == [] and not diag.qualified
    assert f"need {cohort.MIN_QUALIFY_GAMES}" in diag.note
    assert diag.timelines_pulled == 0     # sub-threshold player costs no timelines


def test_collect_note_when_utility_but_none_all_phase():
    games = [("UTILITY", 1000), ("UTILITY", 1200)]  # UTILITY but all short
    c = FakeClient({PUUID: games})
    recs, diag = collect_player_games(c, PUUID, games_per_player=10, scan_cap=60)
    assert recs == [] and not diag.qualified
    assert "no all-phase UTILITY" in diag.note


# --- PlayerDiag.api_calls ----------------------------------------------------

def test_api_calls_totals_every_billable_call():
    d = PlayerDiag(puuid="p", details_pulled=12, timelines_pulled=10, summoner_calls=1)
    assert d.api_calls == 1 + 12 + 10 + 1  # +1 for the id-list call


# --- projection math ---------------------------------------------------------

def _qual(calls_detail, calls_tl):
    d = PlayerDiag(puuid="q", details_pulled=calls_detail, timelines_pulled=calls_tl,
                   all_phase_kept=calls_tl, qualified=True)
    return d


def test_project_scales_calls_from_measured_ratios():
    # 2 qualified (21 calls each), 1 screened-out (16 calls). Project to 100.
    quals = [_qual(10, 10), _qual(10, 10)]
    screened = PlayerDiag(puuid="s", details_pulled=15, qualified=False, note="not support-main")
    report = project([*quals, screened], elapsed_s=100.0, target_players=100, seed_calls=1)

    assert report["measured_ratios"]["calls_per_qualified_player"] == 21.0
    assert report["measured_ratios"]["screened_out_per_qualified"] == 0.5
    # seed(1) + 100*21 + 100*0.5*16 = 2901
    assert report["projection"]["estimated_calls"] == 2901
    assert report["projection"]["target_players"] == 100


def test_ceiling_per_s_takes_tightest_bucket():
    # Dev key: min(20/1, 100/120) == the 2-min bucket at 0.833/s.
    assert cohort._ceiling_per_s("20:1,100:120") == pytest.approx(100 / 120)
    # App key: min(500/10, 30000/600) == 50/s -- ~60x the dev key.
    assert cohort._ceiling_per_s("500:10,30000:600") == pytest.approx(50.0)


def test_ceiling_per_s_defaults_to_devkey_without_header():
    assert cohort._ceiling_per_s(None) == pytest.approx(100 / 120)


def test_project_ceiling_reflects_app_key_header():
    quals = [_qual(10, 10), _qual(10, 10)]
    at_dev = project(quals, elapsed_s=100.0, target_players=100, seed_calls=1)
    at_app = project(quals, elapsed_s=100.0, target_players=100, seed_calls=1,
                     rate_limit_header="500:10,30000:600")
    # Same call estimate, but the app-key ceiling is far faster (fewer hours).
    assert at_app["projection"]["estimated_calls"] == at_dev["projection"]["estimated_calls"]
    assert at_app["projection"]["estimated_hours_at_ratelimit_ceiling"] < \
        at_dev["projection"]["estimated_hours_at_ratelimit_ceiling"]


def test_project_errors_when_nobody_qualified():
    screened = PlayerDiag(puuid="s", details_pulled=15, qualified=False, note="not support-main")
    report = project([screened], elapsed_s=50.0, target_players=100, seed_calls=1)
    assert "error" in report


# --- aggregation: quartiles + cohort shape -----------------------------------

def test_quartiles_inclusive():
    q = cohort._quartiles([1.0, 2.0, 3.0, 4.0, 5.0])
    assert (q["q1"], q["median"], q["q3"], q["mean"], q["n"]) == (2.0, 3.0, 4.0, 3.0, 5)


def test_quartiles_single_value_degenerates():
    assert cohort._quartiles([7.0]) == {
        "q1": 7.0, "median": 7.0, "q3": 7.0, "mean": 7.0, "n": 1, "values": [7.0],
    }


def test_quartiles_carries_sorted_distribution():
    # the anonymous per-player distribution the UI ranks the subject against.
    q = cohort._quartiles([3.0, 1.0, 2.0])
    assert q["values"] == [1.0, 2.0, 3.0]


def _player_metric(deaths_laning, vspm):
    """A minimal per-player metrics dict with one varying phased + one whole-game value."""
    return {
        "games": 6,
        "phased": {m: {"laning": deaths_laning, "mid": 0.0, "late": 0.0} for m in PHASED_METRICS},
        "whole_game": {k: vspm for k in WHOLE_GAME_METRICS},
    }


def test_aggregate_cohort_shape_and_quartiles():
    per_player = [_player_metric(d, float(d)) for d in (1, 2, 3, 4, 5)]
    metrics = aggregate_cohort(per_player)
    # every phased metric carries all three phases; whole-game metrics are flat.
    for m in PHASED_METRICS:
        assert set(metrics[m]) == {"laning", "mid", "late"}
    for k in WHOLE_GAME_METRICS:
        assert set(metrics[k]) == {"q1", "median", "q3", "mean", "n", "values"}
    # median of 1..5 is 3, across distinct players.
    assert metrics["deaths"]["laning"]["median"] == 3.0
    assert metrics["vision_score_per_min"]["median"] == 3.0
    assert metrics["deaths"]["laning"]["n"] == 5


# --- seeding: iterate a whole tier -------------------------------------------

def test_iter_seeds_walks_all_divisions():
    c = FakeClient({}, entries=[{"puuid": "A"}])  # one entry on page 1, then empty
    seeds = list(cohort.iter_seeds(c, "SILVER", ("I", "II", "III", "IV")))
    assert len(seeds) == 4  # one per division


def test_iter_seeds_apex_uses_league_entries():
    c = FakeClient({}, entries=[{"puuid": "A"}, {"puuid": "B"}])
    assert list(cohort.iter_seeds(c, "MASTER", None)) == [{"puuid": "A"}, {"puuid": "B"}]


# --- full crawl end-to-end (client injected) ---------------------------------

def test_crawl_collects_target_and_builds_cohort(monkeypatch):
    entries = [{"puuid": "SCR"}, {"puuid": "Q1"}, {"puuid": "Q2"}]
    games = {
        "SCR": [("MIDDLE", 1800)] * 15,     # screened out at the 15-game mark
        "Q1": [("UTILITY", 1800)] * 6,      # 6 all-phase -> qualifies
        "Q2": [("UTILITY", 1800)] * 6,
    }
    c = FakeClient(games, entries=entries)
    monkeypatch.setattr(cohort, "load_key", lambda *a, **k: "KEY")
    monkeypatch.setattr(cohort, "RiotClient", lambda *a, **k: c)

    cohort_dict = cohort.crawl(
        "SILVER", ("II",), target_players=2, games_per_player=10, scan_cap=60,
        platform=PlatformRoute.SG2, regional=RegionalRoute.SEA,
    )
    meta = cohort_dict["meta"]
    assert meta["player_count"] == 2          # the two qualifiers
    assert meta["games_total"] == 12          # 6 + 6
    assert meta["seeds_tried"] == 3           # burned the screened-out seed first
    assert meta["tier"] == "SILVER" and meta["divisions"] == ["II"]
    assert set(cohort_dict["metrics"]) >= set(PHASED_METRICS) | set(WHOLE_GAME_METRICS)


def test_cohort_json_carries_no_puuids(monkeypatch):
    # Privacy invariant: the committed artifact is aggregate-only.
    entries = [{"puuid": "Q1"}, {"puuid": "Q2"}]
    games = {"Q1": [("UTILITY", 1800)] * 6, "Q2": [("UTILITY", 1800)] * 6}
    c = FakeClient(games, entries=entries)
    monkeypatch.setattr(cohort, "load_key", lambda *a, **k: "KEY")
    monkeypatch.setattr(cohort, "RiotClient", lambda *a, **k: c)

    cohort_dict = cohort.crawl(
        "SILVER", ("II",), target_players=2, games_per_player=10, scan_cap=60,
        platform=PlatformRoute.SG2, regional=RegionalRoute.SEA,
    )
    blob = json.dumps(cohort_dict)
    assert "Q1" not in blob and "Q2" not in blob
