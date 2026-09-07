"""Spec for carryia/personal/ingest.py -- the two new timeline extractors.

Scope: `_wards`, `_objectives`, and the `extract_game` wiring that carries their
output onto the `GameRecord`. The pre-existing `_deaths` / KPI extraction is not
re-specced here; this file pins only the streams added 2026-09-01 (the
cohort-benchmark decision -- every phased metric is judged against the cohort, so
ward activity and objective participation must be carried per-event).

TDD state on first run (the house three-state, as in test_phases.py /
test_riot_client.py):

  - GREEN -- regression locks on the SETTLED contract. The `WardEvent` /
    `ObjectiveEvent` dataclasses and the two new `GameRecord` fields already exist
    in snapshot.py; these tests lock their shape, and lock the hard boundary that
    the events carry a raw `game_time_s` and NO phase (phase is derived at read
    time by the Producer via phases.py -- never stored). Break either and one goes
    red.

  - RED -- contract stubs. `_wards`, `_objectives`, and the `extract_game` wiring
    are NOT implemented yet: the helpers don't exist (AttributeError) and today's
    `extract_game` omits the two fields, so it can't even construct a `GameRecord`
    (TypeError). Both are legitimate "behaviour not written yet" reds -- the
    assertions are forced invariants (routing of event -> stream, the creatorId/
    killerId field asymmetry, the credit rule, `timestamp // 1000`, order), true
    regardless of HOW you implement. Make them green by writing the bodies. No
    NotImplementedError stubs are shipped here on purpose -- this is a tests-only
    handoff.

  - SKIPPED -- decision tests. None remain: the two forks this file raised (the
    ward-type whitelist and the unmapped void-grub monsterType) were both resolved
    2026-09-01 and are now folded into the RED contract tests below. If a fresh
    fork surfaces, re-add it as a skipped test rather than baking a guess.

Hard boundary (do not violate): no test here asserts a phase. The extractor stores
raw `(game_time_s, ...)` facts only.

Run:  .venv/bin/pytest tests/test_personal_ingest.py -v
"""

import dataclasses

import pytest

from carryia.personal import ingest
from carryia.personal.snapshot import GameRecord, ObjectiveEvent, WardEvent

# --- participants ------------------------------------------------------------
# The subject is participant #2; #7 is some other player whose events must never
# leak into the subject's streams. The subject's puuid is the join key detail
# extraction matches on.
SUBJECT_PID = 2
OTHER_PID = 7
PUUID = "PUUID-SUBJECT"


# --- inline timeline fixtures ------------------------------------------------
# Small hand-rolled Match-V5 timeline fragments, in the house zero-dependency
# style (plain dicts, no mocking library). Real WARD_PLACED events carry
# `creatorId`; WARD_KILL events carry `killerId` (never a creatorId) -- the field
# asymmetry the extractor has to respect -- so the builders mirror that.


def _timeline(events=(), *, frames=None):
    """Wrap events into the Match-V5 timeline shape (`info.frames[].events`).

    Pass a flat `events` iterable for a single frame, or `frames=[[...], [...]]`
    to spread events across frames and exercise cross-frame ordering.
    """
    if frames is None:
        frames = [list(events)]
    return {"info": {"frames": [{"events": list(f)} for f in frames]}}


def _ward_placed(creator_id, ts, ward_type="YELLOW_TRINKET"):
    # WARD_PLACED credits the PLACER via `creatorId` (no killerId key exists).
    return {"type": "WARD_PLACED", "creatorId": creator_id, "wardType": ward_type, "timestamp": ts}


def _ward_kill(killer_id, ts, ward_type="CONTROL_WARD"):
    # WARD_KILL credits the CLEARER via `killerId` (no creatorId key exists).
    return {"type": "WARD_KILL", "killerId": killer_id, "wardType": ward_type, "timestamp": ts}


def _elite(monster_type, ts, killer_id=None, assists=None):
    ev = {"type": "ELITE_MONSTER_KILL", "monsterType": monster_type, "timestamp": ts}
    if killer_id is not None:
        ev["killerId"] = killer_id
    if assists is not None:
        ev["assistingParticipantIds"] = assists
    return ev


def _building(building_type, ts, killer_id=None, assists=None):
    ev = {"type": "BUILDING_KILL", "buildingType": building_type, "timestamp": ts}
    if killer_id is not None:
        ev["killerId"] = killer_id
    if assists is not None:
        ev["assistingParticipantIds"] = assists
    return ev


def _detail(*, participant_id=SUBJECT_PID, puuid=PUUID):
    """Minimal Match-V5 detail: just enough for `extract_game` to resolve the
    subject participant and read the KPI line. The stream contents come from the
    timeline argument, not from here."""
    return {
        "metadata": {"matchId": "SG2_5123456789"},
        "info": {
            "gameCreation": 1_700_000_000_000,
            "gameDuration": 1800,
            "gameEndTimestamp": 1_700_000_001_800,  # presence => gameDuration is seconds
            "queueId": 420,
            "participants": [
                {
                    "puuid": puuid,
                    "participantId": participant_id,
                    "championName": "Lux",
                    "win": True,
                    "kills": 1,
                    "deaths": 2,
                    "assists": 11,
                    "challenges": {
                        "kda": 6.0,
                        "visionScorePerMinute": 2.4,
                        "killParticipation": 0.62,
                        "effectiveHealAndShielding": 5000,
                        "teamDamagePercentage": 0.14,
                        "dragonTakedowns": 1,
                        "enemyChampionImmobilizations": 20,
                        "wardTakedowns": 8,
                    },
                },
                {"puuid": "PUUID-OTHER", "participantId": OTHER_PID, "championName": "Jinx"},
            ],
        },
    }


# --- GREEN: the settled dataclass contract these extractors must produce ------


def test_ward_event_shape_is_game_time_and_action_only():
    names = [f.name for f in dataclasses.fields(WardEvent)]
    assert names == ["game_time_s", "action"]


def test_objective_event_shape_is_game_time_and_kind_only():
    names = [f.name for f in dataclasses.fields(ObjectiveEvent)]
    assert names == ["game_time_s", "kind"]


@pytest.mark.parametrize("cls", [WardEvent, ObjectiveEvent])
def test_event_stores_no_phase(cls):
    # Hard boundary: the extractor freezes raw facts; phase is derived at read
    # time via phases.py and never stored. No field may name a phase.
    assert not any("phase" in f.name for f in dataclasses.fields(cls))


@pytest.mark.parametrize("cls", [WardEvent, ObjectiveEvent])
def test_events_are_frozen(cls):
    ev = cls(game_time_s=10, action="placed") if cls is WardEvent else cls(game_time_s=10, kind="dragon")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.game_time_s = 99  # committed artifact -- read, never mutated


def test_game_record_carries_both_new_streams():
    names = {f.name for f in dataclasses.fields(GameRecord)}
    assert {"ward_events", "objective_events"} <= names


# --- RED: _wards -------------------------------------------------------------
# Forced invariants. `_wards(timeline, participant_id) -> tuple[WardEvent, ...]`.


def test_wards_placement_credited_to_creator_id():
    tl = _timeline([_ward_placed(SUBJECT_PID, 65_000)])
    assert ingest._wards(tl, SUBJECT_PID) == (WardEvent(game_time_s=65, action="placed"),)


def test_wards_clear_credited_to_killer_id():
    # The asymmetry: WARD_KILL has no creatorId, so a clear must be read off
    # `killerId`. An implementation that keys both event types on creatorId would
    # silently drop every clear -- this is what stops that.
    tl = _timeline([_ward_kill(SUBJECT_PID, 130_000)])
    assert ingest._wards(tl, SUBJECT_PID) == (WardEvent(game_time_s=130, action="cleared"),)


def test_wards_exclude_other_players_events():
    tl = _timeline([_ward_placed(OTHER_PID, 60_000), _ward_kill(OTHER_PID, 90_000)])
    assert ingest._wards(tl, SUBJECT_PID) == ()


def test_wards_game_time_is_timestamp_floor_divided_by_1000():
    tl = _timeline([_ward_placed(SUBJECT_PID, 65_432)])
    assert ingest._wards(tl, SUBJECT_PID)[0].game_time_s == 65


def test_wards_preserve_timeline_order_across_frames():
    tl = _timeline(
        frames=[
            [_ward_placed(SUBJECT_PID, 30_000), _ward_kill(OTHER_PID, 31_000)],
            [_ward_kill(SUBJECT_PID, 130_000), _ward_placed(SUBJECT_PID, 140_000)],
        ]
    )
    assert ingest._wards(tl, SUBJECT_PID) == (
        WardEvent(game_time_s=30, action="placed"),
        WardEvent(game_time_s=130, action="cleared"),
        WardEvent(game_time_s=140, action="placed"),
    )


def test_wards_empty_when_subject_has_no_ward_events():
    tl = _timeline([_elite("DRAGON", 600_000, killer_id=SUBJECT_PID)])
    assert ingest._wards(tl, SUBJECT_PID) == ()


# --- RED: _objectives --------------------------------------------------------
# Forced invariants. `_objectives(timeline, participant_id) -> tuple[ObjectiveEvent, ...]`.


@pytest.mark.parametrize(
    "monster_type, kind",
    [
        ("DRAGON", "dragon"),
        ("RIFTHERALD", "herald"),
        ("BARON_NASHOR", "baron"),
        ("HORDE", "grub"),  # void grubs -- included, kind='grub'
    ],
)
def test_objectives_map_monster_type_to_kind(monster_type, kind):
    tl = _timeline([_elite(monster_type, 600_000, killer_id=SUBJECT_PID)])
    assert ingest._objectives(tl, SUBJECT_PID) == (ObjectiveEvent(game_time_s=600, kind=kind),)


def test_objectives_tower_from_building_kill():
    tl = _timeline([_building("TOWER_BUILDING", 720_000, killer_id=SUBJECT_PID)])
    assert ingest._objectives(tl, SUBJECT_PID) == (ObjectiveEvent(game_time_s=720, kind="tower"),)


def test_objectives_exclude_inhibitor_building():
    # Only TOWER_BUILDING counts; INHIBITOR_BUILDING (and nexus) are out of scope.
    tl = _timeline([_building("INHIBITOR_BUILDING", 1_500_000, killer_id=SUBJECT_PID)])
    assert ingest._objectives(tl, SUBJECT_PID) == ()


@pytest.mark.parametrize(
    "event",
    [
        _elite("DRAGON", 600_000, killer_id=OTHER_PID, assists=[3, SUBJECT_PID, 5]),
        _building("TOWER_BUILDING", 900_000, killer_id=OTHER_PID, assists=[SUBJECT_PID]),
    ],
)
def test_objectives_credit_subject_via_assisting_ids(event):
    # Uniform credit rule: killerId OR present in assistingParticipantIds -- applied
    # the same way to epic monsters and towers.
    assert len(ingest._objectives(_timeline([event]), SUBJECT_PID)) == 1


@pytest.mark.parametrize(
    "event",
    [
        _elite("BARON_NASHOR", 1_600_000, killer_id=OTHER_PID, assists=[3, 4, 5]),
        _building("TOWER_BUILDING", 800_000, killer_id=OTHER_PID, assists=[6]),
    ],
)
def test_objectives_exclude_uncredited_events(event):
    # Accepted "credit-only" caveat: zoning/setup with
    # no kill or assist credit reads as non-participation.
    assert ingest._objectives(_timeline([event]), SUBJECT_PID) == ()


def test_objectives_game_time_is_timestamp_floor_divided_by_1000():
    tl = _timeline([_elite("DRAGON", 634_999, killer_id=SUBJECT_PID)])
    assert ingest._objectives(tl, SUBJECT_PID)[0].game_time_s == 634


def test_objectives_preserve_timeline_order():
    tl = _timeline(
        frames=[
            [_elite("DRAGON", 600_000, killer_id=SUBJECT_PID)],
            [
                _building("TOWER_BUILDING", 900_000, killer_id=OTHER_PID, assists=[SUBJECT_PID]),
                _elite("BARON_NASHOR", 1_600_000, killer_id=SUBJECT_PID),
            ],
        ]
    )
    assert ingest._objectives(tl, SUBJECT_PID) == (
        ObjectiveEvent(game_time_s=600, kind="dragon"),
        ObjectiveEvent(game_time_s=900, kind="tower"),
        ObjectiveEvent(game_time_s=1_600, kind="baron"),
    )


def test_objectives_missing_assist_key_treated_as_empty():
    # Robustness default, NOT a design fork (see report): an event by another
    # player with the `assistingParticipantIds` key absent must not raise, and
    # reads as uncredited. Forces `.get(..., [])` rather than `ev[...]`.
    ev = {"type": "ELITE_MONSTER_KILL", "monsterType": "DRAGON", "timestamp": 600_000, "killerId": OTHER_PID}
    assert ingest._objectives(_timeline([ev]), SUBJECT_PID) == ()


def test_objectives_missing_assist_key_still_credits_killer():
    # The other side of the same default: with no assist key but the subject as
    # killerId, the event is still credited via killerId.
    ev = {"type": "BUILDING_KILL", "buildingType": "TOWER_BUILDING", "timestamp": 720_000, "killerId": SUBJECT_PID}
    assert ingest._objectives(_timeline([ev]), SUBJECT_PID) == (ObjectiveEvent(game_time_s=720, kind="tower"),)


# --- RED: extract_game wires both new streams onto the GameRecord ------------


def test_extract_game_wires_all_three_streams_additively():
    # A death, a ward, and an objective -- the new wiring must populate ward and
    # objective streams WITHOUT dropping the pre-existing deaths stream.
    tl = _timeline(
        [
            {"type": "CHAMPION_KILL", "victimId": SUBJECT_PID, "timestamp": 522_000, "position": {"x": 100, "y": 200}},
            _ward_placed(SUBJECT_PID, 65_000),
            _elite("DRAGON", 600_000, killer_id=SUBJECT_PID),
        ]
    )
    rec = ingest.extract_game(_detail(), tl, PUUID)
    assert rec.deaths_ctx  # pre-existing stream still populated
    assert rec.ward_events == (WardEvent(game_time_s=65, action="placed"),)
    assert rec.objective_events == (ObjectiveEvent(game_time_s=600, kind="dragon"),)


def test_extract_game_empty_streams_when_subject_uninvolved():
    tl = _timeline([_ward_placed(OTHER_PID, 60_000), _elite("DRAGON", 600_000, killer_id=OTHER_PID)])
    rec = ingest.extract_game(_detail(), tl, PUUID)
    assert rec.ward_events == ()
    assert rec.objective_events == ()


# --- RED: _wards counts only real vision-ward types (decision resolved) ------
# Decision: ward ACTIVITY counts only real vision
# wards -- the whitelist {YELLOW_TRINKET, CONTROL_WARD, SIGHT_WARD, BLUE_TRINKET}
# -- applied to BOTH WARD_PLACED (placed) and WARD_KILL (cleared), since WARD_KILL
# carries a `wardType` too (clearing a mushroom is not vision work). TEEMO_MUSHROOM
# and UNDEFINED are dropped for either action.


@pytest.mark.parametrize("action, build", [("placed", _ward_placed), ("cleared", _ward_kill)])
@pytest.mark.parametrize(
    "ward_type, expected",
    [
        ("YELLOW_TRINKET", 1),
        ("CONTROL_WARD", 1),
        ("SIGHT_WARD", 1),
        ("BLUE_TRINKET", 1),
        ("TEEMO_MUSHROOM", 0),
        ("UNDEFINED", 0),
    ],
)
def test_wards_count_only_vision_ward_types(action, build, ward_type, expected):
    tl = _timeline([build(SUBJECT_PID, 60_000, ward_type=ward_type)])
    assert len(ingest._wards(tl, SUBJECT_PID)) == expected
