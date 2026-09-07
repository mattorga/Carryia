"""Spec for carryia/phases.py -- the phase vocabulary shared by both planes.

TDD state on first run:
  - `Phase`, the boundary constants, and `phase_at` are already implemented, so
    their tests are GREEN immediately -- they lock the spec against regression.
  - `matches` is not implemented, so its tests start RED. Make them green by
    implementing the wildcard rule you choose. The truth-table test is left
    skipped for you to fill in *after* you've decided that rule -- writing the
    expected column IS deciding the semantics, so it's deliberately yours.
"""

import json

import pytest

from carryia.phases import (
    LANING_ENDS_S,
    MID_ENDS_S,
    TEMPORAL,
    Phase,
    matches,
    phase_at,
)


# --- Phase enum -------------------------------------------------------------

def test_phase_values_are_their_lowercase_strings():
    assert (Phase.LANING, Phase.MID, Phase.LATE, Phase.ALL) == (
        "laning",
        "mid",
        "late",
        "all",
    )


def test_phase_is_a_str_so_it_serialises_straight_to_json():
    assert Phase.LANING == "laning"
    assert json.dumps({"phase": Phase.MID}) == '{"phase": "mid"}'


def test_temporal_is_the_three_clock_windows_and_excludes_the_wildcard():
    assert TEMPORAL == (Phase.LANING, Phase.MID, Phase.LATE)
    assert Phase.ALL not in TEMPORAL


# --- boundaries (regression guard on the 14:00 / 25:00 decision) ------------

def test_boundaries_match_the_pinned_decision():
    assert LANING_ENDS_S == 14 * 60  # 840
    assert MID_ENDS_S == 25 * 60     # 1500


# --- phase_at: game-time -> phase (personal-plane / write side) -------------

@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, Phase.LANING),
        (522, Phase.LANING),               # canonical "death at 8:42" example
        (LANING_ENDS_S - 1, Phase.LANING),
        (LANING_ENDS_S, Phase.MID),        # strict < : 14:00 is already mid
        (MID_ENDS_S - 1, Phase.MID),
        (MID_ENDS_S, Phase.LATE),          # strict < : 25:00 is already late
        (60 * 45, Phase.LATE),
    ],
)
def test_phase_at_maps_game_time_to_phase(seconds, expected):
    assert phase_at(seconds) == expected


def test_phase_at_never_emits_the_wildcard():
    for seconds in range(0, 60 * 60, 30):
        assert phase_at(seconds) in TEMPORAL


def test_phase_at_rejects_negative_time():
    with pytest.raises(ValueError):
        phase_at(-1)


# --- matches: phase-scoped retrieval (read side) ----------------------------
# Invariants below hold no matter which wildcard variant you pick -- they are
# forced, so they're asserted for you. Both start RED (matches is unimplemented).

@pytest.mark.parametrize("phase", TEMPORAL)
def test_a_tip_matches_a_query_for_its_own_phase(phase):
    # A laning tip must be retrievable for a laning query, or the tag is pointless.
    assert matches(phase, phase) is True


@pytest.mark.parametrize("tip", list(Phase))
@pytest.mark.parametrize("query", TEMPORAL)
def test_matches_returns_a_bool(tip, query):
    assert isinstance(matches(tip, query), bool)


# --- matches: the wildcard semantics (settled rule) -------------------------
# ALL query = un-phased, matches everything; a temporal query matches its own
# phase or ALL; strict otherwise (no adjacency bleed). See phases.matches.

@pytest.mark.parametrize(
    "tip, query, expected",
    [
        # ALL-tagged tip vs a temporal query -- whole-game advice surfaces.
        (Phase.ALL, Phase.LANING, True),
        (Phase.ALL, Phase.MID, True),
        (Phase.ALL, Phase.LATE, True),
        # Non-adjacent temporal mismatch -- strict, no bleed.
        (Phase.LANING, Phase.LATE, False),
        # Adjacent temporal mismatch -- still no bleed.
        (Phase.MID, Phase.LANING, False),
        # A deliberately un-phased query matches every tip.
        (Phase.LANING, Phase.ALL, True),
    ],
)
def test_matches_wildcard_truth_table(tip, query, expected):
    assert matches(tip, query) is expected
