"""Personal-plane snapshot schema -- the one *code* home for a cleaned game record.

The personal-plane analog of `carryia/schema.py`: where that module is the frozen
contract for a coaching *tip*, this is the frozen contract for one of the
subject's *games*. Each line in `data/snapshot/games.jsonl` deserialises to one
`GameRecord`; `data/snapshot/meta.json` deserialises to one `SnapshotMeta`.

Kept in `carryia/personal/` (not the top-level
`schema.py`) to honour the two-plane separation -- knowledge-plane and
personal-plane records never share a module.

**Frozen because the snapshot is a committed artifact.** The ingest runs once,
offline, with the author's key;
reviewers read the committed files, never re-pull. Once written, a record is read
and aggregated, never mutated.

**Cleaned, not raw.** These fields are the *extraction* -- the KPI stat line
distilled from Match-V5. The raw Riot payloads are persisted separately under
`data/raw/riot/` (write-raw-before-cleaned, so a rate-limit stall never forces a
re-pull). This record is what the Stat-Line Producer (SEAM ②) averages over N
games into the runtime stat line; the average itself is *produced*, not frozen.

**Prefer Riot's `challenges.*` derived values over recomputing them**
(they expose stats you'd otherwise mis-derive). So the KPI fields
here hold Riot's own derived numbers, not raw counts we divided ourselves.

Three field-picks left open at design time ("resolve at build time in the Role
Profile") are resolved here -- implementation mechanics, kept in this docstring:

  1. **CC -> `enemy_immobilizations`** (`challenges.enemyChampionImmobilizations`,
     a *count*), not `timeCCingOthers` (seconds). A count is coachable ("land more
     hooks/stuns"); seconds-of-CC is abstract. The field name states what it is.
  2. **Vision denied -> `ward_takedowns`** (`challenges.wardTakedowns`), the single
     cleanest "vision denied" number, over the raw `wardsKilled` and the
     lane-phase-only `wardTakedownsBefore20M`.
  3. **Lane state @10-14 -> DEFERRED to V2.** It is a *diff vs the paired UTILITY
     opponent*, so it needs the opponent's per-minute frames -- and V1 stores only
     the subject's own stats. It re-enters with the `@game` head-to-head
     work, when a snapshot re-pull adds the opponent fields.

The record also carries three **phased event streams** -- deaths, ward actions,
and objective participation -- each a timeline of `(game_time_s, ...)` facts, not
a per-phase count. The phase is derived at read time (`phases.py`), so the
per-phase stat line the Producer builds can never drift from the boundaries. The
ward + objective streams were added 2026-09-01 (the cohort-benchmark decision:
every phased metric is judged against the cohort, so the ones that phase must be
carried per-event); they extend the original death-only stream and re-extract
from the already-saved raw timelines -- no Riot re-pull.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

__all__ = [
    "DeathEvent",
    "GameRecord",
    "ObjectiveEvent",
    "SnapshotMeta",
    "SubjectProfile",
    "UTILITY",
    "WardEvent",
]

# The Match-V5 `teamPosition` value that means "support" -- the snapshot's
# membership rule: a game is admitted iff the subject
# played this position. One home for the magic string.
UTILITY = "UTILITY"


@dataclass(frozen=True, slots=True)
class DeathEvent:
    """One of the subject's deaths, with the context that makes it coachable.

    "Deaths are the exception that shaped the design" (`architecture-coach.md`):
    a raw death count says nothing, but *when* and *where* you died is a pattern a
    coach can name ("you keep dying in the river during mid game"). Sourced from
    the timeline's `CHAMPION_KILL` events where the subject is the victim.

    We freeze the **facts** (game time + map position) and NOT the derived phase:
    `phases.py` is the single home for the game-time -> phase mapping, so the phase
    is derived at read time from `game_time_s`. Freezing it here would rot the
    record if a phase boundary ever moved.
    """

    game_time_s: int   # seconds from game start (timeline event `timestamp` / 1000)
    x: int             # map x of the kill (event `position.x`)
    y: int             # map y of the kill (event `position.y`)


@dataclass(frozen=True, slots=True)
class WardEvent:
    """One of the subject's ward actions -- a ward placed or a ward cleared.

    The per-phase proxy for vision work: vision *score* is a whole-game Riot
    aggregate that cannot be split by phase, so ward events carry the phased
    signal instead. Sourced from the timeline's
    `WARD_PLACED` events where `creatorId` is the subject (placed) and `WARD_KILL`
    events where `killerId` is the subject (cleared).

    Like `DeathEvent`, we freeze the **fact** (game time) and derive the phase at
    read time via `phases.py`. Timeline ward events carry no map position, so there
    is none to freeze.

    **Only real vision wards are recorded** (2026-09-01): `TEEMO_MUSHROOM` and
    untyped (`UNDEFINED`) placements are dropped as non-vision, so the stream is a
    clean vision proxy. The `wardType` whitelist filter lives in `_wards`.
    """

    game_time_s: int   # seconds from game start (timeline event `timestamp` / 1000)
    action: str        # "placed" (WARD_PLACED, creatorId=subject) | "cleared" (WARD_KILL, killerId=subject)


@dataclass(frozen=True, slots=True)
class ObjectiveEvent:
    """One epic monster or tower the subject was credited for -- the phased
    objective-participation signal.

    Sourced from the timeline's `ELITE_MONSTER_KILL` (dragon / herald / baron, plus
    void grubs → `grub`, included 2026-09-01 as a laning-phase objective) and
    `BUILDING_KILL` (tower) events where the subject is the killer or appears in
    `assistingParticipantIds`. **Credit-only** is an accepted caveat: uncredited
    zoning or setup reads as non-participation.

    Freezes the **facts** (game time + objective kind) and derives the phase at
    read time via `phases.py`.
    """

    game_time_s: int   # seconds from game start (timeline event `timestamp` / 1000)
    kind: str          # "dragon"|"herald"|"baron"|"grub" (ELITE_MONSTER_KILL) | "tower" (BUILDING_KILL)


@dataclass(frozen=True, slots=True)
class GameRecord:
    """One of the subject's support games -- the cleaned KPI stat line + context.

    Field order groups identity/outcome, the raw KDA line, the support KPI set
    (the ✅ rows of the field-mapping table), and the phased event streams
    (deaths, wards, objectives). Every KPI is Riot's own `challenges.*` derived
    value unless noted.
    """

    # --- identity & outcome ------------------------------------------------
    match_id: str          # Match-V5 id (e.g. "NA1_5123456789") -- already stable, no derive
    game_creation: date    # day the game was played (info.gameCreation ms -> date); for recency
    game_duration_s: int   # info.gameDuration (seconds)
    champion: str          # participant.championName
    queue_id: int          # info.queueId (420 = ranked solo, the snapshot's only queue)
    win: bool              # participant.win

    # --- KDA (raw counts + Riot's derived ratio) ---------------------------
    kills: int             # participant.kills
    deaths: int            # participant.deaths
    assists: int           # participant.assists
    kda: float             # challenges.kda (Riot-derived; not recomputed)

    # --- support KPI stat line (challenges.* derived) ----------------------
    vision_score_per_min: float   # challenges.visionScorePerMinute
    kill_participation: float     # challenges.killParticipation (0..1)
    effective_heal_shield: int    # challenges.effectiveHealAndShielding
    team_damage_pct: float        # challenges.teamDamagePercentage (0..1)
    dragon_takedowns: int         # challenges.dragonTakedowns
    enemy_immobilizations: int    # challenges.enemyChampionImmobilizations (CC, see module doc)
    ward_takedowns: int           # challenges.wardTakedowns (vision denied, see module doc)

    # --- phased event streams (timeline; ward/objective added 2026-09-01) ---
    deaths_ctx: tuple[DeathEvent, ...]            # subject-as-victim CHAMPION_KILL events, in order
    ward_events: tuple[WardEvent, ...]            # WARD_PLACED (placed) / WARD_KILL (cleared), subject the actor
    objective_events: tuple[ObjectiveEvent, ...]  # epics + towers the subject was credited for


@dataclass(frozen=True, slots=True)
class SnapshotMeta:
    """The snapshot's subject + provenance -- one record, committed once.

    Holds the constant-across-games identity (repeating it per line would be
    noise) and the reproducibility trail a reviewer needs to know *whose* games
    these are and *when* they were pulled. The `puuid` is the stable join key;
    the `riot_id` is the human-readable handle it resolved from.
    """

    riot_id: str        # "gameName#tagLine" -- the only way in since name lookup was removed
    puuid: str          # Account-V1 puuid -- stable join key for every Match-V5 call
    region: str         # regional route the matches were pulled from (americas/europe/asia/sea)
    queue_id: int       # the queue filter the match list was pulled under (420)
    game_count: int     # N -- number of support games in games.jsonl
    retrieved_at: date  # the day the author ran the ingest


@dataclass(frozen=True, slots=True)
class SubjectProfile:
    """The subject's platform-route profile -- icon, level, rank, and split record.

    Deliberately its own record (and its own `data/snapshot/profile.json`), NOT part
    of `SnapshotMeta`. `SnapshotMeta` is the *regional* match-snapshot provenance
    (`app.py` reads it, so it stays byte-stable); this is the *platform*-route subject
    profile, which churns on Riot's split schedule, not the match pull's. Sourced by
    the one-time author pull `profile_pull.py`:

      - `profile_icon_id`, `summoner_level`  <- Summoner-V4 `by-puuid`
      - `tier` .. `losses`                   <- League-V4 `by-puuid` (RANKED_SOLO_5x5)

    **`wins`/`losses` are the CURRENT split, not lifetime** -- Riot exposes no career
    total, and the counts reset each split. So the only
    honest label a UI can put on the derived win rate is "this split", and it may be a
    *smaller* sample than the committed 10-game snapshot early in a split.

    **Unranked is real state.** If the subject has no RANKED_SOLO_5x5 placement this
    split, League-V4 omits the entry entirely; the rank fields are then `None` and
    consumers show "Unranked" (no win rate to derive). Icon/level still resolve.

    Frozen and committed like the snapshot: the author pulls once with their key,
    reviewers read `profile.json`, never re-pull.
    """

    # --- Summoner-V4 (always present) --------------------------------------
    profile_icon_id: int    # summoner.profileIconId -- Data Dragon `profileicon/{id}.png`
    summoner_level: int     # summoner.summonerLevel -- account level (not rank)
    platform: str           # platform shard the rank was read from (na1/euw1/sg2/...)
    queue_type: str         # the League-V4 queueType read (RANKED_SOLO_5x5)
    retrieved_at: date      # the day the author ran the profile pull (rank is this-split)

    # --- League-V4 RANKED_SOLO_5x5 (None when unranked this split) ----------
    tier: str | None        # "SILVER" | ... | None if unranked
    rank: str | None        # division "I".."IV" (blank for apex tiers) | None
    league_points: int | None  # leaguePoints (LP) | None
    wins: int | None        # cumulative ranked wins THIS SPLIT | None
    losses: int | None      # cumulative ranked losses THIS SPLIT | None
