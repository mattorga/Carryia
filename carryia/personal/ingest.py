"""Personal-plane match ingest -- builds the committed Match Snapshot.

Stage: the personal-plane half of P0-3 (player-data acquisition). This is the
one script that turns the Riot API into `data/snapshot/` -- the frozen games the
whole app grounds in. **Author-only, one-time, offline, with my key**; reviewers
never run it (they read the committed snapshot). A plain Python script, per the
rubric (full marks for ingestion without an orchestrator).

The pipeline is four regional calls:

    Riot ID --Account-V1--> puuid
    puuid   --Match-V5----> match-id list (newest first, queue=420)
    each id --Match-V5----> detail  (the KPI stat line; filter to subject, read teamPosition)
            --Match-V5----> timeline (death-with-context)

**Role is a post-fetch filter, not an API param** (Match-V5 has no role query):
we over-fetch the id list, pull each detail, and keep a game only if the subject
played `UTILITY`. So we scan more raw games than we
commit, stopping once `--count` support games are collected (or `--scan-cap` is
hit). Raw payloads are persisted *before* cleaning, so a rate-limit stall never
forces a re-pull.

Run:

    python -m carryia.personal.ingest --riot-id "guuji#miko" --region sea --count 10
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from carryia.paths import DATA
from carryia.personal.riot_client import RegionalRoute, RiotAPIError, RiotClient
from carryia.personal.snapshot import (
    UTILITY,
    DeathEvent,
    GameRecord,
    ObjectiveEvent,
    SnapshotMeta,
    WardEvent,
)

SNAPSHOT_DIR = DATA / "snapshot"
RAW_DIR = DATA / "raw" / "riot"
QUEUE_RANKED_SOLO = 420
# Below this, a game is a remake/aborted match (a real pull surfaced a 4.7-min,
# 0/0/2 game that Riot did NOT flag `gameEndedInEarlySurrender`, so the flag is
# unreliable and duration is the robust signal). Such a game is noise in the
# pattern-over-N averages -- the same reason Decision 2026-08-29/238 drops
# off-role games. The next-shortest real game is ~19 min, so the floor is wide.
MIN_GAME_DURATION_S = 300


class IngestError(RuntimeError):
    """A fatal problem in the ingest run -- bad Riot ID, expired key, no support
    games found. `main()` maps it to a non-zero exit and a stderr message."""


# --- env ---------------------------------------------------------------------

def load_key(name: str = "RIOT_API_KEY") -> str:
    """Read `name` from the environment, else parse it out of repo `.env`.

    Same read-at-runtime, never-commit pattern as the client's live test -- the
    key stays out of source. Raises rather than returning None: a missing key is
    a fatal misconfiguration, not a skip, when you're deliberately running a pull.
    """
    import os

    if name in os.environ:
        return os.environ[name]
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if line.startswith(f"{name}=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    raise IngestError(f"{name} not set (env or .env) -- dev keys expire every 24h")


# --- extraction: raw Riot payloads -> GameRecord -----------------------------

def _subject(detail: dict[str, Any], puuid: str) -> dict[str, Any]:
    """Return the subject's participant sub-object from a match detail, matched by
    `puuid` (participants are the 10 players; only one is the subject)."""
    for p in detail["info"]["participants"]:
        if p.get("puuid") == puuid:
            return p
    raise IngestError(f"subject puuid not among participants of {detail['metadata']['matchId']}")


def _game_creation(info: dict[str, Any]) -> date:
    """`info.gameCreation` (epoch ms, UTC) -> a calendar date. UTC so the day
    doesn't drift with the author's local timezone."""
    return datetime.fromtimestamp(info["gameCreation"] / 1000, tz=timezone.utc).date()


def _duration_s(info: dict[str, Any]) -> int:
    """Game length in seconds, around Riot's unit quirk: since patch 11.20 matches
    carry `gameEndTimestamp` and `gameDuration` is already seconds; older matches
    lack it and report milliseconds. Ranked-solo pulls are modern, but guard anyway."""
    dur = info["gameDuration"]
    return dur if "gameEndTimestamp" in info else dur // 1000


def _deaths(timeline: dict[str, Any], participant_id: int) -> tuple[DeathEvent, ...]:
    """Walk the timeline for the subject's deaths -- `CHAMPION_KILL` events whose
    `victimId` is the subject -- as `(game_time_s, x, y)` in chronological order.
    The phase each death falls in is derived later from `game_time_s` (phases.py
    is the one home), so it isn't frozen here."""
    events: list[DeathEvent] = []
    for frame in timeline["info"]["frames"]:
        for ev in frame.get("events", []):
            if ev.get("type") == "CHAMPION_KILL" and ev.get("victimId") == participant_id:
                pos = ev.get("position", {})
                events.append(
                    DeathEvent(
                        game_time_s=ev["timestamp"] // 1000,
                        x=pos.get("x", 0),
                        y=pos.get("y", 0),
                    )
                )
    return tuple(events)


# Real vision-ward types (2026-09-01): ward *activity* is the per-phase vision
# proxy, so only genuine vision wards count -- TEEMO_MUSHROOM and untyped
# (UNDEFINED) placements are dropped. Applied to placed and cleared alike.
_VISION_WARD_TYPES = frozenset({"YELLOW_TRINKET", "CONTROL_WARD", "SIGHT_WARD", "BLUE_TRINKET"})

# Epic-monster `monsterType` -> ObjectiveEvent kind. Void grubs (HORDE) are in as a
# laning-phase objective (2026-09-01); an unmapped monsterType is simply not one of
# our objectives and is skipped.
_MONSTER_KIND = {
    "DRAGON": "dragon",
    "RIFTHERALD": "herald",
    "BARON_NASHOR": "baron",
    "HORDE": "grub",
}


def _wards(timeline: dict[str, Any], participant_id: int) -> tuple[WardEvent, ...]:
    """Walk the timeline for the subject's ward *activity* -- `WARD_PLACED` events
    whose `creatorId` is the subject (placed) and `WARD_KILL` events whose
    `killerId` is the subject (cleared); the two event types credit the actor
    through different fields. Only real vision wards count (`_VISION_WARD_TYPES`).
    Phase is derived later from `game_time_s` (phases.py), so it isn't frozen here."""
    events: list[WardEvent] = []
    for frame in timeline["info"]["frames"]:
        for ev in frame.get("events", []):
            if ev.get("wardType") not in _VISION_WARD_TYPES:
                continue
            etype = ev.get("type")
            if etype == "WARD_PLACED" and ev.get("creatorId") == participant_id:
                action = "placed"
            elif etype == "WARD_KILL" and ev.get("killerId") == participant_id:
                action = "cleared"
            else:
                continue
            events.append(WardEvent(game_time_s=ev["timestamp"] // 1000, action=action))
    return tuple(events)


def _objectives(timeline: dict[str, Any], participant_id: int) -> tuple[ObjectiveEvent, ...]:
    """Walk the timeline for objectives the subject was *credited* for -- epic
    monsters (`ELITE_MONSTER_KILL`, incl. void grubs) and towers (`BUILDING_KILL`
    with `buildingType == TOWER_BUILDING`) where the subject is the `killerId` or in
    `assistingParticipantIds`. Credit-only is an accepted caveat (uncredited
    zoning/setup reads as non-participation). Phase is derived later from
    `game_time_s` (phases.py), so it isn't frozen here."""
    events: list[ObjectiveEvent] = []
    for frame in timeline["info"]["frames"]:
        for ev in frame.get("events", []):
            etype = ev.get("type")
            if etype == "ELITE_MONSTER_KILL":
                kind = _MONSTER_KIND.get(ev.get("monsterType"))
            elif etype == "BUILDING_KILL" and ev.get("buildingType") == "TOWER_BUILDING":
                kind = "tower"
            else:
                kind = None
            if kind is None:
                continue
            credited = ev.get("killerId") == participant_id or participant_id in ev.get(
                "assistingParticipantIds", []
            )
            if not credited:
                continue
            events.append(ObjectiveEvent(game_time_s=ev["timestamp"] // 1000, kind=kind))
    return tuple(events)


def extract_game(detail: dict[str, Any], timeline: dict[str, Any], puuid: str) -> GameRecord:
    """Clean one match (detail + timeline) into a `GameRecord` for the subject.

    Every KPI reads Riot's `challenges.*` derived value (the doc's standing rule);
    `.get` with a 0 default keeps a rare missing challenge from crashing the pull.
    Vision-per-minute additionally falls back to `visionScore / minutes` since it's
    the one KPI cheap and safe to recompute."""
    info = detail["info"]
    p = _subject(detail, puuid)
    ch = p.get("challenges", {})
    dur_s = _duration_s(info)

    vspm = ch.get("visionScorePerMinute")
    if vspm is None:
        minutes = max(dur_s / 60, 1e-9)
        vspm = p.get("visionScore", 0) / minutes

    return GameRecord(
        match_id=detail["metadata"]["matchId"],
        game_creation=_game_creation(info),
        game_duration_s=dur_s,
        champion=p.get("championName", ""),
        queue_id=info.get("queueId", 0),
        win=bool(p.get("win", False)),
        kills=p.get("kills", 0),
        deaths=p.get("deaths", 0),
        assists=p.get("assists", 0),
        kda=ch.get("kda", (p.get("kills", 0) + p.get("assists", 0)) / max(p.get("deaths", 0), 1)),
        vision_score_per_min=float(vspm),
        kill_participation=float(ch.get("killParticipation", 0.0)),
        effective_heal_shield=int(ch.get("effectiveHealAndShielding", 0)),
        team_damage_pct=float(ch.get("teamDamagePercentage", 0.0)),
        dragon_takedowns=int(ch.get("dragonTakedowns", 0)),
        enemy_immobilizations=int(ch.get("enemyChampionImmobilizations", 0)),
        ward_takedowns=int(ch.get("wardTakedowns", 0)),
        deaths_ctx=_deaths(timeline, p["participantId"]),
        ward_events=_wards(timeline, p["participantId"]),
        objective_events=_objectives(timeline, p["participantId"]),
    )


# --- collection: scan match ids, keep support games --------------------------

def _keepable(detail: dict[str, Any], puuid: str) -> tuple[bool, str]:
    """Decide whether a scanned game belongs in the snapshot, from detail alone
    (so a reject costs no timeline call). Returns (keep?, reason-for-log).

    Two gates, both realising Decision 238's "keep only clean support games":
      - role: the subject played `UTILITY` (blank = remake/autofill -> drop);
      - length: at least `MIN_GAME_DURATION_S` (a sub-5-min game is a remake/
        aborted match, noise in the averages).
    """
    p = _subject(detail, puuid)
    pos = p.get("teamPosition", "")
    if pos != UTILITY:
        return False, pos or "blank"
    dur = _duration_s(detail["info"])
    if dur < MIN_GAME_DURATION_S:
        return False, f"{dur // 60}m-short"
    return True, "keep"


def collect_support_games(
    client: RiotClient,
    puuid: str,
    target_n: int,
    *,
    queue: int = QUEUE_RANKED_SOLO,
    scan_cap: int = 60,
) -> list[GameRecord]:
    """Scan the subject's ranked-solo match ids newest-first, keeping games where
    they played `UTILITY`, until `target_n` are collected or `scan_cap` ids are
    scanned. Persists each kept game's raw payloads before cleaning. Progress is
    logged to stderr so the pull is observable (and a stall is diagnosable)."""
    match_ids = client.match_ids_by_puuid(puuid, queue=queue, count=scan_cap)
    kept: list[GameRecord] = []
    for i, match_id in enumerate(match_ids, 1):
        if len(kept) >= target_n:
            break
        detail = client.match(match_id)
        keep, reason = _keepable(detail, puuid)
        if not keep:  # off-role, blank (remake/autofill), or too short -> dropped (238)
            print(f"  [{i}/{len(match_ids)}] {match_id}  skip ({reason})", file=sys.stderr, flush=True)
            continue
        timeline = client.match_timeline(match_id)
        _persist_raw(match_id, detail, timeline)
        kept.append(extract_game(detail, timeline, puuid))
        print(f"  [{i}/{len(match_ids)}] {match_id}  keep ({len(kept)}/{target_n})", file=sys.stderr, flush=True)
    if not kept:
        raise IngestError(f"no {UTILITY} games in {len(match_ids)} scanned ids")
    return kept


# --- persistence -------------------------------------------------------------

def _persist_raw(match_id: str, detail: dict[str, Any], timeline: dict[str, Any]) -> None:
    """Write the raw detail + timeline JSON before cleaning, so a mid-run rate-limit
    stall never forces re-fetching what we already paid for."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{match_id}.detail.json").write_text(json.dumps(detail))
    (RAW_DIR / f"{match_id}.timeline.json").write_text(json.dumps(timeline))


def _record_to_dict(rec: GameRecord) -> dict[str, Any]:
    """Frozen dataclass -> a JSON-serialisable dict: `asdict` recurses the nested
    `DeathEvent`s (and turns the tuple into a list); the one non-JSON field, the
    `date`, becomes an ISO string."""
    d = dataclasses.asdict(rec)
    d["game_creation"] = rec.game_creation.isoformat()
    return d


def _dict_to_record(d: dict[str, Any]) -> GameRecord:
    """Inverse of `_record_to_dict`: one deserialised `games.jsonl` line -> a
    `GameRecord`, rehydrating the ISO `game_creation` back to a `date` and the
    three event lists back to tuples of their frozen dataclasses. Field-by-field so
    an unknown key in a stale line fails loud, not silent."""
    return GameRecord(
        match_id=d["match_id"],
        game_creation=date.fromisoformat(d["game_creation"]),
        game_duration_s=d["game_duration_s"],
        champion=d["champion"],
        queue_id=d["queue_id"],
        win=d["win"],
        kills=d["kills"],
        deaths=d["deaths"],
        assists=d["assists"],
        kda=d["kda"],
        vision_score_per_min=d["vision_score_per_min"],
        kill_participation=d["kill_participation"],
        effective_heal_shield=d["effective_heal_shield"],
        team_damage_pct=d["team_damage_pct"],
        dragon_takedowns=d["dragon_takedowns"],
        enemy_immobilizations=d["enemy_immobilizations"],
        ward_takedowns=d["ward_takedowns"],
        deaths_ctx=tuple(DeathEvent(**e) for e in d["deaths_ctx"]),
        ward_events=tuple(WardEvent(**e) for e in d["ward_events"]),
        objective_events=tuple(ObjectiveEvent(**e) for e in d["objective_events"]),
    )


def load_snapshot(path: Path = SNAPSHOT_DIR / "games.jsonl") -> list[GameRecord]:
    """Read the committed `games.jsonl` back into `GameRecord`s -- the reader the
    Stat-Line Producer runs `metrics.player_metrics` over. Reviewers never re-pull,
    so this (not the Riot path) is how the app reaches the subject's games."""
    with path.open() as f:
        return [_dict_to_record(json.loads(line)) for line in f if line.strip()]


def write_snapshot(games: list[GameRecord], meta: SnapshotMeta) -> None:
    """Write `data/snapshot/games.jsonl` (one record per line, newest first) and
    `meta.json` (subject + provenance)."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with (SNAPSHOT_DIR / "games.jsonl").open("w") as f:
        for rec in games:
            f.write(json.dumps(_record_to_dict(rec)) + "\n")
    meta_d = dataclasses.asdict(meta)
    meta_d["retrieved_at"] = meta.retrieved_at.isoformat()
    (SNAPSHOT_DIR / "meta.json").write_text(json.dumps(meta_d, indent=2) + "\n")


def rebuild_from_raw() -> int:
    """Re-extract the snapshot from the persisted raw payloads under `RAW_DIR` --
    no Riot calls -- and rewrite `games.jsonl`. This is the payoff of
    write-raw-before-cleaned: after a schema change the record re-cleans from disk,
    never a re-pull. Provenance in `meta.json` (puuid, riot-id, region, pull date)
    is preserved as-is; only `game_count` is refreshed to what was on disk."""
    meta_d = json.loads((SNAPSHOT_DIR / "meta.json").read_text())
    puuid = meta_d["puuid"]
    details = sorted(RAW_DIR.glob("*.detail.json"))
    if not details:
        raise IngestError(f"no raw payloads under {RAW_DIR} to rebuild from")

    games: list[GameRecord] = []
    for dpath in details:
        match_id = dpath.name.removesuffix(".detail.json")
        tpath = RAW_DIR / f"{match_id}.timeline.json"
        if not tpath.exists():
            raise IngestError(f"raw timeline missing for {match_id}")
        detail = json.loads(dpath.read_text())
        timeline = json.loads(tpath.read_text())
        games.append(extract_game(detail, timeline, puuid))
    # Newest first, as the live pull writes; match_id breaks same-day ties.
    games.sort(key=lambda g: (g.game_creation, g.match_id), reverse=True)

    meta = SnapshotMeta(
        riot_id=meta_d["riot_id"],
        puuid=puuid,
        region=meta_d["region"],
        queue_id=meta_d["queue_id"],
        game_count=len(games),
        retrieved_at=date.fromisoformat(meta_d["retrieved_at"]),
    )
    write_snapshot(games, meta)
    return len(games)


# --- entry point -------------------------------------------------------------

def run(riot_id: str, region: str, count: int, *, queue: int = QUEUE_RANKED_SOLO) -> int:
    """Resolve the subject, collect `count` support games, write the snapshot.
    Returns the number of games actually committed."""
    if "#" not in riot_id:
        raise IngestError(f"riot-id must be 'gameName#tagLine', got {riot_id!r}")
    game_name, tag_line = riot_id.split("#", 1)
    try:
        regional = RegionalRoute(region.lower())
    except ValueError as exc:
        raise IngestError(f"region must be one of {[r.value for r in RegionalRoute]}, got {region!r}") from exc

    client = RiotClient(load_key(), regional)  # regional-only; no platform/rank here
    print(f"resolving {riot_id} on {regional} ...", file=sys.stderr, flush=True)
    acct = client.account_by_riot_id(game_name, tag_line)
    puuid = acct["puuid"]

    print(f"scanning ranked-solo matches for {count} {UTILITY} games ...", file=sys.stderr, flush=True)
    games = collect_support_games(client, puuid, count, queue=queue)

    meta = SnapshotMeta(
        riot_id=riot_id,
        puuid=puuid,
        region=regional.value,
        queue_id=queue,
        game_count=len(games),
        retrieved_at=date.today(),
    )
    write_snapshot(games, meta)
    return len(games)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the committed personal-plane Match Snapshot.")
    parser.add_argument("--from-raw", action="store_true",
                        help="re-extract from saved raw payloads; no Riot calls (re-clean after a schema change)")
    parser.add_argument("--riot-id", help="subject's Riot ID, 'gameName#tagLine' (required unless --from-raw)")
    parser.add_argument("--region", help="regional cluster: americas | europe | asia | sea (required unless --from-raw)")
    parser.add_argument("--count", type=int, default=10, help="support games to commit (default 10)")
    parser.add_argument("--queue", type=int, default=QUEUE_RANKED_SOLO, help="queue id (default 420 = ranked solo)")
    args = parser.parse_args(argv)
    try:
        if args.from_raw:
            n = rebuild_from_raw()
        else:
            if not args.riot_id or not args.region:
                parser.error("--riot-id and --region are required unless --from-raw")
            n = run(args.riot_id, args.region, args.count, queue=args.queue)
    except (IngestError, RiotAPIError) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {n} games to {SNAPSHOT_DIR.relative_to(DATA.parent)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
