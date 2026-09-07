"""One-time author pull for the subject's PLATFORM-route profile -> `profile.json`.

Companion to `ingest.py`. Where that script builds the *regional* match snapshot
(`games.jsonl` + `meta.json`), this one captures the *platform*-route profile the
dashboard header needs -- icon, level, rank, and the split win/loss record -- into
`data/snapshot/profile.json`. **Author-only, one-time, offline, with my key**;
reviewers read the committed `profile.json` and never call Riot.

Two platform calls (rank is platform-routed, e.g. a SEA subject's shard is `sg2`):

    puuid --Summoner-V4 by-puuid--> profileIconId, summonerLevel
    puuid --League-V4  by-puuid--> [LeagueEntryDTO]  (pick RANKED_SOLO_5x5)

The League entry carries `tier`, `rank`, `leaguePoints`, and the cumulative
`wins`/`losses`. **Those counts are the current split, not lifetime** -- Riot has no
career total and they reset each split -- so the header labels the derived win rate
"this split" (`SubjectProfile` docstring). If the subject is unranked this split the
RANKED_SOLO_5x5 entry is absent; the rank fields are then null and the header shows
"Unranked" (icon/level still resolve).

Run:

    python -m carryia.personal.profile_pull --platform sg2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import date

from carryia.paths import DATA
from carryia.personal.ingest import IngestError, load_key
from carryia.personal.riot_client import PlatformRoute, RiotAPIError, RiotClient
from carryia.personal.snapshot import SubjectProfile

SNAPSHOT_DIR = DATA / "snapshot"
SOLO_QUEUE = "RANKED_SOLO_5x5"  # League-V4's queue STRING (not Match-V5's numeric 420)


def _solo_entry(entries: list[dict]) -> dict | None:
    """The RANKED_SOLO_5x5 entry, or None if the subject is unranked in it this
    split (League-V4 omits an unranked queue from the array -- absent, not zeroed)."""
    return next((e for e in entries if e.get("queueType") == SOLO_QUEUE), None)


def build_profile(client: RiotClient, puuid: str, platform: PlatformRoute) -> SubjectProfile:
    """The two platform calls -> a `SubjectProfile`. Icon/level always resolve; the
    rank fields are None when the subject is unranked in solo/duo this split."""
    summoner = client.summoner_by_puuid(puuid)
    solo = _solo_entry(client.league_entries_by_puuid(puuid))
    return SubjectProfile(
        profile_icon_id=summoner["profileIconId"],
        summoner_level=summoner["summonerLevel"],
        platform=platform.value,
        queue_type=SOLO_QUEUE,
        retrieved_at=date.today(),
        tier=solo["tier"] if solo else None,
        rank=solo["rank"] if solo else None,
        league_points=solo["leaguePoints"] if solo else None,
        wins=solo["wins"] if solo else None,
        losses=solo["losses"] if solo else None,
    )


def write_profile(profile: SubjectProfile) -> None:
    """Serialise to `data/snapshot/profile.json` (date -> isoformat, like meta.json)."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    d = dataclasses.asdict(profile)
    d["retrieved_at"] = profile.retrieved_at.isoformat()
    (SNAPSHOT_DIR / "profile.json").write_text(json.dumps(d, indent=2) + "\n")


def run(platform: str) -> SubjectProfile:
    """Read the committed puuid from `meta.json`, pull the profile, write it."""
    try:
        shard = PlatformRoute(platform.lower())
    except ValueError as exc:
        raise IngestError(
            f"platform must be one of {[p.value for p in PlatformRoute]}, got {platform!r}"
        ) from exc
    meta = json.loads((SNAPSHOT_DIR / "meta.json").read_text())
    puuid = meta["puuid"]
    # Regional host is unused here (platform calls only), but RiotClient needs one;
    # derive it from meta so the client is constructed honestly.
    from carryia.personal.riot_client import RegionalRoute

    client = RiotClient(load_key(), RegionalRoute(meta["region"]), platform=shard)
    print(f"pulling profile for {meta['riot_id']} on {shard} ...", file=sys.stderr, flush=True)
    profile = build_profile(client, puuid, shard)
    write_profile(profile)
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-time author pull: subject icon/level/rank/split-record -> profile.json."
    )
    parser.add_argument(
        "--platform", default="sg2",
        help="platform shard the subject ranks on (na1|euw1|kr|sg2|...); default sg2 (SEA subject)",
    )
    args = parser.parse_args(argv)
    try:
        p = run(args.platform)
    except (IngestError, RiotAPIError, KeyError) as exc:
        print(f"profile pull failed: {exc}", file=sys.stderr)
        return 1
    rank = f"{p.tier} {p.rank} ({p.league_points} LP)" if p.tier else "Unranked"
    record = f"{p.wins}W {p.losses}L" if p.wins is not None else "no split record"
    print(f"wrote profile.json -- lvl {p.summoner_level}, {rank}, {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
