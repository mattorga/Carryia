"""Cohort-benchmark crawl -- the SECOND caller of the personal-plane extractor.

Where `ingest.py` pulls the *subject's* games, this pulls a *cohort* of the
subject's rank peers so V1 can place each phased metric in a below/avg/above
bucket against them. Same `extract_game`, two callers.

Three things make the cohort crawl expensive-and-uncertain, which is why it was
piloted before the full run:

  1. **Over-fetch** -- Match-V5 has no role filter, so we scan a player's recent
     ids and pull each *detail* to find their support games; details-per-keeper is
     unknown until measured.
  2. **All-phase yield** -- the cohort keeps only games that *reached late phase*
     (>= `MID_ENDS_S`, so all three phase cells are populated), a stricter floor
     than the subject's 5-min remake cut.
  3. **Support-main screen-out** -- a league band is not a role, so a seed may not
     be a support main; we screen on a majority-UTILITY sample (decide once, on
     details alone) and skip the rest.

This module has three modes, all on the same per-player path:

  - **pilot** (default): run a few players, project 5 -> N (calls + wall-clock),
    write `data/benchmark/pilot_report.json`. The persistent app key measured
    ~0.83/s (dev-tier limits), so ~125 players is ~2.6 h, rate-bound.
  - **--full**: seed the whole tier (all divisions), collect N qualified support
    mains, aggregate -> `data/benchmark/cohort.json` (aggregate quartiles, NO
    puuids -- the committed reviewer input).
  - **--from-raw**: re-aggregate `cohort.json` from the raw payloads persisted
    under `data/raw/cohort/` -- no Riot calls (salvage a crashed crawl / re-clean
    after a schema change).

Run:

    python -m carryia.personal.cohort --tier SILVER --division II --players 5   # pilot
    python -m carryia.personal.cohort --tier SILVER --full --players 125        # full (all Silver divisions)
    python -m carryia.personal.cohort --tier SILVER --full --from-raw           # rebuild from raw
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import date
from statistics import mean as _mean, quantiles as _quantiles
from typing import Any, Iterator

from carryia.paths import DATA
from carryia.phases import MID_ENDS_S, TEMPORAL
from carryia.personal.ingest import (
    IngestError,
    QUEUE_RANKED_SOLO,
    _duration_s,
    _subject,
    extract_game,
    load_key,
)
from carryia.personal.metrics import PHASED_METRICS, WHOLE_GAME_METRICS, player_metrics
from carryia.personal.riot_client import (
    PlatformRoute,
    RegionalRoute,
    RiotAPIError,
    RiotClient,
)
from carryia.personal.snapshot import UTILITY, GameRecord

# The four non-apex divisions, high to low -- a full-tier crawl walks all of them.
ALL_DIVISIONS = ("I", "II", "III", "IV")

COHORT_RAW_DIR = DATA / "raw" / "cohort"
BENCHMARK_DIR = DATA / "benchmark"

# A cohort game must have reached late phase, so all three phase cells (laning /
# mid / late) carry data -- otherwise a short game silently under-fills `late`
#. `MID_ENDS_S` (1500s) is
# the mid->late boundary in phases.py, the one home; a game at or past it is
# "all-phase". Stricter than the subject snapshot's 300s remake floor, on purpose.
ALL_PHASE_FLOOR_S = MID_ENDS_S

# Apex tiers have no divisions -- they route to their own endpoint, not the
# paged band endpoint. One home for the distinction.
APEX_TIERS = frozenset({"CHALLENGER", "GRANDMASTER", "MASTER"})

# Support-main screen: scan this many of a seed's recent games on the (cheap)
# detail call, then decide ONCE -- if fewer than this fraction were UTILITY,
# they aren't a support main, bail before paying a single (heavy) timeline call.
SCREEN_AFTER = 15
SCREEN_MIN_UTILITY_FRAC = 0.4

# A seed counts toward the cohort only if it yields at least this many all-phase
# games -- a 1- or 2-game account is sampling noise, not a peer (2026-09-01 sizing
# wants ~10 games/player). Below it, the player is dropped with zero timeline cost.
MIN_QUALIFY_GAMES = 5


# --- diagnostics -------------------------------------------------------------

@dataclasses.dataclass
class PlayerDiag:
    """One seed player's cost + yield -- the raw material the projection scales.

    Every count is an API-call tally or a games-kept tally, so `project` can turn
    5 players' worth into an N-player estimate honestly (no hidden constants)."""

    puuid: str
    ids_scanned: int = 0        # match ids returned by the one list call
    details_pulled: int = 0     # match-detail calls (the over-fetch numerator)
    timelines_pulled: int = 0   # timeline calls (== games kept, the heavy call)
    summoner_calls: int = 0     # summoner_by_id fallbacks (entry lacked a puuid)
    utility_games: int = 0      # of scanned, how many were UTILITY
    all_phase_kept: int = 0     # of UTILITY, how many reached late (kept)
    qualified: bool = False     # passed the support-main screen and yielded games
    note: str = ""              # why skipped, if not qualified

    @property
    def api_calls(self) -> int:
        """Total Riot calls this player cost -- 1 list + details + timelines +
        any summoner fallbacks. Seeding calls are counted separately (shared)."""
        return 1 + self.details_pulled + self.timelines_pulled + self.summoner_calls


# --- seeding: a rank band -> candidate puuids --------------------------------

def _entry_puuid(client: RiotClient, entry: dict[str, Any], diag: PlayerDiag | None = None) -> str | None:
    """A league entry -> puuid. Newer entries carry `puuid` directly; older ones
    only `summonerId`, so fall back to Summoner-V4 (counted, since it's a call)."""
    if entry.get("puuid"):
        return entry["puuid"]
    sid = entry.get("summonerId")
    if not sid:
        return None
    if diag is not None:
        diag.summoner_calls += 1
    return client.summoner_by_id(sid).get("puuid")


def seed_players(
    client: RiotClient, tier: str, division: str | None, *, queue: str = "RANKED_SOLO_5x5"
) -> list[dict[str, Any]]:
    """One band -> its ranked entries (each a dict with `summonerId`/`puuid`).

    Apex tiers (no divisions) use the apex endpoint; the rest use the paged band
    endpoint (page 1 only here -- one page is far more than a 5-player pilot needs,
    and the full crawl pages on demand). The entries are the seed pool; the caller
    screens each for support-main-ness."""
    tier = tier.upper()
    if tier in APEX_TIERS:
        return client.apex_league(tier, queue=queue).get("entries", [])
    if division is None:
        raise IngestError(f"non-apex tier {tier} needs a --division (I..IV)")
    return client.entries_by_queue(tier, division.upper(), queue=queue)


# --- per-player collection: scan, screen, keep all-phase support games --------

def collect_player_games(
    client: RiotClient,
    puuid: str,
    *,
    games_per_player: int,
    scan_cap: int,
    queue: int = QUEUE_RANKED_SOLO,
) -> tuple[list[GameRecord], PlayerDiag]:
    """Scan one seed's recent ranked games, screen for support-main, and keep up to
    `games_per_player` all-phase UTILITY games. Returns (games, diagnostics).

    The identical path the full crawl runs, instrumented, in two phases so a
    rejected seed never pays for a (heavy) timeline call:

      1. **Screen + select on details alone.** Role and game length both come from
         the detail call, so the whole scan runs on details. At the `SCREEN_AFTER`
         mark we decide ONCE: below `SCREEN_MIN_UTILITY_FRAC` UTILITY -> not a
         support main, bail immediately (details only, zero timelines). Otherwise
         commit -- keep scanning for all-phase UTILITY candidates, never re-screen.
      2. **Extract.** Only once the seed has passed the screen AND cleared the
         `MIN_QUALIFY_GAMES` bar do we pull a timeline per kept candidate.

    So a screened-out or under-sampled seed costs ~`SCREEN_AFTER` detail calls and
    nothing more; timelines are spent only on a player who actually joins the cohort."""
    diag = PlayerDiag(puuid=puuid)
    match_ids = client.match_ids_by_puuid(puuid, queue=queue, count=scan_cap)
    diag.ids_scanned = len(match_ids)

    # --- phase 1: screen + select candidates (detail calls only) ---
    candidates: list[tuple[str, dict[str, Any]]] = []
    screened = False
    for scanned, match_id in enumerate(match_ids, 1):
        if len(candidates) >= games_per_player:
            break
        detail = client.match(match_id)
        diag.details_pulled += 1
        try:
            p = _subject(detail, puuid)
        except IngestError:
            continue  # puuid not in this match (shouldn't happen) -- skip defensively
        if p.get("teamPosition", "") == UTILITY:
            diag.utility_games += 1
            if _duration_s(detail["info"]) >= ALL_PHASE_FLOOR_S:
                candidates.append((match_id, detail))  # all-phase support game
        # Decide-once support-main verdict, exactly at the screen mark.
        if not screened and scanned >= SCREEN_AFTER:
            screened = True
            if diag.utility_games / scanned < SCREEN_MIN_UTILITY_FRAC:
                diag.note = f"not support-main ({diag.utility_games}/{scanned} UTILITY)"
                return [], diag  # zero timelines paid

    # Under-sampled (passed or too few games to screen) -> drop before any timeline.
    if len(candidates) < MIN_QUALIFY_GAMES:
        diag.note = (
            f"no all-phase UTILITY games in {diag.details_pulled} scanned"
            if not candidates
            else f"only {len(candidates)} all-phase games (need {MIN_QUALIFY_GAMES})"
        )
        return [], diag

    # --- phase 2: extract kept candidates (the heavy timeline calls) ---
    games: list[GameRecord] = []
    for match_id, detail in candidates:
        timeline = client.match_timeline(match_id)
        diag.timelines_pulled += 1
        _persist_raw(puuid, match_id, detail, timeline)
        games.append(extract_game(detail, timeline, puuid))
        diag.all_phase_kept += 1
    diag.qualified = True
    return games, diag


# --- persistence -------------------------------------------------------------

def _persist_raw(puuid: str, match_id: str, detail: dict[str, Any], timeline: dict[str, Any]) -> None:
    """Persist a cohort game's raw payloads under a per-player dir, before cleaning
    -- so the full crawl (and a re-extract after a schema change) never re-pays for
    what the pilot already pulled. Kept separate from the subject's `raw/riot/`."""
    pdir = COHORT_RAW_DIR / puuid
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / f"{match_id}.detail.json").write_text(json.dumps(detail))
    (pdir / f"{match_id}.timeline.json").write_text(json.dumps(timeline))


# --- projection: 5 players -> N ----------------------------------------------

def _ceiling_per_s(header: str | None) -> float:
    """The sustained request ceiling (req/s) from a Riot `X-App-Rate-Limit` header
    like `20:1,100:120` -- the TIGHTEST bucket, min(count/window) across them (a
    dev key is 0.83/s; a persistent app key is far higher). Falls back to the
    dev-key 100/120 only if the header wasn't captured."""
    if not header:
        return 100 / 120
    rates = []
    for bucket in header.split(","):
        count, window = bucket.split(":")
        rates.append(int(count) / int(window))
    return min(rates) if rates else 100 / 120


def project(
    diags: list[PlayerDiag],
    elapsed_s: float,
    *,
    target_players: int,
    seed_calls: int,
    rate_limit_header: str | None = None,
) -> dict[str, Any]:
    """Turn the pilot's measured cost into an N-player estimate -- calls + hours.

    Honest scaling from three measured ratios: calls per *qualified* player, the
    screen-out overhead (seeds burned per keeper), and the effective call rate the
    pilot actually achieved (which already bakes in 429 backoff). The ceiling line
    reads the key's OWN limit from `rate_limit_header`, so it's right whether the
    key is a dev key or a persistent app key."""
    qualified = [d for d in diags if d.qualified]
    screened_out = [d for d in diags if not d.qualified]
    n_qual = len(qualified)
    if n_qual == 0:
        return {"error": "no players qualified in the pilot -- cannot project", "seeds_tried": len(diags)}

    calls_per_qual = sum(d.api_calls for d in qualified) / n_qual
    games_per_qual = sum(d.all_phase_kept for d in qualified) / n_qual
    # Screen-out overhead: burned seeds and their cost, per keeper.
    screenout_per_qual = len(screened_out) / n_qual
    calls_per_screenout = (sum(d.api_calls for d in screened_out) / len(screened_out)) if screened_out else 0.0
    total_calls = sum(d.api_calls for d in diags) + seed_calls
    effective_rate = total_calls / elapsed_s if elapsed_s > 0 else 0.0

    est_calls = (
        seed_calls
        + target_players * calls_per_qual
        + target_players * screenout_per_qual * calls_per_screenout
    )
    est_hours_measured = (est_calls / effective_rate / 3600) if effective_rate > 0 else None
    # Ceiling from the key's own reported limit (dev vs app key differ ~60x).
    ceiling_rate = _ceiling_per_s(rate_limit_header)
    est_hours_ceiling = est_calls / ceiling_rate / 3600

    return {
        "pilot": {
            "seeds_tried": len(diags),
            "qualified": n_qual,
            "screened_out": len(screened_out),
            "total_calls": total_calls,
            "elapsed_s": round(elapsed_s, 1),
            "effective_rate_per_s": round(effective_rate, 3),
        },
        "measured_ratios": {
            "calls_per_qualified_player": round(calls_per_qual, 1),
            "all_phase_games_per_qualified_player": round(games_per_qual, 1),
            "screened_out_per_qualified": round(screenout_per_qual, 2),
            "calls_per_screened_out": round(calls_per_screenout, 1),
        },
        "projection": {
            "target_players": target_players,
            "estimated_calls": round(est_calls),
            "ceiling_rate_per_s": round(ceiling_rate, 3),
            "estimated_hours_at_pilot_rate": round(est_hours_measured, 2) if est_hours_measured else None,
            "estimated_hours_at_ratelimit_ceiling": round(est_hours_ceiling, 2),
        },
    }


# --- aggregation: per-player metrics -> cohort distribution ------------------

def _quartiles(values: list[float]) -> dict[str, Any]:
    """q1 / median / q3 (+ mean, n, and the full sorted `values`) over a cohort
    distribution. One value per player, so this is the across-players spread the
    subject is bucketed against. `values` is the anonymous per-player distribution
    (no puuids) the UI ranks the subject against for a true empirical percentile --
    quartiles alone can't yield one. Degenerates gracefully for tiny inputs (a real
    cohort is ~100+)."""
    n = len(values)
    if n == 0:
        return {"q1": 0.0, "median": 0.0, "q3": 0.0, "mean": 0.0, "n": 0, "values": []}
    vals = sorted(round(v, 4) for v in values)
    if n == 1:
        v = vals[0]
        return {"q1": v, "median": v, "q3": v, "mean": v, "n": 1, "values": vals}
    q1, med, q3 = _quantiles(values, n=4, method="inclusive")
    return {
        "q1": round(q1, 4), "median": round(med, 4), "q3": round(q3, 4),
        "mean": round(_mean(values), 4), "n": n, "values": vals,
    }


def aggregate_cohort(per_player: list[dict]) -> dict[str, Any]:
    """Cross-player quartiles for every metric -- the committed cohort distribution.
    Each player contributes ONE value per metric (its per-game mean from
    `player_metrics`), so the spread is over distinct players (09-01 sizing)."""
    metrics: dict[str, Any] = {}
    for m in PHASED_METRICS:
        metrics[m] = {
            p.value: _quartiles([pp["phased"][m][p.value] for pp in per_player])
            for p in TEMPORAL
        }
    for m in WHOLE_GAME_METRICS:
        metrics[m] = _quartiles([pp["whole_game"][m] for pp in per_player])
    return metrics


def _build_cohort(
    per_player: list[dict], *, tier: str, divisions, platform, regional,
    seeds_tried: int, elapsed_s: float,
) -> dict[str, Any]:
    """Assemble the committed `cohort.json` -- provenance meta + aggregate metrics.
    Deliberately carries NO puuids: the artifact a reviewer reads is the
    distribution, not the individuals."""
    return {
        "meta": {
            "tier": tier.upper(),
            "divisions": list(divisions) if divisions else None,
            "platform": str(platform),
            "region": str(regional),
            "queue": QUEUE_RANKED_SOLO,
            "player_count": len(per_player),
            "games_total": sum(pp["games"] for pp in per_player),
            "seeds_tried": seeds_tried,
            "built": date.today().isoformat(),
            "elapsed_s": round(elapsed_s, 1),
        },
        "metrics": aggregate_cohort(per_player),
    }


def _write_cohort(cohort: dict[str, Any]) -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_DIR / "cohort.json").write_text(json.dumps(cohort, indent=2) + "\n")


# --- full crawl: seed all divisions -> collect -> cohort.json -----------------

def iter_seeds(
    client: RiotClient, tier: str, divisions, *, queue: str = "RANKED_SOLO_5x5", max_pages: int = 20
) -> Iterator[dict[str, Any]]:
    """Lazily yield candidate entries across a whole tier -- apex as one league,
    non-apex paging each division until a short/empty page. Lazy so the crawl stops
    seeding the moment it has enough qualified players (no over-fetch of the pool)."""
    tier = tier.upper()
    if tier in APEX_TIERS:
        yield from client.apex_league(tier, queue=queue).get("entries", [])
        return
    for div in divisions:
        for page in range(1, max_pages + 1):
            entries = client.entries_by_queue(tier, div, queue=queue, page=page)
            if not entries:
                break
            yield from entries


def crawl(
    tier: str, divisions, *, target_players: int, games_per_player: int, scan_cap: int,
    platform: PlatformRoute, regional: RegionalRoute,
) -> dict[str, Any]:
    """The full cohort crawl: seed the tier, collect `target_players` qualified
    support mains via the tightened screen, aggregate -> `cohort.json` dict.

    Same per-player path the pilot measured, run to N. Raw payloads are persisted
    per kept game (in `collect_player_games`), so a crash is salvageable with
    `rebuild_cohort_from_raw` -- no re-pull."""
    client = RiotClient(load_key(), regional, platform)
    band = "/".join(divisions) if divisions else "(apex)"
    print(f"crawling {tier.upper()} {band} on {platform} for {target_players} support mains ...",
          file=sys.stderr, flush=True)

    per_player: list[dict] = []
    seeds_tried = 0
    started = time.monotonic()
    for entry in iter_seeds(client, tier, divisions):
        if len(per_player) >= target_players:
            break
        seeds_tried += 1
        puuid = _entry_puuid(client, entry)
        if not puuid:
            continue
        try:
            games, diag = collect_player_games(
                client, puuid, games_per_player=games_per_player, scan_cap=scan_cap
            )
        except RiotAPIError as exc:
            print(f"  {puuid[:8]}..  error ({exc}) -- skipping", file=sys.stderr, flush=True)
            continue
        if not diag.qualified:
            continue
        per_player.append(player_metrics(games))
        rl = client.last_app_rate_limit_count or "?"
        print(f"  [{len(per_player)}/{target_players}] {puuid[:8]}..  {diag.all_phase_kept}g  "
              f"[seeds {seeds_tried}, rl {rl}]", file=sys.stderr, flush=True)
    elapsed = time.monotonic() - started

    if not per_player:
        raise IngestError(f"crawl collected no qualified players in {seeds_tried} seeds")
    return _build_cohort(per_player, tier=tier, divisions=divisions, platform=platform,
                         regional=regional, seeds_tried=seeds_tried, elapsed_s=elapsed)


def rebuild_cohort_from_raw(tier: str, divisions, platform, regional) -> dict[str, Any]:
    """Re-aggregate `cohort.json` from the persisted raw payloads under
    `COHORT_RAW_DIR` -- no Riot calls. Salvages a crashed crawl (raw is written per
    kept game) and re-cleans after a schema change. Only qualified players ever get
    raw (a screened-out seed pulls no timeline), so each puuid dir is a member; the
    `MIN_QUALIFY_GAMES` bar is re-checked defensively. Meta comes from the args."""
    player_dirs = [d for d in sorted(COHORT_RAW_DIR.glob("*")) if d.is_dir()]
    if not player_dirs:
        raise IngestError(f"no raw payloads under {COHORT_RAW_DIR} to rebuild from")

    per_player: list[dict] = []
    for pdir in player_dirs:
        puuid = pdir.name
        games: list[GameRecord] = []
        for dpath in sorted(pdir.glob("*.detail.json")):
            match_id = dpath.name.removesuffix(".detail.json")
            tpath = pdir / f"{match_id}.timeline.json"
            if not tpath.exists():
                continue
            games.append(extract_game(json.loads(dpath.read_text()), json.loads(tpath.read_text()), puuid))
        if len(games) >= MIN_QUALIFY_GAMES:
            per_player.append(player_metrics(games))
    if not per_player:
        raise IngestError("no players with enough games in raw to rebuild")
    return _build_cohort(per_player, tier=tier, divisions=divisions, platform=platform,
                         regional=regional, seeds_tried=len(player_dirs), elapsed_s=0.0)


# --- entry point -------------------------------------------------------------

def run_pilot(
    tier: str,
    division: str | None,
    *,
    players: int,
    games_per_player: int,
    scan_cap: int,
    project_to: int,
    platform: PlatformRoute,
    regional: RegionalRoute,
) -> dict[str, Any]:
    """Seed a band, collect `players` qualified support mains, and project to
    `project_to`. Returns the report dict (also written to disk by `main`)."""
    client = RiotClient(load_key(), regional, platform)
    print(f"seeding {tier}{'/' + division if division else ''} on {platform} ...", file=sys.stderr, flush=True)
    pool = seed_players(client, tier, division)
    seed_calls = 1  # one apex/band call built the pool
    if not pool:
        raise IngestError(f"no entries seeded for {tier} {division or ''}".strip())
    print(f"  {len(pool)} entries in pool; screening for {players} support mains ...", file=sys.stderr, flush=True)

    diags: list[PlayerDiag] = []
    qualified = 0
    started = time.monotonic()
    for entry in pool:
        if qualified >= players:
            break
        diag_probe = PlayerDiag(puuid="")
        puuid = _entry_puuid(client, entry, diag_probe)
        if not puuid:
            continue
        try:
            _, diag = collect_player_games(
                client, puuid, games_per_player=games_per_player, scan_cap=scan_cap
            )
        except RiotAPIError as exc:
            print(f"  {puuid[:8]}..  error ({exc}) -- skipping", file=sys.stderr, flush=True)
            continue
        diag.summoner_calls += diag_probe.summoner_calls
        seed_calls += diag_probe.summoner_calls  # the puuid-fallback call is seeding cost
        diags.append(diag)
        if diag.qualified:
            qualified += 1
        status = f"keep {diag.all_phase_kept}g" if diag.qualified else f"skip ({diag.note})"
        head = f"{qualified}/{players}" if diag.qualified else "-"
        rl = client.last_app_rate_limit_count or "?"
        print(
            f"  [{head}] {puuid[:8]}..  {status}  "
            f"[scanned {diag.details_pulled}, rl {rl}]",
            file=sys.stderr, flush=True,
        )
    elapsed = time.monotonic() - started

    report = project(
        diags, elapsed, target_players=project_to, seed_calls=seed_calls,
        rate_limit_header=client.last_app_rate_limit,
    )
    report["per_player"] = [dataclasses.asdict(d) for d in diags]
    report["last_app_rate_limit"] = client.last_app_rate_limit
    return report


def _write_report(report: dict[str, Any]) -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_DIR / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n")


def _divisions_for(args) -> tuple[str, ...] | None:
    """Which divisions a full crawl / rebuild walks. Apex has none; otherwise the
    one `--division` if given, else the whole tier (all four)."""
    if args.tier.upper() in APEX_TIERS:
        return None
    return (args.division.upper(),) if args.division else ALL_DIVISIONS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cohort crawl: pilot (measure+project), full crawl (-> cohort.json), or rebuild-from-raw.")
    parser.add_argument("--tier", required=True, help="rank tier: IRON..DIAMOND (needs --division), or CHALLENGER/GRANDMASTER/MASTER")
    parser.add_argument("--division", help="division I..IV; a full crawl without it walks the whole tier")
    parser.add_argument("--full", action="store_true", help="run the full crawl and write cohort.json (default is the pilot)")
    parser.add_argument("--from-raw", action="store_true", help="rebuild cohort.json from persisted raw payloads; no Riot calls")
    parser.add_argument("--players", type=int, default=5, help="qualified support mains to collect (pilot 5; a full crawl wants ~125)")
    parser.add_argument("--games-per-player", type=int, default=10, help="all-phase support games per player (default 10)")
    parser.add_argument("--scan-cap", type=int, default=60, help="max recent ids scanned per player (default 60)")
    parser.add_argument("--project-to", type=int, default=125, help="pilot only: player count to project the full crawl to (default 125)")
    parser.add_argument("--platform", default="sg2", help="platform shard to seed on (default sg2)")
    parser.add_argument("--region", default="sea", help="regional cluster for match calls (default sea)")
    args = parser.parse_args(argv)

    try:
        platform = PlatformRoute(args.platform.lower())
        regional = RegionalRoute(args.region.lower())
    except ValueError as exc:
        print(f"cohort failed: {exc}", file=sys.stderr)
        return 1

    # --- rebuild-from-raw: no Riot calls, re-aggregate cohort.json ---
    if args.from_raw:
        try:
            cohort = rebuild_cohort_from_raw(args.tier, _divisions_for(args), platform, regional)
        except (IngestError, RiotAPIError) as exc:
            print(f"rebuild failed: {exc}", file=sys.stderr)
            return 1
        _write_cohort(cohort)
        m = cohort["meta"]
        print(f"rebuilt cohort.json from raw: {m['player_count']} players, {m['games_total']} games")
        return 0

    # --- full crawl: seed the tier, collect N, write cohort.json ---
    if args.full:
        try:
            cohort = crawl(
                args.tier, _divisions_for(args),
                target_players=args.players,
                games_per_player=args.games_per_player,
                scan_cap=args.scan_cap,
                platform=platform,
                regional=regional,
            )
        except (IngestError, RiotAPIError) as exc:
            print(f"crawl failed: {exc}", file=sys.stderr)
            return 1
        _write_cohort(cohort)
        m = cohort["meta"]
        print(f"\nwrote {(BENCHMARK_DIR / 'cohort.json').relative_to(DATA.parent)}: "
              f"{m['player_count']} players, {m['games_total']} games, {m['seeds_tried']} seeds, "
              f"{round(m['elapsed_s'] / 3600, 2)}h")
        return 0

    # --- pilot: measure the cost on a few players, project 5 -> N ---
    try:
        report = run_pilot(
            args.tier, args.division,
            players=args.players,
            games_per_player=args.games_per_player,
            scan_cap=args.scan_cap,
            project_to=args.project_to,
            platform=platform,
            regional=regional,
        )
    except (IngestError, RiotAPIError) as exc:
        print(f"pilot failed: {exc}", file=sys.stderr)
        return 1

    _write_report(report)
    proj = report.get("projection")
    if proj:
        print(
            f"\nprojection -> {proj['target_players']} players: "
            f"~{proj['estimated_calls']} calls, "
            f"~{proj['estimated_hours_at_pilot_rate']}h at pilot rate "
            f"(~{proj['estimated_hours_at_ratelimit_ceiling']}h at the "
            f"{proj['ceiling_rate_per_s']}/s key ceiling)"
        )
    print(f"wrote {(BENCHMARK_DIR / 'pilot_report.json').relative_to(DATA.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
