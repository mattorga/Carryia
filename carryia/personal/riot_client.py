"""Typed client for the Riot Developer API — the personal-plane's only door out.

One function per endpoint, so the endpoint *list* lives as code, not a prose doc
that drifts from Riot's reference. Call order, routing, rate budget, and field
mapping are all encoded here. Author-only: every call here runs once, offline, with my key,
to build the committed Match Snapshot. Reviewers never invoke it.

The routing gotcha, encoded in the type system so it can't be gotten wrong:

  - Match-V5 is REGIONAL -- host is a cluster (americas/europe/asia/sea). A SEA
    player's matches live on `sea`. The personal-plane pull.
  - Account-V1 is GLOBAL but is served ONLY on the continental clusters
    (americas/asia/europe), NOT `sea` -- calling it on `sea` 403s. So it routes to
    the region's continental home (`sea -> asia`; the others already are one),
    held as its own `_account_host`. This split is the SEA routing gotcha.
  - Summoner-V4 and League-V4 are PLATFORM -- host is a shard (na1/euw1/kr/...).
    Rank only; belongs to the Benchmark Crawl, kept here for one home but off the
    personal plane.

A NA subject: match calls -> `americas`, rank calls -> `na1`. Mixing them is the
classic 404/403. `RiotClient` holds one host of each kind, so a caller picks the
pair once at construction and every method routes itself.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

import requests

__all__ = [
    "RegionalRoute",
    "PlatformRoute",
    "RiotClient",
    "RiotAPIError",
]


class RegionalRoute(StrEnum):
    """Regional cluster hosts — for Account-V1 and Match-V5. Value is the
    subdomain of `api.riotgames.com`."""

    AMERICAS = "americas"
    EUROPE = "europe"
    ASIA = "asia"
    SEA = "sea"


class PlatformRoute(StrEnum):
    """Platform shard hosts — for Summoner-V4 and League-V4 (rank). Value is the
    subdomain of `api.riotgames.com`. Extend as needed; the common ones only.

    The SEA shards (SG2/PH2/TH2/TW2/VN2) all sit under the `sea` regional cluster
    and are what the cohort crawl seeds from — the subject plays on `sg2`."""

    NA1 = "na1"
    EUW1 = "euw1"
    EUN1 = "eun1"
    KR = "kr"
    SG2 = "sg2"
    PH2 = "ph2"
    TH2 = "th2"
    TW2 = "tw2"
    VN2 = "vn2"


class RiotAPIError(RuntimeError):
    """A non-success response from the Riot API. Carries the HTTP status so
    callers (and `_get`'s own retry logic) can branch on it -- 404 = subject not
    found, 403 = key missing/expired, 429 = rate-limited, 5xx = Riot's problem."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"[{status}] {message}")
        self.status = status


class RiotClient:
    """One player's-worth of Riot calls, bound to a regional + platform host pair.

    Construct with the subject's home routing (e.g. AMERICAS + NA1), then call the
    endpoint methods. Every method delegates to `_get`, which is the single place
    auth, error mapping, and rate-limiting live -- change the policy once, there.
    """

    _BASE = "https://{host}.api.riotgames.com"
    _MAX_RETRIES = 8  # cap on reactive 429 + network-blip retries (a long crawl needs headroom)
    _TIMEOUT_S = (10, 30)  # (connect, read) -- 30s read gives large timeline payloads room
    # Account-V1 isn't served on `sea`; route it to the continental home instead.
    # Every other region already IS continental, so it maps to itself (via .get).
    _ACCOUNT_CLUSTER = {RegionalRoute.SEA: RegionalRoute.ASIA}

    def __init__(
        self,
        api_key: str,
        regional: RegionalRoute,
        platform: PlatformRoute | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key
        self._regional_host = self._BASE.format(host=regional)
        # Account-V1's host, distinct from the match host: it's global but not on
        # `sea`, so it routes to the continental home (sea -> asia; else itself).
        self._account_host = self._BASE.format(host=self._ACCOUNT_CLUSTER.get(regional, regional))
        # Platform host is for rank only (Summoner/League-V4), which is the
        # Benchmark Crawl, not the personal plane. It's optional so a regional-only
        # caller (the snapshot ingest) needn't invent a shard -- and some regions
        # (e.g. SEA) have no single platform value anyway. `None` until rank needs it.
        self._platform_host = (
            self._BASE.format(host=platform) if platform is not None else None
        )
        # Reuse one connection pool across the ~2N calls of an ingest run.
        self._session = session or requests.Session()
        # Rate-limit telemetry: the most recent X-App-Rate-Limit[-Count] headers
        # Riot handed back, captured by `_get` on every response. The cohort-crawl
        # pilot reads these to measure how close the crawl runs to the app budget
        # before projecting 5 -> N. Format is Riot's `count:window,count:window`.
        self.last_app_rate_limit: str | None = None
        self.last_app_rate_limit_count: str | None = None

    # --- the one HTTP chokepoint -------------------------------------------

    def _get(self, host: str, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET `{host}{path}` with the API key, and return parsed JSON.

        The whole client's auth, error, and rate-limit policy in one place:

          - Auth: the key rides as the `X-Riot-Token` header, not a query param.
          - Errors: a non-2xx status becomes `RiotAPIError(status, message)` --
            404 = subject not found, 403 = key missing/expired, 5xx = Riot's fault.
          - Rate limits: REACTIVE. This is a one-time author pull, so rather than
            model Riot's 20/s + 100/2min token buckets, we wait out the window Riot
            hands back on a 429 (`Retry-After`) and retry -- bounded by
            `_MAX_RETRIES` so an expired/throttled key surfaces instead of looping.
          - Network blips: a read/connect timeout or dropped connection is
            transient and near-certain over a multi-thousand-call crawl, so it is
            caught and retried with exponential backoff (same `_MAX_RETRIES` cap) --
            one hiccup mustn't abort a 2.6h run.

        Returns `resp.json()` -- a dict for detail calls, a JSON array for the
        id-list call, hence the `Any`.
        """
        url = f"{host}{path}"
        headers = {"X-Riot-Token": self._api_key}
        network_err: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                resp = self._session.get(
                    url, headers=headers, params=params, timeout=self._TIMEOUT_S
                )
            except requests.exceptions.RequestException as exc:
                # Transient: read/connect timeout, dropped connection. Back off
                # (1,2,4,8,...s, capped) and retry rather than crash the crawl.
                network_err = exc
                time.sleep(min(2 ** attempt, 30))
                continue
            # Capture the app-limit headers on EVERY response -- especially a 429,
            # which is exactly when the count is the number the pilot cares about.
            self.last_app_rate_limit = resp.headers.get("X-App-Rate-Limit")
            self.last_app_rate_limit_count = resp.headers.get("X-App-Rate-Limit-Count")
            if resp.status_code == 429:
                time.sleep(int(resp.headers.get("Retry-After", "1")))
                continue
            if resp.status_code >= 400:
                raise RiotAPIError(resp.status_code, self._error_message(resp))
            return resp.json()
        if network_err is not None:  # exhausted retries on network errors
            raise RiotAPIError(0, f"network error after {self._MAX_RETRIES} retries: {network_err}")
        raise RiotAPIError(429, "rate-limited: retries exhausted")

    @staticmethod
    def _error_message(resp: requests.Response) -> str:
        """Pull Riot's `status.message` out of an error body, defensively -- the
        body may be a non-dict or unparseable, so fall back to the status code."""
        try:
            body = resp.json()
            return body.get("status", {}).get("message", str(body))
        except Exception:  # noqa: BLE001 -- any parse failure -> generic message
            return f"HTTP {resp.status_code}"

    # --- Account-V1 (regional) --------------------------------------------

    def account_by_riot_id(self, game_name: str, tag_line: str) -> dict[str, Any]:
        """Riot ID -> account, whose `puuid` is the join key for everything else.

        Run once and cache the PUUID -- it's stable, and name lookup is gone, so
        this is the only way in. Routes to `_account_host` (the continental cluster),
        NOT `_regional_host`: Account-V1 403s on `sea`, so a SEA match region resolves
        its account on `asia`.
        """
        path = f"/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        return self._get(self._account_host, path)

    # --- Match-V5 (regional) ----------------------------------------------

    def match_ids_by_puuid(
        self,
        puuid: str,
        *,
        queue: int | None = 420,
        count: int = 20,
        start: int = 0,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[str]:
        """PUUID -> match-ID list (newest first) -- the N games in the snapshot.

        Defaults to `queue=420` (ranked solo) so the sample stays apples-to-apples
        with the ranked benchmark; pass `queue=None` to drop the filter.

        Riot caps `count` at 100 per call, so this LOOPS in pages of 100 -- a single
        call for the common `count <= 100` snapshot, but it transparently handles a
        larger request and stops early on a short page (the subject has no more
        matches). Only the `queue` / time filters that are set are sent.
        """
        path = f"/lol/match/v5/matches/by-puuid/{puuid}/ids"
        ids: list[str] = []
        offset, remaining = start, count
        while remaining > 0:
            page = min(remaining, 100)
            params: dict[str, Any] = {"start": offset, "count": page}
            if queue is not None:
                params["queue"] = queue
            if start_time is not None:
                params["startTime"] = start_time
            if end_time is not None:
                params["endTime"] = end_time
            chunk = self._get(self._regional_host, path, params)
            ids.extend(chunk)
            if len(chunk) < page:  # short page = subject has no more matches
                break
            offset += len(chunk)
            remaining -= len(chunk)
        return ids

    def match(self, match_id: str) -> dict[str, Any]:
        """One match's full detail -- `info.participants[]` (+ `challenges`) and
        `info.teams[]`. The source of the support stat line; filter to the subject
        by `puuid`, then read `teamPosition` (`UTILITY` = support). 1 call/match."""
        path = f"/lol/match/v5/matches/{match_id}"
        return self._get(self._regional_host, path)

    def match_timeline(self, match_id: str) -> dict[str, Any]:
        """One match's timeline -- per-minute `frames[]` (positions, gold, xp) and
        the `events[]` stream. Home of death-with-context (`CHAMPION_KILL`) and
        lane state @10-14. 1 call/match; large payload, so the ingest script should
        persist the raw response before cleaning it."""
        path = f"/lol/match/v5/matches/{match_id}/timeline"
        return self._get(self._regional_host, path)

    # --- Rank: Summoner-V4 + League-V4 (PLATFORM) -------------------------
    # Not the personal plane -- the Benchmark Crawl uses these to read the
    # subject's tier and pick which bands to collect. Note the platform host.

    def summoner_by_puuid(self, puuid: str) -> dict[str, Any]:
        """PUUID -> summoner, whose `id` (encrypted summonerId) feeds League-V4.
        Platform-routed. Step 1 of the two-step rank chain."""
        path = f"/lol/summoner/v4/summoners/by-puuid/{puuid}"
        return self._get(self._platform_host, path)

    def league_entries(self, summoner_id: str) -> list[dict[str, Any]]:
        """summonerId -> ranked entries (`tier`, `rank`, `leaguePoints`) per queue.
        Platform-routed. Step 2 of the OLD two-step rank chain.

        ⚠️ Riot **removed** `/entries/by-summoner/{summonerId}` on 2025-06-20 in the
        PUUID migration -- prefer `league_entries_by_puuid` below, which is the
        current one-call form. Kept only for reference / any un-migrated caller."""
        path = f"/lol/league/v4/entries/by-summoner/{summoner_id}"
        return self._get(self._platform_host, path)

    def league_entries_by_puuid(self, puuid: str) -> list[dict[str, Any]]:
        """PUUID -> ranked entries per queue (one `LeagueEntryDTO` each). One call,
        no Summoner-V4 hop -- the current form since `by-summoner` was removed.

        Each entry carries `queueType` (`RANKED_SOLO_5x5` / `RANKED_FLEX_SR`),
        `tier`, `rank`, `leaguePoints`, and the cumulative `wins` / `losses` for the
        CURRENT split (they reset each split -- not lifetime). An **unranked** queue
        is *absent from the array*, not zeroed, so callers must handle "not found."
        Platform-routed."""
        path = f"/lol/league/v4/entries/by-puuid/{puuid}"
        return self._get(self._platform_host, path)

    # --- Cohort-crawl seeding: League-V4 + Summoner-V4 (PLATFORM) ----------
    # The Benchmark Crawl (09-01) walks a rank band on the subject's shard to
    # seed candidate cohort players, then bridges each to a puuid for the match
    # crawl. NOTE the queue STRING here (`RANKED_SOLO_5x5`), not Match-V5's
    # numeric 420 -- the two APIs spell the same queue differently, the classic
    # League-V4 gotcha, so the default is spelled out rather than reused.

    _APEX_LEAGUE_PATH = {
        "CHALLENGER": "challengerleagues",
        "GRANDMASTER": "grandmasterleagues",
        "MASTER": "masterleagues",
    }

    def apex_league(
        self, tier: str, *, queue: str = "RANKED_SOLO_5x5"
    ) -> dict[str, Any]:
        """One apex tier's whole league (CHALLENGER / GRANDMASTER / MASTER) as a
        LeagueListDTO -- `entries[]` are the players (each carrying `summonerId`,
        and a `puuid` on the newer schema). Platform-routed. The seed's top end;
        apex tiers have no divisions, so they get their own endpoint (no paging)."""
        path = f"/lol/league/v4/{self._APEX_LEAGUE_PATH[tier]}/by-queue/{queue}"
        return self._get(self._platform_host, path)

    def entries_by_queue(
        self,
        tier: str,
        division: str,
        *,
        queue: str = "RANKED_SOLO_5x5",
        page: int = 1,
    ) -> list[dict[str, Any]]:
        """One page of ranked entries at a non-apex band (IRON..DIAMOND x I..IV)
        -- LeagueEntryDTO list, each with `summonerId` (+ `puuid` on the newer
        schema). Platform-routed. `page` starts at 1; an empty list means past the
        last page. The seed's bulk -- most players (the subject included) live in
        one of these bands, so this is the cohort's main source."""
        path = f"/lol/league/v4/entries/{queue}/{tier}/{division}"
        return self._get(self._platform_host, path, {"page": page})

    def summoner_by_id(self, summoner_id: str) -> dict[str, Any]:
        """summonerId -> summoner, whose `puuid` is the join key into Match-V5.
        Platform-routed. Bridges a league entry (which keys on `summonerId`) to the
        regional match crawl -- the fallback when an entry carries no `puuid`."""
        path = f"/lol/summoner/v4/summoners/{summoner_id}"
        return self._get(self._platform_host, path)
