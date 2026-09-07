"""Spec for carryia/personal/riot_client.py -- the typed Riot client behind the Match Snapshot.

TDD state on first run (mirrors test_phases.py):
  - Routing and getter *wiring* are implemented, so those tests are GREEN -- they
    lock the regional-vs-platform gotcha against regression. If a rank call ever
    gets routed to the regional cluster (or vice versa), one goes red.
  - `_get` and `match_ids_by_puuid` are NotImplementedError stubs, so their
    contract tests start RED. Make them green by implementing the bodies; the
    assertions are forced invariants that hold regardless of *how* you implement.
  - The rate-limit STRATEGY (reactive vs proactive) is your decision, so its test
    is left skipped -- fill it once you've chosen, like the wildcard table in
    test_phases.py.
  - The live test hits the real API for Heidi#8481; it skips without RIOT_API_KEY,
    and (like everything else) is RED until `_get` exists.

Run:  .venv/bin/pytest tests/test_riot_client.py -v
"""

import os
from pathlib import Path

import pytest
import requests

from carryia.personal.riot_client import (
    PlatformRoute,
    RegionalRoute,
    RiotAPIError,
    RiotClient,
)

# A throwaway key for the pure/mocked tests -- never the real one.
FAKE_KEY = "RGAPI-test-0000"


def _client(
    regional: RegionalRoute = RegionalRoute.AMERICAS,
    platform: PlatformRoute = PlatformRoute.NA1,
) -> RiotClient:
    """A client on the subject's real routing (NA -> AMERICAS + NA1), fake key."""
    return RiotClient(FAKE_KEY, regional, platform)


# --- routing: host construction (GREEN -- regression lock) ------------------


def test_hosts_built_from_routes():
    c = _client()
    assert c._regional_host == "https://americas.api.riotgames.com"
    assert c._platform_host == "https://na1.api.riotgames.com"


def test_a_different_pair_routes_differently():
    c = _client(RegionalRoute.EUROPE, PlatformRoute.EUW1)
    assert c._regional_host == "https://europe.api.riotgames.com"
    assert c._platform_host == "https://euw1.api.riotgames.com"


def test_account_host_equals_match_host_for_continental_regions():
    # Americas/asia/europe serve Account-V1 directly -- account and match share a host.
    assert _client(RegionalRoute.AMERICAS)._account_host == "https://americas.api.riotgames.com"


def test_sea_routes_account_to_asia_but_matches_to_sea():
    # The SEA gotcha: Account-V1 403s on `sea`, so it resolves on `asia` while the
    # match host stays `sea`.
    c = _client(RegionalRoute.SEA)
    assert c._regional_host == "https://sea.api.riotgames.com"
    assert c._account_host == "https://asia.api.riotgames.com"


def test_platform_optional_for_regional_only_clients():
    # The snapshot ingest is regional-only; no platform/shard needed.
    assert RiotClient(FAKE_KEY, RegionalRoute.SEA)._platform_host is None


@pytest.mark.parametrize(
    "route, subdomain",
    [
        (RegionalRoute.AMERICAS, "americas"),
        (RegionalRoute.SEA, "sea"),
        (PlatformRoute.EUW1, "euw1"),
        (PlatformRoute.KR, "kr"),
        (PlatformRoute.SG2, "sg2"),  # the subject's shard; the cohort crawl seeds here
    ],
)
def test_route_values_are_subdomains(route, subdomain):
    assert route == subdomain


def test_sea_shards_present():
    # All the SEA platform shards route under the `sea` regional cluster.
    assert {r.value for r in PlatformRoute} >= {"sg2", "ph2", "th2", "tw2", "vn2"}


# --- getter wiring: right host, right path (GREEN -- the gotcha as a test) ---
# `_get` is spied, so these assert routing + path only, independent of HTTP.


def _record(c: RiotClient) -> list[tuple]:
    """Replace the HTTP chokepoint with a spy; return the list it appends to."""
    calls: list[tuple] = []

    def fake_get(host, path, params=None):
        calls.append((host, path, params))
        return {"ok": True}

    c._get = fake_get  # type: ignore[method-assign]
    return calls


@pytest.mark.parametrize(
    "invoke, host_attr, path",
    [
        (
            lambda c: c.account_by_riot_id("Heidi", "8481"),
            "_account_host",  # global service, its own host (not the match region)
            "/riot/account/v1/accounts/by-riot-id/Heidi/8481",
        ),
        (
            lambda c: c.match("NA1_4900000001"),
            "_regional_host",
            "/lol/match/v5/matches/NA1_4900000001",
        ),
        (
            lambda c: c.match_timeline("NA1_4900000001"),
            "_regional_host",
            "/lol/match/v5/matches/NA1_4900000001/timeline",
        ),
        (
            lambda c: c.summoner_by_puuid("PUUID123"),
            "_platform_host",  # rank lives on the PLATFORM shard, not regional
            "/lol/summoner/v4/summoners/by-puuid/PUUID123",
        ),
        (
            lambda c: c.league_entries("SUMMONER123"),
            "_platform_host",
            "/lol/league/v4/entries/by-summoner/SUMMONER123",
        ),
        (
            lambda c: c.apex_league("CHALLENGER"),
            "_platform_host",  # cohort seed, platform shard, League-V4 queue STRING
            "/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5",
        ),
        (
            lambda c: c.summoner_by_id("SUMMONER123"),
            "_platform_host",
            "/lol/summoner/v4/summoners/SUMMONER123",
        ),
    ],
)
def test_getter_routes_to_correct_host_and_path(invoke, host_attr, path):
    c = _client()
    calls = _record(c)
    result = invoke(c)
    assert calls == [(getattr(c, host_attr), path, None)]
    assert result == {"ok": True}  # getter passes `_get`'s return straight through


# --- cohort-crawl seeding: routing + the League-V4 queue-string gotcha --------


@pytest.mark.parametrize(
    "tier, sub_path",
    [
        ("CHALLENGER", "challengerleagues"),
        ("GRANDMASTER", "grandmasterleagues"),
        ("MASTER", "masterleagues"),
    ],
)
def test_apex_league_routes_each_tier_to_its_endpoint(tier, sub_path):
    c = _client(RegionalRoute.SEA, PlatformRoute.SG2)
    calls = _record(c)
    c.apex_league(tier)
    assert calls == [
        (c._platform_host, f"/lol/league/v4/{sub_path}/by-queue/RANKED_SOLO_5x5", None)
    ]


def test_entries_by_queue_routes_with_band_and_page():
    # Non-apex band: tier/division in the path, page as a query param, and the
    # League-V4 queue STRING (not Match-V5's 420).
    c = _client(RegionalRoute.SEA, PlatformRoute.SG2)
    calls = _record(c)
    c.entries_by_queue("DIAMOND", "I", page=2)
    assert calls == [
        (c._platform_host, "/lol/league/v4/entries/RANKED_SOLO_5x5/DIAMOND/I", {"page": 2})
    ]


def test_entries_by_queue_defaults_to_page_one():
    c = _client(RegionalRoute.SEA, PlatformRoute.SG2)
    calls = _record(c)
    c.entries_by_queue("GOLD", "IV")
    assert calls[0][2] == {"page": 1}


# --- RiotAPIError (GREEN) ---------------------------------------------------


def test_riot_api_error_carries_status():
    err = RiotAPIError(403, "Forbidden")
    assert err.status == 403
    assert "403" in str(err)


# --- `_get` contract: forced invariants (RED until you implement `_get`) -----
# These hold whatever rate-limit strategy you pick -- they don't test strategy.


class _FakeResponse:
    """Minimal requests.Response stand-in: `.status_code`, `.headers`, `.json()`,
    `.raise_for_status()`. Enough for any reasonable `_get` implementation."""

    def __init__(self, status: int, payload: object) -> None:
        self.status_code = status
        self.headers: dict[str, str] = {}
        self._payload = payload

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error")


def _client_returning(status: int, payload: object) -> RiotClient:
    c = _client()
    c._session.get = lambda *a, **k: _FakeResponse(status, payload)  # type: ignore[method-assign]
    return c


def test_get_success_returns_parsed_object():
    c = _client_returning(200, {"puuid": "abc"})
    assert c._get(c._regional_host, "/x") == {"puuid": "abc"}


def test_get_success_returns_a_bare_list():
    # The id-list endpoint returns a JSON array, not an object.
    c = _client_returning(200, ["NA1_1", "NA1_2"])
    assert c._get(c._regional_host, "/ids") == ["NA1_1", "NA1_2"]


def test_get_captures_app_rate_limit_headers():
    # The pilot reads these to measure crawl headroom before projecting 5 -> N.
    resp = _FakeResponse(200, {"ok": True})
    resp.headers["X-App-Rate-Limit"] = "20:1,100:120"
    resp.headers["X-App-Rate-Limit-Count"] = "3:1,42:120"
    c = _client()
    c._session.get = lambda *a, **k: resp  # type: ignore[method-assign]
    c._get(c._platform_host, "/x")
    assert c.last_app_rate_limit == "20:1,100:120"
    assert c.last_app_rate_limit_count == "3:1,42:120"


def test_get_captures_rate_limit_count_on_429(monkeypatch):
    # A 429 is exactly when the count matters -- capture it before waiting. On a
    # retries-exhausted run the throttle's own count is what survives.
    from carryia.personal import riot_client as rc

    monkeypatch.setattr(rc.time, "sleep", lambda *_: None)
    throttled = _FakeResponse(429, {})
    throttled.headers["Retry-After"] = "0"
    throttled.headers["X-App-Rate-Limit-Count"] = "100:120"
    c = _client()
    c._session.get = lambda *a, **k: throttled  # type: ignore[method-assign]
    with pytest.raises(RiotAPIError):
        c._get(c._platform_host, "/x")
    assert c.last_app_rate_limit_count == "100:120"


@pytest.mark.parametrize("status", [403, 404, 500])
def test_get_maps_error_status_to_riot_api_error(status):
    # 403 = key missing/expired, 404 = subject not found, 5xx = Riot's problem.
    c = _client_returning(status, {"status": {"message": "nope"}})
    with pytest.raises(RiotAPIError) as exc:
        c._get(c._platform_host, "/x")
    assert exc.value.status == status


# --- match_ids_by_puuid: forced invariants (RED until implemented) ----------


def test_match_ids_routes_regionally_and_defaults_to_ranked_queue():
    c = _client()
    calls: list[tuple] = []

    def fake_get(host, path, params=None):
        calls.append((host, path, params))
        return ["NA1_1", "NA1_2"]

    c._get = fake_get  # type: ignore[method-assign]
    out = c.match_ids_by_puuid("PUUID123", count=2)

    assert out == ["NA1_1", "NA1_2"]
    host, path, params = calls[0]
    assert host == c._regional_host
    assert path == "/lol/match/v5/matches/by-puuid/PUUID123/ids"
    assert isinstance(params, dict) and params.get("queue") == 420


# --- rate-limit strategy: YOUR decision (skipped, like the wildcard table) ---


def test_get_reactive_retry_on_429(monkeypatch):
    """The strategy `_get` implements: a 429 is waited out (per `Retry-After`) and
    retried, then succeeds. `time.sleep` is patched so the test doesn't actually
    wait."""
    from carryia.personal import riot_client as rc

    monkeypatch.setattr(rc.time, "sleep", lambda *_: None)

    throttled = _FakeResponse(429, {})
    throttled.headers["Retry-After"] = "0"
    ok = _FakeResponse(200, {"puuid": "abc"})
    responses = iter([throttled, ok])

    c = _client()
    c._session.get = lambda *a, **k: next(responses)  # type: ignore[method-assign]
    assert c._get(c._regional_host, "/x") == {"puuid": "abc"}


def test_get_retries_on_network_error(monkeypatch):
    """A transient read/connect timeout is retried, then succeeds -- one blip must
    not abort a long crawl. `time.sleep` patched so the backoff doesn't wait."""
    from carryia.personal import riot_client as rc

    monkeypatch.setattr(rc.time, "sleep", lambda *_: None)
    ok = _FakeResponse(200, {"puuid": "abc"})
    steps = iter([requests.exceptions.ReadTimeout("read timed out"), ok])

    def flaky(*a, **k):
        step = next(steps)
        if isinstance(step, Exception):
            raise step
        return step

    c = _client()
    c._session.get = flaky  # type: ignore[method-assign]
    assert c._get(c._regional_host, "/x") == {"puuid": "abc"}


def test_get_gives_up_after_network_retries(monkeypatch):
    """Endless network errors surface as a RiotAPIError (status 0), not a crash."""
    from carryia.personal import riot_client as rc

    monkeypatch.setattr(rc.time, "sleep", lambda *_: None)

    def always_timeout(*a, **k):
        raise requests.exceptions.ConnectionError("read timed out")

    c = _client()
    c._session.get = always_timeout  # type: ignore[method-assign]
    with pytest.raises(RiotAPIError) as exc:
        c._get(c._regional_host, "/x")
    assert exc.value.status == 0
    assert "network error" in str(exc.value)


def test_get_gives_up_after_max_retries(monkeypatch):
    """Bounded: endless 429s surface as a RiotAPIError, not an infinite loop."""
    from carryia.personal import riot_client as rc

    monkeypatch.setattr(rc.time, "sleep", lambda *_: None)

    always_throttled = _FakeResponse(429, {})
    always_throttled.headers["Retry-After"] = "0"

    c = _client()
    c._session.get = lambda *a, **k: always_throttled  # type: ignore[method-assign]
    with pytest.raises(RiotAPIError) as exc:
        c._get(c._regional_host, "/x")
    assert exc.value.status == 429


# --- live smoke: one real round-trip (opt-in) -------------------------------


def _load_env_key(name: str) -> str | None:
    """Read `name` from the environment, falling back to a parse of repo `.env`.
    Keeps the real key out of the source -- read at runtime, never committed."""
    if name in os.environ:
        return os.environ[name]
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if line.startswith(f"{name}=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip()
    return None


_LIVE_KEY = _load_env_key("RIOT_API_KEY")


@pytest.mark.skipif(not _LIVE_KEY, reason="live: set RIOT_API_KEY (dev keys expire every 24h)")
def test_live_resolve_heidi_8481():
    """Real identity resolution: Heidi#8481 -> a PUUID. Confirms the key works and
    the chain is wired end-to-end. One call, rate-limit friendly. RED until `_get`
    exists; a 403 here after that means the dev key expired."""
    c = RiotClient(_LIVE_KEY, RegionalRoute.AMERICAS, PlatformRoute.NA1)
    acct = c.account_by_riot_id("Heidi", "8481")
    assert isinstance(acct.get("puuid"), str)
    assert acct["puuid"], "puuid should be non-empty"
