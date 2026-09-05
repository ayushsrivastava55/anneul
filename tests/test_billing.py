"""Offline tests for anneal.billing: no network, no key, no sleeping.

Every HTTP interaction is served by ``httpx.MockTransport``; the one test that exercises the
default client construction monkeypatches ``anneal.billing.httpx.Client`` so the bearer header
and timeout are still asserted without a socket.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest

from anneal import billing
from anneal.billing import (
    MAX_EVENTS_PER_INGEST,
    BillingClient,
    BudgetExhausted,
    BudgetGuard,
    DodoError,
    UsageEvent,
    event_id,
)

KEY = "sk_test_do_not_log_me"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry backoff must never actually sleep in tests."""
    monkeypatch.setattr(billing.time, "sleep", lambda _seconds: None)


@pytest.fixture(autouse=True)
def _cache_to_tmp(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the dashboard balance cache out of the repo's runs/ directory."""
    monkeypatch.setattr(billing, "BALANCE_CACHE", tmp_path / "balance.json")


def make_client(handler: Any, **kwargs: Any) -> BillingClient:
    """A live-mode client whose transport is the given request handler."""
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://test.example")
    kwargs.setdefault("api_key", KEY)
    kwargs.setdefault("customer_id", "cus_1")
    return BillingClient(base_url="https://test.example", http=http, **kwargs)


# --- deterministic event ids ---------------------------------------------------------------


def test_event_id_is_stable_and_field_sensitive() -> None:
    first = event_id("airline", 2, "cand-a", "task-7")
    assert first == event_id("airline", 2, "cand-a", "task-7")
    assert first == UsageEvent("airline", 2, "cand-a", "task-7", 10).id
    assert first != event_id("airline", 3, "cand-a", "task-7")
    assert first != event_id("airline", 2, "cand-b", "task-7")
    assert first != event_id("hotel", 2, "cand-a", "task-7")


def test_event_id_stable_across_client_instances() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={})

    for _ in range(2):
        client = make_client(handler)
        client.debit("airline", 1, "cand-a", "task-1", 5)
        client.flush()
    assert calls[0]["events"][0]["event_id"] == calls[1]["events"][0]["event_id"]


def test_event_payload_carries_numeric_tokens() -> None:
    payload = UsageEvent("airline", 0, "cand-a", "task-1", 42).payload("cus_1")
    assert payload["event_name"] == "anneal_run"
    assert payload["customer_id"] == "cus_1"
    assert payload["metadata"]["tokens"] == 42.0
    assert isinstance(payload["metadata"]["tokens"], float)


def test_repeated_debit_is_charged_once() -> None:
    ingested: list[list[dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ingested.append(json.loads(request.content)["events"])
        return httpx.Response(200, json={})

    client = make_client(handler)
    first = client.debit("airline", 0, "cand-a", "task-1", 100)
    second = client.debit("airline", 0, "cand-a", "task-1", 100)
    assert first == second
    assert client.flush() == 1
    assert client.ledger.spent == 100.0
    assert len(ingested) == 1 and len(ingested[0]) == 1


# --- batching ------------------------------------------------------------------------------


def test_ingest_splits_at_1000_events() -> None:
    batches: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/events/ingest":
            batches.append(len(json.loads(request.content)["events"]))
        return httpx.Response(200, json={})

    client = make_client(handler)
    for index in range(MAX_EVENTS_PER_INGEST + 1):
        client.debit("airline", 0, "cand-a", f"task-{index}", 1)
    client.flush()
    assert batches == [MAX_EVENTS_PER_INGEST, 1]


def test_debit_autoflushes_when_the_buffer_fills() -> None:
    batches: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        batches.append(len(json.loads(request.content)["events"]))
        return httpx.Response(200, json={})

    client = make_client(handler)
    for index in range(MAX_EVENTS_PER_INGEST):
        client.debit("airline", 0, "cand-a", f"task-{index}", 1)
    assert batches == [MAX_EVENTS_PER_INGEST]
    assert client.flush() == 0


# --- transport: retries, auth, no key in logs -----------------------------------------------


def test_retries_5xx_then_succeeds() -> None:
    statuses = [503, 500, 200]
    seen: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        status = statuses[len(seen)]
        seen.append(status)
        return httpx.Response(status, json={} if status == 200 else None)

    client = make_client(handler)
    client.request("POST", "/events/ingest", json_body={"events": []})
    assert seen == [503, 500, 200]


def test_gives_up_after_max_attempts_on_5xx() -> None:
    attempts: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(500, text="boom")

    client = make_client(handler)
    with pytest.raises(DodoError):
        client.request("GET", "/customers")
    assert len(attempts) == billing.MAX_ATTEMPTS


def test_4xx_raises_without_retrying() -> None:
    attempts: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(422, text="bad request")

    client = make_client(handler)
    with pytest.raises(DodoError, match="422"):
        client.request("GET", "/customers")
    assert len(attempts) == 1


def test_default_client_sets_bearer_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={})

    real_client = httpx.Client

    def fake_client(**kwargs: Any) -> httpx.Client:
        seen["kwargs"] = kwargs
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(billing.httpx, "Client", fake_client)
    client = BillingClient(api_key=KEY, customer_id="cus_1")
    client.request("GET", "/customers")
    assert seen["auth"] == f"Bearer {KEY}"
    assert seen["kwargs"]["timeout"] is billing.DEFAULT_TIMEOUT
    assert seen["kwargs"]["base_url"] == "https://test.dodopayments.com"


def test_key_never_reaches_the_logs(caplog: pytest.LogCaptureFixture) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream exploded")

    client = make_client(handler)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(DodoError) as failed_5xx:
            client.request("GET", "/customers")
        client.http._transport = httpx.MockTransport(  # noqa: SLF001 - swap to a 4xx transport
            lambda _r: httpx.Response(403, text="forbidden")
        )
        with pytest.raises(DodoError) as failed_4xx:
            client.request("GET", "/customers")
    assert KEY not in caplog.text
    assert KEY not in str(failed_5xx.value)
    assert KEY not in str(failed_4xx.value)


# --- a small fake Dodo -----------------------------------------------------------------------


class FakeDodo:
    """Enough of the Dodo API to drive entitlement, ingest, balance and ship end to end."""

    def __init__(self) -> None:
        self.balance = 0.0
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.idempotency_keys: set[str] = set()
        self.meters: list[dict[str, Any]] = []
        self.entitlements: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        path = request.url.path
        self.requests.append((request.method, path, body))
        if request.method == "GET" and path == "/meters":
            return httpx.Response(200, json={"items": self.meters})
        if request.method == "POST" and path == "/meters":
            self.meters.append({**body, "id": "mtr_1"})
            return httpx.Response(200, json={"id": "mtr_1"})
        if request.method == "GET" and path == "/credit-entitlements":
            return httpx.Response(200, json={"items": self.entitlements})
        if request.method == "POST" and path == "/credit-entitlements":
            self.entitlements.append({**body, "id": "ce_1"})
            return httpx.Response(200, json={"id": "ce_1"})
        if path.endswith("/ledger-entries"):
            return self._ledger(body)
        if "/balances/" in path:
            return httpx.Response(200, json={"balance": f"{self.balance}"})
        if path == "/events/ingest":
            return httpx.Response(200, json={})
        if path == "/products":
            return httpx.Response(200, json={"product_id": "prod_1"})
        if path == "/checkouts":
            return httpx.Response(200, json={"checkout_url": "https://test.example/c/sess_1"})
        return httpx.Response(404, text=f"no route for {path}")

    def _ledger(self, body: dict[str, Any]) -> httpx.Response:
        key = body.get("idempotency_key")
        if key in self.idempotency_keys:
            return httpx.Response(409, text="idempotency key already exists")
        self.idempotency_keys.add(str(key))
        amount = float(body["amount"])
        self.balance += amount if body["entry_type"] == "credit" else -amount
        return httpx.Response(200, json={"id": "led_1", "amount": body["amount"]})


# --- entitlement, debit, balance --------------------------------------------------------------


def test_budget_becomes_a_token_grant_and_runs_draw_it_down() -> None:
    dodo = FakeDodo()
    client = make_client(dodo)
    client.ensure_entitlement(0.001)
    assert client.entitlement_id == "ce_1"
    assert client.meter_id == "mtr_1"
    assert client.balance() == 1000.0

    client.debit("airline", 0, "cand-a", "task-1", 400)
    client.flush()
    assert client.balance() == 600.0

    grants = [b for m, p, b in dodo.requests if p.endswith("/ledger-entries")]
    assert [g["entry_type"] for g in grants] == ["credit", "debit"]
    assert grants[0]["amount"] == "1000"


def test_entitlement_tops_up_only_the_shortfall() -> None:
    dodo = FakeDodo()
    client = make_client(dodo)
    client.ensure_entitlement(0.001)
    client.ensure_entitlement(0.001)
    credits = [b for _m, p, b in dodo.requests if p.endswith("/ledger-entries")]
    assert len(credits) == 1
    assert dodo.balance == 1000.0


def test_replayed_ledger_debit_409_is_not_an_error() -> None:
    dodo = FakeDodo()
    client = make_client(dodo)
    client.ensure_entitlement(0.001)
    client.debit("airline", 0, "cand-a", "task-1", 100)
    client.flush()
    # A second identical batch (e.g. a retried process) hits the same idempotency key.
    client._pending.append(UsageEvent("airline", 0, "cand-a", "task-1", 100))
    client.flush()
    assert dodo.balance == 900.0


# --- BudgetGuard halts ------------------------------------------------------------------------


def test_guard_halts_when_the_remote_balance_hits_zero() -> None:
    dodo = FakeDodo()
    guard = BudgetGuard(0.001, make_client(dodo)).start()
    guard.record_run("airline", 0, "cand-a", "task-1", 400, cost_usd=0.0001)
    assert guard.check() == 600.0
    guard.record_run("airline", 0, "cand-a", "task-2", 600, cost_usd=0.0001)
    with pytest.raises(BudgetExhausted, match="balance"):
        guard.check()


def test_guard_halts_when_local_spend_passes_the_budget() -> None:
    dodo = FakeDodo()
    guard = BudgetGuard(1.00, make_client(dodo)).start()
    guard.record_run("airline", 0, "cand-a", "task-1", 10, cost_usd=0.40)
    assert guard.remaining_usd == pytest.approx(0.60)
    guard.record_run("airline", 0, "cand-a", "task-2", 10, cost_usd=0.65)
    with pytest.raises(BudgetExhausted, match="local spend"):
        guard.check()


def test_guard_flushes_before_polling_the_balance() -> None:
    dodo = FakeDodo()
    guard = BudgetGuard(1.00, make_client(dodo)).start()
    guard.record_run("airline", 0, "cand-a", "task-1", 250)
    assert [p for _m, p, _b in dodo.requests if p == "/events/ingest"] == []
    guard.check()
    assert [p for _m, p, _b in dodo.requests if p == "/events/ingest"] == ["/events/ingest"]


def test_guard_falls_back_to_token_cost_when_no_usd_given() -> None:
    guard = BudgetGuard(0.001, BillingClient(api_key="")).start()
    guard.record_run("airline", 0, "cand-a", "task-1", 500)
    assert guard.spent_usd == pytest.approx(0.0005)


# --- no-key path ------------------------------------------------------------------------------


def test_no_key_warns_once_and_still_enforces_the_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        client = BillingClient(api_key="")
    assert not client.enabled
    assert sum("DODO_API_KEY is unset" in r.message for r in caplog.records) == 1

    guard = BudgetGuard(0.001, client).start()
    assert client.balance() == 1000.0
    guard.record_run("airline", 0, "cand-a", "task-1", 400)
    assert guard.check() == 600.0
    guard.record_run("airline", 0, "cand-a", "task-2", 600)
    with pytest.raises(BudgetExhausted):
        guard.check()


def test_local_ledger_persists_and_dedupes_across_processes(tmp_path: Any) -> None:
    path = tmp_path / "local_ledger.json"
    first = BillingClient(api_key="", ledger_path=path)
    first.ensure_entitlement(0.001)
    first.debit("airline", 0, "cand-a", "task-1", 300)
    assert first.balance() == 700.0

    second = BillingClient(api_key="", ledger_path=path)
    second.debit("airline", 0, "cand-a", "task-1", 300)  # same event_id: already charged
    assert second.ledger.spent == 300.0
    assert second.balance() == 700.0


def test_no_key_makes_no_http_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(**_kwargs: Any) -> httpx.Client:
        raise AssertionError("offline billing must not build an HTTP client")

    monkeypatch.setattr(billing.httpx, "Client", explode)
    client = BillingClient(api_key="")
    client.ensure_entitlement(0.01)
    client.debit("airline", 0, "cand-a", "task-1", 5)
    client.flush()
    client.balance()
    client.ship_agent("agent", "specs/harness.example.yaml")
    client.close()


# --- ship_agent -------------------------------------------------------------------------------


def test_ship_agent_creates_a_metered_product_and_returns_the_link() -> None:
    dodo = FakeDodo()
    client = make_client(dodo)
    client.ensure_entitlement(0.001)
    shipped = client.ship_agent("airline-v3", "runs/final/airline/spec.yaml")

    assert shipped.checkout_url == "https://test.example/c/sess_1"
    assert (shipped.product_id, shipped.meter_id, shipped.offline) == ("prod_1", "mtr_1", False)
    product = next(b for _m, p, b in dodo.requests if p == "/products")
    meter = product["price"]["meters"][0]
    assert product["price"]["type"] == "usage_based_price"
    assert meter["meter_id"] == "mtr_1"
    assert meter["credit_entitlement_id"] == "ce_1"
    cart = next(b for _m, p, b in dodo.requests if p == "/checkouts")
    assert cart["product_cart"] == [{"product_id": "prod_1", "quantity": 1}]


def test_ship_agent_offline_returns_a_deterministic_placeholder_link() -> None:
    client = BillingClient(api_key="")
    first = client.ship_agent("airline-v3", "runs/final/airline/spec.yaml")
    second = BillingClient(api_key="").ship_agent("airline-v3", "elsewhere/spec.yaml")
    assert first.offline and first.checkout_url == second.checkout_url
    assert first.checkout_url.startswith("https://test.dodopayments.com/checkout/offline-")


def test_meter_is_reused_when_it_already_exists() -> None:
    dodo = FakeDodo()
    dodo.meters.append({"id": "mtr_existing", "name": "anneal_run_tokens"})
    client = make_client(dodo)
    assert client.ensure_meter() == "mtr_existing"
    assert [m for m, p, _b in dodo.requests if p == "/meters"] == ["GET"]


def test_balance_is_cached_for_the_dashboard() -> None:
    client = BillingClient(api_key="")
    client.ensure_entitlement(0.001)
    client.debit("airline", 0, "cand-a", "task-1", 250)
    client.balance()
    cached = json.loads(billing.BALANCE_CACHE.read_text())
    assert cached == {
        "balance_tokens": 750.0,
        "spent_tokens": 250.0,
        "unit": "tokens",
        "live": False,
        "updated_at": cached["updated_at"],
    }


def test_smoke_halts_and_prints_a_checkout_link(capsys: pytest.CaptureFixture[str]) -> None:
    assert billing._smoke(budget_usd=0.0005) == 0
    printed = capsys.readouterr().out
    assert "HALT:" in printed
    assert "checkout link: https://" in printed


def test_debit_is_thread_safe_under_the_runner_concurrency() -> None:
    dodo = FakeDodo()
    client = make_client(dodo)
    tasks = [f"task-{index}" for index in range(200)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda t: client.debit("airline", 0, "cand-a", t, 3), tasks))

    assert len(set(ids)) == 200
    assert client.flush() == 200
    assert client.ledger.spent == 600.0
