"""Dodo Payments billing: credit entitlement, per-run debits, budget halt, shipped product.

The optimiser runs on a credit budget. ``--budget <usd>`` is granted as credits in the custom
unit ``tokens``; every task run ingests a usage event (``event_name=anneal_run``) whose
``metadata.tokens`` is the model tokens the run consumed, and the loop halts when the balance
reaches zero (``BudgetGuard.check`` raises :class:`BudgetExhausted`).

Design notes:

- **Deterministic ``event_id``.** ``sha1(domain, iteration, candidate, task)``. Dodo treats
  ``event_id`` as an idempotency key, so a retried batch — or a re-run of the same task in the
  same iteration — never double-charges.
- **Offline by default.** ``DODO_API_KEY`` is empty until the account is verified. With no key
  every method degrades to a local ledger (in memory, optionally persisted) that still grants,
  still debits and still halts at the budget, after one warning. Nothing here needs the network
  to be testable.
- **Endpoints in one place.** ``ENDPOINTS`` mirrors the ``dodopayments`` SDK's routes (read off
  the installed SDK, not guessed). We speak to them with ``httpx`` so the transport — timeouts,
  5xx-only retries, tracing — is ours. If a route ever moves, fix it in that dict alone.
- **Balance actually moves.** Usage events only auto-deduct credits through a product whose
  usage-based price links the meter to the credit entitlement
  (``price.meters[].credit_entitlement_id``), which exists only once an agent is shipped. So the
  optimiser's own runs also post a ledger ``debit`` per flushed batch, keyed by an idempotency
  key derived from the event ids. A ``409`` (key already used) means the debit already landed.
- **Never log the key.** No request headers, and no exception text from this module, contain it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from anneal.tracing import tool_span

logger = logging.getLogger("anneal.billing")

ROOT = Path(__file__).resolve().parent.parent

BASE_URLS: dict[str, str] = {
    "test": "https://test.dodopayments.com",
    "live": "https://live.dodopayments.com",
}

#: Routes, as exposed by the dodopayments SDK. Single source of truth for this module.
ENDPOINTS: dict[str, str] = {
    "customers": "/customers",
    "customer": "/customers/{customer_id}",
    "ingest": "/events/ingest",
    "meters": "/meters",
    "credit_entitlements": "/credit-entitlements",
    "balance": "/credit-entitlements/{entitlement_id}/balances/{customer_id}",
    "ledger_entries": "/credit-entitlements/{entitlement_id}/balances/{customer_id}/ledger-entries",
    "products": "/products",
    "checkouts": "/checkouts",
}

EVENT_NAME = "anneal_run"
CREDIT_UNIT = "tokens"
METADATA_TOKENS_KEY = "tokens"
MAX_EVENTS_PER_INGEST = 1000
#: One credit ("token") per model token: $1 of budget ≈ 1M tokens at the mid tier.
TOKENS_PER_USD = 1_000_000.0
DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 0.5
BALANCE_CACHE = ROOT / "runs" / "billing" / "balance.json"
LOCAL_LEDGER = ROOT / "runs" / "billing" / "local_ledger.json"


class DodoError(RuntimeError):
    """A Dodo API call failed. Carries status and body, never credentials."""


class BudgetExhausted(RuntimeError):
    """The credit balance hit zero or local spend passed the budget: stop the loop."""


def event_id(domain: str, iteration: int, candidate: str, task: str) -> str:
    """Deterministic idempotency key for one task run. Stable across processes and retries."""
    key = "\x1f".join([domain, str(iteration), candidate, task])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def balance(path: Path | None = None) -> float | None:
    """Last known credit balance from the cache ``BillingClient`` writes, else ``None``.

    Module-level and read-only on purpose: the dashboard needs a balance without holding a
    client, without a key and without touching the network. Returns ``None`` when no run has
    billed yet or the cache is unreadable.
    """
    try:
        raw = json.loads((path or BALANCE_CACHE).read_text(encoding="utf-8"))
        value = raw["balance_tokens"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return float(value) if isinstance(value, int | float) else None


def tokens_per_usd() -> float:
    """Credits granted per USD of budget; override with ``ANNEAL_TOKENS_PER_USD``."""
    raw = (os.environ.get("ANNEAL_TOKENS_PER_USD") or "").strip()
    return float(raw) if raw else TOKENS_PER_USD


@dataclass(frozen=True)
class UsageEvent:
    """One task run, billed as ``metadata.tokens`` against the ``anneal_run`` meter."""

    domain: str
    iteration: int
    candidate: str
    task: str
    tokens: int

    @property
    def id(self) -> str:
        """The deterministic ``event_id`` Dodo de-duplicates on."""
        return event_id(self.domain, self.iteration, self.candidate, self.task)

    def payload(self, customer_id: str) -> dict[str, Any]:
        """The ``/events/ingest`` event body. Metadata values are scalars only."""
        return {
            "event_id": self.id,
            "customer_id": customer_id,
            "event_name": EVENT_NAME,
            "metadata": {
                METADATA_TOKENS_KEY: float(self.tokens),
                "domain": self.domain,
                "iteration": float(self.iteration),
                "candidate": self.candidate,
                "task": self.task,
            },
        }


@dataclass
class ShippedAgent:
    """A finished agent, billable: usage meter + product + checkout link."""

    name: str
    meter_id: str
    product_id: str
    checkout_url: str
    offline: bool = False


@dataclass
class LocalLedger:
    """The no-key fallback: grants, spends and de-duplicates exactly like the real thing."""

    granted: float = 0.0
    spent: float = 0.0
    events: dict[str, float] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | None) -> LocalLedger:
        """Read the ledger at ``path``; a missing or unreadable file starts an empty one."""
        if path is None or not path.exists():
            return cls(path=path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("local ledger at %s is unreadable; starting empty", path)
            return cls(path=path)
        return cls(
            granted=float(raw.get("granted", 0.0)),
            spent=float(raw.get("spent", 0.0)),
            events={str(k): float(v) for k, v in (raw.get("events") or {}).items()},
            path=path,
        )

    def grant(self, tokens: float) -> None:
        """Top the balance up to at least ``tokens`` credits, keeping what was already spent."""
        self.granted = max(self.granted, self.spent + tokens)
        self.save()

    def record(self, event: UsageEvent) -> bool:
        """Charge ``event`` once. False when this ``event_id`` was already charged."""
        if event.id in self.events:
            return False
        self.events[event.id] = float(event.tokens)
        self.spent += float(event.tokens)
        self.save()
        return True

    @property
    def balance(self) -> float:
        """Credits left, never negative."""
        return max(0.0, self.granted - self.spent)

    def save(self) -> None:
        """Persist when a path was configured; in-memory otherwise."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"granted": self.granted, "spent": self.spent, "events": self.events}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class BillingClient:
    """Dodo test-mode client with an offline fallback.

    ``http`` may be an injected ``httpx.Client`` (tests use ``httpx.MockTransport``); otherwise
    one is built lazily with the bearer key and a timeout.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        customer_id: str | None = None,
        base_url: str | None = None,
        env: str | None = None,
        http: httpx.Client | None = None,
        ledger_path: Path | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        backoff: float = BACKOFF_SECONDS,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self.env = (env or "test").strip() or "test"
        self.base_url = (base_url or BASE_URLS.get(self.env, BASE_URLS["test"])).rstrip("/")
        self.customer_id = (customer_id or "").strip() or None
        self.entitlement_id: str | None = None
        self.meter_id: str | None = None
        self._http = http
        self._owns_http = http is None
        self._max_attempts = max(1, max_attempts)
        self._backoff = backoff
        self._lock = threading.Lock()
        self._pending: list[UsageEvent] = []
        self.ledger = LocalLedger.load(ledger_path)
        if not self.enabled:
            logger.warning(
                "DODO_API_KEY is unset; billing degrades to a local ledger (budget still enforced)"
            )

    @classmethod
    def from_env(cls, **kwargs: Any) -> BillingClient:
        """Build from ``DODO_API_KEY`` / ``DODO_ENV`` / ``DODO_CUSTOMER_ID``."""
        kwargs.setdefault("api_key", os.environ.get("DODO_API_KEY"))
        kwargs.setdefault("env", os.environ.get("DODO_ENV") or "test")
        kwargs.setdefault("customer_id", os.environ.get("DODO_CUSTOMER_ID"))
        kwargs.setdefault("ledger_path", LOCAL_LEDGER)
        return cls(**kwargs)

    @property
    def enabled(self) -> bool:
        """True when a key is configured, i.e. calls go to Dodo rather than the local ledger."""
        return bool(self._api_key)

    # --- transport ---------------------------------------------------------------------

    @property
    def http(self) -> httpx.Client:
        """The HTTP client, built on first use with the bearer header and a timeout."""
        if self._http is None:
            self._http = httpx.Client(
                base_url=self.base_url,
                timeout=DEFAULT_TIMEOUT,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._http

    @tool_span("dodo.request")
    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Call Dodo. Retries 5xx only (``MAX_ATTEMPTS`` with linear backoff); 4xx raises at once.

        Errors quote status and response body — never the request headers, so no key can leak.
        """
        url = self.base_url + path
        last: httpx.Response | None = None
        for attempt in range(1, self._max_attempts + 1):
            response = self.http.request(method, url, json=json_body, params=params)
            if response.status_code < 500:
                return self._decode(response, method, path)
            last = response
            logger.warning(
                "dodo %s %s -> %s (attempt %d/%d)",
                method,
                path,
                response.status_code,
                attempt,
                self._max_attempts,
            )
            if attempt < self._max_attempts:
                time.sleep(self._backoff * attempt)
        raise DodoError(
            f"{method} {path} failed after {self._max_attempts} attempts: "
            f"{last.status_code if last else '?'} {last.text if last else ''}"[:500]
        )

    @staticmethod
    def _decode(response: httpx.Response, method: str, path: str) -> Any:
        """Return the parsed body, or raise :class:`DodoError` on a 4xx."""
        if response.status_code >= 400:
            raise DodoError(f"{method} {path} -> {response.status_code} {response.text}"[:500])
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return {}

    def close(self) -> None:
        """Flush pending events and close the HTTP client we own."""
        self.flush()
        if self._owns_http and self._http is not None:
            self._http.close()
            self._http = None

    # --- setup -------------------------------------------------------------------------

    @tool_span("dodo.ensure_customer")
    def ensure_customer(self, email: str | None = None, name: str | None = None) -> str:
        """Return the customer id, creating the customer when ``DODO_CUSTOMER_ID`` is unset."""
        if not self.enabled:
            self.customer_id = self.customer_id or "local-customer"
            return self.customer_id
        if self.customer_id:
            return self.customer_id
        body = {
            "email": email or os.environ.get("DODO_CUSTOMER_EMAIL") or "anneal@example.com",
            "name": name or "Anneal",
        }
        created = self.request("POST", ENDPOINTS["customers"], json_body=body)
        self.customer_id = str(created.get("customer_id") or created.get("id"))
        return self.customer_id

    def _find_named(self, path: str, name: str, id_keys: tuple[str, ...]) -> str | None:
        """First item on ``path`` whose ``name`` matches, as an id. None when absent."""
        listing = self.request("GET", path)
        items = listing.get("items") if isinstance(listing, dict) else listing
        for item in items or []:
            if isinstance(item, dict) and item.get("name") == name:
                for key in id_keys:
                    if item.get(key):
                        return str(item[key])
        return None

    @tool_span("dodo.ensure_meter")
    def ensure_meter(self, name: str = "anneal_run_tokens") -> str | None:
        """Meter summing ``metadata.tokens`` over ``anneal_run`` events. Idempotent by name."""
        if not self.enabled:
            self.meter_id = self.meter_id or "local-meter"
            return self.meter_id
        existing = self._find_named(ENDPOINTS["meters"], name, ("id", "meter_id"))
        if existing:
            self.meter_id = existing
            return existing
        body = {
            "name": name,
            "event_name": EVENT_NAME,
            "measurement_unit": CREDIT_UNIT,
            "aggregation": {"type": "sum", "key": METADATA_TOKENS_KEY},
            "description": "Model tokens consumed by one Anneal task run",
        }
        created = self.request("POST", ENDPOINTS["meters"], json_body=body)
        self.meter_id = str(created.get("id") or created.get("meter_id"))
        return self.meter_id

    @tool_span("dodo.ensure_entitlement")
    def ensure_entitlement(self, budget_usd: float, name: str = "anneal-budget") -> str | None:
        """Map ``--budget`` to a credit grant in the unit ``tokens``; tops up only the shortfall.

        Returns the credit entitlement id (None offline). Also ensures the meter, so a shipped
        product can link the two for auto-deduction.
        """
        required = budget_usd * tokens_per_usd()
        self.ensure_customer()
        self.ensure_meter()
        if not self.enabled:
            self.ledger.grant(required)
            return None
        if self.entitlement_id is None:
            self.entitlement_id = self._find_named(
                ENDPOINTS["credit_entitlements"], name, ("id", "credit_entitlement_id")
            ) or self._create_entitlement(name)
        shortfall = required - self._balance_or_zero()
        if shortfall > 0:
            self._ledger_entry(
                amount=shortfall,
                entry_type="credit",
                idempotency_key=f"grant:{name}:{uuid.uuid4().hex}",
                reason=f"anneal budget ${budget_usd:.2f}",
            )
        return self.entitlement_id

    def _balance_or_zero(self) -> float:
        """Balance, treating "no balance row yet" (404) as zero so the first grant can land."""
        try:
            return self.balance()
        except DodoError as exc:
            if " 404 " not in f" {exc} ":
                raise
            return 0.0

    def _create_entitlement(self, name: str) -> str:
        """Create the ``tokens`` credit entitlement."""
        body = {
            "name": name,
            "unit": CREDIT_UNIT,
            "precision": 0,
            "overage_enabled": False,
            "rollover_enabled": False,
            "description": "Anneal optimiser run budget",
        }
        created = self.request("POST", ENDPOINTS["credit_entitlements"], json_body=body)
        return str(created.get("id") or created.get("credit_entitlement_id"))

    def _ledger_entry(
        self, *, amount: float, entry_type: str, idempotency_key: str, reason: str
    ) -> None:
        """Post a credit/debit ledger entry. A 409 means the key already landed: not an error."""
        if not (self.enabled and self.entitlement_id and self.customer_id):
            return
        path = ENDPOINTS["ledger_entries"].format(
            entitlement_id=self.entitlement_id, customer_id=self.customer_id
        )
        body = {
            "credit_entitlement_id": self.entitlement_id,
            "amount": f"{amount:.0f}",
            "entry_type": entry_type,
            "idempotency_key": idempotency_key,
            "reason": reason,
        }
        try:
            self.request("POST", path, json_body=body)
        except DodoError as exc:
            if " 409 " not in f" {exc} ":
                raise
            logger.info("ledger entry %s already applied", idempotency_key)

    # --- metering ----------------------------------------------------------------------

    def debit(self, domain: str, iteration: int, candidate: str, task: str, tokens: int) -> str:
        """Charge one task run. Returns the deterministic ``event_id``.

        Events are buffered and ingested in batches of ``MAX_EVENTS_PER_INGEST``; the buffer is
        thread-safe because the runner executes tasks concurrently. Local spend is recorded
        immediately so the budget holds even if a flush never happens.
        """
        event = UsageEvent(domain, iteration, candidate, task, int(tokens))
        with self._lock:
            fresh = self.ledger.record(event)
            if fresh:
                self._pending.append(event)
            due = len(self._pending) >= MAX_EVENTS_PER_INGEST
        if due:
            self.flush()
        return event.id

    @tool_span("dodo.ingest")
    def flush(self) -> int:
        """Ingest buffered events, ≤1000 per call. Returns how many events were sent."""
        with self._lock:
            batch, self._pending = self._pending, []
        if not batch:
            return 0
        if not self.enabled:
            return len(batch)
        customer_id = self.ensure_customer()
        for start in range(0, len(batch), MAX_EVENTS_PER_INGEST):
            chunk = batch[start : start + MAX_EVENTS_PER_INGEST]
            self.request(
                "POST",
                ENDPOINTS["ingest"],
                json_body={"events": [event.payload(customer_id) for event in chunk]},
            )
            self._debit_credits(chunk)
        return len(batch)

    def _debit_credits(self, chunk: list[UsageEvent]) -> None:
        """Deduct the chunk's tokens from the credit balance.

        Usage events auto-deduct only through a product whose price links the meter to the
        entitlement, which exists once an agent is shipped; the optimiser's own runs debit the
        ledger directly, idempotently keyed by the chunk's event ids.
        """
        if not self.entitlement_id:
            return
        total = float(sum(event.tokens for event in chunk))
        if total <= 0:
            return
        key = hashlib.sha1("|".join(sorted(e.id for e in chunk)).encode("utf-8")).hexdigest()
        reason = f"{EVENT_NAME}: {len(chunk)} task runs"
        try:
            self._ledger_entry(
                amount=total, entry_type="debit", idempotency_key=f"run:{key}", reason=reason
            )
        except DodoError as exc:
            if " 400 " not in f" {exc} ":
                raise
            # Insufficient balance: draw the rest down to zero so BudgetGuard halts, not crashes.
            remaining = self.balance()
            logger.warning("debit of %.0f exceeds balance; drawing down %.0f", total, remaining)
            if remaining > 0:
                self._ledger_entry(
                    amount=remaining,
                    entry_type="debit",
                    idempotency_key=f"run:{key}:clamp",
                    reason=reason,
                )

    @tool_span("dodo.balance")
    def balance(self) -> float:
        """Credits left. Offline that is grant − spend. Cached to disk for the dashboard."""
        if self.enabled and self.entitlement_id and self.customer_id:
            path = ENDPOINTS["balance"].format(
                entitlement_id=self.entitlement_id, customer_id=self.customer_id
            )
            body = self.request("GET", path)
            value = max(0.0, float(body.get("balance", 0.0)))
        else:
            value = self.ledger.balance
        self._cache_balance(value)
        return value

    def _cache_balance(self, value: float) -> None:
        """Write ``runs/billing/balance.json`` for the dashboard. Failures are not fatal."""
        try:
            BALANCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            BALANCE_CACHE.write_text(
                json.dumps(
                    {
                        "balance_tokens": value,
                        "spent_tokens": self.ledger.spent,
                        "unit": CREDIT_UNIT,
                        "live": self.enabled,
                        "updated_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - dashboard courtesy only
            logger.debug("balance cache not written: %s", exc)

    # --- ship --------------------------------------------------------------------------

    @tool_span("dodo.ship_agent")
    def ship_agent(
        self, name: str, spec_path: str | Path, price_per_1k_usd: float = 0.02
    ) -> ShippedAgent:
        """Publish the finished agent: usage meter + usage-priced product, and a checkout link.

        The product's usage price links the ``anneal_run`` meter to the credit entitlement, so
        every invocation of the shipped agent draws down credits the buyer paid for. Offline the
        ids and the link are deterministic placeholders, clearly marked.
        """
        spec = Path(spec_path)
        meter_id = self.ensure_meter() or "local-meter"
        if not self.enabled:
            digest = hashlib.sha1(f"{name}|{spec.name}".encode()).hexdigest()[:12]
            url = f"{self.base_url}/checkout/offline-{digest}?agent={name}"
            logger.warning("no DODO_API_KEY: %s is a placeholder checkout link", url)
            return ShippedAgent(name, meter_id, f"local-product-{digest}", url, offline=True)
        product = self.request(
            "POST",
            ENDPOINTS["products"],
            json_body=self._product_body(name, spec, meter_id, price_per_1k_usd),
        )
        product_id = str(product.get("product_id") or product.get("id"))
        session = self.request(
            "POST",
            ENDPOINTS["checkouts"],
            json_body={"product_cart": [{"product_id": product_id, "quantity": 1}]},
        )
        url = str(session.get("checkout_url") or session.get("url") or "")
        return ShippedAgent(name, meter_id, product_id, url)

    def _product_body(
        self, name: str, spec: Path, meter_id: str, price_per_1k_usd: float
    ) -> dict[str, Any]:
        """Usage-based product body: fixed $0 subscription plus per-token metered usage."""
        meter: dict[str, Any] = {
            "meter_id": meter_id,
            "price_per_unit": f"{price_per_1k_usd / 1000.0:.9f}",
        }
        if self.entitlement_id:
            meter["credit_entitlement_id"] = self.entitlement_id
            meter["meter_units_per_credit"] = "1"
        return {
            "name": f"Anneal agent: {name}",
            "description": f"Agent harness annealed by Anneal ({spec.name}); billed per token.",
            "tax_category": "saas",
            "price": {
                "type": "usage_based_price",
                "currency": "USD",
                "discount": 0,
                "fixed_price": 0,
                "payment_frequency_count": 1,
                "payment_frequency_interval": "Month",
                "subscription_period_count": 1,
                "subscription_period_interval": "Month",
                "meters": [meter],
            },
        }


class BudgetGuard:
    """Budget enforcement the CLI calls once per iteration.

    ``record_run`` debits a task; ``check`` flushes the buffer, polls the balance and raises
    :class:`BudgetExhausted` when credits are gone or local spend has passed ``--budget``. The
    local condition is authoritative even with no key, so the loop always halts.
    """

    def __init__(
        self,
        budget_usd: float,
        client: BillingClient | None = None,
        *,
        min_balance_tokens: float = 0.0,
    ) -> None:
        self.budget_usd = float(budget_usd)
        self.client = client or BillingClient.from_env()
        self.min_balance_tokens = min_balance_tokens
        self._spend_usd = 0.0
        self._lock = threading.Lock()

    def start(self) -> BudgetGuard:
        """Ensure the customer and the credit grant for this budget. Safe to call twice."""
        self.client.ensure_entitlement(self.budget_usd)
        return self

    def record_run(
        self,
        domain: str,
        iteration: int,
        candidate: str,
        task: str,
        tokens: int,
        cost_usd: float | None = None,
    ) -> str:
        """Debit one task run. ``cost_usd`` (from the runner's per-backend prices) wins over the
        token-derived estimate for the local spend check."""
        event = self.client.debit(domain, iteration, candidate, task, tokens)
        spend = cost_usd if cost_usd is not None else tokens / tokens_per_usd()
        with self._lock:
            self._spend_usd += float(spend)
        return event

    @property
    def spent_usd(self) -> float:
        """USD spent so far according to the runs recorded here."""
        return self._spend_usd

    @property
    def remaining_usd(self) -> float:
        """Budget left, never negative."""
        return max(0.0, self.budget_usd - self._spend_usd)

    def check(self) -> float:
        """Flush, poll the balance, and raise :class:`BudgetExhausted` at the threshold."""
        self.client.flush()
        balance = self.client.balance()
        if self._spend_usd >= self.budget_usd:
            raise BudgetExhausted(
                f"local spend ${self._spend_usd:.4f} reached budget ${self.budget_usd:.2f}"
            )
        if balance <= self.min_balance_tokens:
            raise BudgetExhausted(
                f"credit balance {balance:.0f} {CREDIT_UNIT} at or below "
                f"{self.min_balance_tokens:.0f}; halting the loop"
            )
        return balance

    def close(self) -> None:
        """Flush and release the HTTP client."""
        self.client.close()


def _smoke(budget_usd: float = 0.001) -> int:
    """Offline demo: grant a tiny budget, debit until it halts, print the checkout link."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = BillingClient.from_env(ledger_path=None)
    guard = BudgetGuard(budget_usd, client).start()
    print(f"live={client.enabled} budget=${budget_usd} balance={client.balance():.0f} tokens")
    halted = False
    for index in range(10):
        guard.record_run("smoke", 0, "cand-0", f"task-{index}", tokens=200)
        try:
            print(f"  after task-{index}: balance={guard.check():.0f} tokens")
        except BudgetExhausted as exc:
            print(f"  HALT: {exc}")
            halted = True
            break
    shipped = client.ship_agent("smoke-agent", "specs/harness.example.yaml")
    print(f"checkout link: {shipped.checkout_url}")
    client.close()
    return 0 if halted else 1


if __name__ == "__main__":
    raise SystemExit(_smoke())
