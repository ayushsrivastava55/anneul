"""Shared test setup.

The suite is offline by contract. Every test that needs a model injects
``tests.fakes.FakeClient``; nothing should reach a network.

That contract used to hold only by accident. Modules pick their live-vs-local path by
looking for a credential -- ``diagnose.default_trace_source`` returns the real Neatlogs
MCP client when ``NEATLOGS_API_KEY`` is set and ``LocalTraces`` otherwise, and billing and
tracing branch the same way. So the moment a real key landed in ``.env``, tests that meant
to run offline started calling sponsor APIs with fixture data, and
``test_four_failures_produce_four_ledger_entries`` failed with the Neatlogs API rejecting
``trace-t-wrongtool`` as a malformed trace id. The suite's behaviour must not depend on
whether the machine running it happens to be configured.

So: strip sponsor credentials for every test. A test that wants the live path can set the
var itself with monkeypatch, which is explicit and local.
"""

from __future__ import annotations

import pytest

# Credentials that flip a module from its local/offline path to a live sponsor API.
LIVE_KEYS = (
    "NEATLOGS_API_KEY",
    "DODO_API_KEY",
    "DODO_CUSTOMER_ID",
    "DODO_CUSTOMER_EMAIL",
)


@pytest.fixture(autouse=True)
def _offline_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove sponsor credentials so no test silently goes live."""
    for key in LIVE_KEYS:
        monkeypatch.delenv(key, raising=False)
