"""Tests for anneal.diagnose: deterministic pre-checks, LLM classifier, ledger upsert/rank.

All offline: the LLM classifier is a fake callable, traces come from the rows themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anneal import diagnose
from anneal.diagnose import (
    LocalTraces,
    NeatlogsMCP,
    load_ledger,
    load_taxonomy,
    rank,
    upsert,
)
from anneal.spec import HarnessSpec
from tests.fakes import FakeClient

SPEC = HarnessSpec.model_validate(
    {
        "id": "cand-1",
        "topology": "single",
        "step_budget": 8,
        "nodes": [
            {
                "name": "executor",
                "role": "executor",
                "model_tier": "mid",
                "system_prompt_ref": "airline_executor@v1",
                "tools": ["get_reservation_details", "cancel_reservation"],
                "max_steps": 8,
            }
        ],
    }
)


@dataclass
class FakeTask:
    id: str
    input: dict[str, Any]
    split: str = "search"
    expected: dict[str, Any] = field(default_factory=dict)


def _row(task_id: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task_id": task_id,
        "candidate_id": "cand-1",
        "iteration": 0,
        "score": 0.0,
        "hard_fail": False,
        "hit_step_budget": False,
        "schema_error": False,
        "tokens_in": 10,
        "tokens_out": 5,
        "per_node": {"executor": {"tokens_in": 10, "tokens_out": 5, "backend": None, "ms": 5}},
        "latency_ms": 5,
        "trace_id": f"trace-{task_id}",
        "output": {"answer": "done"},
        "trace": [{"tool": "cancel_reservation", "args": {"reservation_id": "X"}, "result": "ok"}],
    }
    base.update(over)
    return base


ROWS = [
    _row("t-pass", score=1.0),
    _row("t-unsafe", hard_fail=True),
    _row("t-schema", schema_error=True),
    _row("t-loop", hit_step_budget=True),
    _row("t-wrongtool"),
]


def fake_domain(task_ids: list[str]) -> Any:
    tasks = [FakeTask(id=t, input={"instruction": f"do {t}"}) for t in task_ids]

    def load_tasks(split: str | None = None) -> list[FakeTask]:
        return [t for t in tasks if split is None or t.split == split]

    return SimpleNamespace(name="fake", eval=SimpleNamespace(THRESHOLD=1.0, load_tasks=load_tasks))


WRONG_TOOL = json.dumps({"class": "wrong_tool", "node": "executor"})


def fake_client(reply: str, turns: int = 1) -> FakeClient:
    """Shared offline OpenAI stand-in scripted to answer ``reply`` ``turns`` times."""
    return FakeClient(turns=[reply] * turns)


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "ledger.json"


def test_four_failures_produce_four_ledger_entries(ledger_path: Path) -> None:
    fake = fake_client(WRONG_TOOL)
    domain = fake_domain([r["task_id"] for r in ROWS])
    issues = diagnose.diagnose(ROWS, domain, SPEC, ledger_path=ledger_path, client=fake)

    by_class = {i["class"]: i for i in issues}
    assert set(by_class) == {"unsafe_action", "output_format", "loop_or_timeout", "wrong_tool"}
    assert all(i["count"] == 1 and i["status"] == "open" for i in issues)
    assert by_class["unsafe_action"]["evidence"] == ["trace-t-unsafe"]
    assert by_class["wrong_tool"]["node"] == "executor"
    assert [i["id"] for i in issues] == ["L-0001", "L-0002", "L-0003", "L-0004"]
    # Only the LLM-detected failure needed a model call; deterministic classes did not.
    assert len(fake.calls) == 1
    prompt = json.dumps(fake.calls[0]["messages"])
    assert "do t-wrongtool" in prompt and "cancel_reservation" in prompt
    assert json.loads(ledger_path.read_text()) == issues


def test_rerun_upserts_counts_and_dedupes_evidence(ledger_path: Path) -> None:
    fake = fake_client(WRONG_TOOL, turns=2)
    domain = fake_domain([r["task_id"] for r in ROWS])
    diagnose.diagnose(ROWS, domain, SPEC, ledger_path=ledger_path, client=fake)
    second = diagnose.diagnose(ROWS, domain, SPEC, ledger_path=ledger_path, client=fake)

    assert len(second) == 4
    assert all(i["count"] == 2 for i in second)
    assert all(len(i["evidence"]) == 1 for i in second)
    ids = [i["id"] for i in load_ledger(ledger_path)]
    assert ids == ["L-0001", "L-0002", "L-0003", "L-0004"]


def test_rank_orders_by_count_times_severity() -> None:
    taxonomy = load_taxonomy()
    ledger: list[dict[str, Any]] = []
    for ev in ("a", "b", "c", "d"):
        upsert(ledger, "output_format", "executor", [ev])  # 4 * 3 = 12
    for _ in range(3):
        upsert(ledger, "unsafe_action", "executor", ["z"])  # 3 * 5 = 15
    upsert(ledger, "context_overflow", "executor", ["y"])  # 1 * 2 = 2
    upsert(ledger, "wrong_tool", "executor", ["x"])  # 1 * 4 = 4
    ledger[-1]["status"] = "fixed"

    ranked = rank(ledger, taxonomy)
    assert [i["class"] for i in ranked] == ["unsafe_action", "output_format", "context_overflow"]
    assert ledger[0]["count"] == 4 and ledger[1]["count"] == 3


def test_unknown_llm_class_falls_back_to_first_llm_class(ledger_path: Path) -> None:
    fake = fake_client("definitely not json")
    domain = fake_domain(["t-wrongtool"])
    issues = diagnose.diagnose(
        [_row("t-wrongtool")], domain, SPEC, ledger_path=ledger_path, client=fake
    )
    assert len(issues) == 1
    assert issues[0]["class"] in load_taxonomy()


def test_rows_outside_search_split_are_never_read(ledger_path: Path) -> None:
    fake = fake_client(WRONG_TOOL)
    domain = fake_domain(["t-wrongtool"])  # t-other is not a search task
    rows = [_row("t-wrongtool"), _row("t-other")]
    issues = diagnose.diagnose(rows, domain, SPEC, ledger_path=ledger_path, client=fake)
    assert len(issues) == 1 and issues[0]["count"] == 1
    assert len(fake.calls) == 1


def test_local_traces_use_row_trace() -> None:
    src = LocalTraces(ROWS)
    ctx = src.get_trace_context("trace-t-unsafe")
    assert ctx is not None and ctx["trace"][0]["tool"] == "cancel_reservation"
    assert src.get_trace_context("missing") is None


def test_neatlogs_mcp_calls_tool_with_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": [{"type": "text", "text": json.dumps({"spans": [1]})}]},
            }

    class FakeHttp:
        def post(self, url: str, *, json: Any, headers: dict[str, str]) -> FakeResponse:
            sent.append({"url": url, "json": json, "headers": headers})
            return FakeResponse()

    src = NeatlogsMCP("secret", http=FakeHttp(), min_interval_s=0.0)
    ctx = src.get_trace_context("abc")
    assert ctx == {"spans": [1]}
    assert sent[0]["headers"]["Authorization"] == "Bearer secret"
    assert sent[0]["json"]["params"]["name"] == "get_trace_context"
    assert "abc" in json.dumps(sent[0]["json"]["params"]["arguments"])


def test_default_source_is_local_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    assert isinstance(diagnose.default_trace_source(ROWS), LocalTraces)
    monkeypatch.setenv("NEATLOGS_API_KEY", "k")
    assert isinstance(diagnose.default_trace_source(ROWS), NeatlogsMCP)
