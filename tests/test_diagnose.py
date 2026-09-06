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
        def __init__(self, result: dict[str, Any], session: str | None = None) -> None:
            self.headers = {"content-type": "application/json"}
            if session:
                self.headers["mcp-session-id"] = session
            self._result = result

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"jsonrpc": "2.0", "id": 1, "result": self._result}

    class FakeHttp:
        def post(self, url: str, *, json: Any, headers: dict[str, str]) -> FakeResponse:
            sent.append({"url": url, "json": json, "headers": dict(headers)})
            if json["method"] == "initialize":
                return FakeResponse({"protocolVersion": "2025-03-26"}, session="sess-1")
            text = __import__("json").dumps({"spans": [1]})
            return FakeResponse({"content": [{"type": "text", "text": text}]})

    src = NeatlogsMCP("secret", http=FakeHttp(), min_interval_s=0.0)
    ctx = src.get_trace_context("abc")
    assert ctx == {"spans": [1]}
    assert [s["json"]["method"] for s in sent] == ["initialize", "tools/call"]
    assert sent[0]["headers"]["Authorization"] == "Bearer secret"
    assert sent[1]["headers"]["Mcp-Session-Id"] == "sess-1"
    assert sent[1]["json"]["params"]["name"] == "get_trace_context"
    assert "abc" in json.dumps(sent[1]["json"]["params"]["arguments"])
    src.get_trace_context("def")
    assert len(sent) == 3  # initialize happens once


def test_default_source_is_local_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    assert isinstance(diagnose.default_trace_source(ROWS), LocalTraces)
    monkeypatch.setenv("NEATLOGS_API_KEY", "k")
    assert isinstance(diagnose.default_trace_source(ROWS), NeatlogsMCP)


# --- diagnosis confidence -----------------------------------------------------------------


def test_reported_confidence_is_parsed_clamped_and_carried() -> None:
    reply = json.dumps({"class": "wrong_tool", "node": "executor", "confidence": 0.9})
    cls, node, conf = diagnose.classify_with_llm(
        [{"role": "user", "content": "x"}], ["wrong_tool"], ["executor"], fake_client(reply)
    )
    assert (cls, node, conf) == ("wrong_tool", "executor", 0.9)
    # Out-of-range values are clamped rather than trusted or discarded.
    for raw, want in ((5, 1.0), (-2, 0.0)):
        body = json.dumps({"class": "wrong_tool", "node": "executor", "confidence": raw})
        assert diagnose.classify_with_llm(
            [{"role": "user", "content": "x"}], ["wrong_tool"], ["executor"], fake_client(body)
        )[2] == want


@pytest.mark.parametrize("raw", ["null", '"high"', "{}"])
def test_unreported_confidence_is_unknown_not_low(raw: str) -> None:
    """Absent or junk confidence must read as None. It is the difference between "we do not
    know how sure the classifier was" and "the classifier told us it was unsure", and only
    the latter is allowed to block a repair."""
    reply = f'{{"class": "wrong_tool", "node": "executor", "confidence": {raw}}}'
    assert diagnose.classify_with_llm(
        [{"role": "user", "content": "x"}], ["wrong_tool"], ["executor"], fake_client(reply)
    )[2] is None


def test_a_guessed_class_scores_as_the_low_confidence_diagnosis_it_is() -> None:
    """An unusable reply falls back to a heuristic guess. The ledger already recorded that as
    `fallback` and then nothing acted on it; a guess is a low-confidence diagnosis."""
    ledger: list[Any] = []
    issue = diagnose.upsert(
        ledger, "wrong_tool", "executor", ["t1"], fallback=True,
        confidence=diagnose.FALLBACK_CONFIDENCE,
    )
    assert issue["fallback"] is True
    assert diagnose.confidence_of(issue) == diagnose.FALLBACK_CONFIDENCE
    assert not diagnose.actionable(issue), "we must not spend an operator on a guess"


def test_confidence_is_a_running_mean_so_one_shaky_row_cannot_sink_an_issue() -> None:
    ledger: list[Any] = []
    diagnose.upsert(ledger, "wrong_tool", "executor", ["t1"], confidence=0.9)
    for i in range(8):
        diagnose.upsert(ledger, "wrong_tool", "executor", [f"t{i + 2}"], confidence=0.9)
    issue = diagnose.upsert(ledger, "wrong_tool", "executor", ["t99"], confidence=0.1)
    assert issue["count"] == 10 and issue["confidence_n"] == 10
    assert diagnose.confidence_of(issue) == pytest.approx(0.82)
    assert diagnose.actionable(issue), "nine solid diagnoses outweigh one shaky one"


def test_repeated_shaky_diagnoses_do_sink_an_issue() -> None:
    ledger: list[Any] = []
    for i in range(6):
        issue = diagnose.upsert(ledger, "wrong_tool", "executor", [f"t{i}"], confidence=0.2)
    assert diagnose.confidence_of(issue) == pytest.approx(0.2)
    assert not diagnose.actionable(issue)


def test_unscored_issue_is_actionable_so_a_silent_classifier_cannot_stall_the_loop() -> None:
    """The failure mode this guards is severe: if absence blocked, a model that ignores the
    confidence field would halt every repair while looking like a principled refusal."""
    ledger: list[Any] = []
    issue = diagnose.upsert(ledger, "wrong_tool", "executor", ["t1"], confidence=None)
    assert not diagnose.confidence_known(issue)
    assert diagnose.confidence_of(issue) == diagnose.DEFAULT_CONFIDENCE
    assert diagnose.DEFAULT_CONFIDENCE < diagnose.CONFIDENCE_FLOOR, "default is below the floor"
    assert diagnose.actionable(issue), "unknown must not block"


def test_deterministic_classes_are_certain() -> None:
    """A forbidden tool call is observed by rule, not inferred, so there is nothing to doubt."""
    row = {"task_id": "t", "score": 0.0, "hard_fail": True}
    cls, _node, fell_back, conf = diagnose._classify_row(
        row, None, SPEC, LocalTraces([row]), diagnose.load_taxonomy(), None
    )
    assert cls == "unsafe_action" and not fell_back
    assert conf == diagnose.CERTAIN_CONFIDENCE
    assert diagnose.actionable({"confidence": conf, "confidence_n": 1})


def test_rank_weights_by_confidence_without_letting_it_override_severity() -> None:
    tax = diagnose.load_taxonomy()
    unsure_severe = {
        "id": "L-0001", "class": "unsafe_action", "node": "executor", "count": 4,
        "status": "open", "confidence": 0.6, "confidence_n": 4,
    }
    certain_mild = {
        "id": "L-0002", "class": "context_overflow", "node": "executor", "count": 1,
        "status": "open", "confidence": 1.0, "confidence_n": 1,
    }
    # 4 * 5 * 0.6 = 12.0 still beats 1 * 2 * 1.0 = 2.0: confidence weights, it does not veto.
    assert [i["id"] for i in diagnose.rank([certain_mild, unsure_severe], tax)] == [
        "L-0001", "L-0002",
    ]
    # But between two equally common failures, the believed one goes first.
    doubted = {**unsure_severe, "id": "L-0003", "confidence": 0.3}
    believed = {**unsure_severe, "id": "L-0004", "confidence": 0.95}
    assert [i["id"] for i in diagnose.rank([doubted, believed], tax)] == ["L-0004", "L-0003"]
