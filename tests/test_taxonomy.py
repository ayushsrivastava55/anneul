"""Tests for the full failure taxonomy: every class id in the yaml is reachable offline.

Deterministic classes are decided from run flags and trace shape with zero model calls; the
rest come from the classifier, which is a scripted ``tests.fakes.FakeClient`` here. An
unusable reply must land on a taxonomy id and be flagged as a fallback in the ledger.
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
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    context_token_limit,
    deterministic_class,
    deterministic_classes,
    llm_classes,
    load_taxonomy,
    missing_capability_signal,
)
from anneal.spec import HarnessSpec
from tests.fakes import FakeClient

TAXONOMY = load_taxonomy()

SPEC = HarnessSpec.model_validate(
    {
        "id": "cand-tax",
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

OK_STEP = {"tool": "get_reservation_details", "args": {"id": "R1"}, "result": "ok"}


@dataclass
class FakeTask:
    id: str
    input: dict[str, Any]
    split: str = "search"
    expected: dict[str, Any] = field(default_factory=dict)


def _row(task_id: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task_id": task_id,
        "candidate_id": "cand-tax",
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
        "trace": [dict(OK_STEP)],
    }
    base.update(over)
    return base


def _domain(task_ids: list[str]) -> Any:
    tasks = [FakeTask(id=t, input={"instruction": f"do {t}"}) for t in task_ids]

    def load_tasks(split: str | None = None) -> list[FakeTask]:
        return [t for t in tasks if split is None or t.split == split]

    return SimpleNamespace(name="fake", eval=SimpleNamespace(THRESHOLD=1.0, load_tasks=load_tasks))


def _diagnose(row: dict[str, Any], ledger_path: Path, client: Any | None) -> dict[str, Any]:
    issues = diagnose.diagnose(
        [row], _domain([row["task_id"]]), SPEC, ledger_path=ledger_path, client=client
    )
    assert len(issues) == 1
    return issues[0]


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "ledger.json"


# --- taxonomy coverage --------------------------------------------------------------------


def test_every_class_is_wired_to_exactly_one_detector() -> None:
    det, llm_ids = deterministic_classes(), llm_classes(TAXONOMY)
    assert set(det) | set(llm_ids) == set(TAXONOMY), "a taxonomy class has no detector"
    assert not set(det) & set(llm_ids)
    assert all(TAXONOMY[c]["detect"] == "deterministic" for c in det)
    assert all(TAXONOMY[c]["detect"] == "llm" for c in llm_ids)


# --- deterministic classes ----------------------------------------------------------------


# loop_or_timeout is intentionally absent: a step-budget death is a symptom, so the row is
# either reclassed deterministically as missing_capability (ungranted-tool call in the trace)
# or sent to the model with loop_or_timeout still on the menu. See test_diagnose.py.
DETERMINISTIC_ROWS: list[tuple[str, dict[str, Any]]] = [
    ("unsafe_action", {"hard_fail": True}),
    ("output_format", {"schema_error": True}),
    ("context_overflow", {"output": "error: BadRequestError('context_length_exceeded')"}),
]


@pytest.mark.parametrize(
    ("expected", "over"), DETERMINISTIC_ROWS, ids=[c for c, _ in DETERMINISTIC_ROWS]
)
def test_deterministic_row_classifies_without_a_model_call(
    expected: str, over: dict[str, Any], ledger_path: Path
) -> None:
    fake = FakeClient(turns=[])  # any call would raise "no scripted turns left"
    issue = _diagnose(_row(f"t-{expected}", **over), ledger_path, fake)
    assert (issue["class"], issue["fallback"]) == (expected, False)
    assert fake.calls == []


def test_context_overflow_fires_on_a_token_total_over_the_limit() -> None:
    row = _row("t-big", tokens_in=DEFAULT_CONTEXT_TOKEN_LIMIT, tokens_out=1)
    assert deterministic_class(row) == "context_overflow"
    assert deterministic_class(_row("t-small")) is None


def test_context_token_limit_reads_env_and_survives_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(diagnose.CONTEXT_TOKEN_LIMIT_ENV, "500")
    assert context_token_limit() == 500
    assert deterministic_class(_row("t-big", tokens_in=600)) == "context_overflow"
    for bad in ("not-a-number", "", "-1"):
        monkeypatch.setenv(diagnose.CONTEXT_TOKEN_LIMIT_ENV, bad)
        assert context_token_limit() == DEFAULT_CONTEXT_TOKEN_LIMIT
    monkeypatch.delenv(diagnose.CONTEXT_TOKEN_LIMIT_ENV)
    assert context_token_limit() == DEFAULT_CONTEXT_TOKEN_LIMIT


def test_unsafe_action_outranks_the_other_deterministic_checks() -> None:
    row = _row("t-all", hard_fail=True, schema_error=True, hit_step_budget=True, tokens_in=10**9)
    assert deterministic_class(row) == "unsafe_action"


# --- missing_capability heuristics --------------------------------------------------------


REFUND = {
    "tool": "issue_refund",
    "args": {"id": "R1"},
    "result": "Error: unknown tool 'issue_refund'",
}
FAILED_CALL = {
    "tool": "cancel_reservation",
    "args": {"id": "R1", "why": "x"},
    "result": "Error: no",
}


def test_signal_fires_for_an_action_no_tool_provides() -> None:
    assert missing_capability_signal(_row("t", trace=[dict(OK_STEP), REFUND]), SPEC)
    # Same story with a tool the spec never granted, whatever the result text says.
    ungranted = {"tool": "book_flight", "args": {}, "result": "booked"}
    assert missing_capability_signal(_row("t", trace=[ungranted]), SPEC)


def test_signal_fires_for_an_identical_failing_call_retried() -> None:
    reordered = {
        "tool": "cancel_reservation",
        "args": {"why": "x", "id": "R1"},
        "result": "Error: no",
    }
    assert missing_capability_signal(_row("t", trace=[FAILED_CALL, reordered]), SPEC)
    # One failure is a blip, and different args are a different attempt.
    assert not missing_capability_signal(_row("t", trace=[FAILED_CALL]), SPEC)
    other = {"tool": "cancel_reservation", "args": {"id": "R2"}, "result": "Error: no"}
    assert not missing_capability_signal(_row("t", trace=[FAILED_CALL, other]), SPEC)
    assert not missing_capability_signal(_row("t", trace=[dict(OK_STEP), dict(OK_STEP)]), SPEC)


def test_heuristics_hint_the_classifier_without_deciding_for_it(ledger_path: Path) -> None:
    row = _row("t-cap", trace=[REFUND])
    assert deterministic_class(row) is None  # the heuristic must not short-circuit the model
    fake = FakeClient(turns=[json.dumps({"class": "wrong_tool", "node": "executor"})])
    issue = _diagnose(row, ledger_path, fake)
    assert len(fake.calls) == 1
    assert "missing_capability" in json.dumps(fake.calls[0]["messages"])
    assert (issue["class"], issue["fallback"]) == ("wrong_tool", False)


# --- classifier classes -------------------------------------------------------------------


@pytest.mark.parametrize("expected", llm_classes(TAXONOMY))
def test_classifier_row_reaches_every_llm_class(expected: str, ledger_path: Path) -> None:
    fake = FakeClient(turns=[json.dumps({"class": expected, "node": "executor"})])
    issue = _diagnose(_row(f"t-{expected}"), ledger_path, fake)
    assert (issue["class"], issue["node"], issue["fallback"]) == (expected, "executor", False)
    system = fake.calls[0]["messages"][0]["content"]
    assert expected in system and TAXONOMY[expected]["description"][:30] in system
    for det in deterministic_classes():
        assert det not in system  # the model may not pick a deterministic-only class


# --- fallback -----------------------------------------------------------------------------


UNUSABLE = ["definitely not json", json.dumps({"class": "hallucinated_class"}), "{}"]


@pytest.mark.parametrize("reply", UNUSABLE, ids=["garbage", "invented_id", "empty_object"])
def test_unusable_reply_falls_back_to_a_taxonomy_class(reply: str, ledger_path: Path) -> None:
    fake = FakeClient(turns=[reply])
    issue = _diagnose(_row("t-fallback"), ledger_path, fake)
    assert issue["class"] in TAXONOMY
    assert issue["class"] == "wrong_tool"  # highest-severity classifier class, no signal fired
    assert issue["fallback"] is True
    assert issue["node"] == "executor"  # blamed by default_node, not by the model


def test_fallback_prefers_a_fired_heuristic_over_the_default(ledger_path: Path) -> None:
    fake = FakeClient(turns=["I cannot answer that"])
    issue = _diagnose(_row("t-fallback-cap", trace=[REFUND]), ledger_path, fake)
    assert (issue["class"], issue["fallback"]) == ("missing_capability", True)


def test_fallback_flag_is_sticky_and_persisted(ledger_path: Path) -> None:
    good = json.dumps({"class": "wrong_tool", "node": "executor"})
    _diagnose(_row("t-a"), ledger_path, FakeClient(turns=[good]))
    issue = _diagnose(_row("t-b"), ledger_path, FakeClient(turns=["garbage"]))
    assert issue["count"] == 2 and issue["fallback"] is True
    _diagnose(_row("t-c"), ledger_path, FakeClient(turns=[good]))
    stored = json.loads(ledger_path.read_text())
    assert len(stored) == 1 and stored[0]["fallback"] is True and stored[0]["count"] == 3
