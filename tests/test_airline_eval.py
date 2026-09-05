"""Acceptance tests for domains/airline/eval.py. No LLM calls; never touches holdout."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import yaml

from domains.airline import eval as airline_eval
from domains.airline.fixtures import tools as airline_tools

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT / "domains" / "airline"


def _train_task(tag: str, without: str = "transfer") -> airline_eval.Task:
    for task in airline_eval.load_tasks("train") + airline_eval.load_tasks("search"):
        if tag in task.tags and without not in task.tags:
            return task
    raise AssertionError(f"no non-holdout task tagged {tag} without {without}")


def _replay(task: airline_eval.Task) -> list[dict]:
    trace = []
    for action in task.expected["actions"]:
        fn = getattr(airline_tools, action["name"])
        fn(**action["kwargs"])
        trace.append({"tool": action["name"], "args": action["kwargs"]})
    return trace


def test_replaying_expected_actions_scores_one() -> None:
    task = _train_task("cancel")
    airline_eval.setup(task)
    trace = _replay(task)
    assert airline_eval.score(task, "done") == 1.0
    assert airline_eval.is_hard_fail(task, trace) is False


def test_doing_nothing_on_a_change_task_scores_zero() -> None:
    task = _train_task("write")
    airline_eval.setup(task)
    assert airline_eval.score(task, "I could not help with that.") == 0.0


def test_doing_nothing_on_a_no_change_task_scores_one() -> None:
    task = _train_task("no_write")
    airline_eval.setup(task)
    assert airline_eval.score(task, "Sorry, policy does not allow that.") == 1.0


def test_setup_resets_state_between_runs() -> None:
    task = _train_task("cancel")
    airline_eval.setup(task)
    before = airline_tools.db_hash()
    _replay(task)
    assert airline_tools.db_hash() != before
    airline_eval.setup(task)
    assert airline_tools.db_hash() == before


def test_unexpected_cancel_is_hard_fail() -> None:
    task = _train_task("no_write")
    trace = [
        {"tool": "get_user_details", "args": {"user_id": task.input["user_id"]}},
        {"tool": "cancel_reservation", "args": {"reservation_id": "ZFA04Y"}},
    ]
    assert airline_eval.is_hard_fail(task, trace) is True


def test_unexpected_transfer_is_hard_fail() -> None:
    task = _train_task("no_write")
    assert "transfer" not in task.tags
    trace = [{"tool": "transfer_to_human_agents", "args": {"summary": "help"}}]
    assert airline_eval.is_hard_fail(task, trace) is True


def test_reads_only_are_never_hard_fail() -> None:
    task = _train_task("write")
    trace = [
        {"tool": "get_user_details", "args": {"user_id": task.input["user_id"]}},
        {"tool": "search_direct_flight", "args": {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}},
        {"tool": "calculate", "args": {"expression": "1+1"}},
    ]
    assert airline_eval.is_hard_fail(task, trace) is False


def test_split_sizes_and_threshold() -> None:
    assert airline_eval.THRESHOLD == 1.0
    assert len(airline_eval.load_tasks("train")) == 20
    assert len(airline_eval.load_tasks("search")) == 10
    lines = (DOMAIN / "tasks.jsonl").read_text().splitlines()
    assert len(lines) == 40
    assert {json.loads(line)["split"] for line in lines} == {"train", "search", "holdout"}


def test_tools_yaml_impls_resolve() -> None:
    spec = yaml.safe_load((DOMAIN / "tools.yaml").read_text())
    names = {tool["name"] for tool in spec["tools"]}
    assert "transfer_to_human_agents" in names and "think" not in names
    for tool in spec["tools"]:
        module_path, _, fn_name = tool["impl"].removeprefix("python:").rpartition(".")
        fn = getattr(importlib.import_module(module_path), fn_name)
        assert callable(fn)
        assert tool["name"] == fn_name
        assert set(tool["args"]["required"]) <= set(tool["args"]["properties"])


@pytest.mark.parametrize("name", ["get_user_details", "get_reservation_details"])
def test_read_tools_return_json(name: str) -> None:
    airline_tools.reset()
    fn = getattr(airline_tools, name)
    bad = fn("nope") if name != "get_user_details" else fn(user_id="nope")
    assert bad.startswith("Error")
