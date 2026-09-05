"""Tests for anneal.runtime: single + planner_executor topologies with a scripted fake client.

Everything runs offline via ``tests.fakes.FakeClient``. Only train-split airline tasks are
used (holdout is sacred).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anneal.domain import Domain, load_domain
from anneal.runtime import TaskResult, run_task, shallow_check
from anneal.spec import HarnessSpec
from tests.fakes import FakeClient

ROOT = Path(__file__).resolve().parents[1]
AIRLINE = ROOT / "domains" / "airline"
TASK_IDS = ("airline-43", "airline-01", "airline-28")

# --- scripted fake client (tests/fakes.py, shared with the llm-gateway session) -------------

Turn = str | list[dict[str, str]] | dict[str, Any]


def tool_turn(name: str, **kwargs: Any) -> list[dict[str, str]]:
    return [{"name": name, "arguments": json.dumps(kwargs)}]


def factory_for(client: FakeClient) -> Any:
    return lambda tier: client


# --- fixtures -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def domain() -> Domain:
    return load_domain(AIRLINE)


def _tool_names(domain: Domain) -> list[str]:
    return [t.name for t in domain.tools.tools]


def single_spec(domain: Domain, *, step_budget: int = 20, max_steps: int = 20) -> HarnessSpec:
    return HarnessSpec.model_validate(
        {
            "id": "cand-test-single",
            "topology": "single",
            "step_budget": step_budget,
            "nodes": [
                {
                    "name": "executor",
                    "role": "executor",
                    "model_tier": "mid",
                    "system_prompt_ref": "anneal/airline/executor@v1",
                    "tools": _tool_names(domain),
                    "max_steps": max_steps,
                }
            ],
        }
    )


def planner_spec(domain: Domain) -> HarnessSpec:
    return HarnessSpec.model_validate(
        {
            "id": "cand-test-pe",
            "topology": "planner_executor",
            "step_budget": 12,
            "nodes": [
                {
                    "name": "planner",
                    "role": "planner",
                    "model_tier": "frontier",
                    "system_prompt_ref": "anneal/airline/planner@v1",
                    "max_steps": 2,
                },
                {
                    "name": "executor",
                    "role": "executor",
                    "model_tier": "mid",
                    "system_prompt_ref": "anneal/airline/executor@v1",
                    "tools": _tool_names(domain),
                    "max_steps": 8,
                },
            ],
        }
    )


def _task(domain: Domain, task_id: str) -> Any:
    tasks = {t.id: t for t in domain.eval.load_tasks("train")}
    return tasks[task_id]


def _replay_script(task: Any) -> list[Turn]:
    turns: list[Turn] = [
        tool_turn(a["name"], **a["kwargs"]) for a in task.expected["actions"]
    ]
    turns.append("Done: applied the requested changes.")
    return turns


# --- acceptance: 3 airline train tasks end to end -----------------------------------------


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_single_topology_runs_airline_task_end_to_end(domain: Domain, task_id: str) -> None:
    task = _task(domain, task_id)
    client = FakeClient(_replay_script(task))
    result = run_task(single_spec(domain), task, domain, client_factory=factory_for(client))

    assert isinstance(result, TaskResult)
    assert domain.eval.score(task, result.output) == 1.0
    assert domain.eval.is_hard_fail(task, result.trace) is False
    assert [s["tool"] for s in result.trace] == [a["name"] for a in task.expected["actions"]]
    assert all("result" in s and "args" in s for s in result.trace)
    assert result.output == "Done: applied the requested changes."
    assert result.steps == len(task.expected["actions"]) + 1
    assert result.hit_step_budget is False
    assert result.schema_error is False
    assert result.latency_ms >= 0


def test_per_node_accounting_from_usage(domain: Domain) -> None:
    task = _task(domain, "airline-43")
    client = FakeClient(_replay_script(task))
    result = run_task(single_spec(domain), task, domain, client_factory=factory_for(client))
    acct = result.per_node["executor"]
    n_calls = len(task.expected["actions"]) + 1
    assert acct["tokens_in"] == 10 * n_calls
    assert acct["tokens_out"] == 5 * n_calls
    assert acct["backend"] == "fake-backend"
    assert acct["ms"] >= 0


def test_llm_sees_goal_task_and_tools(domain: Domain) -> None:
    task = _task(domain, "airline-43")
    client = FakeClient(["Nothing to do."])
    run_task(single_spec(domain), task, domain, client_factory=factory_for(client))
    kw = client.calls[0]
    assert kw["messages"][0]["role"] == "system"
    assert "airline" in kw["messages"][0]["content"].lower()
    assert task.input["user_id"] in kw["messages"][1]["content"]
    assert {t["function"]["name"] for t in kw["tools"]} == set(_tool_names(domain))


# --- step budget ------------------------------------------------------------------------


def test_step_budget_stops_infinite_tool_loop(domain: Domain) -> None:
    task = _task(domain, "airline-43")
    client = FakeClient([tool_turn("get_user_details", user_id=task.input["user_id"])] * 50)
    result = run_task(
        single_spec(domain, step_budget=3), task, domain, client_factory=factory_for(client)
    )
    assert result.hit_step_budget is True
    assert result.steps == 3
    assert len(result.trace) == 3
    assert result.output is None


def test_node_max_steps_also_counts_as_budget_hit(domain: Domain) -> None:
    task = _task(domain, "airline-43")
    client = FakeClient([tool_turn("list_all_airports")] * 50)
    spec = single_spec(domain, step_budget=50, max_steps=2)
    result = run_task(spec, task, domain, client_factory=factory_for(client))
    assert result.hit_step_budget is True
    assert result.steps == 2


# --- tool dispatch -----------------------------------------------------------------------


def test_invalid_tool_args_return_error_without_calling_impl(domain: Domain) -> None:
    task = _task(domain, "airline-43")
    client = FakeClient(
        [
            [{"name": "get_reservation_details", "arguments": "{}"}],
            [{"name": "get_reservation_details", "arguments": json.dumps({"reservation_id": 7})}],
            [{"name": "no_such_tool", "arguments": "{}"}],
            [{"name": "calculate", "arguments": "not json"}],
            "gave up",
        ]
    )
    result = run_task(single_spec(domain), task, domain, client_factory=factory_for(client))
    results = [s["result"] for s in result.trace]
    assert all(r.startswith("Error") for r in results), results
    assert "reservation_id" in results[0]
    assert "string" in results[1]
    assert "no_such_tool" in results[2]
    # the error result was fed back to the model as a tool message
    tool_msgs = [m for m in client.calls[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith("Error")


def test_mcp_tools_are_not_implemented() -> None:
    from anneal import runtime
    from anneal.spec import ToolSpec

    tool = ToolSpec(name="remote", description="x", impl="mcp:server/tool")
    with pytest.raises(NotImplementedError):
        runtime.invoke_tool(tool, {})


# --- planner_executor --------------------------------------------------------------------


def test_planner_executor_plans_executes_and_stops_on_done(domain: Domain) -> None:
    task = _task(domain, "airline-43")
    actions = task.expected["actions"]
    turns: list[Turn] = ["1. Look up reservation 3RK2T9\n2. Update the passenger name"]
    turns += [tool_turn(a["name"], **a["kwargs"]) for a in actions]
    turns += ["Changed Mei Lee to Mei Garcia on 3RK2T9.", "DONE"]
    client = FakeClient(turns)

    result = run_task(planner_spec(domain), task, domain, client_factory=factory_for(client))

    assert domain.eval.score(task, result.output) == 1.0
    assert result.output == "Changed Mei Lee to Mei Garcia on 3RK2T9."
    assert result.steps == len(actions) + 3
    assert set(result.per_node) == {"planner", "executor"}
    assert result.per_node["planner"]["tokens_in"] == 20
    # planner never sees tools; executor does
    assert "tools" not in client.calls[0] or not client.calls[0]["tools"]
    assert client.calls[1]["tools"]
    # the executor saw the plan
    exec_msgs = client.calls[1]["messages"]
    assert any("Look up reservation" in (m.get("content") or "") for m in exec_msgs)


def test_planner_executor_replans_once(domain: Domain) -> None:
    task = _task(domain, "airline-43")
    actions = task.expected["actions"]
    turns: list[Turn] = ["1. Look up reservation 3RK2T9"]
    turns += [tool_turn(actions[0]["name"], **actions[0]["kwargs"]), "Looked it up."]
    turns += ["1. Now update the passengers"]
    turns += [tool_turn(actions[1]["name"], **actions[1]["kwargs"]), "Updated."]
    client = FakeClient(turns)

    result = run_task(planner_spec(domain), task, domain, client_factory=factory_for(client))

    assert result.output == "Updated."
    assert domain.eval.score(task, result.output) == 1.0
    assert result.steps == 6
    assert result.hit_step_budget is False
    # replan happens at most once: the second planner reply is followed by no third planning call
    assert len(client.calls) == 6


# --- output parsing + schema -------------------------------------------------------------


def test_json_output_is_parsed(domain: Domain) -> None:
    task = _task(domain, "airline-43")
    client = FakeClient(['{"summary": "nothing changed", "changed": false}'])
    result = run_task(single_spec(domain), task, domain, client_factory=factory_for(client))
    assert result.output == {"summary": "nothing changed", "changed": False}


def test_schema_error_when_output_fails_shallow_check(
    domain: Domain, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}, "changed": {"type": "boolean"}},
        "required": ["summary", "changed"],
    }
    monkeypatch.setattr(domain.eval, "output_schema", schema, raising=False)
    spec = single_spec(domain)
    spec.nodes[0].schema_ref = "output_schema"
    task = _task(domain, "airline-43")

    bad = run_task(spec, task, domain, client_factory=factory_for(FakeClient(["plain text"])))
    assert bad.schema_error is True
    missing = run_task(
        spec, task, domain, client_factory=factory_for(FakeClient(['{"summary": "x"}']))
    )
    assert missing.schema_error is True
    good = run_task(
        spec,
        task,
        domain,
        client_factory=factory_for(FakeClient(['{"summary": "x", "changed": true}'])),
    )
    assert good.schema_error is False


def test_shallow_check() -> None:
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "integer"}, "b": {"type": "array"}},
    }
    assert shallow_check({"a": 1, "b": []}, schema) is None
    assert "a" in (shallow_check({}, schema) or "")
    assert "integer" in (shallow_check({"a": "1"}, schema) or "")
    assert "integer" in (shallow_check({"a": True}, schema) or "")  # bool is not an integer
    assert shallow_check("nope", schema) is not None
    assert shallow_check(3.0, {"type": "number"}) is None


def test_setup_is_called_before_each_run(domain: Domain, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real_setup = domain.eval.setup

    def spy(task: Any) -> None:
        calls.append(task.id)
        real_setup(task)

    monkeypatch.setattr(domain.eval, "setup", spy)
    task = _task(domain, "airline-01")
    run_task(single_spec(domain), task, domain, client_factory=factory_for(FakeClient(["ok"])))
    assert calls == ["airline-01"]
