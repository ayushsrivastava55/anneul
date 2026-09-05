"""Tests for the critic_loop and tool_router topologies plus the validator/escalate roles.

Everything runs offline through ``tests.fakes.FakeClient``. Only train-split airline tasks
are used (holdout is sacred).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anneal.domain import Domain, load_domain
from anneal.runtime import ROUTE_TRACE_TOOL, SINGLE_GROUP, run_task, tool_groups
from anneal.spec import HarnessSpec, ToolSpec
from tests.fakes import FakeClient

ROOT = Path(__file__).resolve().parents[1]
AIRLINE = ROOT / "domains" / "airline"

Turn = str | list[dict[str, str]] | dict[str, Any]


def tool_turn(name: str, **kwargs: Any) -> list[dict[str, str]]:
    return [{"name": name, "arguments": json.dumps(kwargs)}]


def factory_for(client: FakeClient) -> Any:
    return lambda tier: client


def texts(call: dict[str, Any]) -> str:
    """All message text of one recorded request (the runtime appends to the same list)."""
    return "\n".join(str(m.get("content") or "") for m in call["messages"])


@pytest.fixture(scope="module")
def domain() -> Domain:
    return load_domain(AIRLINE)


def _tool_names(domain: Domain) -> list[str]:
    return [t.name for t in domain.tools.tools]


def _task(domain: Domain, task_id: str) -> Any:
    tasks = {t.id: t for t in domain.eval.load_tasks("train")}
    return tasks[task_id]


def _replay(task: Any, final: str) -> list[Turn]:
    turns: list[Turn] = [tool_turn(a["name"], **a["kwargs"]) for a in task.expected["actions"]]
    turns.append(final)
    return turns


def _node(name: str, role: str, **kw: Any) -> dict[str, Any]:
    node = {
        "name": name,
        "role": role,
        "model_tier": "cheap",
        "system_prompt_ref": f"anneal/airline/{name}@v1",
        "tools": [],
        "max_steps": 1,
    }
    node.update(kw)
    return node


def critic_spec(
    domain: Domain, *, step_budget: int = 20, extra: list[dict[str, Any]] | None = None
) -> HarnessSpec:
    nodes = [
        _node("executor", "executor", model_tier="mid", tools=_tool_names(domain), max_steps=8),
        _node("critic", "critic", max_steps=4),
    ]
    return HarnessSpec.model_validate(
        {
            "id": "cand-test-critic",
            "topology": "critic_loop",
            "step_budget": step_budget,
            "nodes": nodes + (extra or []),
        }
    )


def router_spec(
    domain: Domain, *, step_budget: int = 20, extra: list[dict[str, Any]] | None = None
) -> HarnessSpec:
    nodes = [
        _node("router", "router"),
        _node("executor", "executor", model_tier="mid", tools=_tool_names(domain), max_steps=8),
    ]
    return HarnessSpec.model_validate(
        {
            "id": "cand-test-router",
            "topology": "tool_router",
            "step_budget": step_budget,
            "nodes": nodes + (extra or []),
        }
    )


def single_spec(
    domain: Domain, *, step_budget: int = 20, extra: list[dict[str, Any]]
) -> HarnessSpec:
    nodes = [
        _node("executor", "executor", model_tier="mid", tools=_tool_names(domain), max_steps=8)
    ]
    return HarnessSpec.model_validate(
        {
            "id": "cand-test-single",
            "topology": "single",
            "step_budget": step_budget,
            "nodes": nodes + extra,
        }
    )


# --- critic_loop ---------------------------------------------------------------------------


def test_critic_loop_retries_once_then_passes(domain: Domain) -> None:
    task = _task(domain, "airline-43")
    client = FakeClient(
        [
            "First answer.",
            "FAIL: you never updated the passenger name.",
            "Second answer.",
            "PASS: the passenger was updated.",
        ]
    )
    result = run_task(critic_spec(domain), task, domain, client_factory=factory_for(client))

    assert result.output == "Second answer."
    assert result.steps == 4
    assert result.hit_step_budget is False
    assert set(result.per_node) == {"executor", "critic"}
    assert result.per_node["critic"]["tokens_in"] == 20  # two critic turns
    # the critic saw the first answer; the retrying executor saw the critique
    assert "First answer." in texts(client.calls[1])
    assert "never updated" in texts(client.calls[2])
    # the critic node never gets tools
    assert not client.calls[1].get("tools")


def test_critic_loop_gives_up_after_two_retries(domain: Domain) -> None:
    task = _task(domain, "airline-43")
    client = FakeClient(
        ["a1", "FAIL: nope.", "a2", "FAIL: still nope.", "a3", "FAIL: never.", "a4"]
    )
    result = run_task(critic_spec(domain), task, domain, client_factory=factory_for(client))

    assert result.output == "a3"  # two retries, then the last answer stands
    assert result.steps == 5
    assert len(client.calls) == 5  # no third critique
    assert result.hit_step_budget is False


def test_critic_loop_keeps_answer_when_budget_ends_before_verdict(domain: Domain) -> None:
    task = _task(domain, "airline-43")
    client = FakeClient(["Only answer.", "PASS"])
    result = run_task(
        critic_spec(domain, step_budget=1), task, domain, client_factory=factory_for(client)
    )
    assert result.output == "Only answer."
    assert result.hit_step_budget is True


def test_critic_loop_scores_one_on_airline_task(domain: Domain) -> None:
    task = _task(domain, "airline-43")
    client = FakeClient([*_replay(task, "Renamed Mei Lee to Mei Garcia."), "PASS: correct."])
    result = run_task(critic_spec(domain), task, domain, client_factory=factory_for(client))

    assert domain.eval.score(task, result.output) == 1.0
    assert domain.eval.is_hard_fail(task, result.trace) is False
    assert [s["tool"] for s in result.trace] == [a["name"] for a in task.expected["actions"]]
    assert result.steps == len(task.expected["actions"]) + 2
    assert result.schema_error is False
    # the critic reviews the executor's tool calls, not just its final text
    assert "update_reservation_passengers" in texts(client.calls[-1])


# --- tool groups ---------------------------------------------------------------------------


def test_tool_groups_cluster_by_name_prefix(domain: Domain) -> None:
    groups = tool_groups(domain.tools.tools)
    assert groups["update"] == [
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    ]
    assert groups["get"] == ["get_reservation_details", "get_user_details"]
    assert groups["cancel"] == ["cancel_reservation"]
    # every tool lands in exactly one group
    assert sorted(n for names in groups.values() for n in names) == sorted(_tool_names(domain))


def test_tool_groups_restricted_to_the_executors_tools(domain: Domain) -> None:
    allowed = ["cancel_reservation", "get_user_details"]
    assert tool_groups(domain.tools.tools, allowed) == {
        "get": ["get_user_details"],
        "cancel": ["cancel_reservation"],
    }


def test_tool_groups_fall_back_to_one_group() -> None:
    tools = [
        ToolSpec(name="get_a", description="a", impl="python:x.a"),
        ToolSpec(name="get_b", description="b", impl="python:x.b"),
    ]
    assert tool_groups(tools) == {SINGLE_GROUP: ["get_a", "get_b"]}
    assert tool_groups(tools, []) == {}


def test_tool_groups_prefer_an_explicit_group_field() -> None:
    tools = [
        SimpleNamespace(name="alpha", group="reads"),
        SimpleNamespace(name="beta", group="writes"),
    ]
    assert tool_groups(tools) == {"reads": ["alpha"], "writes": ["beta"]}  # type: ignore[arg-type]


# --- tool_router ---------------------------------------------------------------------------


def test_tool_router_restricts_the_executor_tool_list(domain: Domain) -> None:
    task = _task(domain, "airline-05")
    client = FakeClient(["update", "Nothing else to do."])
    result = run_task(router_spec(domain), task, domain, client_factory=factory_for(client))

    exposed = {t["function"]["name"] for t in client.calls[1]["tools"]}
    assert exposed == {
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    }
    assert exposed < set(_tool_names(domain))
    assert not client.calls[0].get("tools")  # the router itself never gets tools
    # the choice is recorded in the trace
    route = result.trace[0]
    assert route["tool"] == ROUTE_TRACE_TOOL
    assert route["result"] == "update"
    assert "update" in route["args"]["groups"]


def test_tool_router_falls_back_to_the_first_group(domain: Domain) -> None:
    task = _task(domain, "airline-05")
    client = FakeClient(["I am not sure.", "Nothing to do."])
    result = run_task(router_spec(domain), task, domain, client_factory=factory_for(client))
    first = next(iter(tool_groups(domain.tools.tools, _tool_names(domain))))
    assert result.trace[0]["result"] == first


def test_tool_router_scores_one_on_airline_task(domain: Domain) -> None:
    task = _task(domain, "airline-05")
    client = FakeClient(["update", *_replay(task, "Applied all three updates.")])
    result = run_task(router_spec(domain), task, domain, client_factory=factory_for(client))

    assert domain.eval.score(task, result.output) == 1.0
    assert domain.eval.is_hard_fail(task, result.trace) is False
    assert [s["tool"] for s in result.trace[1:]] == [a["name"] for a in task.expected["actions"]]
    assert result.steps == len(task.expected["actions"]) + 2
    assert result.hit_step_budget is False
    assert set(result.per_node) == {"router", "executor"}


def test_tool_router_honours_the_step_budget(domain: Domain) -> None:
    task = _task(domain, "airline-05")
    turns: list[Turn] = ["update"]
    turns += [tool_turn("update_reservation_passengers", reservation_id="x", passengers=[])] * 50
    spec = router_spec(domain, step_budget=3)
    result = run_task(spec, task, domain, client_factory=factory_for(FakeClient(turns)))
    assert result.hit_step_budget is True
    assert result.steps == 3
    assert result.output is None


# --- validator + escalate --------------------------------------------------------------------


@pytest.fixture
def output_schema(domain: Domain, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }
    monkeypatch.setattr(domain.eval, "output_schema", schema, raising=False)
    return schema


def test_validator_forces_one_retry(domain: Domain, output_schema: dict[str, Any]) -> None:
    task = _task(domain, "airline-43")
    spec = single_spec(
        domain, extra=[_node("validator", "validator", schema_ref="output_schema")]
    )
    client = FakeClient(["plain text", '{"summary": "renamed the passenger"}'])
    result = run_task(spec, task, domain, client_factory=factory_for(client))

    assert result.output == {"summary": "renamed the passenger"}
    assert result.schema_error is False
    assert result.steps == 2
    assert "schema" in texts(client.calls[1])


def test_validator_retries_at_most_once(domain: Domain, output_schema: dict[str, Any]) -> None:
    task = _task(domain, "airline-43")
    spec = single_spec(
        domain, extra=[_node("validator", "validator", schema_ref="output_schema")]
    )
    client = FakeClient(["still text", "text again", "and again"])
    result = run_task(spec, task, domain, client_factory=factory_for(client))

    assert result.output == "text again"
    assert result.schema_error is True
    assert len(client.calls) == 2


def test_escalate_node_returns_a_reason_when_the_budget_runs_out(domain: Domain) -> None:
    task = _task(domain, "airline-43")
    spec = single_spec(domain, step_budget=2, extra=[_node("escalate", "escalate")])
    client = FakeClient([tool_turn("list_all_airports")] * 5)
    result = run_task(spec, task, domain, client_factory=factory_for(client))

    assert result.hit_step_budget is True
    assert isinstance(result.output, dict)
    assert "budget" in result.output["escalate"]


def test_escalate_node_fires_when_the_critic_never_passes(domain: Domain) -> None:
    task = _task(domain, "airline-43")
    spec = critic_spec(domain, extra=[_node("escalate", "escalate")])
    client = FakeClient(["a1", "FAIL: nope.", "a2", "FAIL: still nope.", "a3"])
    result = run_task(spec, task, domain, client_factory=factory_for(client))

    assert result.output == {"escalate": "critic still failing after 2 retries"}
    assert result.steps == 5  # the escalate node costs no LLM step
