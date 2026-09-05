"""Unit tests for the structural mutation operators in anneal.mutate (offline).

One test per operator: it makes the structural change the taxonomy promises, the mutated
spec validates, lineage is set and the input spec is untouched. Prompts are written to a
tmp_path store, so nothing here touches the repo's ``prompts/`` directory or the network.
The deterministic operators must not spend anything at the gateway: their tests assert the
scripted client was never called.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from anneal import mutate
from anneal.mutate import DEFERRED, OPERATORS, LocalPromptStore, apply
from anneal.spec import HarnessSpec, ToolsManifest
from tests.fakes import FakeClient
from tests.test_mutate import EVIDENCE, make_domain, make_issue, make_spec

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = yaml.safe_load((ROOT / "specs" / "failure_taxonomy.yaml").read_text())

BASE_PROMPT = "You are the airline executor.\n"


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalPromptStore:
    """A prompt store rooted in tmp_path, holding the executor's v1 prompt."""
    local = LocalPromptStore(tmp_path / "prompts")
    monkeypatch.setattr(mutate, "PROMPT_STORE", local)
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    folder = tmp_path / "prompts" / "anneal" / "airline" / "executor"
    folder.mkdir(parents=True)
    (folder / "v1.md").write_text(BASE_PROMPT)
    return local


def domain_with_escalate() -> SimpleNamespace:
    """make_domain() plus an ``escalate`` hand-off tool in the manifest."""
    domain = make_domain()
    domain.tools = ToolsManifest.model_validate(
        {
            "tools": [
                *[tool.model_dump() for tool in domain.tools.tools],
                {
                    "name": "escalate",
                    "description": "Hand the case to a human agent.",
                    "impl": "python:x.y",
                    "mutates": True,
                },
            ]
        }
    )
    return domain


def node_named(spec: HarnessSpec, name: str) -> Any:
    return next(node for node in spec.nodes if node.name == name)


# --- registry vs taxonomy ----------------------------------------------------------------


def test_every_taxonomy_operator_is_registered_or_deferred() -> None:
    declared = set(TAXONOMY["operators"])
    assert set(OPERATORS) | set(DEFERRED) == declared
    assert not set(OPERATORS) & set(DEFERRED), "an operator is either implemented or deferred"
    assert DEFERRED == {}, f"nothing is deferred any more, found {sorted(DEFERRED)}"
    for cls in TAXONOMY["classes"]:
        for name in cls["operators"]:
            assert name in OPERATORS or name in DEFERRED, f"{cls['id']}: {name} is unaccounted for"


def test_missing_capability_selects_synthesize_tool(store: LocalPromptStore) -> None:
    # The operator itself (AO worker, generated tool, adoption) is covered offline in
    # tests/test_synth_tool.py; here we only pin the taxonomy -> operator routing.
    assert mutate.select_operator(make_issue("missing_capability")) == "synthesize_tool"
    with pytest.raises(mutate.NoOperatorAvailable):
        mutate.select_operator(make_issue("missing_capability", ["synthesize_tool"]))


# --- add_validator_node -------------------------------------------------------------------


def test_add_validator_node_appends_cheap_schema_checked_node(store: LocalPromptStore) -> None:
    client = FakeClient(turns=[])
    spec = make_spec()
    out = apply(spec, make_issue("output_format"), EVIDENCE, make_domain(), client=client)

    HarnessSpec.model_validate(out.model_dump())
    validator = node_named(out, "validator")
    assert validator.role == "validator"
    assert validator.model_tier == "cheap"
    assert validator.schema_ref == "output_schema"
    assert validator.tools == []
    assert validator.system_prompt_ref == "anneal/airline/validator@v1"
    assert out.step_budget == spec.step_budget + mutate.VALIDATOR_STEPS
    assert (store.root / "anneal" / "airline" / "validator" / "v1.md").exists()
    assert store.label_of("anneal/airline/validator", 1) == "staging"
    assert out.lineage is not None and out.lineage.operator == "add_validator_node"
    assert out.lineage.parent == "cand-01" and out.lineage.ledger_issue == "L-0003"
    assert [n.name for n in spec.nodes] == ["executor"]  # input spec untouched
    assert client.calls == []  # deterministic: no gateway spend


def test_add_validator_node_refuses_a_second_validator(store: LocalPromptStore) -> None:
    issue = make_issue("output_format")
    out = apply(make_spec(), issue, EVIDENCE, make_domain(), client=FakeClient(turns=[]))
    with pytest.raises(mutate.NoOperatorAvailable, match="validator"):
        mutate.add_validator_node(out, issue, EVIDENCE, make_domain(), None)


# --- add_cite_or_abstain -------------------------------------------------------------------


def test_add_cite_or_abstain_bumps_the_prompt(store: LocalPromptStore) -> None:
    client = FakeClient(turns=[])
    spec = make_spec()
    out = apply(spec, make_issue("unsupported_claim"), EVIDENCE, make_domain(), client=client)

    HarnessSpec.model_validate(out.model_dump())
    assert out.nodes[0].system_prompt_ref == "anneal/airline/executor@v2"
    text = (store.root / "anneal" / "airline" / "executor" / "v2.md").read_text()
    assert text.startswith(BASE_PROMPT.rstrip())
    assert "Cite or abstain" in text and "abstain" in text
    assert store.label_of("anneal/airline/executor", 2) == "staging"
    assert out.lineage is not None and out.lineage.operator == "add_cite_or_abstain"
    assert spec.nodes[0].system_prompt_ref == "anneal/airline/executor@v1"
    assert client.calls == []


def test_add_cite_or_abstain_is_not_applied_twice(store: LocalPromptStore) -> None:
    issue = make_issue("unsupported_claim")
    out = apply(make_spec(), issue, EVIDENCE, make_domain(), client=FakeClient(turns=[]))
    with pytest.raises(mutate.NoOperatorAvailable, match="already carries"):
        mutate.add_cite_or_abstain(out, issue, EVIDENCE, make_domain(), None)


# --- add_step_budget_and_critic ------------------------------------------------------------


def test_add_step_budget_and_critic_raises_budget_and_switches_topology(
    store: LocalPromptStore,
) -> None:
    client = FakeClient(turns=[])
    spec = make_spec()
    out = apply(spec, make_issue("loop_or_timeout"), EVIDENCE, make_domain(), client=client)

    HarnessSpec.model_validate(out.model_dump())
    critic = node_named(out, "critic")
    assert critic.role == "critic" and critic.tools == []
    assert critic.max_steps == mutate.CRITIC_STEPS
    assert critic.system_prompt_ref == "anneal/airline/critic@v1"
    assert out.topology == "critic_loop"
    assert out.step_budget == spec.step_budget + mutate.CRITIC_STEPS + mutate.STEP_BUDGET_BUMP
    assert out.lineage is not None and out.lineage.operator == "add_step_budget_and_critic"
    assert spec.topology == "single" and spec.step_budget == 8
    assert client.calls == []


# --- add_memory -----------------------------------------------------------------------------


def test_add_memory_enables_episodic_then_widens_top_k(store: LocalPromptStore) -> None:
    client = FakeClient(turns=[])
    spec = make_spec()
    issue = make_issue("context_overflow")
    out = apply(spec, issue, EVIDENCE, make_domain(), client=client)

    HarnessSpec.model_validate(out.model_dump())
    assert out.memory.enabled is True
    assert out.memory.kind == "episodic"
    assert out.memory.top_k == mutate.MEMORY_TOP_K
    assert out.lineage is not None and out.lineage.operator == "add_memory"
    assert spec.memory.enabled is False and spec.memory.kind == "none"

    again = apply(out, issue, EVIDENCE, make_domain(), client=client)
    assert again.memory.top_k == mutate.MEMORY_TOP_K + mutate.MEMORY_TOP_K_STEP
    assert client.calls == []


# --- add_escalation_node ---------------------------------------------------------------------


def test_add_escalation_node_appends_node_and_instructs_the_executor(
    store: LocalPromptStore,
) -> None:
    client = FakeClient(turns=[])
    spec = make_spec()
    out = apply(spec, make_issue("unsafe_action"), EVIDENCE, domain_with_escalate(), client=client)

    HarnessSpec.model_validate(out.model_dump())
    escalate = node_named(out, "escalate")
    assert escalate.role == "escalate"
    assert escalate.model_tier == "cheap"
    assert escalate.tools == ["escalate"]  # wired to the manifest's hand-off tool
    assert escalate.system_prompt_ref == "anneal/airline/escalate@v1"
    assert out.step_budget == spec.step_budget + mutate.ESCALATE_STEPS
    executor = node_named(out, "executor")
    assert executor.system_prompt_ref == "anneal/airline/executor@v2"
    text = (store.root / "anneal" / "airline" / "executor" / "v2.md").read_text()
    assert text.startswith(BASE_PROMPT.rstrip())
    assert "Escalate instead of acting" in text and "`escalate`" in text
    assert out.lineage is not None and out.lineage.operator == "add_escalation_node"
    assert [n.name for n in spec.nodes] == ["executor"]
    assert client.calls == []


def test_add_escalation_node_without_a_hand_off_tool(store: LocalPromptStore) -> None:
    out = apply(make_spec(), make_issue("unsafe_action"), EVIDENCE, make_domain(),
                client=FakeClient(turns=[]))
    assert node_named(out, "escalate").tools == []  # no such tool in this manifest
    text = (store.root / "anneal" / "airline" / "executor" / "v2.md").read_text()
    assert "escalate node" in text


# --- switch_topology --------------------------------------------------------------------------


def switch_issue() -> dict[str, Any]:
    """A wrong_tool issue whose earlier operators are spent, so switch_topology is next."""
    return make_issue("wrong_tool", tried=["rewrite_tool_desc", "add_fewshots"])


def test_switch_topology_walks_the_ladder_and_stops_at_the_top(store: LocalPromptStore) -> None:
    client = FakeClient(turns=[])
    spec = make_spec()
    issue = switch_issue()

    planner_exec = apply(spec, issue, EVIDENCE, make_domain(), client=client)
    HarnessSpec.model_validate(planner_exec.model_dump())
    assert planner_exec.topology == "planner_executor"
    planner = node_named(planner_exec, "planner")
    assert planner.role == "planner" and planner.max_steps == mutate.PLANNER_STEPS
    assert planner.system_prompt_ref == "anneal/airline/planner@v1"
    carried = (store.root / "anneal" / "airline" / "planner" / "v1.md").read_text()
    assert BASE_PROMPT.strip() in carried  # executor prompt carried across
    assert planner_exec.step_budget == spec.step_budget + mutate.PLANNER_STEPS
    assert planner_exec.lineage is not None
    assert planner_exec.lineage.operator == "switch_topology"

    critic_loop = apply(planner_exec, issue, EVIDENCE, make_domain(), client=client)
    HarnessSpec.model_validate(critic_loop.model_dump())
    assert critic_loop.topology == "critic_loop"
    assert node_named(critic_loop, "critic").role == "critic"
    assert [n.name for n in critic_loop.nodes] == ["executor", "planner", "critic"]

    with pytest.raises(mutate.NoOperatorAvailable, match="top of the ladder"):
        mutate.switch_topology(critic_loop, issue, EVIDENCE, make_domain(), None)

    assert spec.topology == "single"
    assert client.calls == []


def test_switch_topology_rejects_a_topology_off_the_ladder(store: LocalPromptStore) -> None:
    data = make_spec().model_dump()
    data["topology"] = "tool_router"
    data["nodes"][0].update(name="router", role="router")
    router_spec = HarnessSpec.model_validate(data)
    with pytest.raises(mutate.NoOperatorAvailable, match="not on the ladder"):
        mutate.switch_topology(router_spec, switch_issue(), EVIDENCE, make_domain(), None)


def test_switch_topology_reuses_an_existing_critic(store: LocalPromptStore) -> None:
    """planner_executor that already has a critic only moves the topology, no new node."""
    data = make_spec().model_dump()
    data["topology"] = "planner_executor"
    data["nodes"] += [
        {"name": "planner", "role": "planner", "model_tier": "mid",
         "system_prompt_ref": "anneal/airline/planner@v1", "tools": [], "max_steps": 2},
        {"name": "critic", "role": "critic", "model_tier": "mid",
         "system_prompt_ref": "anneal/airline/critic@v1", "tools": [], "max_steps": 4},
    ]
    spec = HarnessSpec.model_validate(data)
    out = mutate.switch_topology(spec, switch_issue(), EVIDENCE, make_domain(), None)
    assert out.topology == "critic_loop"
    assert len(out.nodes) == len(spec.nodes)
    assert out.step_budget == spec.step_budget
