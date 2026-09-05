"""Tests for anneal.mutate: typed operators driven by the failure taxonomy (offline).

Uses a scripted fake OpenAI-style client and a tmp_path prompt store, so nothing here
touches the network, the repo's ``prompts/`` directory, or any task split other than
``train`` and ``search``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from anneal import mutate
from anneal.mutate import OPERATORS, LocalPromptStore, apply, load_taxonomy, operators_for
from anneal.spec import HarnessSpec, ToolsManifest, load_spec
from tests.fakes import FakeClient

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = yaml.safe_load((ROOT / "specs" / "failure_taxonomy.yaml").read_text())

REWRITTEN_DESC = (
    "Cancel a reservation. Use when the user explicitly asks to cancel and the policy "
    "allows a refund. Do not use when the user only asks about cancellation options."
)
FEWSHOTS = (
    "### Example 1\nUser: cancel 8JX2WO\nCorrect: check policy first, then cancel.\n"
    "### Example 2\nUser: what if I cancel?\nCorrect: explain policy, do not cancel."
)


# --- fakes ---------------------------------------------------------------------------


class FakeEval:
    """Stands in for domains/<name>/eval.py; records which splits were requested."""

    def __init__(self) -> None:
        self.requested: list[str | None] = []

    def load_tasks(self, split: str | None = None) -> list[Any]:
        self.requested.append(split)
        assert split == "train", f"mutate must only read the train split, asked for {split!r}"
        return [
            SimpleNamespace(
                id="t-1",
                input={"instruction": "Cancel my reservation 8JX2WO"},
                expected={"actions": [{"name": "cancel_reservation", "kwargs": {}}]},
                split="train",
            ),
            SimpleNamespace(
                id="t-2",
                input={"instruction": "Can I cancel my flight?"},
                expected={"actions": []},
                split="train",
            ),
        ]


def make_domain() -> SimpleNamespace:
    tools = ToolsManifest.model_validate(
        {
            "tools": [
                {
                    "name": "cancel_reservation",
                    "description": "Cancel a reservation.",
                    "impl": "python:x.y",
                    "mutates": True,
                },
                {
                    "name": "get_reservation_details",
                    "description": "Get details of a reservation.",
                    "impl": "python:x.y",
                },
            ]
        }
    )
    return SimpleNamespace(name="airline", goal="Be a helpful airline agent.", tools=tools,
                           eval=FakeEval())


def make_spec() -> HarnessSpec:
    return HarnessSpec.model_validate(
        {
            "id": "cand-01",
            "topology": "single",
            "step_budget": 8,
            "nodes": [
                {
                    "name": "executor",
                    "role": "executor",
                    "model_tier": "mid",
                    "system_prompt_ref": "anneal/airline/executor@v1",
                    "tools": ["cancel_reservation", "get_reservation_details"],
                    "max_steps": 6,
                }
            ],
            "lineage": {"parent": "seed", "iteration": 2, "operator": "architect"},
        }
    )


def make_issue(cls: str, tried: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": "L-0003",
        "class": cls,
        "node": "executor",
        "count": 4,
        "evidence": ["trace-a", "trace-b"],
        "status": "open",
        "operators_tried": list(tried or []),
    }


EVIDENCE = [
    {
        "task_id": "s-7",
        "split": "search",
        "output": "I cancelled your reservation.",
        "trace": [
            {"tool": "get_reservation_details", "args": {"reservation_id": "X"}, "result": {}},
            {"tool": "cancel_reservation", "args": {"reservation_id": "X"}, "result": "ok"},
        ],
        "expected": {"actions": []},
    },
    {
        "task_id": "s-9",
        "split": "search",
        "output": "Done.",
        "trace": [{"tool": "cancel_reservation", "args": {"reservation_id": "Y"}, "result": "ok"}],
    },
]


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalPromptStore:
    local = LocalPromptStore(tmp_path / "prompts")
    monkeypatch.setattr(mutate, "PROMPT_STORE", local)
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    (tmp_path / "prompts" / "anneal" / "airline" / "executor").mkdir(parents=True)
    (tmp_path / "prompts" / "anneal" / "airline" / "executor" / "v1.md").write_text(
        "You are the airline executor.\n"
    )
    return local


# --- registry ----------------------------------------------------------------------------


def test_registry_matches_taxonomy_yaml() -> None:
    taxonomy = load_taxonomy()
    yaml_ops = set(TAXONOMY["operators"])
    assert set(taxonomy["operators"]) == yaml_ops
    assert set(OPERATORS) <= yaml_ops, "every implemented operator must be declared in the yaml"
    assert {"rewrite_tool_desc", "add_fewshots"} <= set(OPERATORS)
    for cls in TAXONOMY["classes"]:
        assert set(cls["operators"]) <= yaml_ops
        assert operators_for(cls["id"]) == cls["operators"]


def test_operators_for_unknown_class_raises() -> None:
    with pytest.raises(KeyError):
        operators_for("not_a_class")


# --- spec field ------------------------------------------------------------------------


def test_spec_tool_overrides_round_trip(tmp_path: Path) -> None:
    spec = make_spec()
    assert spec.tool_overrides == {}
    data = spec.model_dump()
    data["tool_overrides"] = {"cancel_reservation": "new text"}
    mutated = HarnessSpec.model_validate(data)
    path = tmp_path / "spec.yaml"
    path.write_text(mutated.to_yaml())
    assert load_spec(path).tool_overrides == {"cancel_reservation": "new text"}


# --- rewrite_tool_desc ------------------------------------------------------------------


def test_rewrite_tool_desc_sets_override_and_lineage(store: LocalPromptStore) -> None:
    client = FakeClient(turns=[REWRITTEN_DESC])
    spec = make_spec()
    out = apply(spec, make_issue("wrong_tool"), EVIDENCE, make_domain(), client=client)

    HarnessSpec.model_validate(out.model_dump())  # mutated spec validates
    assert out.id != spec.id
    assert out.tool_overrides == {"cancel_reservation": REWRITTEN_DESC}
    assert out.lineage is not None
    assert out.lineage.parent == "cand-01"
    assert out.lineage.iteration == 3
    assert out.lineage.operator == "rewrite_tool_desc"
    assert out.lineage.ledger_issue == "L-0003"
    assert spec.tool_overrides == {}  # input spec untouched

    prompt = json.dumps(client.calls[0]["messages"])
    assert "Cancel a reservation." in prompt  # current description shown to the LLM
    assert "s-7" in prompt and "I cancelled your reservation." in prompt  # failing evidence


def test_rewrite_tool_desc_honours_explicit_tool(store: LocalPromptStore) -> None:
    issue = {**make_issue("wrong_tool"), "tool": "get_reservation_details"}
    out = apply(make_spec(), issue, EVIDENCE, make_domain(), client=FakeClient(turns=["new desc"]))
    assert out.tool_overrides == {"get_reservation_details": "new desc"}


# --- add_fewshots ----------------------------------------------------------------------


def test_add_fewshots_writes_prompt_v2_from_train_only(store: LocalPromptStore) -> None:
    client = FakeClient(turns=[FEWSHOTS])
    domain = make_domain()
    spec = make_spec()
    out = apply(spec, make_issue("policy_misread"), EVIDENCE, domain, client=client)

    HarnessSpec.model_validate(out.model_dump())
    assert out.nodes[0].system_prompt_ref == "anneal/airline/executor@v2"
    v2 = store.root / "anneal" / "airline" / "executor" / "v2.md"
    assert v2.exists()
    text = v2.read_text()
    assert text.startswith("You are the airline executor.")
    assert FEWSHOTS in text
    assert store.label_of("anneal/airline/executor", 2) == "staging"
    assert domain.eval.requested == ["train"]
    assert out.lineage is not None and out.lineage.operator == "add_fewshots"
    assert out.lineage.ledger_issue == "L-0003"
    assert spec.nodes[0].system_prompt_ref == "anneal/airline/executor@v1"

    prompt = json.dumps(client.calls[0]["messages"])
    assert "Cancel my reservation 8JX2WO" in prompt  # train tasks shown
    assert "s-7" in prompt  # failing evidence shown


def test_add_fewshots_without_v1_file_still_bumps_to_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = LocalPromptStore(tmp_path / "empty")
    monkeypatch.setattr(mutate, "PROMPT_STORE", local)
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    out = apply(
        make_spec(), make_issue("output_format", tried=["add_validator_node"]), EVIDENCE,
        make_domain(), client=FakeClient(turns=[FEWSHOTS]),
    )
    assert out.nodes[0].system_prompt_ref == "anneal/airline/executor@v2"
    assert (tmp_path / "empty" / "anneal" / "airline" / "executor" / "v2.md").exists()


# --- operator selection ---------------------------------------------------------------


def test_apply_skips_tried_operators(store: LocalPromptStore) -> None:
    issue = make_issue("wrong_tool", tried=["rewrite_tool_desc"])
    out = apply(make_spec(), issue, EVIDENCE, make_domain(), client=FakeClient(turns=[FEWSHOTS]))
    assert out.lineage is not None and out.lineage.operator == "add_fewshots"


def test_apply_raises_when_no_operator_left(store: LocalPromptStore) -> None:
    # wrong_tool also lists switch_topology (implemented in task 2.2), so spend that too.
    issue = make_issue("wrong_tool", tried=["rewrite_tool_desc", "add_fewshots",
                                            "switch_topology"])
    with pytest.raises(mutate.NoOperatorAvailable):
        apply(make_spec(), issue, EVIDENCE, make_domain(), client=FakeClient(turns=[]))


def test_apply_rejects_evidence_from_other_splits(store: LocalPromptStore) -> None:
    bad = [{**EVIDENCE[0], "split": "hold" + "out"}]
    client = FakeClient(turns=[])
    with pytest.raises(ValueError, match="search"):
        apply(make_spec(), make_issue("wrong_tool"), bad, make_domain(), client=client)


def test_apply_unknown_node_raises(store: LocalPromptStore) -> None:
    issue = {**make_issue("wrong_tool"), "node": "ghost"}
    with pytest.raises(KeyError, match="ghost"):
        apply(make_spec(), issue, EVIDENCE, make_domain(), client=FakeClient(turns=["x"]))
