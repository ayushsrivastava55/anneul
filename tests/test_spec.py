"""Tests for anneal.spec: harness spec and tools manifest models."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from anneal.spec import HarnessSpec, ToolsManifest, dump_spec, load_spec, load_tools

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "specs" / "harness.example.yaml"
INVOICES_TOOLS = ROOT / "domains" / "invoices" / "tools.yaml"


def example_dict() -> dict[str, Any]:
    return yaml.safe_load(EXAMPLE.read_text())


def test_parse_example_spec() -> None:
    spec = load_spec(EXAMPLE)
    assert spec.id == "cand-02"
    assert spec.topology == "planner_executor"
    assert spec.step_budget == 12
    assert spec.memory.enabled is False
    assert spec.memory.kind == "none"
    assert [n.name for n in spec.nodes] == ["planner", "executor", "validator"]
    expected_tools = ["lookup_po", "lookup_receipt", "post_entry", "escalate"]
    assert spec.nodes[1].tools == expected_tools
    assert spec.nodes[2].schema_ref == "output_schema"
    assert spec.nodes[0].schema_ref is None
    assert spec.lineage is not None
    assert spec.lineage.operator == "add_validator_node"
    assert spec.lineage.ledger_issue == "L-0007"


def test_reject_bad_topology() -> None:
    data = example_dict()
    data["topology"] = "foo"
    with pytest.raises(ValidationError):
        HarnessSpec.model_validate(data)


def test_reject_duplicate_node_names() -> None:
    data = example_dict()
    data["nodes"][1]["name"] = "planner"
    with pytest.raises(ValidationError, match="unique"):
        HarnessSpec.model_validate(data)


def test_reject_empty_nodes() -> None:
    data = example_dict()
    data["nodes"] = []
    with pytest.raises(ValidationError):
        HarnessSpec.model_validate(data)


def test_reject_bad_role_and_tier() -> None:
    data = example_dict()
    data["nodes"][0]["role"] = "wizard"
    with pytest.raises(ValidationError):
        HarnessSpec.model_validate(data)
    data = example_dict()
    data["nodes"][0]["model_tier"] = "huge"
    with pytest.raises(ValidationError):
        HarnessSpec.model_validate(data)


def test_single_requires_exactly_one_executor() -> None:
    data = example_dict()
    data["topology"] = "single"
    data["lineage"] = None
    # planner + executor + validator: one executor -> valid despite validator
    spec = HarnessSpec.model_validate(data)
    assert spec.topology == "single"
    data["nodes"][0]["role"] = "executor"  # now two executors
    with pytest.raises(ValidationError, match="exactly one executor"):
        HarnessSpec.model_validate(data)
    data["nodes"] = [data["nodes"][2]]  # validator only, zero executors
    with pytest.raises(ValidationError, match="exactly one executor"):
        HarnessSpec.model_validate(data)


def test_planner_executor_requires_planner_and_executor() -> None:
    data = example_dict()
    data["nodes"][0]["role"] = "critic"
    with pytest.raises(ValidationError, match="planner"):
        HarnessSpec.model_validate(data)
    data = example_dict()
    data["nodes"][1]["role"] = "router"
    with pytest.raises(ValidationError, match="executor"):
        HarnessSpec.model_validate(data)


def test_spec_round_trip(tmp_path: Path) -> None:
    spec = load_spec(EXAMPLE)
    out = tmp_path / "spec.yaml"
    dump_spec(spec, out)
    assert load_spec(out) == spec
    assert yaml.safe_load(spec.to_yaml()) == yaml.safe_load(out.read_text())


def test_parse_invoices_tools() -> None:
    manifest = load_tools(INVOICES_TOOLS)
    assert isinstance(manifest, ToolsManifest)
    expected = ["lookup_po", "lookup_receipt", "post_entry", "escalate"]
    assert [t.name for t in manifest.tools] == expected
    first = manifest.tools[0]
    assert first.impl.startswith("python:")
    assert first.args["required"] == ["po_number"]
    assert "purchase order" in first.description


def test_reject_bad_tool_impl() -> None:
    data = yaml.safe_load(INVOICES_TOOLS.read_text())
    data["tools"][0]["impl"] = "shell:foo"
    with pytest.raises(ValidationError, match="python:"):
        ToolsManifest.model_validate(data)
