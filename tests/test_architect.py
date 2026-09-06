"""Tests for anneal.architect with an offline fake client (task 1.3 acceptance)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anneal import architect
from anneal.spec import HarnessSpec, ToolsManifest, load_tools

ROOT = Path(__file__).resolve().parent.parent
AIRLINE = ROOT / "domains" / "airline"


# A reply that passes validate_elaboration: long enough, prose, names a real tool.
GOOD = (
    "Start by calling get_user_details to load the traveller before anything else, then work "
    "through the reservation tools one at a time. Never guess a confirmation number: look it "
    "up. Only after every read has come back should you take an action that changes state, and "
    "that action must be your last. Finish with the exact output the goal asks for, on its own, "
    "with no commentary around it."
)
BARE_JSON = '{"vendor": "str", "invoice_number": "str", "total": "number"}'

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["vendor", "decision"],
    "properties": {"vendor": {"type": "string"}, "decision": {"type": "string"}},
}


@dataclass
class FakeDomain:
    name: str
    goal: str
    tools: ToolsManifest
    eval: Any = None


class FakeCompletions:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        text = self.replies[(len(self.calls) - 1) % len(self.replies)]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeClient:
    """OpenAI-shaped fake: ``client.chat.completions.create(...)``."""

    def __init__(self, replies: list[str]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(replies))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chat.completions.calls


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)


@pytest.fixture
def domain() -> FakeDomain:
    return FakeDomain(
        name="airline",
        goal=(AIRLINE / "goal.md").read_text(),
        tools=load_tools(AIRLINE / "tools.yaml"),
    )


def test_propose_three_candidates(domain: FakeDomain, tmp_path: Path) -> None:
    client = FakeClient([GOOD])
    specs = architect.propose(domain, 3, client=client, prompts_root=tmp_path)

    assert len(specs) == 3
    assert all(isinstance(s, HarnessSpec) for s in specs)
    assert [s.id for s in specs] == ["cand-01", "cand-02", "cand-03"]
    assert len({s.topology for s in specs}) >= 2
    assert {"single", "planner_executor"} <= {s.topology for s in specs}
    # every spec re-validates from its own YAML dump
    for s in specs:
        HarnessSpec.model_validate(s.model_dump(mode="json"))
        assert s.step_budget == 12
        assert s.memory.enabled is False
        assert s.lineage is not None and s.lineage.iteration == 0
        assert sum(n.max_steps for n in s.nodes) <= s.step_budget


def test_node_tiers_and_tools(domain: FakeDomain, tmp_path: Path) -> None:
    specs = architect.propose(domain, 3, client=FakeClient([GOOD]), prompts_root=tmp_path)
    all_tools = [t.name for t in domain.tools.tools]
    for s in specs:
        for node in s.nodes:
            expected_tier = {"planner": "frontier", "executor": "mid", "critic": "cheap"}
            assert node.model_tier == expected_tier[node.role]
            assert node.tools == (all_tools if node.role == "executor" else [])


def test_prompts_written_once_per_node(domain: FakeDomain, tmp_path: Path) -> None:
    client = FakeClient([GOOD])
    specs = architect.propose(domain, 3, client=client, prompts_root=tmp_path)

    assert len(client.calls) == 3  # planner, executor, critic: one accepted call each
    refs = {n.system_prompt_ref for s in specs for n in s.nodes}
    assert refs == {
        "anneal/airline/executor@v1",
        "anneal/airline/planner@v1",
        "anneal/airline/critic@v1",
    }
    for node in ("planner", "executor", "critic"):
        path = tmp_path / "anneal" / "airline" / node / "v1.md"
        assert path.is_file()
        text = path.read_text()
        # the deterministic template is always there, the elaboration only on top of it
        assert "## Output contract" in text and "## Working rules" in text
        assert "get_user_details" in text
        assert text.startswith(f"# You are the {node.upper()}")
        assert text.endswith(GOOD)
    assert all(n.prompt_source == "elaborated" for s in specs for n in s.nodes)


def test_template_states_role_goal_contract_and_tools(domain: FakeDomain) -> None:
    text = architect.build_template(domain, "executor")
    assert "EXECUTOR" in text
    assert domain.goal.strip()[:60] in text  # the goal, in the domain's own words
    assert "## Output contract" in text
    for tool in domain.tools.tools:
        assert tool.name in text
    assert "Call tools instead of guessing" in text
    assert "traceable to a tool result" in text


def test_template_lists_read_tools_before_write_tools(domain: FakeDomain) -> None:
    text = architect.build_template(domain, "executor")
    reads = text.index("Read-only tools")
    writes = text.index("Write tools")
    assert reads < writes
    assert "only as the very last action" in text


def test_bare_json_blob_falls_back_to_template(domain: FakeDomain, tmp_path: Path) -> None:
    """Root cause 1: a small model returning a bare schema must not become the prompt."""
    client = FakeClient([BARE_JSON])
    specs = architect.propose(domain, 1, client=client, prompts_root=tmp_path)

    assert len(client.calls) == 2  # first attempt, then one stricter retry
    retry = client.calls[1]["messages"][-1]["content"]
    assert "bare JSON blob" in retry
    text = (tmp_path / "anneal/airline/executor/v1.md").read_text()
    assert text == architect.build_template(domain, "executor")
    assert BARE_JSON not in text
    assert specs[0].nodes[0].prompt_source == "template"


def test_retry_is_accepted_when_the_second_reply_is_good(
    domain: FakeDomain, tmp_path: Path
) -> None:
    client = FakeClient([BARE_JSON, GOOD, GOOD, GOOD])
    specs = architect.propose(domain, 1, client=client, prompts_root=tmp_path)
    assert specs[0].nodes[0].prompt_source == "elaborated_retry"
    assert (tmp_path / "anneal/airline/executor/v1.md").read_text().endswith(GOOD)


def test_code_fence_only_reply_is_rejected(domain: FakeDomain, tmp_path: Path) -> None:
    fence = "```json\n" + BARE_JSON + "\n```"
    specs = architect.propose(domain, 1, client=FakeClient([fence]), prompts_root=tmp_path)
    assert specs[0].nodes[0].prompt_source == "template"


def test_validate_elaboration_reasons(domain: FakeDomain) -> None:
    names = [t.name for t in domain.tools.tools]
    assert architect.validate_elaboration(GOOD, names) is None
    assert "characters long" in str(architect.validate_elaboration("too short", names))
    assert "bare JSON" in str(architect.validate_elaboration(BARE_JSON, names))
    long_no_tool = "Think carefully about the request and answer it well. " * 8
    assert "never mentioned any of the tools" in str(
        architect.validate_elaboration(long_no_tool, names)
    )


def test_schema_ref_set_when_domain_exposes_a_schema(domain: FakeDomain, tmp_path: Path) -> None:
    """Root cause 2: without schema_ref runtime can never raise schema_error."""
    domain.eval = SimpleNamespace(OUTPUT_SCHEMA=OUTPUT_SCHEMA)
    specs = architect.propose(domain, 3, client=FakeClient([GOOD]), prompts_root=tmp_path)
    for spec in specs:
        answer_nodes = [n for n in spec.nodes if n.role == "executor"]
        assert answer_nodes and all(n.schema_ref == "OUTPUT_SCHEMA" for n in answer_nodes)
        assert all(n.schema_ref is None for n in spec.nodes if n.role != "executor")
    # the required keys reach the prompt so the node knows the contract
    text = (tmp_path / "anneal/airline/executor/v1.md").read_text()
    assert "`vendor`" in text and "`decision`" in text


def test_schema_ref_none_when_domain_has_no_schema(domain: FakeDomain, tmp_path: Path) -> None:
    specs = architect.propose(domain, 3, client=FakeClient([GOOD]), prompts_root=tmp_path)
    assert all(n.schema_ref is None for s in specs for n in s.nodes)
    assert architect.find_output_schema(domain) is None


def test_prompt_request_contains_goal_and_tools(domain: FakeDomain, tmp_path: Path) -> None:
    client = FakeClient([GOOD])
    architect.propose(domain, 1, client=client, prompts_root=tmp_path)
    user_msg = client.calls[0]["messages"][-1]["content"]
    assert "airline support agent" in user_msg
    assert "get_user_details" in user_msg


def test_empty_reply_falls_back(domain: FakeDomain, tmp_path: Path) -> None:
    specs = architect.propose(domain, 1, client=FakeClient([""]), prompts_root=tmp_path)
    text = (tmp_path / "anneal/airline/executor/v1.md").read_text()
    assert text.strip() and "EXECUTOR" in text
    assert text == architect.build_template(domain, "executor")
    assert specs[0].topology == "single"
    assert specs[0].nodes[0].prompt_source == "template"


def test_a_dead_model_still_yields_a_usable_prompt(domain: FakeDomain, tmp_path: Path) -> None:
    class Boom:
        def create(self, **_: Any) -> Any:
            raise RuntimeError("connection refused")

    client = SimpleNamespace(chat=SimpleNamespace(completions=Boom()))
    specs = architect.propose(domain, 1, client=client, prompts_root=tmp_path)
    text = (tmp_path / "anneal/airline/executor/v1.md").read_text()
    assert text == architect.build_template(domain, "executor")
    assert specs[0].nodes[0].prompt_source == "template"


def test_bad_n_raises(domain: FakeDomain, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        architect.propose(domain, 0, client=FakeClient([GOOD]), prompts_root=tmp_path)
    with pytest.raises(ValueError):
        architect.propose(domain, 99, client=FakeClient([GOOD]), prompts_root=tmp_path)
