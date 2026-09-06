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


@dataclass
class FakeDomain:
    name: str
    goal: str
    tools: ToolsManifest


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
    client = FakeClient(["```\nYou are the node. Follow the goal.\n```"])
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
    specs = architect.propose(domain, 3, client=FakeClient(["prompt"]), prompts_root=tmp_path)
    all_tools = [t.name for t in domain.tools.tools]
    for s in specs:
        for node in s.nodes:
            expected_tier = {"planner": "frontier", "executor": "mid", "critic": "cheap"}
            assert node.model_tier == expected_tier[node.role]
            assert node.tools == (all_tools if node.role == "executor" else [])


def test_prompts_written_once_per_node(domain: FakeDomain, tmp_path: Path) -> None:
    client = FakeClient(["canned prompt text"])
    specs = architect.propose(domain, 3, client=client, prompts_root=tmp_path)

    assert len(client.calls) == 3  # planner, executor, critic: one call each, no live path
    refs = {n.system_prompt_ref for s in specs for n in s.nodes}
    assert refs == {
        "anneal/airline/executor@v1",
        "anneal/airline/planner@v1",
        "anneal/airline/critic@v1",
    }
    for node in ("planner", "executor", "critic"):
        path = tmp_path / "anneal" / "airline" / node / "v1.md"
        assert path.is_file()
        assert path.read_text() == "canned prompt text"


def test_prompt_request_contains_goal_and_tools(domain: FakeDomain, tmp_path: Path) -> None:
    client = FakeClient(["p"])
    architect.propose(domain, 1, client=client, prompts_root=tmp_path)
    user_msg = client.calls[0]["messages"][-1]["content"]
    assert "airline support agent" in user_msg
    assert "get_user_details" in user_msg


def test_empty_reply_falls_back(domain: FakeDomain, tmp_path: Path) -> None:
    specs = architect.propose(domain, 1, client=FakeClient([""]), prompts_root=tmp_path)
    text = (tmp_path / "anneal/airline/executor/v1.md").read_text()
    assert text.strip() and "EXECUTOR" in text
    assert specs[0].topology == "single"


def test_bad_n_raises(domain: FakeDomain, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        architect.propose(domain, 0, client=FakeClient(["p"]), prompts_root=tmp_path)
    with pytest.raises(ValueError):
        architect.propose(domain, 99, client=FakeClient(["p"]), prompts_root=tmp_path)
