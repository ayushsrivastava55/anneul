"""Tests for mutate.synthesize_tool: AO writes a tool, the operator adopts it (offline).

No AO session is ever spawned here. A ``FakeAO`` stands in for ``anneal.ao``: it records the
spawn, and its ``wait_for_branch`` either accepts (writing the stub tool the real worker
would have committed) or rejects. The domain lives in ``tmp_path``, so the repo's
``domains/`` tree, ``prompts/``, the network and git are all untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from anneal import mutate
from anneal.spec import HarnessSpec, ToolsManifest
from tests.fakes import FakeClient
from tests.test_mutate import EVIDENCE, make_issue, make_spec

TOOL_NAME = "compute_refund"
TOOL_JSON = {
    "name": TOOL_NAME,
    "description": "Compute the refund owed for a cancelled segment. Use before a refund.",
    "args": {
        "type": "object",
        "properties": {"fare": {"type": "number"}, "days_before": {"type": "integer"}},
        "required": ["fare", "days_before"],
    },
    "examples": [
        {"args": {"fare": 100, "days_before": 30}, "expected": 100},
        {"args": {"fare": 100, "days_before": 3}, "expected": 50},
        {"args": {"fare": 0, "days_before": 1}, "expected": 0},
    ],
}
STUB_TOOL = f'''"""Stub written by the fake AO worker."""


def {TOOL_NAME}(fare: float, days_before: int) -> float:
    return fare if days_before >= 7 else fare / 2
'''

HUMAN_TOOLS_YAML = """tools:
- name: cancel_reservation
  description: Cancel a reservation.
  impl: python:x.y
  mutates: true
- name: get_reservation_details
  description: Get details of a reservation.
  impl: python:x.y
  mutates: false
"""


class FakeAO:
    """Stands in for ``anneal.ao``: records the spawn, then accepts or rejects the branch."""

    def __init__(self, domain_dir: Path, *, accept: bool = True, write: bool = True) -> None:
        self.domain_dir = domain_dir
        self.accept = accept
        self.write = write
        self.spawns: list[dict[str, str]] = []
        self.waits: list[dict[str, Any]] = []
        self.git_calls: list[list[str]] = []

    def spawn_worker(self, name: str, branch: str, prompt: str, **_: Any) -> str:
        self.spawns.append({"name": name, "branch": branch, "prompt": prompt})
        return "anneal-fake-1"

    def wait_for_branch(
        self, branch: str, accept_cmd: list[str], *args: Any, **kw: Any
    ) -> tuple[bool, str]:
        self.waits.append({"branch": branch, "accept_cmd": accept_cmd, "kw": kw})
        if not self.accept:
            return False, "1 failed, 0 passed"
        if self.write:  # what the accepted commit would leave in the worktree
            module = self.domain_dir / mutate.GENERATED_PKG / f"{TOOL_NAME}.py"
            module.parent.mkdir(parents=True, exist_ok=True)
            module.write_text(STUB_TOOL)
        return True, "1 passed"

    def _git(self, args: list[str], repo: Path) -> Any:  # only reached for the test file
        self.git_calls.append(args)
        return SimpleNamespace(returncode=1, stdout="", stderr="not found")


@pytest.fixture
def domain(tmp_path: Path) -> SimpleNamespace:
    """A domain rooted in tmp_path with a human-written tools.yaml we must never edit."""
    domain_dir = tmp_path / "domains" / "airline"
    (domain_dir / mutate.GENERATED_PKG).mkdir(parents=True)
    (domain_dir / "tools.yaml").write_text(HUMAN_TOOLS_YAML)
    return SimpleNamespace(
        name="airline",
        goal="Be a helpful airline agent.",
        tools=ToolsManifest.model_validate(yaml.safe_load(HUMAN_TOOLS_YAML)),
        eval=SimpleNamespace(),
        path=domain_dir,
    )


def client_with_tool_spec(payload: Any = None) -> FakeClient:
    return FakeClient(turns=[json.dumps(payload if payload is not None else TOOL_JSON)])


def run(domain: SimpleNamespace, ao: FakeAO, issue: dict[str, Any], **kw: Any) -> HarnessSpec:
    return mutate.synthesize_tool(
        make_spec(), issue, EVIDENCE, domain, client_with_tool_spec(**kw), ao_module=ao
    )


# --- happy path ---------------------------------------------------------------------------


def test_accepted_branch_adds_the_tool_to_the_node_and_the_generated_manifest(
    domain: SimpleNamespace,
) -> None:
    ao = FakeAO(domain.path)
    issue = make_issue("missing_capability")

    out = run(domain, ao, issue)

    HarnessSpec.model_validate(out.model_dump())
    executor = next(n for n in out.nodes if n.name == "executor")
    assert TOOL_NAME in executor.tools
    assert executor.tools[:2] == ["cancel_reservation", "get_reservation_details"]

    generated = yaml.safe_load((domain.path / mutate.GENERATED_TOOLS_YAML).read_text())
    ToolsManifest.model_validate(generated)
    entry = next(t for t in generated["tools"] if t["name"] == TOOL_NAME)
    assert entry["description"] == TOOL_JSON["description"]
    assert entry["args"] == TOOL_JSON["args"]
    assert entry["impl"] == f"python:domains.airline.generated_tools.{TOOL_NAME}.{TOOL_NAME}"
    assert entry["mutates"] is False


def test_human_tools_yaml_is_never_modified(domain: SimpleNamespace) -> None:
    before = (domain.path / "tools.yaml").read_text()

    run(domain, FakeAO(domain.path), make_issue("missing_capability"))

    assert (domain.path / "tools.yaml").read_text() == before
    assert (domain.path / mutate.GENERATED_TOOLS_YAML).exists()


def test_spawn_uses_a_short_name_the_tool_branch_and_a_prompt_naming_both_files(
    domain: SimpleNamespace,
) -> None:
    ao = FakeAO(domain.path)

    run(domain, ao, make_issue("missing_capability"))

    (spawn,) = ao.spawns
    assert spawn["branch"] == f"ao/tool-{TOOL_NAME}"
    assert len(spawn["name"]) <= 20
    assert spawn["name"].startswith("tool-")
    assert f"domains/airline/generated_tools/{TOOL_NAME}.py" in spawn["prompt"]
    assert f"tests/test_{TOOL_NAME}.py" in spawn["prompt"]
    assert "tools.yaml" in spawn["prompt"]  # told not to touch it

    (wait,) = ao.waits
    assert wait["accept_cmd"][1:] == ["-m", "pytest", "-q", f"tests/test_{TOOL_NAME}.py"]
    assert wait["kw"]["session_id"] == "anneal-fake-1"


def test_session_names_stay_unique_and_within_the_ao_limit(domain: SimpleNamespace) -> None:
    names = {mutate._session_name("a_very_long_tool_name_indeed") for _ in range(20)}

    assert len(names) == 20
    assert all(len(name) <= 20 for name in names)


def with_fake_ao(ao: FakeAO) -> Any:
    """Bind ``ao`` into the registered operator so ``apply`` never reaches a real daemon."""
    original = mutate.OPERATORS["synthesize_tool"]

    def bound(spec: Any, iss: Any, ev: Any, dom: Any, cli: Any) -> Any:
        return mutate.synthesize_tool(spec, iss, ev, dom, cli, ao_module=ao)

    return original, bound


def test_lineage_is_set_when_applied_through_apply(domain: SimpleNamespace) -> None:
    ao = FakeAO(domain.path)
    issue = make_issue("missing_capability")
    original, bound = with_fake_ao(ao)
    mutate.OPERATORS["synthesize_tool"] = bound
    try:
        out = mutate.apply(make_spec(), issue, EVIDENCE, domain, client=client_with_tool_spec())
    finally:
        mutate.OPERATORS["synthesize_tool"] = original

    HarnessSpec.model_validate(out.model_dump())
    assert out.lineage is not None
    assert out.lineage.operator == "synthesize_tool"
    assert out.lineage.parent == "cand-01"
    assert out.lineage.ledger_issue == issue["id"]
    assert TOOL_NAME in next(n for n in out.nodes if n.name == "executor").tools


# --- failure paths ------------------------------------------------------------------------


def test_failed_acceptance_leaves_the_spec_unchanged_and_marks_the_issue(
    domain: SimpleNamespace,
) -> None:
    ao = FakeAO(domain.path, accept=False)
    spec = make_spec()
    issue = make_issue("missing_capability")

    out = mutate.synthesize_tool(
        spec, issue, EVIDENCE, domain, client_with_tool_spec(), ao_module=ao
    )

    assert out is spec
    assert out.model_dump() == make_spec().model_dump()
    assert not (domain.path / mutate.GENERATED_TOOLS_YAML).exists()
    assert issue["operators_tried"] == ["synthesize_tool"]
    assert issue["last_attempt"]["operator"] == "synthesize_tool"
    assert issue["last_attempt"]["ok"] is False
    assert "not accepted" in issue["last_attempt"]["reason"]


def test_apply_refuses_to_promote_a_no_op_mutation(domain: SimpleNamespace) -> None:
    ao = FakeAO(domain.path, accept=False)
    issue = make_issue("missing_capability")
    original, bound = with_fake_ao(ao)
    mutate.OPERATORS["synthesize_tool"] = bound
    try:
        with pytest.raises(mutate.NoOperatorAvailable, match="changed nothing"):
            mutate.apply(make_spec(), issue, EVIDENCE, domain, client=client_with_tool_spec())
    finally:
        mutate.OPERATORS["synthesize_tool"] = original
    assert issue["operators_tried"] == ["synthesize_tool"]


def test_accepted_branch_without_the_tool_file_is_a_failed_attempt(
    domain: SimpleNamespace,
) -> None:
    ao = FakeAO(domain.path, write=False)  # accepted, but nothing landed in the worktree
    spec = make_spec()
    issue = make_issue("missing_capability")

    out = mutate.synthesize_tool(
        spec, issue, EVIDENCE, domain, client_with_tool_spec(), ao_module=ao
    )

    assert out is spec
    assert not (domain.path / mutate.GENERATED_TOOLS_YAML).exists()
    assert "could not adopt" in issue["last_attempt"]["reason"]


def test_ao_error_during_spawn_is_a_failed_attempt(domain: SimpleNamespace) -> None:
    class Unreachable(FakeAO):
        def spawn_worker(self, *args: Any, **kw: Any) -> str:
            raise RuntimeError("AO daemon unreachable at http://127.0.0.1:3001")

    spec = make_spec()
    issue = make_issue("missing_capability")

    out = mutate.synthesize_tool(
        spec, issue, EVIDENCE, domain, client_with_tool_spec(), ao_module=Unreachable(domain.path)
    )

    assert out is spec
    assert "unreachable" in issue["last_attempt"]["reason"]


# --- the model's tool spec ------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {**TOOL_JSON, "name": "Compute Refund"},  # not snake_case
        {**TOOL_JSON, "name": "cancel_reservation"},  # collides with the human manifest
        {**TOOL_JSON, "description": ""},
        {**TOOL_JSON, "examples": []},
        {**TOOL_JSON, "args": "a string"},
    ],
)
def test_unusable_tool_specs_are_rejected_before_any_session_is_spawned(
    domain: SimpleNamespace, payload: dict[str, Any]
) -> None:
    ao = FakeAO(domain.path)

    with pytest.raises(RuntimeError):
        run(domain, ao, make_issue("missing_capability"), payload=payload)

    assert ao.spawns == []


def test_a_fenced_json_reply_is_still_parsed(domain: SimpleNamespace) -> None:
    ao = FakeAO(domain.path)
    fenced = FakeClient(turns=[f"```json\n{json.dumps(TOOL_JSON)}\n```"])

    out = mutate.synthesize_tool(
        make_spec(), make_issue("missing_capability"), EVIDENCE, domain, fenced, ao_module=ao
    )

    assert TOOL_NAME in next(n for n in out.nodes if n.name == "executor").tools


def test_a_second_synthesis_replaces_rather_than_duplicates_the_manifest_entry(
    domain: SimpleNamespace,
) -> None:
    ao = FakeAO(domain.path)

    run(domain, ao, make_issue("missing_capability"))
    run(domain, ao, make_issue("missing_capability"))

    generated = yaml.safe_load((domain.path / mutate.GENERATED_TOOLS_YAML).read_text())
    assert [t["name"] for t in generated["tools"]] == [TOOL_NAME]
