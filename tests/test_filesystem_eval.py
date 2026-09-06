"""Tests for the filesystem demonstration domain.

The evaluator tests are pure disk state and always run. The last test is the acceptance
test for `mcp:` dispatch end to end: a scripted (offline) model drives the runtime, the
runtime calls the **real** third-party `@modelcontextprotocol/server-filesystem` over stdio,
and the domain's own evaluator then scores the resulting directory. It skips cleanly when
npx or that package is unavailable.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from anneal import mcp
from anneal.domain import Domain, load_domain
from anneal.spec import HarnessSpec
from tests.fakes import FakeClient

ROOT = Path(__file__).resolve().parents[1]
FILESYSTEM = ROOT / "domains" / "filesystem"
SPLITS = {"train": 6, "search": 3, "holdout": 3}


@pytest.fixture(scope="module")
def domain() -> Domain:
    return load_domain(FILESYSTEM)


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the domain at a throwaway sandbox root for the duration of one test."""
    root = (tmp_path / "sandbox").resolve()
    root.mkdir()
    monkeypatch.setenv("ANNEAL_FS_SANDBOX", str(root))
    return root


# --- tasks -----------------------------------------------------------------------------------


def test_tasks_split_50_25_25(domain: Domain) -> None:
    tasks = domain.eval.load_tasks()
    counts = {split: len(domain.eval.load_tasks(split)) for split in SPLITS}
    assert counts == SPLITS
    assert len(tasks) == sum(SPLITS.values())
    assert len({t.id for t in tasks}) == len(tasks)


def test_every_task_declares_expected_state(domain: Domain) -> None:
    for task in domain.eval.load_tasks():
        assert task.expected.get("files"), task.id
        assert task.input["instruction"].strip(), task.id


def test_workdir_is_resolved_per_task(domain: Domain, sandbox: Path) -> None:
    tasks = domain.eval.load_tasks("train")
    assert tasks[0].input["workdir"] == str(sandbox / tasks[0].id)
    assert "{workdir}" not in json.dumps(tasks[0].input)
    assert len({t.input["workdir"] for t in tasks}) == len(tasks)


def test_tools_are_all_served_by_the_mcp_server(domain: Domain) -> None:
    """Every tool of this domain is third-party: nothing is a local python impl."""
    assert domain.tools.tools
    assert all(t.impl.startswith("mcp:fs/") for t in domain.tools.tools)
    servers = mcp.parse_servers(domain.tools.servers)
    assert servers["fs"].is_stdio
    assert servers["fs"].args[1] == "@modelcontextprotocol/server-filesystem"


# --- setup / score / hard fail ------------------------------------------------------------


def _task(domain: Domain, task_id: str) -> Any:
    return next(t for t in domain.eval.load_tasks() if t.id == task_id)


def test_setup_seeds_and_wipes_the_task_directory(domain: Domain, sandbox: Path) -> None:
    task = _task(domain, "fs-03")  # seeded with draft.txt
    work = sandbox / task.id
    domain.eval.setup(task)
    assert (work / "draft.txt").read_text() == "chapter one\nchapter two\n"
    (work / "leftover.txt").write_text("stale")
    domain.eval.setup(task)
    assert not (work / "leftover.txt").exists()


def test_score_is_one_only_for_the_exact_end_state(domain: Domain, sandbox: Path) -> None:
    task = _task(domain, "fs-01")
    domain.eval.setup(task)
    work = sandbox / task.id
    assert domain.eval.score(task, "done") == 0.0  # nothing written yet
    (work / "notes.txt").write_text("remember to water the plants\n")  # trailing \n tolerated
    assert domain.eval.score(task, "done") == 1.0
    (work / "notes.txt").write_text("remember to water the plants (I added this)")
    assert domain.eval.score(task, "done") == 0.0


def test_score_requires_absent_paths_to_be_gone(domain: Domain, sandbox: Path) -> None:
    task = _task(domain, "fs-03")
    domain.eval.setup(task)
    work = sandbox / task.id
    shutil.copy(work / "draft.txt", work / "final.txt")
    assert domain.eval.score(task, "") == 0.0  # draft.txt still there
    (work / "draft.txt").unlink()
    assert domain.eval.score(task, "") == 1.0


def test_score_ignores_the_final_message(domain: Domain, sandbox: Path) -> None:
    task = _task(domain, "fs-01")
    domain.eval.setup(task)
    (sandbox / task.id / "notes.txt").write_text("remember to water the plants")
    assert domain.eval.score(task, {"anything": "at all"}) == 1.0


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"path": "{work}/ok.txt"}, False),
        ({"path": "{work}/nested/ok.txt"}, False),
        ({"source": "{work}/a", "destination": "{work}/b"}, False),
        ({"path": "/etc/passwd"}, True),
        ({"path": "{work}/../fs-99/steal.txt"}, True),
        ({"destination": "{sandbox}/elsewhere.txt"}, True),
    ],
)
def test_is_hard_fail_catches_paths_outside_the_task_directory(
    domain: Domain, sandbox: Path, args: dict[str, str], expected: bool
) -> None:
    task = _task(domain, "fs-01")
    work = sandbox / task.id
    work.mkdir(parents=True, exist_ok=True)
    filled = {k: v.format(work=work, sandbox=sandbox) for k, v in args.items()}
    trace = [{"tool": "write_file", "args": filled, "result": "ok"}]
    assert domain.eval.is_hard_fail(task, trace) is expected


def test_is_hard_fail_ignores_traces_without_paths(domain: Domain, sandbox: Path) -> None:
    task = _task(domain, "fs-01")
    assert domain.eval.is_hard_fail(task, []) is False
    assert domain.eval.is_hard_fail(task, [{"tool": "route", "args": None}]) is False


# --- acceptance: a task completed through the real third-party MCP server ------------------


def _spec(domain: Domain) -> HarnessSpec:
    return HarnessSpec.model_validate(
        {
            "id": "cand-fs-single",
            "topology": "single",
            "step_budget": 10,
            "nodes": [
                {
                    "name": "executor",
                    "role": "executor",
                    "model_tier": "mid",
                    "system_prompt_ref": "anneal/filesystem/executor@v1",
                    "tools": [t.name for t in domain.tools.tools],
                    "max_steps": 6,
                }
            ],
        }
    )


def _tool_turn(name: str, **kwargs: Any) -> list[dict[str, str]]:
    return [{"name": name, "arguments": json.dumps(kwargs)}]


def _real_server_available(sandbox: Path) -> bool:
    if shutil.which("npx") is None:
        return False
    cfg = mcp.ServerConfig(
        name="probe",
        command="npx",
        args=("-y", "@modelcontextprotocol/server-filesystem", str(sandbox)),
        startup_timeout_s=180.0,
    )
    client = mcp.Client(cfg)
    try:
        return "write_file" in client.list_tools()
    except (mcp.MCPError, OSError):
        return False
    finally:
        client.close()


@pytest.mark.integration
def test_task_completes_through_the_real_mcp_server(domain: Domain, sandbox: Path) -> None:
    """fs-02 (mkdir + write) end to end: runtime -> real MCP server -> evaluator scores 1.0.

    Only the model is faked. Every tool call leaves the process and is executed by
    ``@modelcontextprotocol/server-filesystem``.
    """
    from anneal.runtime import run_task

    if not _real_server_available(sandbox):
        pytest.skip("npx / @modelcontextprotocol/server-filesystem unavailable offline")
    mcp.close_all()  # the sandbox path changed, so any pooled server points at the old root

    task = _task(domain, "fs-02")
    work = Path(task.input["workdir"])
    client = FakeClient(
        turns=[
            _tool_turn("create_directory", path=str(work / "reports")),
            _tool_turn("write_file", path=str(work / "reports" / "summary.md"),
                       content="# Q3 summary"),
            "Created reports/summary.md.",
        ]
    )
    try:
        result = run_task(_spec(domain), task, domain, client_factory=lambda tier: client)
        assert [step["tool"] for step in result.trace] == ["create_directory", "write_file"]
        assert not any(str(step["result"]).startswith("Error") for step in result.trace), (
            result.trace
        )
        assert (work / "reports" / "summary.md").read_text() == "# Q3 summary"
        assert domain.eval.score(task, result.output) == 1.0
        assert domain.eval.is_hard_fail(task, result.trace) is False

        # the schemas the model was offered came from the server's own tools/list
        offered = {t["function"]["name"]: t["function"] for t in client.calls[0]["tools"]}
        assert offered["write_file"]["parameters"]["required"] == ["path", "content"]
        assert "overwrite" in offered["write_file"]["description"].lower()
    finally:
        mcp.close_all()


def test_a_dead_server_degrades_to_a_tool_error_not_a_crash(
    domain: Domain, sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the server unavailable the run still completes; the model just sees errors."""
    from anneal.runtime import run_task

    mcp.close_all()
    monkeypatch.setitem(
        domain.tools.__dict__, "servers", {"fs": {"command": "definitely-not-a-real-binary-xyz"}}
    )
    task = _task(domain, "fs-01")
    client = FakeClient(
        turns=[
            _tool_turn("write_file", path=str(Path(task.input["workdir"]) / "notes.txt"),
                       content="x"),
            "I could not reach the filesystem server.",
        ]
    )
    try:
        result = run_task(_spec(domain), task, domain, client_factory=lambda tier: client)
        assert result.trace[0]["result"].startswith("Error:")
        assert domain.eval.score(task, result.output) == 0.0
    finally:
        mcp.close_all()
