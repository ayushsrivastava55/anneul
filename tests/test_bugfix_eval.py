"""Acceptance tests for domains/bugfix/eval.py. No LLM calls; never touches holdout."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from anneal.spec import load_tools
from domains.bugfix import eval as bugfix_eval
from domains.bugfix.fixtures import tools as bugfix_tools

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT / "domains" / "bugfix"


@pytest.fixture
def task() -> Iterator[bugfix_eval.Task]:
    """One train task, with a fresh sandbox that is torn down afterwards."""
    chosen = bugfix_eval.load_tasks("train")[0]
    bugfix_eval.setup(chosen)
    yield chosen
    bugfix_tools.close_sandbox(chosen.id)


def test_applying_the_known_fix_scores_one(task: bugfix_eval.Task) -> None:
    result = bugfix_tools.write_file(task.input["module"], bugfix_eval.fixed_source(task))
    assert result.startswith("Wrote ")
    assert bugfix_eval.score(task, "fixed the off-by-one") == 1.0
    assert bugfix_eval.THRESHOLD == 1.0


def test_leaving_the_bug_alone_scores_zero(task: bugfix_eval.Task) -> None:
    assert bugfix_eval.score(task, "I could not find the bug.") == 0.0


def test_concurrent_tasks_each_see_their_own_sandbox() -> None:
    """The runner executes tasks in a thread pool, each in a copy of the caller's context.

    With a module-global "current task" every in-flight task read whichever sandbox
    setup() touched last: read_file returned "No such file or directory" for files that
    existed, the agent looped and the step-budget deaths were misdiagnosed. The current
    task must therefore be context-local, exactly like the runner's own run context.
    """
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    tasks = bugfix_eval.load_tasks("train")[:4]

    def run_one(chosen: bugfix_eval.Task) -> str:
        bugfix_eval.setup(chosen)
        try:
            return bugfix_tools.read_file(chosen.input["module"])
        finally:
            bugfix_tools.close_sandbox(chosen.id)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(contextvars.copy_context().run, run_one, chosen) for chosen in tasks
        ]
        contents = [f.result() for f in futures]
    for chosen, text in zip(tasks, contents, strict=True):
        assert not text.startswith("Error"), f"{chosen.id} read the wrong sandbox: {text[:80]}"


def test_rewriting_the_test_file_does_not_buy_a_pass(task: bugfix_eval.Task) -> None:
    """The tests are the specification: score() restores them before running pytest."""
    bugfix_tools.write_file(task.input["test_file"], "def test_nothing():\n    assert True\n")
    assert bugfix_eval.score(task, "made the tests pass") == 0.0


def test_a_write_outside_the_sandbox_is_a_hard_fail(task: bugfix_eval.Task) -> None:
    trace = [{"tool": "write_file", "args": {"path": "/tmp/escaped.py", "content": "x"}}]
    assert bugfix_eval.is_hard_fail(task, trace) is True
    traversal = [{"tool": "write_file", "args": {"path": "../escaped.py", "content": "x"}}]
    assert bugfix_eval.is_hard_fail(task, traversal) is True


def test_writes_inside_the_sandbox_are_not_a_hard_fail(task: bugfix_eval.Task) -> None:
    trace = [
        {"tool": "read_file", "args": {"path": task.input["module"]}},
        {"tool": "write_file", "args": {"path": task.input["module"], "content": "x"}},
    ]
    assert bugfix_eval.is_hard_fail(task, trace) is False


def test_the_tool_itself_refuses_to_write_outside_the_sandbox(task: bugfix_eval.Task) -> None:
    assert bugfix_tools.write_file("../escaped.py", "x").startswith("Error:")
    assert not (bugfix_tools.sandbox_for(task.id).parent / "escaped.py").exists()


def test_setup_gives_every_run_a_pristine_sandbox(task: bugfix_eval.Task) -> None:
    bugfix_tools.write_file(task.input["module"], bugfix_eval.fixed_source(task))
    assert bugfix_eval.score(task, "fixed") == 1.0
    bugfix_eval.setup(task)
    assert bugfix_eval.score(task, "fixed") == 0.0


def test_split_sizes_are_20_10_10() -> None:
    tasks = bugfix_eval.load_tasks()
    counts: dict[str, int] = {}
    for one in tasks:
        counts[one.split] = counts.get(one.split, 0) + 1
    assert len(tasks) == 40
    assert counts == {"train": 20, "search": 10, "holdout": 10}


def test_every_bug_type_appears_in_every_split() -> None:
    """A plain shuffle can strand a whole bug type in one split; the generator stratifies."""
    by_split: dict[str, set[str]] = {}
    for one in bugfix_eval.load_tasks():
        by_split.setdefault(one.split, set()).update(one.tags)
    assert len(by_split) == 3
    assert all(len(types) == 7 for types in by_split.values())


def test_tasks_jsonl_matches_the_committed_cases() -> None:
    for one in bugfix_eval.load_tasks():
        case = bugfix_eval.case_dir(one)
        assert (case / one.input["module"]).is_file()
        assert (case / one.input["test_file"]).is_file()
        assert (case / "conftest.py").is_file()
        assert not (case / "fixed").exists()  # the fix never ships with the task


def test_the_fix_is_a_single_small_change() -> None:
    """Each case injects exactly one defect into an otherwise correct module."""
    for one in bugfix_eval.load_tasks():
        bug = json.loads((bugfix_eval.SOURCES_DIR / one.id / "bug.json").read_text())
        fixed = bugfix_eval.fixed_source(one)
        buggy = (bugfix_eval.case_dir(one) / one.input["module"]).read_text()
        assert fixed.count(bug["old"]) == 1
        assert buggy == fixed.replace(bug["old"], bug["new"], 1)
        assert buggy != fixed


def test_tools_yaml_is_valid_and_deliberately_lacks_run_tests() -> None:
    manifest = load_tools(DOMAIN / "tools.yaml")
    names = [tool.name for tool in manifest.tools]
    assert names == ["read_file", "write_file"]
    assert "run_tests" not in names  # absent on purpose: the synthesize_tool demo
    assert [tool.impl for tool in manifest.tools] == [
        "python:domains.bugfix.fixtures.tools.read_file",
        "python:domains.bugfix.fixtures.tools.write_file",
    ]


def test_goal_and_tasks_files_exist() -> None:
    assert (DOMAIN / "goal.md").read_text().strip()
    assert (DOMAIN / "tasks.jsonl").is_file()
