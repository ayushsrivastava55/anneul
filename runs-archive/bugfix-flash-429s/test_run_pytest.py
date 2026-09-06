"""Acceptance tests for the synthesised bugfix tool `run_pytest`.

Every case runs against a real per-task sandbox built from `domains/bugfix/fixtures/cases/`,
so pytest really is executed in a subprocess -- just never in the repo tree. Cases that must
pass get the known-good module from `fixtures/sources/` written over the buggy one first
(that is `eval.fixed_source`, the same fix `test_bugfix_eval.py` applies); cases that must
fail are left pristine. No LLM calls, no network, and holdout tasks are never touched: the
fixtures are read straight from disk by task id.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from domains.bugfix import eval as bugfix_eval
from domains.bugfix.fixtures import tools as bugfix_tools
from domains.bugfix.generated_tools.run_pytest import run_pytest

RESULT_KEYS = {"passed", "return_code", "output"}

OpenCase = Callable[..., Path]


def _task(task_id: str) -> bugfix_eval.Task:
    for task in bugfix_eval.load_tasks():
        if task.id == task_id:
            return task
    raise AssertionError(f"no such bugfix task: {task_id}")


@pytest.fixture
def open_case() -> Iterator[OpenCase]:
    """Open a sandbox for a task id, optionally with the bug already fixed."""
    opened: list[str] = []

    def _open(task_id: str, *, fixed: bool = False) -> Path:
        task = _task(task_id)
        root = bugfix_tools.open_sandbox(task_id, bugfix_eval.case_dir(task))
        opened.append(task_id)
        if fixed:
            (root / task.input["module"]).write_text(bugfix_eval.fixed_source(task))
        return root

    yield _open
    for task_id in opened:
        bugfix_tools.close_sandbox(task_id)


def test_a_passing_test_file_reports_passed(open_case: OpenCase) -> None:
    open_case("bugfix_01", fixed=True)
    result = run_pytest(path="tests/test_bugfix_01.py")
    assert set(result) == RESULT_KEYS
    assert result["passed"] is True
    assert result["return_code"] == 0
    assert "collected" in result["output"]
    assert "passed" in result["output"]
    assert "tests/test_bugfix_01.py" in result["output"]


def test_a_failing_test_file_reports_the_failure(open_case: OpenCase) -> None:
    open_case("bugfix_06")  # pristine: truncate() is one character too long
    result = run_pytest(path="tests/test_bugfix_06.py")
    assert result["passed"] is False
    assert result["return_code"] == 1
    assert "FAILURES" in result["output"]
    assert "failed" in result["output"]
    assert "test_long_text_is_cut_to_the_limit" in result["output"]


def test_a_directory_with_extra_args_selects_matching_tests(open_case: OpenCase) -> None:
    open_case("bugfix_03", fixed=True)
    result = run_pytest(path="tests", extra_args=["-k", "bugfix_03"])
    assert result["passed"] is True
    assert result["return_code"] == 0
    assert "collected" in result["output"]
    assert "tests/test_bugfix_03.py" in result["output"]


def test_extra_args_really_reach_pytest(open_case: OpenCase) -> None:
    """`-k` on a single test name must narrow the run, not just be accepted."""
    open_case("bugfix_03", fixed=True)
    everything = run_pytest(path="tests/test_bugfix_03.py")
    narrowed = run_pytest(path="tests/test_bugfix_03.py", extra_args=["-k", "page_count"])
    assert "collected 4 items" in everything["output"]
    assert "1 passed" in narrowed["output"]
    assert narrowed["passed"] is True


def test_the_result_is_json_serialisable(open_case: OpenCase) -> None:
    open_case("bugfix_01", fixed=True)
    result = run_pytest(path="tests/test_bugfix_01.py")
    assert json.loads(json.dumps(result)) == result


def test_the_sandbox_module_is_what_runs(open_case: OpenCase) -> None:
    """The run happens in the sandbox, so the sandbox copy is what gets imported and scored."""
    root = open_case("bugfix_01", fixed=True)
    pristine = bugfix_eval.case_dir(_task("bugfix_01")) / "chunk_list.py"
    before = pristine.read_text()
    (root / "chunk_list.py").write_text("def chunk_list(items, size):\n    return 'broken'\n")
    result = run_pytest(path="tests/test_bugfix_01.py")
    assert result["passed"] is False
    assert pristine.read_text() == before  # the committed fixture is never touched


@pytest.mark.parametrize(
    "path",
    ["", "   ", "../escaped.py", "/etc/passwd", "tests/test_bugfix_99.py"],
)
def test_bad_paths_raise_value_error(open_case: OpenCase, path: str) -> None:
    open_case("bugfix_01")
    with pytest.raises(ValueError):
        run_pytest(path=path)


@pytest.mark.parametrize("path", [None, 3, ["tests"]])
def test_a_non_string_path_raises_value_error(open_case: OpenCase, path: object) -> None:
    open_case("bugfix_01")
    with pytest.raises(ValueError):
        run_pytest(path=path)  # type: ignore[arg-type]


@pytest.mark.parametrize("extra_args", ["-v", 7, ["-k", ""], ["-k", None], [["-v"]]])
def test_bad_extra_args_raise_value_error(open_case: OpenCase, extra_args: object) -> None:
    open_case("bugfix_01", fixed=True)
    with pytest.raises(ValueError):
        run_pytest(path="tests/test_bugfix_01.py", extra_args=extra_args)  # type: ignore[arg-type]


def test_extra_args_defaults_to_nothing(open_case: OpenCase) -> None:
    """Omitting extra_args behaves like passing []; only the run's timing may differ."""
    open_case("bugfix_01", fixed=True)
    default = run_pytest(path="tests/test_bugfix_01.py")
    explicit = run_pytest(path="tests/test_bugfix_01.py", extra_args=[])
    assert (default["passed"], default["return_code"]) == (True, 0)
    assert (default["passed"], default["return_code"]) == (
        explicit["passed"],
        explicit["return_code"],
    )
    assert "collected 4 items" in default["output"]
    assert "collected 4 items" in explicit["output"]


def test_no_open_sandbox_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bugfix_tools, "_CURRENT", None)
    with pytest.raises(ValueError, match="no task sandbox"):
        run_pytest(path="tests/test_bugfix_01.py")
