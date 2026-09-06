"""Sandboxed `read_file` / `write_file` impls plus the per-task sandbox registry.

State lives in `_SANDBOXES`, one directory per task id. `eval.setup(task)` calls
`open_sandbox(task_id, source_dir)` before every run, which throws away any previous sandbox
for that task and copies the pristine case files into a fresh temp directory outside the repo
tree. The tools only ever touch paths inside the current sandbox; anything that escapes it is
refused (and separately flagged by `eval.is_hard_fail`, which sees the attempt in the trace).

Which sandbox is "current" is a `ContextVar`, not a module global: the runner executes tasks
concurrently in a thread pool and copies the caller's context into each pool thread, so a
global would make every in-flight task read whichever sandbox `setup()` touched last. (That
exact bug shipped: every `read_file` returned "No such file or directory" because the agent
was looking in another task's sandbox, and the resulting step-budget deaths were misread as
the executor looping. Airline's DB uses a ContextVar for the same reason.)

Every wrapper returns a string, airline style (contents on success, "Error: ..." on failure),
so the runtime never sees an exception.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from contextvars import ContextVar
from pathlib import Path

_SANDBOXES: dict[str, Path] = {}
_CURRENT: ContextVar[str | None] = ContextVar("bugfix_current_task", default=None)
_MAX_BYTES = 200_000


def open_sandbox(task_id: str, source_dir: Path) -> Path:
    """Copy `source_dir` into a fresh temp sandbox for `task_id` and make it current."""
    close_sandbox(task_id)
    root = Path(tempfile.mkdtemp(prefix=f"anneal-bugfix-{task_id}-")).resolve()
    shutil.copytree(source_dir, root, dirs_exist_ok=True)
    _SANDBOXES[task_id] = root
    _CURRENT.set(task_id)
    return root


def close_sandbox(task_id: str) -> None:
    """Remove the sandbox for `task_id`, if any."""
    root = _SANDBOXES.pop(task_id, None)
    if root is not None:
        shutil.rmtree(root, ignore_errors=True)


def sandbox_for(task_id: str) -> Path | None:
    """The live sandbox directory for `task_id`, or None if setup() has not run."""
    return _SANDBOXES.get(task_id)


def current_sandbox() -> Path | None:
    """The sandbox of the task the runtime is executing right now."""
    task_id = _CURRENT.get()
    return _SANDBOXES.get(task_id) if task_id is not None else None


def resolve_in_sandbox(root: Path, path: str) -> Path | None:
    """Resolve `path` under `root`, or None if it escapes the sandbox.

    Absolute paths and `..` traversal both escape. `root` is resolved too, so the macOS
    /var -> /private/var symlink does not produce a false positive.
    """
    root = Path(root).resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    return resolved if resolved == root or root in resolved.parents else None


def read_file(path: str) -> str:
    """Read a file from the current task sandbox."""
    root = current_sandbox()
    if root is None:
        return "Error: no task sandbox is open"
    target = resolve_in_sandbox(root, path)
    if target is None:
        return f"Error: {path} is outside the task directory"
    try:
        return target.read_text()
    except OSError as exc:
        return f"Error: cannot read {path}: {exc.strerror or exc}"


def write_file(path: str, content: str) -> str:
    """Overwrite a file in the current task sandbox. Refuses to escape it."""
    root = current_sandbox()
    if root is None:
        return "Error: no task sandbox is open"
    target = resolve_in_sandbox(root, path)
    if target is None:
        return f"Error: refusing to write {path}: outside the task directory"
    if len(content) > _MAX_BYTES:
        return f"Error: refusing to write {path}: over {_MAX_BYTES} bytes"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    except OSError as exc:
        return f"Error: cannot write {path}: {exc.strerror or exc}"
    return f"Wrote {path} ({len(content)} bytes)."


@atexit.register
def _cleanup() -> None:
    for root in list(_SANDBOXES.values()):
        shutil.rmtree(root, ignore_errors=True)
    _SANDBOXES.clear()
