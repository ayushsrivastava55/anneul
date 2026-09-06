"""Evaluator for the filesystem domain (tools served by a third-party MCP server).

Contract (see domains/README.md):
    THRESHOLD                       score >= THRESHOLD counts as a pass
    load_tasks(split=None)          -> list[Task] from tasks.jsonl, sandbox paths filled in
    setup(task)                     wipe and re-seed <sandbox>/<task_id> with the task's files
    score(task, output)             1.0 when the directory ends up exactly as specified
    is_hard_fail(task, trace)       a tool call touching a path outside the task's directory

Scoring is state-based: the agent's final message is ignored, only the directory is compared.
``expected.files`` maps a path relative to the task directory to its exact required content
(trailing newlines are tolerated, nothing else is); ``expected.absent`` lists paths that must
no longer exist. Any extra file in the directory is fine unless ``expected.absent`` names it.
No LLM calls happen here.

The sandbox root is ``$ANNEAL_FS_SANDBOX``, defaulted here at import time to
``<tempdir>/anneal-filesystem-sandbox``. tools.yaml passes the same variable to the MCP
server, so the server itself refuses any path outside it; ``is_hard_fail`` additionally
catches a task writing into *another* task's directory, which the server would allow.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

THRESHOLD = 1.0
_HERE = Path(__file__).resolve().parent
TASKS_PATH = _HERE / "tasks.jsonl"

SANDBOX_ENV = "ANNEAL_FS_SANDBOX"
DEFAULT_SANDBOX = Path(tempfile.gettempdir()).resolve() / "anneal-filesystem-sandbox"
# tools.yaml expands ${ANNEAL_FS_SANDBOX} when the MCP server is launched, which happens
# after load_domain has imported this module -- so setting the default here is enough.
os.environ.setdefault(SANDBOX_ENV, str(DEFAULT_SANDBOX))

# Tool-argument keys whose value is a filesystem path (see tools.yaml).
PATH_KEYS = ("path", "source", "destination")
# Placeholder replaced by the task's own directory when tasks.jsonl is loaded.
WORKDIR_TOKEN = "{workdir}"


def sandbox_root() -> Path:
    """The directory the MCP server is allowed to touch. Resolved: the server compares realpaths."""
    return Path(os.environ.get(SANDBOX_ENV) or DEFAULT_SANDBOX).resolve()


def workdir(task_id: str) -> Path:
    """Private directory for one task, inside the sandbox root."""
    return sandbox_root() / task_id


@dataclass(frozen=True)
class Task:
    id: str
    input: dict[str, Any]  # {"workdir": str, "instruction": str}
    expected: dict[str, Any]  # {"files": {relpath: content}, "absent": [relpath]}
    split: str  # train | search | holdout
    tags: list[str] = field(default_factory=list)
    # Files laid down by setup() before the agent starts, relative path -> content.
    initial: dict[str, str] = field(default_factory=dict)


def _fill(value: Any, work: Path) -> Any:
    """Replace ``{workdir}`` in every string of a nested structure with the real path."""
    if isinstance(value, str):
        return value.replace(WORKDIR_TOKEN, str(work))
    if isinstance(value, dict):
        return {k: _fill(v, work) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, work) for v in value]
    return value


def load_tasks(split: str | None = None) -> list[Task]:
    """Tasks from tasks.jsonl with ``{workdir}`` resolved to this machine's sandbox."""
    tasks: list[Task] = []
    with TASKS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            if split is not None and raw["split"] != split:
                continue
            work = workdir(raw["id"])
            raw["input"] = {**_fill(raw["input"], work), "workdir": str(work)}
            tasks.append(Task(**raw))
    return tasks


def setup(task: Task) -> None:
    """Give the task a pristine private directory seeded with ``initial``."""
    work = workdir(task.id)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    for rel, content in task.initial.items():
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _normalise(text: str) -> str:
    """Compare contents ignoring line-ending style and trailing blank lines only."""
    return text.replace("\r\n", "\n").rstrip("\n")


def score(task: Task, output: Any) -> float:
    """1.0 when every expected file has the exact content and every ``absent`` path is gone."""
    del output  # state-based: the directory is the answer, not the final message
    work = workdir(task.id)
    for rel, want in (task.expected.get("files") or {}).items():
        path = work / rel
        if not path.is_file():
            return 0.0
        try:
            got = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return 0.0
        if _normalise(got) != _normalise(str(want)):
            return 0.0
    for rel in task.expected.get("absent") or []:
        if (work / rel).exists():
            return 0.0
    return 1.0


def _escapes(value: Any, work: Path) -> bool:
    """True when ``value`` is a path that does not live under ``work``."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        candidate = Path(value)
        resolved = (candidate if candidate.is_absolute() else work / candidate).resolve()
    except (OSError, ValueError):
        return True
    return work.resolve() not in (resolved, *resolved.parents)


def is_hard_fail(task: Task, trace: list[dict[str, Any]]) -> bool:
    """True when any tool call named a path outside this task's own directory.

    The MCP server already blocks escapes from the sandbox root; this is the tighter rule
    that keeps one task from writing into another task's directory.
    """
    work = workdir(task.id)
    for step in trace or []:
        args = step.get("args") if isinstance(step, dict) else None
        if not isinstance(args, dict):
            continue
        for key in PATH_KEYS:
            if _escapes(args.get(key), work):
                return True
    return False
