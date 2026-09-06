"""Run pytest inside the current bugfix task sandbox and report the verdict.

The human-written `domains/bugfix/tools.yaml` deliberately gives the agent `read_file` and
`write_file` only, so after rewriting a module it has no way to find out whether the target
test now passes: it has to reason about the test file by hand. This tool closes that gap. It
runs pytest in the live per-task sandbox created by `eval.setup()` -- the same directory
`write_file` edits -- and returns whether the run passed plus the captured output.

Pure stdlib, no network, no new dependencies. `path` is resolved inside the sandbox exactly
like `read_file` does, so a path that escapes it is refused (with a `ValueError`) rather than
executed. The subprocess gets the same minimal environment `eval.py` uses for scoring, so it
cannot pick up repo-level pytest configuration.

Returns `{"passed": bool, "return_code": int, "output": str}`.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from domains.bugfix.fixtures import tools as _tools

TIMEOUT_S = 30
TIMEOUT_RETURN_CODE = -1  # pytest never exits negative, so these cannot collide
SPAWN_RETURN_CODE = -2
MAX_OUTPUT_CHARS = 8_000
HEAD_CHARS = 2_000
# --no-header keeps the platform/rootdir preamble out of the model's context; the collection
# line, the per-test progress line, the FAILURES section and the summary all survive.
PYTEST_ARGS = ("-p", "no:cacheprovider", "--no-header")
SUBPROCESS_ENV = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}


def run_pytest(path: str, extra_args: Sequence[str] | None = None) -> dict[str, Any]:
    """Run pytest on `path` (a file or directory) inside the current task sandbox.

    `extra_args` are passed to pytest ahead of `path`, e.g. `["-k", "bugfix_03"]`.
    Raises ValueError if no sandbox is open, if `path` is not a relative path that exists
    inside it, or if `extra_args` is not a list of non-empty strings.
    """
    root = _sandbox_root()
    target = _resolve_target(root, path)
    args = _clean_extra_args(extra_args)
    command = [sys.executable, "-m", "pytest", *PYTEST_ARGS, *args, _relative(root, target)]
    try:
        proc = subprocess.run(  # fixed argv, no shell, cwd inside the sandbox
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            env=SUBPROCESS_ENV,
        )
    except subprocess.TimeoutExpired as exc:
        partial = _as_text(exc.stdout) + _as_text(exc.stderr)
        timed_out = f"{partial}\nError: pytest timed out after {TIMEOUT_S}s."
        return _result(TIMEOUT_RETURN_CODE, timed_out)
    except OSError as exc:
        return _result(SPAWN_RETURN_CODE, f"Error: cannot run pytest: {exc}")
    return _result(proc.returncode, proc.stdout + proc.stderr)


def _result(return_code: int, output: str) -> dict[str, Any]:
    return {"passed": return_code == 0, "return_code": return_code, "output": _clip(output)}


def _sandbox_root() -> Path:
    root = _tools.current_sandbox()
    if root is None:
        raise ValueError("no task sandbox is open")
    return Path(root).resolve()


def _resolve_target(root: Path, path: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string relative to the task directory")
    target = _tools.resolve_in_sandbox(root, path)
    if target is None:
        raise ValueError(f"path {path!r} is outside the task directory")
    if not target.exists():
        raise ValueError(f"path {path!r} does not exist in the task directory")
    return target


def _clean_extra_args(extra_args: Sequence[str] | None) -> list[str]:
    if extra_args is None:
        return []
    if isinstance(extra_args, str | bytes) or not isinstance(extra_args, Sequence):
        raise ValueError("extra_args must be a list of strings")
    args = list(extra_args)
    for arg in args:
        if not isinstance(arg, str) or not arg.strip():
            raise ValueError(f"extra_args entries must be non-empty strings, got {arg!r}")
    return args


def _relative(root: Path, target: Path) -> str:
    """`target` as a sandbox-relative posix path, or "." for the sandbox root itself."""
    rel = target.relative_to(root).as_posix()
    return rel or "."


def _as_text(value: str | bytes | None) -> str:
    """Timeout output is str under text=True, but bytes on some paths; accept both."""
    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode("utf-8", "replace")


def _clip(output: str) -> str:
    """Trim the transcript to a size worth putting in a prompt, keeping head and tail."""
    text = output.strip()
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    omitted = len(text) - MAX_OUTPUT_CHARS
    tail = text[-(MAX_OUTPUT_CHARS - HEAD_CHARS) :]
    return f"{text[:HEAD_CHARS]}\n... [{omitted} characters omitted] ...\n{tail}"
