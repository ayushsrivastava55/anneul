"""AO as executor: spawn a worker session, poll it, accept its branch by running tests.

Anneal's code-level mutations (tool synthesis, new glue) are delegated to AO worker
sessions: a session gets a prompt and a branch, and the branch is accepted only when a
command we choose passes against a checkout of it. Nothing here knows about domains.

Observed AO surface (daemon 127.0.0.1:${AO_PORT:-3001}, CLI ``~/.local/bin/ao``,
recorded 2026-09-05 against project ``anneal``; every block below is pasted output)::

    $ ao spawn --help
    Usage:
      ao spawn [flags]
    Flags:
          --branch string    Branch for git project sessions (default: ao/<session-id>/root)
          --claim-pr string  Immediately claim an existing PR for the spawned session
          --harness string   Agent harness / --agent: claude-code, codex, aider, ... (default:
                             project worker.agent)
          --issue string     Issue id to associate with the session
          --kind string      Session role: worker or orchestrator (default: worker)
          --mode string      Initial session interface: chat or tui
          --model string     Agent model override for this session only
          --name string      Display name shown in the sidebar (required, max 20 characters)
          --project string   Project id to spawn the session in
          --prompt string    Initial prompt for the agent
          --skip-agent-check Skip advisory agent catalog preflight
          --tracker-provider Issue tracker provider: github or gitlab

There is **no** ``--json`` flag on ``ao spawn`` (docs/ARCHITECTURE.md is wrong about that);
spawn prints one human line, e.g. ``spawned session anneal-22 (idle) [prompt 31 B, system
7980 B]``. Redirecting the CLI to a capture server did not work either: neither ``AO_PORT``
nor a fake ``HOME`` moved it, because it discovers the daemon through ``~/.ao/running.json``
(``{"pid":5619,"port":3001,...}``). The request body below was therefore reconstructed by
probing the daemon's validation errors, and confirmed by the response: the CLI's
``[prompt 31 B, system 7980 B]`` is exactly the ``promptBytes`` / ``systemPromptBytes`` the
API returns, so both paths hit the same endpoint.

Probing ``POST /api/v1/sessions``::

    {}                                            -> 400 PROJECT_ID_REQUIRED
    {"projectId":"anneal"}                        -> 400 AGENT_REQUIRED
    {"projectId":"anneal","displayName":"probe-shape","kind":"worker"}
                                                  -> 400 AGENT_REQUIRED
    {... ,"displayName":"this-name-is-way-too-long-x","harness":"claude-code"}
                                                  -> 400 DISPLAY_NAME_TOO_LONG
                                                     "displayName must be 20 characters or fewer"

The accepted body and its 201 response::

    POST /api/v1/sessions
    {"projectId":"anneal","displayName":"probe-shape","kind":"worker",
     "harness":"claude-code","branch":"ao/probe-shape","prompt":"echo nothing; ..."}
    201 {"session":{"id":"anneal-22","projectId":"anneal","kind":"worker",
         "harness":"claude-code","autoReviewEnabled":false,"displayName":"probe-shape",
         "mode":"chat","activity":{"state":"idle","lastActivityAt":"...Z"},
         "isTerminated":false,"terminateOnPrMerge":false,"autoInjectReview":true,
         "autoInjectCI":true,"createdAt":"...Z","updatedAt":"...Z","isPinned":false,
         "status":"idle","kanbanColumn":"building","displayStatus":"Awaiting PR",
         "branch":"ao/probe-shape","lastUserMessageAt":"...Z","prs":[]},
         "promptBytes":31,"systemPromptBytes":7980}

``GET /api/v1/sessions/{id}`` returns the same ``{"session": {...}}`` envelope. After
``ao session kill anneal-22`` it reads ``"status":"terminated"``, ``"isTerminated":true``,
``"activity":{"state":"exited",...}``, ``"kanbanColumn":"archive"``. Statuses seen over a
session's life: ``idle`` -> ``working`` -> ``terminated``; ``activity.state``: ``idle`` ->
``active`` -> ``exited``. The CLI's own views of the same data::

    $ ao session ls --json
    {"data":[{"id":"anneal-10","projectId":"anneal","role":"worker","status":"idle",
              "harness":"claude-code","isTerminated":false,"lastActivityAt":"...Z",
              "createdAt":"...Z","updatedAt":"...Z"}, ...]}

    $ ao session get anneal-17            $ ao session get anneal-17 --json
    id: anneal-17                         {"session":{"id":"anneal-17","projectId":"anneal",
    project: anneal                        "kind":"worker","harness":"claude-code",
    name: ao-spawn                         "displayName":"ao-spawn","activity":{"state":
    role: worker                           "active","lastActivityAt":"...Z"},
    status: working                        "isTerminated":false,"createdAt":"...Z",
    activity: active                       "updatedAt":"...Z","status":"working"}}
    harness: claude-code
    terminated: false

Messages (``ao send --session <id> --message <s>``) go to
``POST /api/v1/sessions/{id}/conversation/messages``. The field is ``text``; ``message``,
``prompt`` and ``content`` are all rejected with
``400 {"error":"validation","code":"CHAT_MESSAGE_EMPTY","message":"message text is
required"}``. A good call returns
``202 {"turnId":"3008...","providerTurnId":"eb13...","state":"running","duplicate":false}``.

One trap worth recording: AO creates the branch **at spawn time**, before the agent has
committed anything (``git branch --list ao/probe-shape`` matched seconds after the spawn).
Branch existence is therefore not a completion signal, so :func:`wait_for_branch` accepts a
branch only once its tip has moved past ``base_ref`` and only re-runs ``accept_cmd`` when the
tip sha changes.

Verified end to end with ``uv run python -m anneal.ao --selftest``: session ``anneal-27``
was spawned on ``ao/selftest-f836bf``, wrote and committed a trivial tool plus its test,
and the commit ``70dfa93`` was accepted by pytest in a detached worktree (``1 passed``);
the session was then killed and the branch deleted.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx

from anneal.config import env
from anneal.tracing import tool_span

logger = logging.getLogger("anneal.ao")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = "3001"
DEFAULT_PROJECT = "anneal"
DEFAULT_AGENT = "claude-code"
MAX_NAME_LEN = 20
HTTP_TIMEOUT_S = 10.0
ACCEPT_TIMEOUT_S = 900.0
GIT_TIMEOUT_S = 120.0


class AOError(RuntimeError):
    """The AO daemon refused a request or answered with a body we cannot use."""


def base_url() -> str:
    """Base URL of the local AO daemon (``AO_PORT`` overrides the default 3001)."""
    return f"http://127.0.0.1:{env('AO_PORT') or DEFAULT_PORT}"


# --- http -------------------------------------------------------------------------------


def _request(
    method: str,
    path: str,
    *,
    body: dict[str, object] | None = None,
    client: httpx.Client | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> dict[str, object]:
    """Call the daemon and return the decoded body. Never blocks longer than `timeout_s`."""
    owned = client is None
    http = client or httpx.Client(timeout=timeout_s)
    try:
        response = http.request(method, f"{base_url()}{path}", json=body)
    except httpx.HTTPError as exc:
        raise AOError(f"AO daemon unreachable at {base_url()}: {exc}") from exc
    finally:
        if owned:
            http.close()
    try:
        payload = response.json()
    except ValueError as exc:
        raise AOError(f"AO {method} {path}: non-JSON reply ({response.status_code})") from exc
    if response.status_code >= 400:
        code = payload.get("code") if isinstance(payload, dict) else None
        message = payload.get("message") if isinstance(payload, dict) else None
        raise AOError(f"AO {method} {path}: {response.status_code} {code}: {message}")
    if not isinstance(payload, dict):
        raise AOError(f"AO {method} {path}: expected an object, got {type(payload).__name__}")
    return payload


# --- sessions ---------------------------------------------------------------------------


@tool_span("ao.spawn")
def spawn_worker(
    name: str,
    branch: str,
    prompt: str,
    *,
    project: str = DEFAULT_PROJECT,
    agent: str = DEFAULT_AGENT,
    client: httpx.Client | None = None,
) -> str:
    """Spawn an AO worker session on `branch` and return its session id.

    `name` is the sidebar display name; the daemon rejects anything over 20 characters, so
    we reject it first with a `ValueError` rather than burning a round trip.
    """
    name = name.strip()
    if not name:
        raise ValueError("AO session name must not be empty")
    if len(name) > MAX_NAME_LEN:
        raise ValueError(f"AO session name {name!r} is {len(name)} chars, max {MAX_NAME_LEN}")
    if not branch:
        raise ValueError("AO session branch must not be empty")
    payload = _request(
        "POST",
        "/api/v1/sessions",
        body={
            "projectId": project,
            "displayName": name,
            "kind": "worker",
            "harness": agent,
            "branch": branch,
            "prompt": prompt,
        },
        client=client,
    )
    session = payload.get("session")
    if not isinstance(session, dict) or not session.get("id"):
        raise AOError(f"AO spawn returned no session id: {payload}")
    session_id = str(session["id"])
    logger.info(json.dumps({"event": "ao.spawn", "session": session_id, "branch": branch}))
    return session_id


@tool_span("ao.send")
def send(session_id: str, message: str, *, client: httpx.Client | None = None) -> str:
    """Send `message` to a running session; return the turn id the daemon assigns."""
    payload = _request(
        "POST",
        f"/api/v1/sessions/{session_id}/conversation/messages",
        body={"text": message},
        client=client,
    )
    return str(payload.get("turnId", ""))


@tool_span("ao.get_session")
def get_session(session_id: str, *, client: httpx.Client | None = None) -> dict[str, object]:
    """Return the daemon's session record (the inside of the ``{"session": ...}`` envelope)."""
    payload = _request("GET", f"/api/v1/sessions/{session_id}", client=client)
    session = payload.get("session")
    if not isinstance(session, dict):
        raise AOError(f"AO get session {session_id}: no session in reply")
    return session


@tool_span("ao.status")
def status(session_id: str, *, client: httpx.Client | None = None) -> str:
    """Return the session status: ``idle``, ``working``, ``terminated``, ..."""
    return str(get_session(session_id, client=client).get("status", "unknown"))


# --- git --------------------------------------------------------------------------------


def _run(
    cmd: list[str], *, cwd: str | Path | None = None, timeout_s: float = GIT_TIMEOUT_S
) -> subprocess.CompletedProcess[str]:
    """Single subprocess seam: everything git and every accept command goes through here."""
    return subprocess.run(  # noqa: S603
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _git(args: list[str], repo: Path) -> subprocess.CompletedProcess[str]:
    return _run(["git", "-C", str(repo), *args])


def _rev_parse(ref: str, repo: Path) -> str | None:
    """Resolve `ref` to a commit sha, or None when it does not exist."""
    done = _git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], repo)
    sha = done.stdout.strip()
    return sha if done.returncode == 0 and sha else None


def _branch_tip(branch: str, repo: Path) -> str | None:
    """Tip of `branch`, looking at the local ref first and then at origin after a fetch."""
    sha = _rev_parse(branch, repo)
    if sha:
        return sha
    _git(["fetch", "--quiet", "origin"], repo)
    return _rev_parse(branch, repo) or _rev_parse(f"origin/{branch}", repo)


def _run_accept(sha: str, accept_cmd: list[str], repo: Path, timeout_s: float) -> tuple[bool, str]:
    """Check `sha` out in a throwaway detached worktree and run `accept_cmd` inside it.

    The worktree is always removed, whether the command passes, fails or explodes.
    """
    parent = tempfile.mkdtemp(prefix="anneal-ao-")
    work = Path(parent) / "wt"
    try:
        added = _git(["worktree", "add", "--detach", str(work), sha], repo)
        if added.returncode != 0:
            return False, f"git worktree add failed: {added.stdout}{added.stderr}"
        try:
            done = _run(accept_cmd, cwd=work, timeout_s=timeout_s)
        except subprocess.TimeoutExpired:
            return False, f"accept command timed out after {timeout_s}s: {accept_cmd}"
        return done.returncode == 0, f"{done.stdout}{done.stderr}"
    finally:
        _git(["worktree", "remove", "--force", str(work)], repo)
        shutil.rmtree(parent, ignore_errors=True)


@tool_span("ao.wait_for_branch")
def wait_for_branch(
    branch: str,
    accept_cmd: list[str],
    timeout_s: float = 1800.0,
    poll_s: float = 15.0,
    *,
    session_id: str | None = None,
    repo: Path | str = ROOT,
    base_ref: str = "main",
    accept_timeout_s: float = ACCEPT_TIMEOUT_S,
    client: httpx.Client | None = None,
) -> tuple[bool, str]:
    """Poll a worker until its branch carries work, then accept it with `accept_cmd`.

    Each poll reads the session (when `session_id` is given) and resolves the branch tip.
    `accept_cmd` runs in a detached checkout of that tip, once per distinct sha; a tip still
    equal to `base_ref` means AO created the branch but the agent has not committed yet.
    Returns ``(ok, output)`` and gives up when the session terminates or `timeout_s` passes.
    """
    repo = Path(repo)
    deadline = time.monotonic() + timeout_s
    base_sha = _rev_parse(base_ref, repo)
    tested: set[str] = set()
    last = ""
    while True:
        terminated = False
        if session_id is not None:
            try:
                terminated = bool(get_session(session_id, client=client).get("isTerminated"))
            except AOError as exc:  # a flaky daemon must not abort the wait
                logger.warning(json.dumps({"event": "ao.poll_failed", "error": str(exc)}))
        sha = _branch_tip(branch, repo)
        if sha and sha != base_sha and sha not in tested:
            tested.add(sha)
            ok, last = _run_accept(sha, accept_cmd, repo, accept_timeout_s)
            logger.info(json.dumps({"event": "ao.accept", "branch": branch, "sha": sha, "ok": ok}))
            if ok:
                return True, last
        if terminated:
            return False, last or f"session {session_id} terminated without an accepted commit"
        if time.monotonic() >= deadline:
            return False, last or f"timed out after {timeout_s}s waiting for {branch}"
        time.sleep(poll_s)


# --- selftest (real daemon; never collected by pytest) ------------------------------------

_SELFTEST_PROMPT = """Create exactly two files at the repository root, nothing else:

1. `{module}.py` containing:

```python
\"\"\"Trivial tool generated by an AO worker for the anneal ao.py selftest.\"\"\"


def add(a: int, b: int) -> int:
    return a + b
```

2. `test_{module}.py` containing:

```python
from {module} import add


def test_add() -> None:
    assert add(2, 3) == 5
```

Then `git add` both files and commit them on the current branch with the message
"{name}: add trivial tool for the ao.py selftest". Do not push, do not open a PR, do not
touch any other file, and do not run any other command.
"""


def _ao_cli() -> str | None:
    return shutil.which("ao") or (
        str(Path.home() / ".local/bin/ao") if (Path.home() / ".local/bin/ao").exists() else None
    )


def _selftest(timeout_s: float, poll_s: float) -> int:
    """Really spawn a worker, wait for its commit, accept it with pytest, then clean up."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    token = uuid.uuid4().hex[:6]
    name = f"selftest-{token}"
    branch = f"ao/{name}"
    module = f"ao_selftest_{token}"
    prompt = _SELFTEST_PROMPT.format(module=module, name=name)

    session_id = spawn_worker(name, branch, prompt)
    print(f"spawned {session_id} on {branch} (status={status(session_id)})")
    accept_cmd = [sys.executable, "-m", "pytest", "-q", f"test_{module}.py"]
    try:
        ok, output = wait_for_branch(
            branch,
            accept_cmd,
            timeout_s,
            poll_s,
            session_id=session_id,
            accept_timeout_s=300.0,
        )
    finally:
        cli = _ao_cli()
        if cli:
            _run([cli, "session", "kill", session_id])
        _git(["branch", "-D", branch], ROOT)
    print(output.strip()[-2000:])
    print(f"selftest {'PASSED' if ok else 'FAILED'}: {branch}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m anneal.ao", description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="real end-to-end AO spawn")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--poll", type=float, default=15.0)
    args = parser.parse_args(argv)
    if not args.selftest:
        parser.print_help()
        return 0
    return _selftest(args.timeout, args.poll)


if __name__ == "__main__":  # pragma: no cover - exercised by --selftest only
    raise SystemExit(main())
