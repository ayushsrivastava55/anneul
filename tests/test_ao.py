"""Offline tests for anneal.ao: no daemon, no git, no subprocesses.

HTTP goes through an injected ``httpx.Client`` on a ``MockTransport``; every subprocess
(git and the accept command) goes through the single ``anneal.ao._run`` seam, which the
fake below replaces so we can assert on the exact argv, including worktree cleanup.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from anneal import ao

SHA_EMPTY = "0" * 40  # base_ref tip: AO made the branch, the agent has not committed yet
SHA_WORK = "a" * 40


def session_body(session_id: str = "anneal-99", **over: object) -> dict[str, object]:
    """The daemon's ``{"session": {...}}`` envelope, as observed on 2026-09-05."""
    session = {
        "id": session_id,
        "projectId": "anneal",
        "kind": "worker",
        "harness": "claude-code",
        "displayName": "probe",
        "status": "idle",
        "isTerminated": False,
        "branch": "ao/probe",
        "prs": [],
    }
    session.update(over)
    return {"session": session, "promptBytes": 31, "systemPromptBytes": 7980}


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class FakeRun:
    """Records every argv and answers from a scripted table of returncodes/stdout."""

    def __init__(self, script=None) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self.script = script or {}

    def __call__(self, cmd, *, cwd=None, timeout_s=None):
        self.calls.append((list(cmd), str(cwd) if cwd else None))
        key = self.key(cmd)
        # default: a resolved tip is NOT an ancestor of main, i.e. the agent did commit
        rc, out = self.script.get(key, (1, "") if key.startswith("merge-base") else (0, ""))
        if isinstance(rc, Exception):
            raise rc
        return subprocess.CompletedProcess(cmd, rc, out, "")

    @staticmethod
    def key(cmd: list[str]) -> str:
        """Collapse an argv to a stable key: git subcommand+ref, or 'accept'."""
        if cmd[0] == "git":
            rest = cmd[3:]  # drop ["git", "-C", <repo>]
            if rest[0] == "rev-parse":
                return f"rev-parse:{rest[-1]}"
            if rest[0] == "merge-base":
                return f"merge-base:{rest[-2]}"
            return ":".join(rest[:2])
        return "accept"

    def argvs(self) -> list[list[str]]:
        return [c for c, _ in self.calls]

    def ran(self, needle: str) -> list[list[str]]:
        return [c for c in self.argvs() if needle in " ".join(c)]


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Make polling instant and time deterministic: each monotonic() call advances 10s."""
    slept: list[float] = []
    monkeypatch.setattr(ao.time, "sleep", slept.append)
    ticks = iter(range(0, 100000, 10))
    monkeypatch.setattr(ao.time, "monotonic", lambda: float(next(ticks)))
    return slept


# --- spawn_worker -------------------------------------------------------------------------


def test_spawn_rejects_name_over_20_chars_before_any_http() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("spawn must validate the name before calling the daemon")

    with pytest.raises(ValueError, match="21 chars, max 20"):
        ao.spawn_worker("x" * 21, "ao/x", "hi", client=make_client(handler))


def test_spawn_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ao.spawn_worker("   ", "ao/x", "hi")


def test_spawn_posts_the_observed_payload_and_returns_the_session_id() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json=session_body("anneal-42"))

    session_id = ao.spawn_worker(
        "tool-add", "ao/tool-add", "make a tool", client=make_client(handler)
    )

    assert session_id == "anneal-42"
    assert seen["method"] == "POST"
    assert seen["url"] == f"{ao.base_url()}/api/v1/sessions"
    assert seen["body"] == {
        "projectId": "anneal",
        "displayName": "tool-add",
        "kind": "worker",
        "harness": "claude-code",
        "branch": "ao/tool-add",
        "prompt": "make a tool",
    }


def test_spawn_raises_aoerror_with_the_daemon_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": "bad_request", "code": "AGENT_REQUIRED", "message": "nope"}
        )

    with pytest.raises(ao.AOError, match="AGENT_REQUIRED"):
        ao.spawn_worker("tool-add", "ao/tool-add", "x", client=make_client(handler))


# --- send / status ------------------------------------------------------------------------


def test_send_uses_the_text_field_on_the_conversation_endpoint() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(202, json={"turnId": "t-1", "state": "running"})

    assert ao.send("anneal-42", "ping", client=make_client(handler)) == "t-1"
    assert seen["url"].endswith("/api/v1/sessions/anneal-42/conversation/messages")
    assert seen["body"] == {"text": "ping"}


def test_status_reads_the_session_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=session_body("anneal-42", status="working"))

    assert ao.status("anneal-42", client=make_client(handler)) == "working"


# --- wait_for_branch ----------------------------------------------------------------------


def wait(fake: FakeRun, monkeypatch: pytest.MonkeyPatch, **kw):
    monkeypatch.setattr(ao, "_run", fake)
    return ao.wait_for_branch(
        "ao/tool-add", ["pytest", "-q"], kw.pop("timeout_s", 100.0), 15.0, repo=Path("/repo"), **kw
    )


def test_waits_while_the_branch_only_holds_the_base_commit(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """AO creates the branch at spawn time; a tip equal to main must not trigger accept."""
    counter = {"n": 0}
    fake = FakeRun()

    def script(cmd, *, cwd=None, timeout_s=None):
        key = FakeRun.key(cmd)
        if key == "rev-parse:main^{commit}":
            return subprocess.CompletedProcess(cmd, 0, SHA_EMPTY, "")
        if key == "rev-parse:ao/tool-add^{commit}":
            counter["n"] += 1
            # empty branch for two polls, then the worker's commit lands
            return subprocess.CompletedProcess(
                cmd, 0, SHA_EMPTY if counter["n"] <= 2 else SHA_WORK, ""
            )
        if key == f"merge-base:{SHA_EMPTY}":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return fake(cmd, cwd=cwd, timeout_s=timeout_s)

    monkeypatch.setattr(ao, "_run", script)
    ok, out = ao.wait_for_branch("ao/tool-add", ["pytest", "-q"], 100.0, 15.0, repo=Path("/repo"))

    assert ok is True
    assert len(no_sleep) == 2, "should have polled twice before the commit appeared"
    assert len(fake.ran("pytest")) == 1, "accept must run exactly once, on the new sha"
    assert fake.ran("worktree add")[0][-1] == SHA_WORK


def test_a_tip_reachable_from_main_is_not_accepted_even_with_a_different_sha(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """AO may branch from a commit newer than our local main; ancestry, not equality, decides."""
    fake = FakeRun(
        {
            "rev-parse:main^{commit}": (0, SHA_EMPTY),
            "rev-parse:ao/tool-add^{commit}": (0, SHA_WORK),
            f"merge-base:{SHA_WORK}": (0, ""),  # tip already reachable from main => no work
        }
    )
    ok, out = wait(fake, monkeypatch, timeout_s=25.0)

    assert ok is False
    assert "timed out" in out
    assert not fake.ran("pytest")


def test_accept_failure_returns_false_with_output_and_still_removes_the_worktree(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    fake = FakeRun(
        {
            "rev-parse:main^{commit}": (0, SHA_EMPTY),
            "rev-parse:ao/tool-add^{commit}": (0, SHA_WORK),
            "accept": (1, "1 failed"),
        }
    )
    ok, out = wait(fake, monkeypatch, timeout_s=10.0)

    assert ok is False
    assert "1 failed" in out
    assert fake.ran("worktree remove"), "the worktree must be removed on the failing path"


def test_accept_pass_removes_the_worktree_too(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    fake = FakeRun(
        {
            "rev-parse:main^{commit}": (0, SHA_EMPTY),
            "rev-parse:ao/tool-add^{commit}": (0, SHA_WORK),
            "accept": (0, "1 passed"),
        }
    )
    ok, out = wait(fake, monkeypatch)

    assert (ok, "1 passed" in out) == (True, True)
    removes = fake.ran("worktree remove")
    assert len(removes) == 1
    assert removes[0][-1] == fake.ran("worktree add")[0][-2], "same path added and removed"


def test_accept_runs_inside_the_detached_worktree_not_the_repo(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    fake = FakeRun(
        {
            "rev-parse:main^{commit}": (0, SHA_EMPTY),
            "rev-parse:ao/tool-add^{commit}": (0, SHA_WORK),
        }
    )
    wait(fake, monkeypatch)
    cwd = next(cwd for cmd, cwd in fake.calls if "pytest" in cmd)
    assert cwd is not None and cwd != "/repo" and cwd.endswith("/wt")


def test_worktree_is_removed_when_the_accept_command_times_out(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    fake = FakeRun(
        {
            "rev-parse:main^{commit}": (0, SHA_EMPTY),
            "rev-parse:ao/tool-add^{commit}": (0, SHA_WORK),
            "accept": (subprocess.TimeoutExpired(["pytest"], 1.0), ""),
        }
    )
    ok, out = wait(fake, monkeypatch, timeout_s=10.0)

    assert ok is False
    assert "timed out" in out
    assert fake.ran("worktree remove")


def test_missing_branch_falls_back_to_origin_after_a_fetch(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    fake = FakeRun(
        {
            "rev-parse:main^{commit}": (0, SHA_EMPTY),
            "rev-parse:ao/tool-add^{commit}": (1, ""),
            "rev-parse:origin/ao/tool-add^{commit}": (0, SHA_WORK),
        }
    )
    ok, _ = wait(fake, monkeypatch)

    assert ok is True
    assert fake.ran("fetch"), "must fetch before giving up on the local ref"


def test_times_out_without_ever_running_accept(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    fake = FakeRun(
        {"rev-parse:main^{commit}": (0, SHA_EMPTY), "rev-parse:ao/tool-add^{commit}": (1, "")}
    )
    ok, out = wait(fake, monkeypatch, timeout_s=25.0)

    assert ok is False
    assert "timed out" in out
    assert not fake.ran("pytest")
    assert not fake.ran("worktree add")


def test_gives_up_when_the_session_terminates_without_a_commit(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=session_body("anneal-42", status="terminated", isTerminated=True)
        )

    fake = FakeRun(
        {"rev-parse:main^{commit}": (0, SHA_EMPTY), "rev-parse:ao/tool-add^{commit}": (1, "")}
    )
    ok, out = wait(fake, monkeypatch, session_id="anneal-42", client=make_client(handler))

    assert ok is False
    assert "terminated" in out
    assert not no_sleep, "a terminated session must not be polled again"


def test_a_daemon_hiccup_does_not_abort_the_wait(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("daemon restarting")

    fake = FakeRun(
        {
            "rev-parse:main^{commit}": (0, SHA_EMPTY),
            "rev-parse:ao/tool-add^{commit}": (0, SHA_WORK),
            "accept": (0, "1 passed"),
        }
    )
    ok, _ = wait(fake, monkeypatch, session_id="anneal-42", client=make_client(handler))

    assert ok is True


# --- misc ---------------------------------------------------------------------------------


def test_base_url_honours_ao_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AO_PORT", "4321")
    assert ao.base_url() == "http://127.0.0.1:4321"


def test_selftest_is_not_collected_by_pytest() -> None:
    """The real end-to-end spawn must never run under `uv run pytest`."""
    assert not hasattr(ao, "test_selftest")
    assert ao._selftest.__name__.startswith("_")
