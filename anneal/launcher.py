"""Start and watch a run that was launched from the browser.

Creating an agent in the UI used to be a dead end: the page told you to open a terminal, the
agent did not appear on the index because the index only listed directories that already had
runs, and there was no way to start one. Everything after "create" happened somewhere else.

This module closes that. It starts `anneal run` as a separate process, writing into the same
runs directory the dashboard reads, and keeps a handle so the pages can say whether an agent is
running, has finished, or failed. A subprocess rather than a thread because a run loads models
and can be killed, and because a crash in it must not take the server with it.

State lives in memory and is deliberately not persisted: a restarted server has no business
claiming a run is still going. What *is* durable is the runs directory itself, which is the
only thing any page reads its numbers from.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from anneal import config

ROOT = Path(__file__).resolve().parent.parent
LOCAL_LADDER = ROOT / "specs" / "models.local.yaml"
HOSTED_LADDER = ROOT / "specs" / "models.yaml"

# What a run started from the browser is allowed to spend and how far it goes. A person
# clicking Run has not been asked for a budget, so these are the defaults the product picks.
DEFAULT_ITERATIONS = 2
DEFAULT_BUDGET = 2.00
# Tool modules a domain brings may keep state in module globals, which races under the default
# concurrency. A run nobody is supervising is run serially; it is slower and it is correct.
DEFAULT_CONCURRENCY = 1

LOG_NAME = "run.log"


def default_ladder() -> Path:
    """The local models unless the hosted ladder's key is actually set.

    Choosing this for the person is the point: they clicked Run, they did not ask to think
    about model providers, and picking the hosted ladder without a key would fail on the first
    call with an unset-key error.
    """
    return HOSTED_LADDER if (config.env("FRONTIER_API_KEY") or "").strip() else LOCAL_LADDER


@dataclass
class Run:
    """One launched run: the process, where it writes, and how it ended."""

    domain: str
    process: subprocess.Popen
    log_path: Path
    returncode: int | None = None

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    def status(self) -> str:
        """``running`` / ``done`` / ``failed``, as a page would say it."""
        if self.running:
            return "running"
        code = self.process.returncode
        return "done" if code == 0 else "failed"

    def tail(self, lines: int = 12) -> str:
        try:
            return "\n".join(self.log_path.read_text(encoding="utf-8").splitlines()[-lines:])
        except OSError:
            return ""


@dataclass
class Launcher:
    """Every run this server started, by domain. One at a time per agent."""

    runs_dir: Path
    domains_dir: Path
    active: dict[str, Run] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def status(self, domain: str) -> str | None:
        """``running`` / ``done`` / ``failed`` for a run this server started, else None."""
        run = self.active.get(domain)
        return run.status() if run else None

    def get(self, domain: str) -> Run | None:
        return self.active.get(domain)

    def start(self, domain: str, *, ladder: Path | None = None) -> Run:
        """Launch `anneal run` for one agent. Raises if it is already running."""
        with self._lock:
            existing = self.active.get(domain)
            if existing and existing.running:
                raise RuntimeError(f"{domain} is already running")
            domain_dir = self.domains_dir / domain
            if not domain_dir.is_dir():
                raise FileNotFoundError(f"no agent named {domain!r}")
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.runs_dir / f"{domain}.{LOG_NAME}"
            command = [
                sys.executable, "-m", "anneal.cli", "run", str(domain_dir),
                "--models", str(ladder or default_ladder()),
                "--runs-dir", str(self.runs_dir),
                "--iterations", str(DEFAULT_ITERATIONS),
                "--budget", f"{DEFAULT_BUDGET:.2f}",
                "--concurrency", str(DEFAULT_CONCURRENCY),
            ]
            environment = dict(os.environ)
            # one model in memory at a time: these runs share a laptop with the server
            environment.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
            environment.setdefault("OLLAMA_NUM_PARALLEL", "1")
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                command, stdout=handle, stderr=subprocess.STDOUT,
                cwd=str(ROOT), env=environment,
            )
            run = Run(domain=domain, process=process, log_path=log_path)
            self.active[domain] = run
            return run

    def stop(self, domain: str) -> bool:
        """Ask a running process to stop. True if there was one to stop."""
        run = self.active.get(domain)
        if not run or not run.running:
            return False
        run.process.terminate()
        return True
