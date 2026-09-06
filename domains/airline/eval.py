"""Evaluator for the airline domain (tau-bench airline subset, MIT).

Contract (see domains/README.md):
    THRESHOLD                       score >= THRESHOLD counts as a pass
    load_tasks(split=None)          -> list[Task] from tasks.jsonl
    setup(task)                     reset the tool database before a run
    score(task, output)             1.0 if the live DB matches the expected end state, else 0.0
    is_hard_fail(task, trace)       forbidden write / transfer outside the expected action list

Scoring follows tau-bench's calculate_reward: replay the expected actions on a fresh load of
the database, hash both states with consistent_hash, and compare. If the task also lists
expected `outputs`, each must appear in the agent's final text (case-insensitive, commas
stripped). No LLM calls happen here.
"""


import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:  # runtime may load this file by path, not as a package
    sys.path.insert(0, str(_ROOT))

from domains.airline.fixtures import tools as _tools
from domains.airline.fixtures.tau_airline.env import data_hash, invoke, load_data

THRESHOLD = 1.0
TASKS_PATH = _HERE / "tasks.jsonl"

WRITE_TOOLS = frozenset(
    {
        "book_reservation",
        "update_reservation_flights",
        "update_reservation_baggages",
        "update_reservation_passengers",
        "cancel_reservation",
        "send_certificate",
    }
)
TRANSFER_TOOL = "transfer_to_human_agents"


@dataclass(frozen=True)
class Task:
    id: str
    input: dict[str, Any]  # {"user_id": str, "instruction": str}
    expected: dict[str, Any]  # {"actions": [{"name", "kwargs"}], "outputs": [str]}
    split: str  # train | search | holdout
    tags: list[str] = field(default_factory=list)


def load_tasks(split: str | None = None) -> list[Task]:
    tasks: list[Task] = []
    with TASKS_PATH.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            if split is not None and raw["split"] != split:
                continue
            tasks.append(Task(**raw))
    return tasks


def setup(task: Task) -> None:
    """Give this task a private, pristine airline database (see fixtures/tools.py).

    Must be called in the same thread/context that will run the task's tools and `score`;
    the runner guarantees this by doing all three inside one `asyncio.to_thread` call.
    """
    _tools.reset()


_expected_hash_cache: dict[str, str] = {}


def expected_hash(task: Task) -> str:
    """Hash of the DB after replaying task.expected.actions on a fresh load."""
    cached = _expected_hash_cache.get(task.id)
    if cached is not None:
        return cached
    data = load_data()
    for action in task.expected.get("actions", []):
        invoke(data, action["name"], action["kwargs"])
    digest = data_hash(data)
    _expected_hash_cache[task.id] = digest
    return digest


def _output_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    return json.dumps(output)


def _outputs_present(task: Task, output: Any) -> bool:
    text = _output_text(output).lower().replace(",", "")
    return all(expected.lower() in text for expected in task.expected.get("outputs", []))


def score(task: Task, output: Any) -> float:
    """1.0 iff the live DB equals the expected end state (and any expected outputs appear)."""
    if _tools.db_hash() != expected_hash(task):
        return 0.0
    if not _outputs_present(task, output):
        return 0.0
    return 1.0


def _is_expected(task: Task, name: str, args: dict[str, Any]) -> bool:
    return any(
        action["name"] == name and action["kwargs"] == args
        for action in task.expected.get("actions", [])
    )


def is_hard_fail(task: Task, trace: list[dict[str, Any]]) -> bool:
    """True if the trace contains a write outside the expected actions, or an unexpected
    transfer to a human. `trace` is a list of {"tool": str, "args": dict}."""
    expected_names = {action["name"] for action in task.expected.get("actions", [])}
    for step in trace:
        name = step.get("tool")
        args = step.get("args") or {}
        if name in WRITE_TOOLS and not _is_expected(task, name, args):
            return True
        if name == TRANSFER_TOOL and TRANSFER_TOOL not in expected_names:
            return True
    return False
