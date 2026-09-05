"""One-shot generator for tasks.jsonl and tools.yaml. Ran once with seed 0; DO NOT RERUN.

Regenerating would reshuffle the train/search/holdout split and invalidate every number
reported from runs/. Kept only so the selection is auditable.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from domains.airline.fixtures.tau_airline.env import TOOLS_BY_NAME
from domains.airline.fixtures.tau_airline.tasks_test import TASKS

DOMAIN = ROOT / "domains" / "airline"
SEED = 0
N_PICK, N_TRAIN, N_SEARCH = 40, 20, 10
WRITES = {
    "book_reservation": "book",
    "cancel_reservation": "cancel",
    "update_reservation_flights": "update_flights",
    "update_reservation_baggages": "update_baggages",
    "update_reservation_passengers": "update_passengers",
    "send_certificate": "certificate",
}


def tags_for(task) -> list[str]:
    names = [a.name for a in task.actions]
    tags = sorted({WRITES[n] for n in names if n in WRITES})
    if "transfer_to_human_agents" in names:
        tags.append("transfer")
    tags.append("write" if tags and tags != ["transfer"] else "no_write")
    if task.outputs:
        tags.append("has_outputs")
    if len(names) > 2:
        tags.append("multi_step")
    return tags


def write_tasks() -> None:
    rng = random.Random(SEED)
    order = list(range(len(TASKS)))
    rng.shuffle(order)
    chosen = order[:N_PICK]
    rows = []
    for pos, idx in enumerate(chosen):
        task = TASKS[idx]
        split = "train" if pos < N_TRAIN else "search" if pos < N_TRAIN + N_SEARCH else "holdout"
        rows.append(
            {
                "id": f"airline-{idx:02d}",
                "input": {"user_id": task.user_id, "instruction": task.instruction},
                "expected": {
                    "actions": [{"name": a.name, "kwargs": a.kwargs} for a in task.actions],
                    "outputs": list(task.outputs),
                },
                "split": split,
                "tags": tags_for(task),
            }
        )
    with (DOMAIN / "tasks.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def write_tools_yaml() -> None:
    tools = []
    for name, tool in TOOLS_BY_NAME.items():
        fn = tool.get_info()["function"]
        tools.append(
            {
                "name": name,
                "description": fn["description"],
                "args": fn["parameters"],
                "impl": f"python:domains.airline.fixtures.tools.{name}",
                "mutates": name in WRITES,
            }
        )
    tools.sort(key=lambda t: (t["mutates"], t["name"]))
    header = (
        "# Airline tools (tau-bench airline, MIT). Generated once by fixtures/make_tasks.py.\n"
        "# `mutates: true` tools change the database; eval.is_hard_fail flags any such call\n"
        "# that is not in the task's expected action list.\n"
    )
    body = yaml.safe_dump({"tools": tools}, sort_keys=False, width=100)
    (DOMAIN / "tools.yaml").write_text(header + body)


if __name__ == "__main__":
    write_tasks()
    write_tools_yaml()
