"""Evaluator for the invoices domain (AP invoice triage).

Contract (see domains/invoices/EVAL_CONTRACT.md and domains/README.md):
    THRESHOLD                       score >= THRESHOLD counts as a pass
    load_tasks(split=None)          -> list[Task] from tasks.jsonl
    setup(task)                     reset the ledger before a run
    score(task, output)             0.5 * all five fields exact + 0.5 * correct decision
    is_hard_fail(task, trace)       posted an invoice that had to be escalated, or posted twice

`expected` records what is *printed on the invoice*, not what the PO says: on a currency
mismatch the expected currency is the invoice's, and on a tolerance breach the expected total
is the printed total. Only the decision reflects the rules. No LLM calls happen here.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:  # runtime may load this file by path, not as a package
    sys.path.insert(0, str(_ROOT))

from domains.invoices.fixtures import tools as _tools

THRESHOLD = 1.0
TASKS_PATH = _HERE / "tasks.jsonl"

FIELDS = ("vendor", "invoice_number", "po_number", "currency", "total")
POST_TOOL = "post_entry"
_NULLISH = {"", "null", "none", "n/a", "na", "not provided", "unknown", "-"}
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class Task:
    id: str
    input: dict[str, Any]  # {"invoice_text": str}
    expected: dict[str, Any]  # five fields + {"decision": str, "rule": str | None}
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
    """Reset the shared AP store so `task` starts from the pristine ledger."""
    _tools.reset()


def _as_dict(output: Any) -> dict[str, Any]:
    """Coerce whatever the harness produced into a dict; {} if it is not JSON at all."""
    if isinstance(output, dict):
        return output
    if not isinstance(output, str):
        return {}
    text = output.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    match = _OBJECT_RE.search(text)
    for candidate in (text, match.group(0) if match else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _norm_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _NULLISH else text


def _norm_total(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    text = str(value).strip().replace(",", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    try:
        return round(float(text), 2)
    except ValueError:
        return None


def _fields_match(expected: dict[str, Any], got: dict[str, Any]) -> bool:
    for name in FIELDS:
        if name == "total":
            if _norm_total(got.get(name)) != _norm_total(expected[name]):
                return False
        elif _norm_text(got.get(name)) != _norm_text(expected[name]):
            return False
    return True


def score(task: Task, output: Any) -> float:
    """0.5 for extracting all five fields exactly, 0.5 for the right decision."""
    got = _as_dict(output)
    total = 0.5 if _fields_match(task.expected, got) else 0.0
    decision = str(got.get("decision", "")).strip().lower()
    if decision == task.expected["decision"]:
        total += 0.5
    return total


def is_hard_fail(task: Task, trace: list[dict[str, Any]]) -> bool:
    """True if the invoice had to be escalated but was posted, or if it was posted twice.

    `trace` is a list of {"tool": str, "args": dict, ...} in call order.
    """
    posts = sum(1 for step in trace if step.get("tool") == POST_TOOL)
    if posts > 1:
        return True
    return posts >= 1 and task.expected["decision"] == "escalate"
