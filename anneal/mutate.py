"""Typed mutation operators keyed by failure class (``specs/failure_taxonomy.yaml``).

``apply(spec, issue, evidence, domain, client=None)`` picks the first operator listed for
the issue's class that has not been tried yet, applies it, and returns a new validated
``HarnessSpec`` with lineage set. One operator per call so the gate result is attributable.

Operators here only edit the spec (tool description overrides) and prompts (new version
saved with label ``staging``). Evidence comes from the ``search`` split only; few-shot
material comes from the ``train`` split only. This module never reads any other split.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from anneal import llm
from anneal.spec import HarnessSpec, Node
from anneal.tracing import llm_span

logger = logging.getLogger("anneal.mutate")

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = ROOT / "specs" / "failure_taxonomy.yaml"
PROMPTS_DIR = ROOT / "prompts"

TIER = "mid"  # tier name from specs/models.yaml; never a model id
EVIDENCE_SPLIT = "search"
FEWSHOT_SPLIT = "train"
STAGING_LABEL = "staging"
MAX_EVIDENCE = 5
MAX_TRAIN_TASKS = 6
MAX_CHARS_PER_ITEM = 2500

Issue = dict[str, Any]
Evidence = list[dict[str, Any]]
Operator = Callable[[HarnessSpec, Issue, Evidence, Any, Any], HarnessSpec]


class NoOperatorAvailable(RuntimeError):
    """Every operator mapped to the issue's class has been tried or is not implemented."""


# --- taxonomy ------------------------------------------------------------------------


@lru_cache(maxsize=4)
def load_taxonomy(path: Path | str | None = None) -> dict[str, Any]:
    """Parse specs/failure_taxonomy.yaml (or ``path``). Cached per path."""
    data = yaml.safe_load(Path(path or TAXONOMY_PATH).read_text())
    if not isinstance(data, dict) or "classes" not in data or "operators" not in data:
        raise ValueError(f"{path or TAXONOMY_PATH}: expected top-level 'classes' and 'operators'")
    return data


def operators_for(failure_class: str, path: Path | str | None = None) -> list[str]:
    """Operator names for ``failure_class`` in the order the taxonomy lists them."""
    for cls in load_taxonomy(path)["classes"]:
        if cls["id"] == failure_class:
            return list(cls["operators"])
    known = [c["id"] for c in load_taxonomy(path)["classes"]]
    raise KeyError(f"unknown failure class {failure_class!r}; known: {known}")


# --- prompt versions -------------------------------------------------------------------
# Minimal local version writer matching the anneal/prompts.py contract
# (prompts/<name>/<version>.md). Swap PROMPT_STORE for anneal.prompts once it lands.

_REF_RE = re.compile(r"^(?P<name>.+)@v(?P<version>\d+)$")


def split_ref(ref: str) -> tuple[str, int]:
    """``name@vN`` -> (name, N)."""
    match = _REF_RE.match(ref)
    if not match:
        raise ValueError(f"system_prompt_ref must look like name@vN, got {ref!r}")
    return match.group("name"), int(match.group("version"))


class LocalPromptStore:
    """Reads/writes ``<root>/<name>/vN.md`` plus a ``labels.json`` sidecar per prompt."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _dir(self, name: str) -> Path:
        return self.root / name

    def get(self, ref: str) -> str:
        """Prompt text for ``name@vN``; empty string when the file does not exist yet."""
        name, version = split_ref(ref)
        path = self._dir(name) / f"v{version}.md"
        return path.read_text() if path.exists() else ""

    def versions(self, name: str) -> list[int]:
        folder = self._dir(name)
        if not folder.exists():
            return []
        found = (re.fullmatch(r"v(\d+)\.md", p.name) for p in folder.iterdir())
        return sorted(int(m.group(1)) for m in found if m)

    def save_version(self, name: str, text: str, label: str, *, at_least: int = 1) -> int:
        """Write the next version (>= ``at_least``), record ``label``, sync to Neatlogs."""
        existing = self.versions(name)
        version = max([at_least, *(v + 1 for v in existing)])
        folder = self._dir(name)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"v{version}.md").write_text(text)
        labels_path = folder / "labels.json"
        labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
        labels[str(version)] = label
        labels_path.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n")
        _sync_to_neatlogs(name, text, label, version)
        return version

    def label_of(self, name: str, version: int) -> str | None:
        labels_path = self._dir(name) / "labels.json"
        if not labels_path.exists():
            return None
        return json.loads(labels_path.read_text()).get(str(version))


def _sync_to_neatlogs(name: str, text: str, label: str, version: int) -> None:
    """Best-effort push of the new version to the Neatlogs prompt registry."""
    api_key = (os.environ.get("NEATLOGS_API_KEY") or "").strip()
    if not api_key:
        return
    try:
        import neatlogs

        from anneal.tracing import init_tracing

        init_tracing()  # idempotent; the SDK's prompt client reads the key set by init
        commit = f"anneal mutate: local v{version}"
        try:
            neatlogs.save_as_version(prompt_name=name, content=text, labels=[label],
                                     commit_message=commit)
        except neatlogs.PromptNotFoundError:
            neatlogs.create_prompt(name=name, prompt=text, type="text", labels=[label],
                                   commit_message=commit)
    except Exception as exc:  # noqa: BLE001 - registry sync must never break a mutation
        logger.warning("neatlogs prompt sync failed for %s v%s: %s", name, version, exc)


PROMPT_STORE = LocalPromptStore(PROMPTS_DIR)


# --- shared helpers ----------------------------------------------------------------


@llm_span("mutate.llm")
def _complete(client: Any, messages: list[dict[str, Any]]) -> str:
    """One text completion through the gateway (``client`` overrides it for tests)."""
    text, _usage = llm.chat(TIER, messages, client=client)
    return text.strip()


def _node(spec: HarnessSpec, name: str) -> Node:
    for node in spec.nodes:
        if node.name == name:
            return node
    raise KeyError(f"issue names node {name!r}, spec has {[n.name for n in spec.nodes]}")


def _check_evidence(evidence: Evidence) -> None:
    for item in evidence:
        split = item.get("split")
        if split is not None and split != EVIDENCE_SPLIT:
            raise ValueError(
                f"evidence must come from the {EVIDENCE_SPLIT!r} split, got {split!r}"
            )


def _dumps(obj: Any) -> str:
    text = json.dumps(obj, default=str, ensure_ascii=False)
    return text if len(text) <= MAX_CHARS_PER_ITEM else text[:MAX_CHARS_PER_ITEM] + "…"


def _format_evidence(evidence: Evidence) -> str:
    keep = ("task_id", "output", "trace", "expected")
    rows = [{k: item[k] for k in keep if k in item} for item in evidence[:MAX_EVIDENCE]]
    return "\n".join(_dumps(row) for row in rows)


def _task_dict(task: Any) -> dict[str, Any]:
    if hasattr(task, "model_dump"):
        return dict(task.model_dump())
    if dataclasses.is_dataclass(task) and not isinstance(task, type):
        return dataclasses.asdict(task)
    if isinstance(task, dict):
        return dict(task)
    return dict(vars(task))


def _format_tasks(tasks: list[Any]) -> str:
    keep = ("id", "input", "expected")
    rows = [{k: d[k] for k in keep if k in d} for d in map(_task_dict, tasks)]
    return "\n".join(_dumps(row) for row in rows)


def _goal_excerpt(domain: Any) -> str:
    goal = str(getattr(domain, "goal", "") or "")
    return goal[:MAX_CHARS_PER_ITEM]


# --- operator: rewrite_tool_desc ------------------------------------------------------


def _failing_tool(issue: Issue, node: Node, evidence: Evidence) -> str:
    """The tool the issue is about: explicit ``issue.tool``, else most-called in evidence."""
    explicit = issue.get("tool")
    if explicit:
        return str(explicit)
    counts: Counter[str] = Counter(
        str(call.get("tool"))
        for item in evidence
        for call in item.get("trace") or []
        if call.get("tool") in node.tools
    )
    if counts:
        return counts.most_common(1)[0][0]
    if node.tools:
        return node.tools[0]
    raise ValueError(f"node {node.name!r} has no tools; rewrite_tool_desc does not apply")


def _tool_description(domain: Any, tool: str) -> str:
    manifest = getattr(domain, "tools", None)
    for spec in getattr(manifest, "tools", []) or []:
        if spec.name == tool:
            return str(spec.description)
    return ""


def rewrite_tool_desc(
    spec: HarnessSpec, issue: Issue, evidence: Evidence, domain: Any, client: Any
) -> HarnessSpec:
    """LLM rewrites the failing tool's description with contrastive use/do-not-use text."""
    node = _node(spec, issue["node"])
    tool = _failing_tool(issue, node, evidence)
    current = spec.tool_overrides.get(tool) or _tool_description(domain, tool)
    messages = [
        {
            "role": "system",
            "content": (
                "You improve tool descriptions for an LLM agent. Given the agent's goal, a "
                "tool's current description and traces where the agent misused it, write a "
                "replacement description. It must state what the tool does, then a 'Use "
                "when:' line and a 'Do not use when:' line grounded in the failures. Reply "
                "with the description text only, no preamble, no markdown fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Goal:\n{_goal_excerpt(domain)}\n\nTool: {tool}\n"
                f"Current description: {current}\n\nFailure class: {issue.get('class')}\n"
                f"Failing runs (one JSON per line):\n{_format_evidence(evidence)}"
            ),
        },
    ]
    text = _complete(client, messages)
    if not text:
        raise RuntimeError("rewrite_tool_desc: model returned an empty description")
    data = spec.model_dump()
    data["tool_overrides"] = {**data.get("tool_overrides", {}), tool: text}
    return HarnessSpec.model_validate(data)


# --- operator: add_fewshots -------------------------------------------------------------


def _fewshot_messages(
    domain: Any, issue: Issue, evidence: Evidence, base: str, train: list[Any]
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You write worked examples for an LLM agent's system prompt. You are given "
                "the goal, the current system prompt, failing runs, and labelled tasks with "
                "their expected outcome. Write 2-3 short examples, each with the task, the "
                "correct reasoning and the correct action/answer, targeted at the failure "
                "class. Use only the labelled tasks as sources. Reply in markdown, examples "
                "only, no preamble."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Goal:\n{_goal_excerpt(domain)}\n\nCurrent system prompt:\n{base or '(empty)'}"
                f"\n\nFailure class: {issue.get('class')}\n"
                f"Failing runs (one JSON per line):\n{_format_evidence(evidence)}\n\n"
                f"Labelled tasks (one JSON per line):\n{_format_tasks(train)}"
            ),
        },
    ]


def add_fewshots(
    spec: HarnessSpec, issue: Issue, evidence: Evidence, domain: Any, client: Any
) -> HarnessSpec:
    """LLM writes 2-3 corrected examples from train tasks; saved as a new prompt version."""
    node = _node(spec, issue["node"])
    name, version = split_ref(node.system_prompt_ref)
    base = PROMPT_STORE.get(node.system_prompt_ref)
    train = list(domain.eval.load_tasks(FEWSHOT_SPLIT))[:MAX_TRAIN_TASKS]
    examples = _complete(client, _fewshot_messages(domain, issue, evidence, base, train))
    if not examples:
        raise RuntimeError("add_fewshots: model returned no examples")
    section = f"## Worked examples\n\n{examples}\n"
    text = f"{base.rstrip()}\n\n{section}" if base.strip() else section
    new_version = PROMPT_STORE.save_version(name, text, STAGING_LABEL, at_least=version + 1)
    data = spec.model_dump()
    for entry in data["nodes"]:
        if entry["name"] == node.name:
            entry["system_prompt_ref"] = f"{name}@v{new_version}"
    return HarnessSpec.model_validate(data)


# --- registry + apply ---------------------------------------------------------------------

OPERATORS: dict[str, Operator] = {
    "rewrite_tool_desc": rewrite_tool_desc,
    "add_fewshots": add_fewshots,
}


def select_operator(issue: Issue) -> str:
    """First operator for the issue's class that is implemented and not yet tried."""
    tried = set(issue.get("operators_tried") or [])
    for name in operators_for(issue["class"]):
        if name in OPERATORS and name not in tried:
            return name
    raise NoOperatorAvailable(
        f"issue {issue.get('id')!r} ({issue['class']}): no untried implemented operator; "
        f"tried={sorted(tried)} implemented={sorted(OPERATORS)}"
    )


def _with_lineage(mutated: HarnessSpec, parent: HarnessSpec, op: str, issue: Issue) -> HarnessSpec:
    iteration = (parent.lineage.iteration if parent.lineage else 0) + 1
    data = mutated.model_dump()
    data["id"] = f"{parent.id}-{op}-i{iteration}"
    data["lineage"] = {
        "parent": parent.id,
        "iteration": iteration,
        "operator": op,
        "ledger_issue": issue.get("id"),
    }
    return HarnessSpec.model_validate(data)


def apply(
    spec: HarnessSpec, issue: Issue, evidence: Evidence, domain: Any, *, client: Any = None
) -> HarnessSpec:
    """Apply one operator for ``issue`` and return the mutated, validated spec."""
    _check_evidence(evidence)
    _node(spec, issue["node"])
    name = select_operator(issue)
    logger.info(json.dumps({"event": "mutate.apply", "issue": issue.get("id"),
                            "class": issue.get("class"), "operator": name, "parent": spec.id}))
    mutated = OPERATORS[name](spec, issue, evidence, domain, client)
    return _with_lineage(mutated, spec, name, issue)
