"""Typed mutation operators keyed by failure class (``specs/failure_taxonomy.yaml``).

``apply(spec, issue, evidence, domain, client=None)`` picks the first operator listed for
the issue's class that has not been tried yet, applies it, and returns a new validated
``HarnessSpec`` with lineage set. One operator per call so the gate result is attributable.

Operators edit the spec (tool description overrides, nodes, topology, step budget, memory)
and prompts (a new version saved with label ``staging``, node refs bumped to it).
Evidence comes from the ``search`` split only; few-shot material comes from the ``train``
split only. This module never reads any other split.
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

from anneal import llm, prompts
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

# Structural knobs used by the orchestration operators. Steps a new node may take are added
# to the spec's step budget so appending a node never starves the nodes already there.
VALIDATOR_STEPS = 1
ESCALATE_STEPS = 1
PLANNER_STEPS = 2
CRITIC_STEPS = 4
STEP_BUDGET_BUMP = 4  # add_step_budget_and_critic raises the budget once, on top of the node
DEFAULT_SCHEMA_REF = "output_schema"  # attribute the runtime looks up on the eval module
ESCALATE_TOOL = "escalate"
MEMORY_TOP_K = 3
MEMORY_TOP_K_STEP = 2
# switch_topology only ever moves right along this ladder.
TOPOLOGY_LADDER = ("single", "planner_executor", "critic_loop")

Issue = dict[str, Any]
Evidence = list[dict[str, Any]]
Operator = Callable[[HarnessSpec, Issue, Evidence, Any, Any], HarnessSpec]


class NoOperatorAvailable(RuntimeError):
    """No operator for the issue's class is implemented, untried and applicable to this spec."""


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
# Reads go through anneal.prompts. Writes use a local writer so the new version is always
# above the node's current ref (anneal.prompts.save_version only does latest+1, which
# would yield v1 when the v1 file is missing locally).

split_ref = prompts.parse_ref


class LocalPromptStore:
    """Reads/writes ``<root>/<name>/vN.md`` plus a ``labels.json`` sidecar per prompt."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _dir(self, name: str) -> Path:
        return self.root / name

    def get(self, ref: str) -> str:
        """Prompt text for ``name@vN``; empty string when the file does not exist yet."""
        try:
            return prompts.get_prompt(ref, root=self.root)
        except FileNotFoundError:
            return ""

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


# --- structural helpers (shared by the orchestration operators) -------------------------

NODE_PROMPTS: dict[str, str] = {
    "validator": (
        "You are the validator. You call no tools. You are given the task and the agent's "
        "final answer. Check the answer against the required output structure and against the "
        "values in the task. Reply with the corrected final answer in the required structure, "
        "or the answer unchanged when it is already correct."
    ),
    "critic": (
        "You are the critic. You call no tools. Review the executor's tool calls and draft "
        "answer against the goal. Reply APPROVE on the first line when the work is complete "
        "and policy-safe; otherwise reply REVISE followed by the concrete corrections the "
        "executor must make, one per line."
    ),
    "escalate": (
        "You are the escalation node. You are reached when a rule blocks the action the user "
        'asked for. Reply with a JSON object {"escalate": <reason>} naming the rule that was '
        "hit and what a human needs to decide. Never perform the blocked action yourself."
    ),
    "planner": (
        "You are the planner. You call no tools. Write a short numbered plan of tool calls and "
        "checks for an executor that has the tools, flagging every rule that must be verified "
        "before a write. When asked to review the executor's result, reply exactly DONE if the "
        "task is complete, otherwise a revised numbered plan."
    ),
}

CITE_SECTION = """## Cite or abstain

Every value you assert in the final answer must come from a tool result or from the task text.
Name the source next to the value, e.g. `(from get_reservation_details)`. If nothing you have
read supports a value, do not guess it: say you could not verify it and state what is missing.
"""


def _prompt_name(spec: HarnessSpec, role: str) -> str:
    """Prompt name for a new ``role`` node, in the same namespace as the spec's first node."""
    name, _version = split_ref(spec.nodes[0].system_prompt_ref)
    head, sep, _tail = name.rpartition("/")
    return f"{head}/{role}" if sep else role


def _role_node(spec: HarnessSpec, role: str) -> Node | None:
    for node in spec.nodes:
        if node.role == role:
            return node
    return None


def _append_node(
    spec: HarnessSpec,
    op: str,
    *,
    name: str,
    role: str,
    model_tier: str,
    prompt_text: str,
    tools: list[str] | None = None,
    max_steps: int = 1,
    schema_ref: str | None = None,
) -> dict[str, Any]:
    """Spec data with a new node appended, its prompt saved and the step budget raised."""
    if any(node.name == name for node in spec.nodes):
        raise NoOperatorAvailable(f"{op}: spec {spec.id!r} already has a node named {name!r}")
    prompt = _prompt_name(spec, role)
    version = PROMPT_STORE.save_version(prompt, prompt_text, STAGING_LABEL)
    data = spec.model_dump()
    data["nodes"] = [
        *data["nodes"],
        {
            "name": name,
            "role": role,
            "model_tier": model_tier,
            "system_prompt_ref": f"{prompt}@v{version}",
            "tools": list(tools or []),
            "max_steps": max_steps,
            "schema_ref": schema_ref,
        },
    ]
    data["step_budget"] = int(data["step_budget"]) + max_steps
    return data


def _bump_prompt(data: dict[str, Any], node: Node, section: str, op: str) -> None:
    """Append ``section`` to ``node``'s prompt as a new version and repoint the node at it."""
    name, version = split_ref(node.system_prompt_ref)
    base = PROMPT_STORE.get(node.system_prompt_ref)
    if section.strip() in base:
        raise NoOperatorAvailable(f"{op}: node {node.name!r} already carries this instruction")
    text = f"{base.rstrip()}\n\n{section}" if base.strip() else section
    new_version = PROMPT_STORE.save_version(name, text, STAGING_LABEL, at_least=version + 1)
    for entry in data["nodes"]:
        if entry["name"] == node.name:
            entry["system_prompt_ref"] = f"{name}@v{new_version}"


# --- operator: add_validator_node -------------------------------------------------------


def add_validator_node(
    spec: HarnessSpec, issue: Issue, evidence: Evidence, domain: Any, client: Any
) -> HarnessSpec:
    """Append a cheap validator node that checks the final output against the schema."""
    data = _append_node(
        spec,
        "add_validator_node",
        name="validator",
        role="validator",
        model_tier="cheap",
        prompt_text=NODE_PROMPTS["validator"],
        max_steps=VALIDATOR_STEPS,
        schema_ref=str(issue.get("schema_ref") or DEFAULT_SCHEMA_REF),
    )
    return HarnessSpec.model_validate(data)


# --- operator: add_cite_or_abstain ------------------------------------------------------


def add_cite_or_abstain(
    spec: HarnessSpec, issue: Issue, evidence: Evidence, domain: Any, client: Any
) -> HarnessSpec:
    """Require every asserted value to cite a tool result, or to abstain (prompt edit)."""
    node = _node(spec, issue["node"])
    data = spec.model_dump()
    _bump_prompt(data, node, CITE_SECTION, "add_cite_or_abstain")
    return HarnessSpec.model_validate(data)


# --- operator: add_step_budget_and_critic -----------------------------------------------


def add_step_budget_and_critic(
    spec: HarnessSpec, issue: Issue, evidence: Evidence, domain: Any, client: Any
) -> HarnessSpec:
    """Raise the step budget once and add a critic, moving to the critic_loop topology."""
    data = _append_node(
        spec,
        "add_step_budget_and_critic",
        name="critic",
        role="critic",
        model_tier=TIER,
        prompt_text=NODE_PROMPTS["critic"],
        max_steps=CRITIC_STEPS,
    )
    data["step_budget"] = int(data["step_budget"]) + STEP_BUDGET_BUMP
    data["topology"] = "critic_loop"  # the critic only runs when the topology reviews
    return HarnessSpec.model_validate(data)


# --- operator: add_memory ---------------------------------------------------------------


def add_memory(
    spec: HarnessSpec, issue: Issue, evidence: Evidence, domain: Any, client: Any
) -> HarnessSpec:
    """Turn on episodic memory, or widen ``top_k`` when it is already on."""
    data = spec.model_dump()
    memory = dict(data["memory"])
    if memory.get("enabled") and memory.get("kind") == "episodic":
        memory["top_k"] = int(memory.get("top_k") or 0) + MEMORY_TOP_K_STEP
    else:
        memory.update(
            enabled=True, kind="episodic", top_k=int(issue.get("top_k") or MEMORY_TOP_K)
        )
    data["memory"] = memory
    return HarnessSpec.model_validate(data)


# --- operator: add_escalation_node ------------------------------------------------------


def _escalation_tools(issue: Issue, domain: Any) -> list[str]:
    """The hand-off tool: the issue's explicit one, else a manifest tool named ``escalate``."""
    explicit = issue.get("tool")
    if explicit:
        return [str(explicit)]
    manifest = getattr(domain, "tools", None)
    names = [tool.name for tool in getattr(manifest, "tools", []) or []]
    return [ESCALATE_TOOL] if ESCALATE_TOOL in names else []


def _escalation_section(tools: list[str]) -> str:
    hand_off = f"call `{tools[0]}`" if tools else "hand the task to the escalate node"
    return (
        "## Escalate instead of acting\n\n"
        "Before any action that changes state, check the rules in the goal that cover it. "
        f"If a rule is not satisfied, or you cannot verify that it is, do not act: {hand_off} "
        "and state which rule was hit and what a human must decide. Escalating is always "
        "preferred to acting on an unverified assumption.\n"
    )


def add_escalation_node(
    spec: HarnessSpec, issue: Issue, evidence: Evidence, domain: Any, client: Any
) -> HarnessSpec:
    """Append an escalate node and tell the failing node to use it when a rule fails."""
    node = _node(spec, issue["node"])
    tools = _escalation_tools(issue, domain)
    data = _append_node(
        spec,
        "add_escalation_node",
        name="escalate",
        role="escalate",
        model_tier="cheap",
        prompt_text=NODE_PROMPTS["escalate"],
        tools=tools,
        max_steps=ESCALATE_STEPS,
    )
    _bump_prompt(data, node, _escalation_section(tools), "add_escalation_node")
    return HarnessSpec.model_validate(data)


# --- operator: switch_topology ----------------------------------------------------------


def _carry_prompt(spec: HarnessSpec, role: str) -> str:
    """New node's prompt: its role brief plus the executor's prompt carried across."""
    brief = NODE_PROMPTS[role]
    executor = _role_node(spec, "executor")
    base = PROMPT_STORE.get(executor.system_prompt_ref) if executor else ""
    if not base.strip():
        return brief
    return f"{brief}\n\n## Context carried from the executor prompt\n\n{base.rstrip()}\n"


def _grow_to(spec: HarnessSpec, target: str) -> dict[str, Any]:
    """Spec data holding the node ``target`` needs, added only when it is not there yet."""
    role, max_steps = {
        "planner_executor": ("planner", PLANNER_STEPS),
        "critic_loop": ("critic", CRITIC_STEPS),
    }[target]
    if _role_node(spec, role) is not None:
        return spec.model_dump()
    return _append_node(
        spec,
        "switch_topology",
        name=role,
        role=role,
        model_tier=TIER,
        prompt_text=_carry_prompt(spec, role),
        max_steps=max_steps,
    )


def switch_topology(
    spec: HarnessSpec, issue: Issue, evidence: Evidence, domain: Any, client: Any
) -> HarnessSpec:
    """Move one rung right along single -> planner_executor -> critic_loop. Never back."""
    if spec.topology not in TOPOLOGY_LADDER:
        raise NoOperatorAvailable(
            f"switch_topology: topology {spec.topology!r} is not on the ladder "
            f"{list(TOPOLOGY_LADDER)}"
        )
    index = TOPOLOGY_LADDER.index(spec.topology) + 1
    if index >= len(TOPOLOGY_LADDER):
        raise NoOperatorAvailable(
            f"switch_topology: {spec.topology!r} is the top of the ladder; the ladder never "
            "moves back down"
        )
    target = TOPOLOGY_LADDER[index]
    data = _grow_to(spec, target)
    data["topology"] = target
    return HarnessSpec.model_validate(data)


# --- operator: synthesize_tool (deferred) -----------------------------------------------


def synthesize_tool(
    spec: HarnessSpec, issue: Issue, evidence: Evidence, domain: Any, client: Any
) -> HarnessSpec:
    """Deferred: an AO worker writes the missing tool plus its tests, accepted on pytest."""
    raise NotImplementedError(
        "synthesize_tool is task 2.5: it spawns an AO worker session (anneal/ao.py) to write "
        "the missing tool and its tests, then accepts the mutation on pytest. Not implemented "
        "in task 2.2 (ops-full)."
    )


# --- registry + apply ---------------------------------------------------------------------

OPERATORS: dict[str, Operator] = {
    "rewrite_tool_desc": rewrite_tool_desc,
    "add_fewshots": add_fewshots,
    "add_validator_node": add_validator_node,
    "add_cite_or_abstain": add_cite_or_abstain,
    "add_step_budget_and_critic": add_step_budget_and_critic,
    "add_memory": add_memory,
    "add_escalation_node": add_escalation_node,
    "switch_topology": switch_topology,
}

# Operators declared in the taxonomy that are deliberately not registered yet: select_operator
# skips them, so a class whose only operator is deferred raises NoOperatorAvailable.
DEFERRED: dict[str, str] = {
    "synthesize_tool": "task 2.5 - AO worker writes the tool + tests (anneal/ao.py)",
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
