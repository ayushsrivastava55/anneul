"""Architect: goal.md + tools.yaml -> N candidate harness specs (iteration 0).

Topologies come from a fixed, deterministic menu so a run always compares at least
``single`` and ``planner_executor``. The LLM is only asked to write the per-node system
prompts; those are versioned through :mod:`anneal.prompts` and referenced by
``system_prompt_ref``. Nothing here reads tasks of any split.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

import anneal.prompts as prompts
from anneal import llm
from anneal.spec import HarnessSpec, Lineage, ModelTier, Node, Role, ToolsManifest
from anneal.tracing import llm_span, node_span

STEP_BUDGET = 12
OPERATOR = "architect"

# role -> (model tier, one-line brief handed to the prompt-writing LLM)
NODE_TIER: dict[Role, ModelTier] = {
    "planner": "frontier",
    "executor": "mid",
    "critic": "cheap",
    "validator": "cheap",
    "router": "cheap",
    "escalate": "cheap",
}
NODE_BRIEF: dict[str, str] = {
    "planner": (
        "The PLANNER sees the task and the tool list but calls no tools. It writes a short,"
        " numbered plan of tool calls and checks for the executor, flagging policy rules that"
        " must be verified before any write."
    ),
    "executor": (
        "The EXECUTOR has every tool. It follows the goal (and any plan it is given), makes one"
        " tool call at a time, reads before it writes, never fabricates ids or values, and"
        " ends with the final answer the goal asks for."
    ),
    "critic": (
        "The CRITIC calls no tools. It reviews the executor's tool calls and draft answer"
        " against the goal, replies APPROVE when the work is complete and policy-safe, otherwise"
        " lists the concrete corrections the executor must make."
    ),
}

# (topology, [(node name, role, max_steps)]) -- max_steps per candidate sum to <= STEP_BUDGET
MENU: list[tuple[str, list[tuple[str, Role, int]]]] = [
    ("single", [("executor", "executor", STEP_BUDGET)]),
    ("planner_executor", [("planner", "planner", 2), ("executor", "executor", 10)]),
    ("critic_loop", [("executor", "executor", 8), ("critic", "critic", 4)]),
]


class DomainLike(Protocol):
    """What Architect needs from a loaded domain (see anneal/domain.py)."""

    name: str
    goal: str
    tools: ToolsManifest


def _tool_manifest_text(tools: ToolsManifest) -> str:
    lines = []
    for tool in tools.tools:
        kind = "write" if tool.mutates else "read-only"
        lines.append(f"- {tool.name} ({kind}): {tool.description.strip()}")
    return "\n".join(lines) or "- (no tools)"


def _prompt_request(domain: DomainLike, node: str) -> list[dict[str, str]]:
    """Messages asking the frontier model to write one node's system prompt."""
    system = (
        "You design system prompts for nodes of a tool-using agent. Reply with the system"
        " prompt text only: no preamble, no markdown fences, no commentary."
    )
    user = (
        f"Domain: {domain.name}\n\n## Goal (verbatim)\n{domain.goal.strip()}\n\n"
        f"## Tools\n{_tool_manifest_text(domain.tools)}\n\n"
        f"## Node\n{NODE_BRIEF[node]}\n\n"
        "Write the system prompt for this node. Keep every rule from the goal that the node"
        " must obey, name the tools it may use by exact name, and state the output format."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n(.*?)\n```$", re.DOTALL)


def _clean(text: str) -> str:
    text = text.strip()
    match = _FENCE_RE.match(text)
    return match.group(1).strip() if match else text


@llm_span("architect.prompt")
def _ask(messages: list[dict[str, str]], client: Any | None) -> str:
    """One frontier call through the gateway, or through the injected OpenAI-shaped fake."""
    if client is None:
        text, _usage = llm.chat("frontier", messages)
        return text
    response = client.chat.completions.create(
        model=llm.resolve_model("frontier"), messages=messages
    )
    return response.choices[0].message.content or ""


def _fallback_prompt(domain: DomainLike, node: str) -> str:
    """Deterministic prompt used when the model returns nothing."""
    return f"{NODE_BRIEF[node]}\n\n## Goal\n{domain.goal.strip()}"


def _write_node_prompts(
    domain: DomainLike, nodes: list[str], client: Any | None, root: Path | None
) -> dict[str, str]:
    """Ask once per distinct node name; return node -> system_prompt_ref."""
    refs: dict[str, str] = {}
    for node in nodes:
        text = _clean(_ask(_prompt_request(domain, node), client))
        if not text:
            text = _fallback_prompt(domain, node)
        refs[node] = prompts.save_version(
            f"anneal/{domain.name}/{node}", text, label="staging", root=root
        )
    return refs


def _build_spec(
    index: int,
    topology: str,
    layout: list[tuple[str, Role, int]],
    tool_names: list[str],
    refs: dict[str, str],
) -> HarnessSpec:
    cand_id = f"cand-{index:02d}"
    nodes = [
        Node(
            name=name,
            role=role,
            model_tier=NODE_TIER[role],
            system_prompt_ref=refs[name],
            tools=list(tool_names) if role == "executor" else [],
            max_steps=max_steps,
        )
        for name, role, max_steps in layout
    ]
    return HarnessSpec(
        id=cand_id,
        topology=topology,  # type: ignore[arg-type]  # validated by pydantic
        step_budget=STEP_BUDGET,
        nodes=nodes,
        lineage=Lineage(parent=cand_id, iteration=0, operator=OPERATOR),
    )


@node_span("architect")
def propose(
    domain: DomainLike,
    n: int = 3,
    *,
    client: Any | None = None,
    prompts_root: Path | None = None,
) -> list[HarnessSpec]:
    """Return ``n`` validated candidate specs for ``domain`` (topologies from ``MENU``)."""
    if not 1 <= n <= len(MENU):
        raise ValueError(f"n must be between 1 and {len(MENU)}, got {n}")
    menu = MENU[:n]
    node_names = list(dict.fromkeys(name for _, layout in menu for name, _, _ in layout))
    refs = _write_node_prompts(domain, node_names, client, prompts_root)
    tool_names = [t.name for t in domain.tools.tools]
    return [
        _build_spec(i, topology, layout, tool_names, refs)
        for i, (topology, layout) in enumerate(menu, start=1)
    ]
