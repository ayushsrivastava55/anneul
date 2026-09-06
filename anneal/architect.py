"""Architect: goal.md + tools.yaml -> N candidate harness specs (iteration 0).

Topologies come from a fixed, deterministic menu so a run always compares at least
``single`` and ``planner_executor``. Nothing here reads tasks of any split.

Prompts are **assembled deterministically, never delegated**. For each node the architect
builds a template from the domain's own ``goal.md`` and tool manifest that always states the
role, the goal verbatim, the output contract, the tools (read-only first, write tools last)
with the instruction to call them rather than guess, and the rule that every fact must come
from a tool result. The LLM is asked only to *elaborate* that template; its reply is validated
(length, mentions a real tool name, is not a bare JSON blob or a lone code fence) and appended
under its own heading. A reply that fails validation is retried once with a stricter
instruction, then dropped in favour of the pure template. Which path was taken is recorded on
``Node.prompt_source`` and logged, so a run can report honestly how its prompts were made.

The node producing the final answer gets ``schema_ref`` whenever the domain's eval module
exposes an output schema (``OUTPUT_SCHEMA`` / ``output_schema``); runtime then raises
``schema_error`` on a malformed answer. Domains without one leave it ``None``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Protocol

import anneal.prompts as prompts
from anneal import llm
from anneal.spec import HarnessSpec, Lineage, ModelTier, Node, Role, ToolsManifest, ToolSpec
from anneal.tracing import llm_span, node_span

logger = logging.getLogger("anneal.architect")

STEP_BUDGET = 12
OPERATOR = "architect"

# Attribute names probed on the domain's eval module for a JSON-schema dict describing the
# final answer. Runtime resolves ``schema_ref`` with ``getattr(domain.eval, ref)``.
SCHEMA_ATTRS = ("OUTPUT_SCHEMA", "output_schema")

# Prompt validation thresholds. A 0.5-3B model that ignores the brief typically returns a bare
# schema of ~150 bytes; anything usable is far longer and names a tool.
MIN_ELABORATION_CHARS = 180

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
# How each node relates to the tool list in its own prompt.
NODE_TOOL_STANCE: dict[str, str] = {
    "planner": (
        "You call no tools yourself. Your plan must name these tools, by exact name, in the"
        " order the executor should call them."
    ),
    "executor": (
        "You may call these tools. Call them: a fact you can look up must never be guessed."
    ),
    "critic": (
        "You call no tools. Judge whether the executor called the right tools, by exact name,"
        " before it answered."
    ),
}

# (topology, [(node name, role, max_steps)]) -- max_steps per candidate sum to <= STEP_BUDGET
MENU: list[tuple[str, list[tuple[str, Role, int]]]] = [
    ("single", [("executor", "executor", STEP_BUDGET)]),
    ("planner_executor", [("planner", "planner", 2), ("executor", "executor", 10)]),
    ("critic_loop", [("executor", "executor", 8), ("critic", "critic", 4)]),
]

# The node whose text becomes the run's output in every topology on the MENU.
ANSWER_ROLE: Role = "executor"


class DomainLike(Protocol):
    """What Architect needs from a loaded domain (see anneal/domain.py)."""

    name: str
    goal: str
    tools: ToolsManifest


# --- schema discovery ------------------------------------------------------------------------


def find_output_schema(domain: DomainLike) -> tuple[str, dict[str, Any]] | None:
    """``(attribute name, schema)`` when the domain's eval exposes an output schema, else None.

    Read by attribute off the already-imported eval module, so the core never imports a domain.
    """
    evaluator = getattr(domain, "eval", None)
    if evaluator is None:
        return None
    for attr in SCHEMA_ATTRS:
        schema = getattr(evaluator, attr, None)
        if isinstance(schema, dict) and schema:
            return attr, schema
    return None


# --- the deterministic template --------------------------------------------------------------


def _tool_lines(tools: list[ToolSpec]) -> str:
    return "\n".join(f"- `{t.name}`: {t.description.strip()}" for t in tools)


def _tool_manifest_text(tools: ToolsManifest) -> str:
    """Flat listing used in the request to the elaborating model."""
    lines = [
        f"- {t.name} ({'write' if t.mutates else 'read-only'}): {t.description.strip()}"
        for t in tools.tools
    ]
    return "\n".join(lines) or "- (no tools)"


def _tools_section(tools: ToolsManifest, node: str) -> str:
    """Read-only tools first (gather facts), write tools last (final action only)."""
    reads = [t for t in tools.tools if not t.mutates]
    writes = [t for t in tools.tools if t.mutates]
    if not reads and not writes:
        return "## Tools you may call\n\nNo tools are available. Answer from the task input alone."
    stance = NODE_TOOL_STANCE.get(node, NODE_TOOL_STANCE["executor"])
    parts = ["## Tools you may call", "", stance, ""]
    if reads:
        parts += [
            "Read-only tools — use these to fetch every fact you need before you decide:",
            _tool_lines(reads),
            "",
        ]
    if writes:
        parts += [
            "Write tools — only when the goal permits it, and only as the very last action:",
            _tool_lines(writes),
            "",
        ]
    return "\n".join(parts).rstrip()


def _contract_section(schema: dict[str, Any] | None) -> str:
    """The output contract, stated in terms of the goal and (if present) the eval schema."""
    lines = [
        "## Output contract",
        "",
        "Your final message must be exactly the finished result the goal above describes,"
        " in exactly the shape it specifies — nothing before it, nothing after it, no"
        " explanation, no markdown code fences, no apology.",
    ]
    required = [k for k in (schema or {}).get("required", []) or []]
    if required:
        lines.append(
            "It must be a single JSON object containing these keys: "
            + ", ".join(f"`{k}`" for k in required)
            + "."
        )
    if (schema or {}).get("type") == "object" and not required:
        lines.append("It must be a single JSON object.")
    return "\n".join(lines)


def build_template(domain: DomainLike, node: str, schema: dict[str, Any] | None = None) -> str:
    """The floor every prompt is built on. Deterministic: no model can make it unusable."""
    sections = [
        f"# You are the {node.upper()} for the `{domain.name}` task.",
        "",
        NODE_BRIEF.get(node, NODE_BRIEF["executor"]),
        "",
        "## Goal (verbatim, from the domain owner)",
        "",
        domain.goal.strip(),
        "",
        _contract_section(schema),
        "",
        _tools_section(domain.tools, node),
        "",
        "## How to work",
        "",
        "1. Call tools instead of guessing. Any fact a tool can return must come from that"
        " tool, never from memory or assumption.",
        "2. Every value in your final answer must be traceable to a tool result or to the"
        " task input you were given. Never invent an id, a number, a name or a date.",
        "3. Make one tool call at a time and read its result before deciding the next call.",
        "4. Follow every rule stated in the goal above; the goal outranks your own judgement.",
        "5. When you have what you need, stop calling tools and reply with the final answer in"
        " the exact format required by the output contract.",
    ]
    return "\n".join(sections).strip()


# --- elaboration -----------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n(.*?)\n```$", re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

STRICTER = (
    "That reply was unusable: {why} Write plain prose instructions for the node — full"
    " sentences, no JSON object, no code fence. Name the tools by their exact names"
    " ({tools}) and say when each should be called. At least six sentences."
)


def _clean(text: str) -> str:
    text = text.strip()
    match = _FENCE_RE.match(text)
    return match.group(1).strip() if match else text


def validate_elaboration(text: str, tool_names: list[str]) -> str | None:
    """Return why ``text`` is unusable as prompt guidance, or None when it is usable."""
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            return "it was a bare JSON blob with no instructions."
    if not _ANY_FENCE_RE.sub("", stripped).strip():
        return "it was nothing but a code fence."
    if len(stripped) < MIN_ELABORATION_CHARS:
        return f"it was only {len(stripped)} characters long."
    if tool_names and not any(name in stripped for name in tool_names):
        return "it never mentioned any of the tools the node can call."
    return None


def _prompt_request(domain: DomainLike, node: str, template: str) -> list[dict[str, str]]:
    """Messages asking the model to elaborate (never replace) the node's template."""
    system = (
        "You improve system prompts for nodes of a tool-using agent. You are given a prompt"
        " that is already complete and correct. Reply with ADDITIONAL guidance only: prose"
        " instructions that make the node more reliable. Do not repeat the prompt, do not"
        " output JSON, do not output a code fence, do not add commentary about your reply."
    )
    user = (
        f"Domain: {domain.name}\n\n## Tools\n{_tool_manifest_text(domain.tools)}\n\n"
        f"## Node\n{NODE_BRIEF.get(node, NODE_BRIEF['executor'])}\n\n"
        f"## Prompt so far\n{template}\n\n"
        "Write extra guidance for this node: the order to call tools in, the traps in this"
        " goal, and how to keep the final answer in the required format. Name tools by their"
        " exact names. Prose only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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


def _elaborate(
    domain: DomainLike, node: str, template: str, client: Any | None
) -> tuple[str, str]:
    """``(elaboration, path)`` where path is elaborated / elaborated_retry / template."""
    tool_names = [t.name for t in domain.tools.tools]
    messages = _prompt_request(domain, node, template)
    for path in ("elaborated", "elaborated_retry"):
        try:
            text = _clean(_ask(messages, client))
        except Exception as exc:  # noqa: BLE001 - a dead model must not kill the run
            logger.warning("architect prompt call failed for %s: %s", node, exc)
            return "", "template"
        why = validate_elaboration(text, tool_names)
        if why is None:
            return text, path
        messages = [
            *messages,
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": STRICTER.format(why=why, tools=", ".join(tool_names) or "none"),
            },
        ]
    return "", "template"


def compose_prompt(
    domain: DomainLike, node: str, client: Any | None, schema: dict[str, Any] | None
) -> tuple[str, str]:
    """``(prompt text, path)``. The template is always present; elaboration is optional."""
    template = build_template(domain, node, schema)
    elaboration, path = _elaborate(domain, node, template, client)
    text = template if not elaboration else f"{template}\n\n## Additional guidance\n\n{elaboration}"
    logger.info(
        json.dumps(
            {
                "event": "architect_prompt",
                "domain": domain.name,
                "node": node,
                "prompt_source": path,
                "chars": len(text),
            }
        )
    )
    return text, path


def _write_node_prompts(
    domain: DomainLike,
    nodes: list[str],
    client: Any | None,
    root: Path | None,
    schema: dict[str, Any] | None,
) -> dict[str, tuple[str, str]]:
    """Ask once per distinct node name; return node -> (system_prompt_ref, prompt_source)."""
    refs: dict[str, tuple[str, str]] = {}
    for node in nodes:
        text, path = compose_prompt(domain, node, client, schema)
        ref = prompts.save_version(f"anneal/{domain.name}/{node}", text, label="staging", root=root)
        refs[node] = (ref, path)
    return refs


def _build_spec(
    index: int,
    topology: str,
    layout: list[tuple[str, Role, int]],
    tool_names: list[str],
    refs: dict[str, tuple[str, str]],
    schema_attr: str | None,
) -> HarnessSpec:
    cand_id = f"cand-{index:02d}"
    nodes = [
        Node(
            name=name,
            role=role,
            model_tier=NODE_TIER[role],
            system_prompt_ref=refs[name][0],
            prompt_source=refs[name][1],
            tools=list(tool_names) if role == "executor" else [],
            max_steps=max_steps,
            schema_ref=schema_attr if role == ANSWER_ROLE else None,
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
    found = find_output_schema(domain)
    schema_attr, schema = found if found else (None, None)
    if schema_attr is None:
        logger.info(
            json.dumps({"event": "architect_no_output_schema", "domain": domain.name})
        )
    refs = _write_node_prompts(domain, node_names, client, prompts_root, schema)
    tool_names = [t.name for t in domain.tools.tools]
    return [
        _build_spec(i, topology, layout, tool_names, refs, schema_attr)
        for i, (topology, layout) in enumerate(menu, start=1)
    ]
