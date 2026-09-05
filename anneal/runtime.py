"""Generic executor for any ``HarnessSpec`` topology.

The runtime only ever sees the spec and the three domain input files (via ``Domain``).
It never scores: ``runner.py`` does that with ``domain.eval.score``.

Topologies implemented here:

- ``single``: one executor node in a ReAct loop (LLM with OpenAI tool calling, dispatch
  the tools, repeat) until a final assistant message without tool calls, or the node's
  ``max_steps`` / ``spec.step_budget`` is hit.
- ``planner_executor``: the planner emits a numbered step list as text, the executor runs
  it with tools, the planner may replan once (any reply other than ``DONE``).

Tool dispatch: ``python:<module>.<fn>`` via ``importlib.import_module`` on the dotted module
path; ``mcp:`` raises ``NotImplementedError`` for now. Arguments get a shallow
required/type check against the JSON schema in tools.yaml (``jsonschema`` is not a
dependency). Every node is a ``tracing.node_span``, every tool call a ``tracing.tool_span``
and every LLM call an ``tracing.llm_span``.
"""

from __future__ import annotations

import importlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from anneal import llm, tracing
from anneal.domain import Domain, ensure_repo_root_on_path
from anneal.spec import HarnessSpec, Node, ToolSpec

ClientFactory = Callable[[str], Any]

# JSON-schema type name -> python types accepted by the shallow check.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}

DEFAULT_PROMPTS: dict[str, str] = {
    "executor": (
        "You are the executor. Complete the task using only the provided tools. "
        "Call tools as needed; when finished, reply with a final message and no tool calls."
    ),
    "planner": (
        "You are the planner. Write a short numbered list of steps for an executor that has "
        "tools. Do not call tools yourself. When asked to review the executor's result, reply "
        "exactly DONE if the task is complete, otherwise a revised numbered plan."
    ),
    "critic": "You are the critic. Reply PASS or FAIL followed by one line of reasoning.",
    "router": "You are the router. Name the single tool group best suited to the task.",
    "validator": "You are the validator. Check the output against the required schema.",
    "escalate": "You escalate. Reply with a JSON object {\"escalate\": <reason>}.",
}


@dataclass
class TaskResult:
    """Everything the runner needs from one task run (scoring happens in the runner)."""

    output: Any
    trace: list[dict[str, Any]]
    per_node: dict[str, dict[str, Any]]
    steps: int
    hit_step_budget: bool
    schema_error: bool
    trace_id: str | None
    latency_ms: float


class StepBudgetExceeded(Exception):
    """Raised inside a node when ``spec.step_budget`` would be exceeded."""


# --- schema helpers ---------------------------------------------------------------------------


def shallow_check(value: Any, schema: dict[str, Any]) -> str | None:
    """Return an error string if ``value`` fails a shallow required/type check, else None.

    Checks the top-level ``type``, then ``required`` keys and each listed property's
    ``type`` one level deep. Nested objects are not descended into.
    """
    expected = schema.get("type")
    if expected is not None and not _is_type(value, expected):
        return f"expected {expected}, got {type(value).__name__}"
    if not isinstance(value, dict):
        return None
    for key in schema.get("required", []) or []:
        if key not in value:
            return f"missing required field {key!r}"
    for key, sub in (schema.get("properties") or {}).items():
        if key in value and "type" in sub and not _is_type(value[key], sub["type"]):
            return f"field {key!r}: expected {sub['type']}, got {type(value[key]).__name__}"
    return None


def _is_type(value: Any, expected: str | list[str]) -> bool:
    names = expected if isinstance(expected, list) else [expected]
    for name in names:
        accepted = _JSON_TYPES.get(name)
        if accepted is None:
            return True  # unknown type name: do not reject
        if isinstance(value, bool) and name in ("integer", "number"):
            continue
        if isinstance(value, accepted):
            return True
    return False


def parse_output(text: str | None) -> Any:
    """Parse the final assistant text as JSON when it looks like JSON, else return it raw."""
    if text is None:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
    if stripped[:1] in ("{", "["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return text
    return text


# --- tool dispatch ---------------------------------------------------------------------------


def _resolve_python_impl(impl: str) -> Callable[..., Any]:
    """``python:pkg.module.fn`` -> the callable, imported by dotted module path."""
    dotted = impl.removeprefix("python:")
    module_path, _, fn_name = dotted.rpartition(".")
    if not module_path or not fn_name:
        raise ValueError(f"malformed python impl {impl!r}")
    ensure_repo_root_on_path()
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def invoke_tool(tool: ToolSpec, args: dict[str, Any]) -> str:
    """Validate ``args`` against ``tool.args`` and run the impl. Errors come back as text."""
    problem = shallow_check(args, tool.args or {"type": "object"})
    if problem is not None:
        return f"Error: invalid arguments for {tool.name}: {problem}"
    if tool.impl.startswith("mcp:"):
        raise NotImplementedError(f"mcp tools are not supported yet ({tool.impl})")
    fn = _resolve_python_impl(tool.impl)
    try:
        result = fn(**args)
    except Exception as exc:  # noqa: BLE001 - tool failures are data for the model
        return f"Error: {tool.name} failed: {exc}"
    return result if isinstance(result, str) else json.dumps(result, default=str)


def _openai_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.args or {"type": "object", "properties": {}},
        },
    }


def _system_prompt(node: Node, domain: Domain) -> str:
    """Prompt text for ``node``: the versioned prompt when available, else a role default."""
    text = _load_prompt_ref(node.system_prompt_ref) or DEFAULT_PROMPTS.get(node.role, "")
    return f"{text}\n\n# Goal\n\n{domain.goal}".strip()


def _load_prompt_ref(ref: str) -> str | None:
    """``anneal.prompts.get_prompt`` when that module exists and knows ``ref``."""
    try:
        from anneal import prompts  # noqa: PLC0415 - owned by the architect session
    except ImportError:
        return None
    try:
        return str(prompts.get_prompt(ref))
    except (FileNotFoundError, KeyError, ValueError):
        return None


def _task_message(task: Any) -> str:
    payload = getattr(task, "input", task)
    return payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)


# --- one task run ---------------------------------------------------------------------------


@dataclass
class _Run:
    """Mutable state for a single ``run_task`` invocation."""

    spec: HarnessSpec
    domain: Domain
    client_factory: ClientFactory
    trace: list[dict[str, Any]] = field(default_factory=list)
    per_node: dict[str, dict[str, Any]] = field(default_factory=dict)
    steps: int = 0
    hit_step_budget: bool = False
    _clients: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tools_by_name = {t.name: t for t in self.domain.tools.tools}

    def node(self, role: str) -> Node:
        for node in self.spec.nodes:
            if node.role == role:
                return node
        raise ValueError(f"spec {self.spec.id} has no {role} node")

    def _client(self, tier: str) -> Any:
        if tier not in self._clients:
            self._clients[tier] = tracing.wrap_client(self.client_factory(tier))
        return self._clients[tier]

    def call_llm(self, node: Node, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """One LLM call for ``node``; appends the assistant message and returns it."""
        if self.steps >= self.spec.step_budget:
            self.hit_step_budget = True
            raise StepBudgetExceeded(node.name)
        tools = [_openai_tool(self.tools_by_name[n]) for n in node.tools if n in self.tools_by_name]
        traced = tracing.llm_span(f"llm.{node.name}")(self._complete)
        message, usage = traced(node.model_tier, messages, tools or None)
        self.steps += 1
        self._account(node.name, usage)
        messages.append(message)
        return message

    def _complete(
        self, tier: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> tuple[dict[str, Any], llm.Usage]:
        """Completion through the gateway (``llm.complete``) with this run's client."""
        return llm.complete(tier, messages, tools, client=self._client(tier))

    def _account(self, node_name: str, usage: llm.Usage) -> None:
        acct = self.per_node.setdefault(
            node_name, {"tokens_in": 0, "tokens_out": 0, "backend": None, "ms": 0.0}
        )
        acct["tokens_in"] += usage.tokens_in
        acct["tokens_out"] += usage.tokens_out
        acct["ms"] = round(acct["ms"] + usage.latency_ms, 1)
        acct["backend"] = usage.backend or acct["backend"]

    def dispatch(self, call: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call from an assistant message; returns the ``tool`` reply message."""
        fn = call.get("function") or {}
        name = str(fn.get("name", ""))
        args, parse_error = _parse_args(fn.get("arguments"))
        tool = self.tools_by_name.get(name)
        if parse_error is not None:
            result = f"Error: invalid arguments for {name}: {parse_error}"
        elif tool is None:
            result = f"Error: unknown tool {name!r}"
        else:
            result = tracing.tool_span(f"tool.{name}")(invoke_tool)(tool, args)
        self.trace.append({"tool": name, "args": args, "result": result})
        return {"role": "tool", "tool_call_id": call.get("id", ""), "content": result}

    def react(self, node: Node, messages: list[dict[str, Any]]) -> str | None:
        """ReAct loop for ``node``. Returns the final text, or None when a budget stopped it."""
        for _ in range(node.max_steps):
            message = self.call_llm(node, messages)
            calls = message.get("tool_calls") or []
            if not calls:
                return message.get("content") or ""
            messages.extend(self.dispatch(call) for call in calls)
        self.hit_step_budget = True
        return None

    def run_node(self, node: Node, fn: Callable[..., Any], *args: Any) -> Any:
        """Execute ``fn`` inside a node span named after ``node``."""
        return tracing.node_span(f"node.{node.name}")(fn)(*args)

    # --- topologies ---

    def run_single(self, task: Any) -> str | None:
        node = self.node("executor")
        messages = [
            {"role": "system", "content": _system_prompt(node, self.domain)},
            {"role": "user", "content": _task_message(task)},
        ]
        return self.run_node(node, self.react, node, messages)

    def run_planner_executor(self, task: Any) -> str | None:
        planner, executor = self.node("planner"), self.node("executor")
        plan_msgs = [
            {"role": "system", "content": _system_prompt(planner, self.domain)},
            {"role": "user", "content": _task_message(task)},
        ]
        exec_msgs = [
            {"role": "system", "content": _system_prompt(executor, self.domain)},
            {"role": "user", "content": _task_message(task)},
        ]
        plan = self.run_node(planner, self.call_llm, planner, plan_msgs).get("content") or ""
        exec_msgs.append({"role": "user", "content": f"Plan:\n{plan}"})
        result = self.run_node(executor, self.react, executor, exec_msgs)
        if result is None or planner.max_steps < 2:
            return result
        plan_msgs.append(
            {
                "role": "user",
                "content": f"Executor result:\n{result}\n\nReply DONE or a revised plan.",
            }
        )
        verdict = self.run_node(planner, self.call_llm, planner, plan_msgs).get("content") or ""
        if verdict.strip().upper().startswith("DONE"):
            return result
        exec_msgs.append({"role": "user", "content": f"Revised plan:\n{verdict}"})
        return self.run_node(executor, self.react, executor, exec_msgs)

    def execute(self, task: Any) -> str | None:
        runners = {
            "single": self.run_single,
            "planner_executor": self.run_planner_executor,
        }
        if self.spec.topology not in runners:
            raise NotImplementedError(f"topology {self.spec.topology!r} is not implemented yet")
        try:
            return runners[self.spec.topology](task)
        except StepBudgetExceeded:
            self.hit_step_budget = True
            return None


def _parse_args(raw: Any) -> tuple[dict[str, Any], str | None]:
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        return {}, f"arguments are not valid JSON ({exc})"
    if not isinstance(parsed, dict):
        return {}, "arguments must be a JSON object"
    return parsed, None


def _schema_error(spec: HarnessSpec, domain: Domain, output: Any) -> bool:
    """True when a node declares ``schema_ref`` and the output fails the shallow check."""
    for node in spec.nodes:
        if not node.schema_ref:
            continue
        schema = getattr(domain.eval, node.schema_ref, None)
        if isinstance(schema, dict) and shallow_check(output, schema) is not None:
            return True
    return False


def run_task(
    spec: HarnessSpec,
    task: Any,
    domain: Domain,
    *,
    seed: int = 0,
    client_factory: ClientFactory | None = None,
) -> TaskResult:
    """Run ``task`` through ``spec`` and return a ``TaskResult`` (unscored).

    ``client_factory(tier)`` returns an OpenAI-compatible client; defaults to
    ``llm.get_client`` (the gateway). Tests inject a scripted fake. ``seed`` is recorded in
    the run context for reproducibility but not forwarded to providers, several of which
    reject unknown sampling parameters.
    """
    iteration = spec.lineage.iteration if spec.lineage else 0
    run = _Run(spec, domain, client_factory or llm.get_client)
    with tracing.run_context(
        candidate_id=spec.id,
        iteration=iteration,
        domain=domain.name,
        split=getattr(task, "split", None),
    ):
        return tracing.node_span("task")(_run_traced)(run, task, seed)


def _run_traced(run: _Run, task: Any, seed: int) -> TaskResult:
    """Body of ``run_task`` executed inside the top-level task span."""
    del seed  # reserved; see run_task docstring
    setup = getattr(run.domain.eval, "setup", None)
    if callable(setup):
        setup(task)
    start = time.perf_counter()
    text = run.execute(task)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    output = parse_output(text)
    return TaskResult(
        output=output,
        trace=run.trace,
        per_node=run.per_node,
        steps=run.steps,
        hit_step_budget=run.hit_step_budget,
        schema_error=_schema_error(run.spec, run.domain, output),
        trace_id=tracing.current_trace_id(),
        latency_ms=latency_ms,
    )
