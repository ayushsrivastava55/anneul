"""`anneal init`: interview a non-developer and generate a runnable domain directory.

Anneal's core takes exactly three files — ``goal.md``, ``tools.yaml``, ``eval.py`` — and until
now the user had to hand-write all three. That made Anneal a developer tool. This module makes
those files an *output*: five questions in, ``domains/<name>/`` out, ready for
``anneal.domain.load_domain`` and ``anneal run``.

**The interview is data, not control flow.** A :class:`Question` (id, prompt, kind, choices,
help, and an optional ``when`` gate) is a value; :data:`SCRIPT` is an ordered list of them;
:class:`Interview` is the answers dict plus whatever tool discovery found. A
:class:`Transport` is the single method ``ask(Question) -> str``. Two ship here —
:class:`RichTransport` (terminal, single-choice menus by number) and
:class:`ScriptedTransport` (tests, and any non-interactive driver) — and a voice or web
frontend is a third implementation of one method, not a rewrite.

**Nothing here is reimplemented.** Tool discovery goes through :mod:`anneal.mcp` (connect,
``tools/list``); validation through :class:`anneal.spec.ToolsManifest`; the goal text through
:mod:`anneal.llm` and the same template-floor-plus-elaboration pattern
:mod:`anneal.architect` already uses (a deterministic template is always written; the model may
only *add* to it, its reply is validated, and a bad or absent model silently leaves the
template). Environment reads go through :func:`anneal.config.env`. The reserved evaluation
split name is imported from :mod:`anneal.gate`, the only module allowed to spell it.

**Generation rules.**

- ``goal.md`` — deterministic template from the job description, the success criterion and the
  discovered tool names, optionally extended with validated model-written guidance.
- ``tools.yaml`` — for MCP, the ``servers:`` block plus the schemas the server itself published
  via ``tools/list``; for Python, ``python:`` impls introspected from the module. Validated by
  ``ToolsManifest`` before the file is written and by ``spec.load_tools`` after.
- ``eval.py`` — rendered from one of four templates, one per success kind, parameterised by the
  fields the examples reveal. Deterministic string comparison only. **There is no
  LLM-as-judge**, in the generated file or anywhere on this path: that is what makes an Anneal
  number mean something.
- ``tasks.jsonl`` — the collected examples, seeded shuffle, split train/search/reserved with at
  least one task in each. Too few examples for an honest split is stated in the console, in
  ``PROVENANCE.md`` and in the generated ``eval.py`` docstring.
- ``PROVENANCE.md`` — every question, every answer, what discovery found and which path the
  goal text took, so a generated domain is auditable rather than magic.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import random
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml

from anneal import architect, llm, mcp
from anneal.gate import HOLDOUT_SPLIT as RESERVED_SPLIT
from anneal.spec import ToolsManifest, load_tools
from anneal.tracing import llm_span

logger = logging.getLogger("anneal.onboard")

TRAIN_SPLIT = "train"
SEARCH_SPLIT = "search"
SPLIT_SEED = 0
MIN_EXAMPLES = 3
# Below this an honest 50/25/25 split cannot put two tasks in each half of the evaluation,
# so the gate's paired test has almost nothing to work with. We still generate; we say so.
MIN_FOR_REAL_SPLIT = 8
MAX_EXAMPLES = 50

QuestionKind = Literal["choice", "text", "examples"]

# Tool names containing one of these are emitted with `mutates: true`, which is what makes
# the architect put them last in the prompt ("write tools, only as the very last action").
WRITE_HINTS = (
    "write", "create", "delete", "remove", "update", "insert", "post", "send", "set",
    "put", "move", "edit", "append", "book", "cancel", "pay", "submit", "add", "upload",
)

_JSON_TYPES: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean",
                               list: "array", dict: "object"}


class OnboardError(RuntimeError):
    """The interview cannot produce a domain (bad name, existing directory, no examples)."""


# --- the interview, as data -----------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    """One question. ``when`` gates it on an earlier answer, so the script stays a list."""

    id: str
    prompt: str
    kind: QuestionKind = "text"
    choices: tuple[tuple[str, str], ...] = ()  # (value, label)
    help: str = ""
    when: tuple[str, str] | None = None  # (question id, required value)

    def labels(self) -> dict[str, str]:
        return dict(self.choices)


# The five questions. `tools_target` and `tools_module` are follow-ups to question 3: they are
# part of the same question, kept in the script (rather than in code) so one list still
# describes the whole interview.
SCRIPT: list[Question] = [
    Question(
        id="name",
        prompt="What should we call this? (a short name for the folder)",
        help="Letters, numbers and spaces. It becomes domains/<name>/.",
    ),
    Question(
        id="job",
        prompt="In one or two sentences, what should the agent do?",
        help="Plain language. Say what it receives and what it must produce.",
    ),
    Question(
        id="tools",
        prompt="What can the agent use to do it?",
        kind="choice",
        choices=(
            ("mcp", "An MCP server I'll name"),
            ("python", "Python functions in a module"),
            ("none", "Nothing yet — it answers from the task text alone"),
        ),
    ),
    Question(
        id="tools_target",
        prompt="How is that server started? Paste the launch command, or its URL.",
        help="e.g. npx -y @modelcontextprotocol/server-filesystem /tmp/box  or  https://host/mcp",
        when=("tools", "mcp"),
    ),
    Question(
        id="tools_module",
        prompt="Which Python module holds them? (dotted path, importable from here)",
        help="e.g. domains.invoices.fixtures.tools",
        when=("tools", "python"),
    ),
    Question(
        id="success",
        prompt="How do we know a run was right?",
        kind="choice",
        choices=(
            ("fields", "Specific fields in the answer match"),
            ("label", "A decision or label matches"),
            ("state", "The final state of the system matches"),
            ("examples", "Just compare against my examples"),
        ),
    ),
    Question(
        id="examples",
        prompt=f"Now some examples — at least {MIN_EXAMPLES}. Leave the input blank to stop.",
        kind="examples",
        help="For each: what the agent receives, then what a correct answer looks like.",
    ),
]


class Transport(Protocol):
    """How answers reach the interview. One method, so a voice frontend is a class."""

    def ask(self, question: Question) -> str:  # pragma: no cover - protocol
        ...


class ScriptedTransport:
    """Replays a flat queue of answers. Used by tests and by piped, non-interactive runs."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.asked: list[str] = []

    def ask(self, question: Question) -> str:
        self.asked.append(question.id)
        if not self._answers:
            raise OnboardError(f"scripted answers exhausted at question {question.id!r}")
        return self._answers.pop(0)


class RichTransport:
    """The terminal. Choices are a numbered menu; everything else is a line of text."""

    def __init__(self, console: Any) -> None:
        self.console = console

    def ask(self, question: Question) -> str:
        self.console.print(f"\n[bold cyan]{question.prompt}[/bold cyan]")
        if question.help:
            self.console.print(f"[dim]{question.help}[/dim]")
        if question.kind != "choice":
            return str(self.console.input("[green]> [/green]")).strip()
        for i, (_value, label) in enumerate(question.choices, start=1):
            self.console.print(f"  [cyan]{i}[/cyan]. {label}")
        return self._choose(question)

    def _choose(self, question: Question) -> str:
        values = [value for value, _label in question.choices]
        while True:
            raw = str(self.console.input("[green]> [/green]")).strip().lower()
            if raw.isdigit() and 1 <= int(raw) <= len(values):
                return values[int(raw) - 1]
            if raw in values:
                return raw
            self.console.print(f"[yellow]Pick 1-{len(values)}.[/yellow]")


@dataclass
class Example:
    """One input/expected pair as the user typed it."""

    given: str
    expected: str


@dataclass
class Interview:
    """The answers, plus what tool discovery turned up while the interview ran."""

    answers: dict[str, Any] = field(default_factory=dict)
    examples: list[Example] = field(default_factory=list)
    tools: list[ToolDraft] = field(default_factory=list)
    servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return slugify(str(self.answers.get("name", "")))

    @property
    def job(self) -> str:
        return str(self.answers.get("job", "")).strip()

    @property
    def success(self) -> str:
        return str(self.answers.get("success", "examples"))

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]


# --- tool discovery -------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDraft:
    """One tool on its way into tools.yaml. Same shape as ``spec.ToolSpec``."""

    name: str
    description: str
    args: dict[str, Any]
    impl: str
    mutates: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "args": self.args,
                "impl": self.impl, "mutates": self.mutates}


def _mutates(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in WRITE_HINTS)


def server_block(target: str) -> dict[str, Any]:
    """A launch command or a URL -> one entry of the ``servers:`` block."""
    target = target.strip()
    if not target:
        raise OnboardError("no launch command or URL given for the MCP server")
    if target.startswith(("http://", "https://")):
        return {"url": target}
    parts = shlex.split(target)
    return {"command": parts[0], "args": parts[1:]}


def discover_mcp(
    server: str, target: str, *, pool_factory: Any = mcp.get_pool
) -> tuple[dict[str, Any], list[ToolDraft], str | None]:
    """Start the server, run ``tools/list``, and turn its own schemas into drafts.

    Returns ``(servers block, tools, error)``. Discovery is the point: a server that already
    describes its tools is never described again by the user. ``error`` is a human-readable
    reason when the server would not start, in which case the caller falls back to no tools.
    """
    block = server_block(target)
    try:
        pool = pool_factory(mcp.parse_servers({server: block}))
        listed = pool.list_tools(server)
    except Exception as exc:  # noqa: BLE001 - any startup failure degrades to "no tools yet"
        logger.warning("mcp discovery failed for %s: %s", server, exc)
        return {}, [], str(exc)
    drafts = [
        ToolDraft(
            name=info.name,
            description=info.description or info.name,
            args=info.input_schema or {"type": "object", "properties": {}},
            impl=f"{mcp.IMPL_PREFIX}{server}/{info.name}",
            mutates=_mutates(info.name),
        )
        for info in listed.values()
    ]
    return {server: block}, drafts, None


def _param_schema(parameter: inspect.Parameter) -> dict[str, Any]:
    annotation = parameter.annotation
    return {"type": _JSON_TYPES.get(annotation, "string")}


def _signature_schema(func: Any) -> dict[str, Any]:
    """A JSON-schema ``args`` block from a Python signature."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return {"type": "object", "properties": {}}
    properties, required = {}, []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        properties[name] = _param_schema(parameter)
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def discover_python(module_path: str) -> tuple[list[ToolDraft], str | None]:
    """Import ``module_path`` and make one draft per public function defined in it."""
    module_path = module_path.strip()
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001 - a bad module name is an answer, not a crash
        logger.warning("python discovery failed for %s: %s", module_path, exc)
        return [], str(exc)
    drafts = []
    for name, value in vars(module).items():
        if name.startswith("_") or not callable(value) or inspect.isclass(value):
            continue
        if getattr(value, "__module__", None) != module_path:
            continue
        doc = (inspect.getdoc(value) or "").strip().splitlines()
        drafts.append(
            ToolDraft(
                name=name,
                description=doc[0] if doc else f"Call {name}.",
                args=_signature_schema(value),
                impl=f"python:{module_path}.{name}",
                mutates=_mutates(name),
            )
        )
    return drafts, None


# --- running the interview ------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """A directory name that is also a valid Python package name (underscores, not hyphens)."""
    slug = _SLUG_RE.sub("_", text.strip().lower()).strip("_")
    if not slug or slug[0].isdigit():
        raise OnboardError(f"{text!r} is not a usable name; start it with a letter")
    return slug


def _skip(question: Question, answers: dict[str, Any]) -> bool:
    return question.when is not None and answers.get(question.when[0]) != question.when[1]


def _collect_examples(transport: Transport, question: Question) -> list[Example]:
    """Ask for input/expected pairs until the input comes back blank.

    A blank before the minimum is a nudge, not the end: the same question is asked once more.
    A second blank is taken as "that is all I have", which below the minimum is an error the
    caller reports rather than a loop the user cannot leave.
    """
    examples: list[Example] = []
    blanks = 0
    for index in range(1, MAX_EXAMPLES + 1):
        given = transport.ask(
            Question(
                id=f"{question.id}_{index}_input",
                prompt=f"Example {index} — input:",
                help="" if len(examples) >= MIN_EXAMPLES or blanks == 0
                else f"{MIN_EXAMPLES - len(examples)} more needed. Blank again to give up.",
            )
        )
        if not given.strip():
            blanks += 1
            if len(examples) >= MIN_EXAMPLES or blanks > 1:
                break
            continue
        blanks = 0
        expected = transport.ask(
            Question(
                id=f"{question.id}_{index}_expected",
                prompt=f"Example {index} — a correct answer:",
                help="JSON is welcome; plain text is fine too.",
            )
        )
        examples.append(Example(given=given.strip(), expected=expected.strip()))
    if len(examples) < MIN_EXAMPLES:
        raise OnboardError(f"need at least {MIN_EXAMPLES} examples, got {len(examples)}")
    return examples


def _do_discovery(interview: Interview, console: Any, pool_factory: Any) -> None:
    """Question 3's real work: connect, list, and show the user what was found."""
    choice = interview.answers.get("tools")
    if choice == "mcp":
        servers, tools, error = discover_mcp(
            interview.slug, str(interview.answers["tools_target"]), pool_factory=pool_factory
        )
        if error:
            interview.answers["tools"] = "none"
            interview.notes.append(f"MCP server would not start ({error}); continued with none.")
        interview.servers, interview.tools = servers, tools
    elif choice == "python":
        tools, error = discover_python(str(interview.answers["tools_module"]))
        if error:
            interview.answers["tools"] = "none"
            interview.notes.append(f"module would not import ({error}); continued with none.")
        interview.tools = tools
    _report_discovery(interview, console)


def _report_discovery(interview: Interview, console: Any) -> None:
    if console is None:
        return
    for note in interview.notes:
        console.print(f"[yellow]{note}[/yellow]")
    if interview.tools:
        console.print(f"[green]Found {len(interview.tools)} tools:[/green]")
        for tool in interview.tools:
            console.print(f"  [cyan]{tool.name}[/cyan] — {tool.description[:80]}")
    elif interview.answers.get("tools") == "none":
        console.print("[dim]No tools; the agent will answer from the task text alone.[/dim]")


def run_interview(
    transport: Transport,
    *,
    domains_dir: Path,
    console: Any = None,
    pool_factory: Any = mcp.get_pool,
) -> Interview:
    """Walk :data:`SCRIPT` once and return the filled-in :class:`Interview`."""
    interview = Interview()
    for question in SCRIPT:
        if _skip(question, interview.answers):
            continue
        if question.kind == "examples":
            interview.examples = _collect_examples(transport, question)
            continue
        interview.answers[question.id] = transport.ask(question)
        if question.id == "name":
            _check_free(interview.slug, domains_dir)
        # Discovery happens the moment we know where to look -- so the user sees the tools
        # before being asked how to score them, not after.
        if question.id in ("tools_target", "tools_module"):
            _do_discovery(interview, console, pool_factory)
        elif question.id == "tools" and interview.answers["tools"] == "none":
            _report_discovery(interview, console)
    return interview


def _check_free(slug: str, domains_dir: Path) -> None:
    if (domains_dir / slug).exists():
        raise OnboardError(
            f"domains/{slug} already exists; pick another name or delete that directory"
        )


# --- normalising the examples ----------------------------------------------------------------


def _parse_jsonish(text: str) -> Any:
    """Whatever the user typed, as a Python value. Plain prose stays a string."""
    stripped = text.strip().strip("`")
    if stripped.lower().startswith("json"):
        stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except ValueError:
        return text.strip()


def normalise_expected(kind: str, raw: str) -> dict[str, Any]:
    """One example's expected answer as a dict, whatever shape the user typed it in."""
    parsed = _parse_jsonish(raw)
    if kind == "state":
        if isinstance(parsed, dict) and "state" in parsed:
            return parsed
        return {"state": parsed}
    if isinstance(parsed, dict):
        return parsed
    return {"label" if kind == "label" else "answer": parsed}


def expected_fields(expecteds: list[dict[str, Any]]) -> tuple[str, ...]:
    """The keys every example agrees on — the fields an evaluator can actually check."""
    if not expecteds:
        return ("answer",)
    shared = set(expecteds[0])
    for item in expecteds[1:]:
        shared &= set(item)
    if not shared:
        shared = {key for item in expecteds for key in item}
    return tuple(sorted(shared)) or ("answer",)


def split_examples(count: int, *, seed: int = SPLIT_SEED) -> list[str]:
    """Seeded 50/25/25 assignment with at least one task in every split."""
    reserved = max(1, round(count * 0.25))
    searching = max(1, round(count * 0.25))
    training = max(1, count - reserved - searching)
    while training + searching + reserved > count:  # tiny sets: shrink the reserved half first
        if reserved > 1:
            reserved -= 1
        elif searching > 1:
            searching -= 1
        else:
            training -= 1
    labels = ([TRAIN_SPLIT] * training + [SEARCH_SPLIT] * searching
              + [RESERVED_SPLIT] * reserved)
    labels += [TRAIN_SPLIT] * (count - len(labels))
    random.Random(seed).shuffle(labels)
    return labels


def build_tasks(interview: Interview) -> list[dict[str, Any]]:
    """The examples as tasks.jsonl rows, ids in the order they were given."""
    kind = interview.success
    expecteds = [normalise_expected(kind, e.expected) for e in interview.examples]
    splits = split_examples(len(expecteds))
    return [
        {
            "id": f"{interview.slug}-{index:02d}",
            "input": {"request": example.given},
            "expected": expected,
            "split": split,
            "tags": [kind],
        }
        for index, (example, expected, split) in enumerate(
            zip(interview.examples, expecteds, splits, strict=True)
        )
    ]


# --- goal.md -----------------------------------------------------------------------------------

DONE_TEXT = {
    "fields": (
        "Return a single JSON object containing these keys: {fields}. Every value must match"
        " the source exactly — no rounding, no paraphrase, no extra commentary."
    ),
    "label": (
        "Return a single JSON object with the key `{label}`. Its value is your decision, and"
        " it must be one of: {labels}."
    ),
    "state": (
        "Leave the system in exactly the state the request asks for. Then report what the end"
        " state is as a single JSON object under the key `state`."
    ),
    "examples": (
        "Return the finished answer and nothing else, in exactly the shape the examples show."
    ),
}

GOAL_TEMPLATE = """# Goal: @@TITLE@@

@@JOB@@

## What done means
@@DONE@@

## Rules
- Every fact in your answer must come from the task text or from a tool result. Never guess.
- Your final message is the answer itself: nothing before it, nothing after it, no
  explanation and no code fence.
@@RULES@@

## Tools
@@TOOLS@@
"""

GOAL_SYSTEM = (
    "You write the goal document for a tool-using agent. You are given a goal that is already"
    " complete and correct. Reply with ADDITIONAL rules only: short markdown bullets, one per"
    " line, that make the agent more reliable at this job. Do not repeat the goal, do not"
    " output JSON, do not output a code fence, do not comment on your reply."
)


def _goal_title(interview: Interview) -> str:
    return interview.slug.replace("_", " ").strip() or "the agent"


def _done_text(interview: Interview, fields: tuple[str, ...], labels: list[str]) -> str:
    template = DONE_TEXT.get(interview.success, DONE_TEXT["examples"])
    return template.format(
        fields=", ".join(f"`{f}`" for f in fields),
        label=fields[0] if fields else "label",
        labels=", ".join(f"`{v}`" for v in labels) or "the labels in the examples",
    )


def _tools_text(interview: Interview) -> str:
    if not interview.tools:
        return "None. Answer from the task text alone."
    lines = [f"- `{t.name}`: {t.description.strip().splitlines()[0]}" for t in interview.tools]
    return "\n".join(lines) + "\n\nSee tools.yaml for the exact argument schemas."


def goal_template(interview: Interview, fields: tuple[str, ...], labels: list[str]) -> str:
    """The deterministic floor: written whatever the model does or does not say."""
    text = GOAL_TEMPLATE
    for token, value in (
        ("@@TITLE@@", _goal_title(interview)),
        ("@@JOB@@", interview.job or "(no description was given)"),
        ("@@DONE@@", _done_text(interview, fields, labels)),
        ("@@RULES@@", ""),
        ("@@TOOLS@@", _tools_text(interview)),
    ):
        text = text.replace(token, value)
    return re.sub(r"\n{3,}", "\n\n", text)


@llm_span("onboard.goal")
def _ask(messages: list[dict[str, str]], client: Any | None) -> str:
    """One frontier call through the gateway, or through an injected OpenAI-shaped fake."""
    text, _usage = llm.chat("frontier", messages, client=client)
    return text


def elaborate_goal(interview: Interview, template: str, client: Any | None) -> tuple[str, str]:
    """``(extra rules, source)`` where source is ``elaborated`` or ``template``.

    Same contract as ``architect.compose_prompt``: the template is the floor, the model may
    only add to it, its reply is validated, and any failure leaves the template alone.
    """
    messages = [
        {"role": "system", "content": GOAL_SYSTEM},
        {
            "role": "user",
            "content": (
                f"The job, in the user's words: {interview.job}\n\n"
                f"Tools available: {', '.join(interview.tool_names) or 'none'}\n\n"
                f"## Goal so far\n{template}\n\n"
                "Write three to six extra rules for this agent as markdown bullets: the order"
                " to work in, the traps in this job, and what must never happen. Name tools by"
                " their exact names. Bullets only."
            ),
        },
    ]
    try:
        text = architect._clean(_ask(messages, client))
    except Exception as exc:  # noqa: BLE001 - no key, no network: the template still stands
        logger.warning("goal elaboration failed: %s", exc)
        return "", "template"
    why = architect.validate_elaboration(text, interview.tool_names)
    if why is not None:
        logger.warning("goal elaboration rejected: %s", why)
        return "", "template"
    return text, "elaborated"


def compose_goal(
    interview: Interview, fields: tuple[str, ...], labels: list[str], client: Any | None
) -> tuple[str, str]:
    """``(goal.md text, source)``."""
    template = goal_template(interview, fields, labels)
    extra, source = elaborate_goal(interview, template, client)
    if not extra:
        return template, source
    bullets = "\n".join(
        line if line.lstrip().startswith(("-", "*")) else f"- {line.strip()}"
        for line in extra.splitlines()
        if line.strip()
    )
    return template.replace("## Tools", f"{bullets}\n\n## Tools", 1), source


# --- tools.yaml ---------------------------------------------------------------------------------

TOOLS_HEADER = """# Generated by `anneal init` from the onboarding interview (see PROVENANCE.md).
# @@ORIGIN@@
"""

TOOLS_ORIGIN = {
    "mcp": ("Every tool below is served by the MCP server in the `servers:` block; the"
            " descriptions and\n# schemas are the server's own `tools/list` output, not ours."),
    "python": "Every tool below is a Python function introspected from the module named below.",
    "none": "The interview found no tools, so the agent answers from the task text alone.",
}


def build_manifest(interview: Interview) -> ToolsManifest:
    """The interview's tools as a validated :class:`ToolsManifest`."""
    data = {
        "servers": interview.servers,
        "tools": [tool.as_dict() for tool in interview.tools],
    }
    return ToolsManifest.model_validate(data)


def render_tools(interview: Interview, manifest: ToolsManifest) -> str:
    origin = TOOLS_ORIGIN.get(str(interview.answers.get("tools")), TOOLS_ORIGIN["none"])
    header = TOOLS_HEADER.replace("@@ORIGIN@@", origin)
    body = yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False, width=100)
    return header + body


# --- eval.py ------------------------------------------------------------------------------------
#
# One template, four score bodies -- one per `success` kind. Slots are @@TOKEN@@ rather than
# str.format fields because the template is Python source full of braces.

EVAL_TEMPLATE = r'''"""Evaluator for the @@SLUG@@ domain — generated by `anneal init`.

@@SUMMARY@@

Contract (see domains/README.md):
    THRESHOLD                       score >= THRESHOLD counts as a pass
    load_tasks(split=None)          -> list[Task] from tasks.jsonl
    score(task, output)             @@SCORE_DOC@@
    is_hard_fail(task, trace)       True when the run called a tool in FORBIDDEN_TOOLS
@@SETUP_DOC@@
Scoring is deterministic comparison of strings and numbers. No model is asked to judge an
answer here, and none ever should be: an Anneal number is only worth something because
nothing inside the loop can talk it up.
@@WARNING@@"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

THRESHOLD = @@THRESHOLD@@
_HERE = Path(__file__).resolve().parent
TASKS_PATH = _HERE / "tasks.jsonl"

# Tools that must never be called for these tasks. The interview cannot know them; add tool
# names here and any run that calls one is a hard failure, reported separately from the score.
FORBIDDEN_TOOLS: tuple[str, ...] = ()

@@CONSTANTS@@

_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class Task:
    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    split: str  # train | search | reserved evaluation split
    tags: list[str] = field(default_factory=list)


def load_tasks(split: str | None = None) -> list[Task]:
    """Every task in tasks.jsonl, or only those of one split."""
    tasks: list[Task] = []
    with TASKS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            if split is not None and raw["split"] != split:
                continue
            tasks.append(Task(**raw))
    return tasks


def _as_dict(output: Any) -> dict[str, Any]:
    """Whatever the harness produced, as a dict; {} when it is not JSON at all."""
    if isinstance(output, dict):
        return output
    if not isinstance(output, str):
        return {}
    text = output.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    match = _OBJECT_RE.search(text)
    for candidate in (text.strip(), match.group(0) if match else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _norm(value: Any) -> str:
    """Compare as text: case, surrounding whitespace and number formatting do not count."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).lower()
    return " ".join(str(value).split()).lower()


@@SCORE@@

def is_hard_fail(task: Task, trace: list[dict[str, Any]]) -> bool:
    """True when the run called a tool listed in FORBIDDEN_TOOLS.

    `trace` is a list of {"tool": str, "args": dict, ...} in call order.
    """
    del task
    return any(step.get("tool") in FORBIDDEN_TOOLS for step in trace or [])
@@SETUP@@'''

SCORE_FIELDS = '''def score(task: Task, output: Any) -> float:
    """The fraction of FIELDS the answer got right; 1.0 only when every field matches."""
    got = _as_dict(output)
    if not FIELDS:
        return 0.0
    hits = sum(1 for name in FIELDS if _norm(got.get(name)) == _norm(task.expected.get(name)))
    return hits / len(FIELDS)

'''

SCORE_LABEL = '''def score(task: Task, output: Any) -> float:
    """1.0 when the answer carries the expected label, 0.0 otherwise."""
    got = _as_dict(output)
    value = got.get(LABEL_FIELD, output if isinstance(output, str) else None)
    return 1.0 if _norm(value) == _norm(task.expected.get(LABEL_FIELD)) else 0.0

'''

SCORE_STATE = '''def read_state(task: Task, output: Any) -> dict[str, Any]:
    """The end state to compare against `expected["state"]`.

    GENERATED STUB: it believes the agent's own report of what it changed. Replace this body
    with a real probe of your system — a database query, a directory listing, an API read —
    when the agent's word is not evidence. Nothing else in this file needs to change.
    """
    del task
    reported = _as_dict(output)
    inner = reported.get("state")
    return inner if isinstance(inner, dict) else reported


def score(task: Task, output: Any) -> float:
    """The fraction of the expected end state that is actually there."""
    want = task.expected.get("state")
    got = read_state(task, output)
    if not isinstance(want, dict):
        return 1.0 if _norm(got) == _norm(want) else 0.0
    if not want:
        return 0.0
    hits = sum(1 for key, value in want.items() if _norm(got.get(key)) == _norm(value))
    return hits / len(want)

'''

SCORE_EXAMPLES = '''def score(task: Task, output: Any) -> float:
    """1.0 when the answer matches the example, after normalising case and whitespace."""
    want = task.expected.get(ANSWER_FIELD, task.expected)
    got = _as_dict(output)
    candidate = got.get(ANSWER_FIELD, got) if got else output
    return 1.0 if _norm(candidate) == _norm(want) else 0.0

'''

SETUP_STATE = '''

def setup(task: Task) -> None:
    """Reset the system before the task runs.

    GENERATED STUB: state scoring is only honest if every task starts from the same place.
    Put your reset here (truncate the table, wipe the directory, restore the fixture).
    """
    del task
'''

SCORE_BODIES = {
    "fields": SCORE_FIELDS, "label": SCORE_LABEL,
    "state": SCORE_STATE, "examples": SCORE_EXAMPLES,
}
SCORE_DOCS = {
    "fields": "fraction of FIELDS that match exactly",
    "label": "1.0 when LABEL_FIELD matches, else 0.0",
    "state": "fraction of the expected end state that is present",
    "examples": "1.0 on an exact (normalised) match with the example",
}
EVAL_SUMMARY = {
    "fields": "A run is right when every field in FIELDS matches what the example says.",
    "label": "A run is right when the decision in LABEL_FIELD matches the example.",
    "state": "A run is right when the system ends up in the state the example describes.",
    "examples": "A run is right when the answer matches the example it was given.",
}

TOO_FEW_WARNING = (
    "\n\nWARNING — this domain was generated from only @@N@@ examples. That is enough to run,\n"
    "but not enough for the reserved evaluation split to say anything trustworthy: the gate's\n"
    "paired test needs more tasks before a promotion means much. Add examples to tasks.jsonl\n"
    "(same shape, same split tags) before quoting a number from this domain.\n"
)


def _output_schema(fields: tuple[str, ...]) -> dict[str, Any]:
    """The answer shape, read by ``architect.find_output_schema`` off the eval module."""
    return {
        "type": "object",
        "properties": {name: {"type": "string"} for name in fields},
        "required": list(fields),
    }


def eval_constants(kind: str, fields: tuple[str, ...]) -> str:
    """The kind-specific constants block of the generated evaluator."""
    if kind == "fields":
        schema = json.dumps(_output_schema(fields), indent=4)
        return f"# The keys every example agreed on.\nFIELDS = {fields!r}\nOUTPUT_SCHEMA = {schema}"
    if kind == "label":
        label = fields[0] if fields else "label"
        schema = json.dumps(_output_schema((label,)), indent=4)
        return (
            f"# The one key that carries the decision.\nLABEL_FIELD = {label!r}\n"
            f"OUTPUT_SCHEMA = {schema}"
        )
    if kind == "examples":
        return "# Where a plain-text answer is stored in tasks.jsonl.\nANSWER_FIELD = \"answer\""
    return "# State scoring compares expected['state'] with read_state() below."


def render_eval(slug: str, kind: str, fields: tuple[str, ...], *, warning: str = "") -> str:
    """The generated evaluator, from the one template and the score body for ``kind``."""
    kind = kind if kind in SCORE_BODIES else "examples"
    text = EVAL_TEMPLATE
    for token, value in (
        ("@@SLUG@@", slug),
        ("@@SUMMARY@@", EVAL_SUMMARY[kind]),
        ("@@SCORE_DOC@@", SCORE_DOCS[kind]),
        ("@@SETUP_DOC@@", "    setup(task)                     reset the system before the run\n"
                          if kind == "state" else ""),
        ("@@THRESHOLD@@", "1.0"),
        ("@@CONSTANTS@@", eval_constants(kind, fields)),
        ("@@SCORE@@", SCORE_BODIES[kind]),
        ("@@SETUP@@", SETUP_STATE if kind == "state" else ""),
        ("@@WARNING@@", warning),
    ):
        text = text.replace(token, value)
    return text


# --- PROVENANCE.md --------------------------------------------------------------------------


def render_provenance(interview: Interview, tasks: list[dict[str, Any]], goal_source: str) -> str:
    """Every question and answer, so a generated domain can be audited rather than believed."""
    lines = [
        f"# How domains/{interview.slug} was generated",
        "",
        "`anneal init` wrote every file in this directory from the interview below. Nothing",
        "here was hand-written; nothing here is hidden.",
        "",
        "## The interview",
        "",
    ]
    for question in SCRIPT:
        if question.kind == "examples" or _skip(question, interview.answers):
            continue
        answer = interview.answers.get(question.id)
        label = question.labels().get(str(answer), str(answer))
        lines += [f"**{question.prompt}**", "", f"> {label}", ""]
    lines += ["## Tools discovered", ""]
    lines += [f"- `{t.name}` (`{t.impl}`) — {t.description.splitlines()[0]}"
              for t in interview.tools] or ["- none"]
    counts = {split: sum(1 for t in tasks if t["split"] == split) for t in tasks
              for split in [t["split"]]}
    lines += [
        "",
        "## Examples and splits",
        "",
        f"- {len(tasks)} examples given, split seeded with seed {SPLIT_SEED}: "
        + ", ".join(f"{split} {n}" for split, n in sorted(counts.items())),
        f"- success criterion: `{interview.success}` (evaluator template used for eval.py)",
        f"- goal.md source: `{goal_source}` "
        "(`template` = deterministic only; `elaborated` = model-written rules appended)",
        "",
    ]
    lines += [f"> {note}" for note in interview.notes]
    return "\n".join(lines).rstrip() + "\n"


# --- generation ------------------------------------------------------------------------------


PACKAGE_INIT = '"""Generated by `anneal init`. Makes this domain importable as a package."""\n'


def _labels_seen(tasks: list[dict[str, Any]], field_name: str) -> list[str]:
    seen = [str(t["expected"].get(field_name)) for t in tasks if field_name in t["expected"]]
    return sorted(dict.fromkeys(seen))


def generate_domain(
    interview: Interview, *, domains_dir: Path, client: Any | None = None
) -> Path:
    """Write ``domains/<slug>/`` from a finished interview and return its path.

    Everything is validated before it lands: the manifest through ``ToolsManifest``, the
    written tools.yaml again through ``spec.load_tools``. The caller can then
    ``load_domain(path)`` and ``anneal run`` it with no further edits.
    """
    slug = interview.slug
    path = Path(domains_dir) / slug
    _check_free(slug, Path(domains_dir))
    tasks = build_tasks(interview)
    fields = expected_fields([t["expected"] for t in tasks])
    labels = _labels_seen(tasks, fields[0]) if interview.success == "label" else []
    warning = TOO_FEW_WARNING.replace("@@N@@", str(len(tasks))) if (
        len(tasks) < MIN_FOR_REAL_SPLIT) else ""
    if warning:
        interview.notes.append(
            f"Only {len(tasks)} examples: enough to run, too few for the reserved evaluation"
            f" split to mean anything. Add more before quoting a number (see PROVENANCE.md)."
        )
    manifest = build_manifest(interview)
    goal, goal_source = compose_goal(interview, fields, labels, client)

    path.mkdir(parents=True)
    (path / "__init__.py").write_text(PACKAGE_INIT, encoding="utf-8")
    (path / "goal.md").write_text(goal, encoding="utf-8")
    (path / "tools.yaml").write_text(render_tools(interview, manifest), encoding="utf-8")
    (path / "eval.py").write_text(
        render_eval(slug, interview.success, fields, warning=warning), encoding="utf-8"
    )
    (path / "tasks.jsonl").write_text(
        "".join(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n" for task in tasks),
        encoding="utf-8",
    )
    (path / "PROVENANCE.md").write_text(
        render_provenance(interview, tasks, goal_source), encoding="utf-8"
    )
    load_tools(path / "tools.yaml")  # the file on disk, not the object we dumped
    logger.info(
        json.dumps({"event": "onboard_generated", "domain": slug, "tasks": len(tasks),
                    "tools": len(interview.tools), "goal_source": goal_source})
    )
    return path


def init_domain(
    transport: Transport,
    *,
    domains_dir: Path,
    console: Any = None,
    client: Any | None = None,
    pool_factory: Any = mcp.get_pool,
) -> Path:
    """Interview, then generate. The whole of ``anneal init`` in one call."""
    interview = run_interview(
        transport, domains_dir=Path(domains_dir), console=console, pool_factory=pool_factory
    )
    return generate_domain(interview, domains_dir=Path(domains_dir), client=client)
