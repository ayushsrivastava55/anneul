"""`anneal init`: the interview, tool discovery, and the domains it generates.

Everything here is offline. The interview is driven by ``ScriptedTransport``, MCP discovery by
a fake pool, and the one end-to-end test drives the real ``anneal run`` loop with a scripted
gateway in place of the model — so the acceptance claim ("a generated domain loads and runs")
is checked against the real runner and the real evaluator, not a mock of either.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anneal import cli, mcp, onboard, runtime
from anneal.domain import load_domain
from tests.fakes import DEFAULT_BACKEND, FakeClient, FakeRawResponse, build_completion

ROOT = Path(__file__).resolve().parent.parent

# One example pair per line: (what the agent receives, what a correct answer looks like).
FIELD_EXAMPLES = [
    ('Order 1001 from Acme, total 42.50', '{"order": "1001", "customer": "Acme"}'),
    ('Order 1002 from Borg, total 10.00', '{"order": "1002", "customer": "Borg"}'),
    ('Order 1003 from Cyan, total 99.99', '{"order": "1003", "customer": "Cyan"}'),
    ('Order 1004 from Delta, total 5.00', '{"order": "1004", "customer": "Delta"}'),
]


def answers(
    *,
    name: str = "order desk",
    job: str = "Read one order line and pull out the order number and the customer name.",
    tools: str = "none",
    tools_extra: list[str] | None = None,
    success: str = "fields",
    examples: list[tuple[str, str]] | None = None,
) -> list[str]:
    """A flat answer queue in the order ``SCRIPT`` asks for it."""
    queue = [name, job, tools, *(tools_extra or []), success]
    for given, expected in examples if examples is not None else FIELD_EXAMPLES:
        queue += [given, expected]
    queue.append("")  # blank input ends the examples loop
    return queue


def generate(tmp_path: Path, **kw: Any) -> Path:
    """Run the whole interview into ``tmp_path`` and return the generated domain directory."""
    transport = onboard.ScriptedTransport(answers(**kw))
    return onboard.init_domain(transport, domains_dir=tmp_path)


# --- the interview ----------------------------------------------------------------------------


def test_script_is_ordered_data_not_control_flow() -> None:
    ids = [q.id for q in onboard.SCRIPT]
    assert ids == ["name", "job", "tools", "tools_target", "tools_module", "success", "examples"]
    assert [q.kind for q in onboard.SCRIPT if q.kind == "choice"] == ["choice", "choice"]


def test_scripted_interview_generates_a_loadable_domain(tmp_path: Path) -> None:
    path = generate(tmp_path)
    assert path.name == "order_desk"
    for name in ("goal.md", "tools.yaml", "eval.py", "tasks.jsonl", "PROVENANCE.md"):
        assert (path / name).is_file(), f"missing {name}"

    domain = load_domain(path)
    assert domain.name == "order_desk"
    assert "order number" in domain.goal
    tasks = domain.eval.load_tasks()
    assert len(tasks) == len(FIELD_EXAMPLES)
    assert {t.split for t in tasks} == {"train", "search", onboard.RESERVED_SPLIT}
    assert domain.eval.FIELDS == ("customer", "order")


def test_provenance_records_every_question_and_answer(tmp_path: Path) -> None:
    text = (generate(tmp_path) / "PROVENANCE.md").read_text()
    for question in onboard.SCRIPT:
        if question.kind != "examples" and question.when is None:
            assert question.prompt in text
    assert "Nothing yet" in text  # the label of the chosen option, not its internal value
    assert "goal.md source: `template`" in text  # no model was reachable in this test


def test_refuses_to_overwrite_an_existing_domain(tmp_path: Path) -> None:
    generate(tmp_path)
    with pytest.raises(onboard.OnboardError, match="already exists"):
        generate(tmp_path)


def test_too_few_examples_is_stated_in_the_console_and_in_the_files(tmp_path: Path) -> None:
    transport = onboard.ScriptedTransport(answers(examples=FIELD_EXAMPLES[:3]))
    interview = onboard.run_interview(transport, domains_dir=tmp_path)
    path = onboard.generate_domain(interview, domains_dir=tmp_path)
    assert any("too few" in note for note in interview.notes), interview.notes
    assert "WARNING" in (path / "eval.py").read_text()
    assert "too few" in (path / "PROVENANCE.md").read_text()


def test_fewer_than_three_examples_is_refused(tmp_path: Path) -> None:
    queue = answers(examples=FIELD_EXAMPLES[:2])
    queue.append("")  # a second blank: "that is all I have"
    transport = onboard.ScriptedTransport(queue)
    with pytest.raises(onboard.OnboardError, match="at least 3 examples"):
        onboard.run_interview(transport, domains_dir=tmp_path)


def test_every_split_gets_at_least_one_task() -> None:
    for count in range(3, 30):
        labels = onboard.split_examples(count)
        assert len(labels) == count
        assert set(labels) == {"train", "search", onboard.RESERVED_SPLIT}


# --- tool discovery ---------------------------------------------------------------------------


class FakePool:
    """A ``mcp.Pool`` stand-in that publishes a fixed ``tools/list``, or refuses to start."""

    def __init__(self, tools: dict[str, mcp.ToolInfo] | None, error: str | None = None) -> None:
        self._tools, self._error = tools or {}, error
        self.asked: list[str] = []

    def list_tools(self, server: str) -> dict[str, mcp.ToolInfo]:
        self.asked.append(server)
        if self._error:
            raise mcp.MCPError(self._error)
        return self._tools


SERVER_TOOLS = {
    "read_note": mcp.ToolInfo(
        name="read_note",
        description="Read one note by id.",
        input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
    ),
    "write_note": mcp.ToolInfo(name="write_note", description="Overwrite one note."),
}


def test_mcp_discovery_uses_the_servers_own_schemas(tmp_path: Path) -> None:
    pool = FakePool(SERVER_TOOLS)
    transport = onboard.ScriptedTransport(
        answers(name="notes", tools="mcp", tools_extra=["npx -y notes-server /tmp/notes"])
    )
    interview = onboard.run_interview(
        transport, domains_dir=tmp_path, pool_factory=lambda servers: pool
    )
    assert pool.asked == ["notes"]
    assert interview.tool_names == ["read_note", "write_note"]
    # the description is the server's, never one we asked the user to write
    assert interview.tools[0].description == "Read one note by id."
    assert interview.tools[0].args == {"type": "object", "properties": {"id": {"type": "string"}}}
    assert interview.tools[1].mutates is True  # write_* is a write action
    assert interview.servers == {"notes": {"command": "npx", "args": ["-y", "notes-server",
                                                                     "/tmp/notes"]}}

    path = onboard.generate_domain(interview, domains_dir=tmp_path)
    tools = load_domain(path).tools
    assert [t.impl for t in tools.tools] == ["mcp:notes/read_note", "mcp:notes/write_note"]
    assert tools.servers["notes"]["command"] == "npx"


def test_running_out_of_piped_input_is_a_message_not_a_traceback() -> None:
    class DeadConsole:
        def print(self, *_a: Any, **_kw: Any) -> None:
            pass

        def input(self, *_a: Any, **_kw: Any) -> str:
            raise EOFError

    with pytest.raises(onboard.OnboardError, match="ran out of input"):
        onboard.RichTransport(DeadConsole()).ask(onboard.SCRIPT[0])


def test_mcp_url_target_becomes_a_url_server() -> None:
    assert onboard.server_block("https://host/mcp") == {"url": "https://host/mcp"}


def test_a_server_that_will_not_start_falls_back_to_no_tools(tmp_path: Path) -> None:
    pool = FakePool(None, error="npx: command not found")
    transport = onboard.ScriptedTransport(
        answers(name="notes", tools="mcp", tools_extra=["npx -y notes-server"])
    )
    interview = onboard.run_interview(
        transport, domains_dir=tmp_path, pool_factory=lambda servers: pool
    )
    assert interview.tools == []
    assert interview.answers["tools"] == "none"
    assert any("would not start" in note for note in interview.notes)
    path = onboard.generate_domain(interview, domains_dir=tmp_path)
    assert load_domain(path).tools.tools == []


MODULE_SOURCE = '''
"""Fixture module for python tool discovery."""


def lookup_order(order_id: str, verbose: bool = False) -> dict:
    """Fetch one order by id."""
    return {"id": order_id, "verbose": verbose}


def post_order(order_id: str) -> str:
    """Write the order to the ledger."""
    return "ok"


def _private(x: int) -> int:
    return x
'''


def test_python_discovery_introspects_public_functions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "onboard_fixture_tools.py").write_text(MODULE_SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    tools, error = onboard.discover_python("onboard_fixture_tools")
    assert error is None
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"lookup_order", "post_order"}
    assert by_name["lookup_order"].description == "Fetch one order by id."
    assert by_name["lookup_order"].impl == "python:onboard_fixture_tools.lookup_order"
    assert by_name["lookup_order"].args["required"] == ["order_id"]
    assert by_name["lookup_order"].args["properties"]["verbose"] == {"type": "boolean"}
    assert by_name["post_order"].mutates is True


def test_python_discovery_resolves_string_annotations(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool module with `from __future__ import annotations` still gets real JSON types.

    Under that import every annotation arrives as a string, so a naive lookup types every
    argument as `string` and the model is told a float field is text.
    """
    source = "from __future__ import annotations\n" + MODULE_SOURCE.replace(
        "verbose: bool = False", "amount: float = 0.0"
    )
    (tmp_path / "onboard_future_tools.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    tools, error = onboard.discover_python("onboard_future_tools")
    assert error is None
    args = next(t for t in tools if t.name == "lookup_order").args
    assert args["properties"] == {"order_id": {"type": "string"}, "amount": {"type": "number"}}


def test_python_discovery_reports_a_bad_module_instead_of_crashing() -> None:
    tools, error = onboard.discover_python("no.such.module.anywhere")
    assert tools == []
    assert error


# --- the four evaluator templates ---------------------------------------------------------------

# (success kind, examples, a right answer, a wrong answer)
EVAL_CASES = [
    (
        "fields",
        FIELD_EXAMPLES,
        '{"order": "1001", "customer": "Acme"}',
        '{"order": "9999", "customer": "Nobody"}',
    ),
    (
        "label",
        [("Invoice A, no PO", "escalate"), ("Invoice B, PO ok", "approve"),
         ("Invoice C, PO ok", "approve")],
        '{"label": "escalate"}',
        '{"label": "approve"}',
    ),
    (
        "state",
        [("Archive note 1", '{"note_1": "archived"}'), ("Archive note 2", '{"note_2": "archived"}'),
         ("Archive note 3", '{"note_3": "archived"}')],
        '{"state": {"note_1": "archived"}}',
        '{"state": {"note_1": "open"}}',
    ),
    (
        "examples",
        [("Say hello", "hello there"), ("Say goodbye", "goodbye"), ("Say yes", "yes")],
        "hello there",
        "something else entirely",
    ),
]


@pytest.mark.parametrize(("kind", "examples", "good", "bad"), EVAL_CASES)
def test_each_eval_template_scores_good_1_and_bad_0(
    tmp_path: Path, kind: str, examples: list[tuple[str, str]], good: str, bad: str
) -> None:
    path = generate(tmp_path, name=f"probe {kind}", success=kind, examples=examples)
    evaluator = load_domain(path).eval
    task = next(t for t in evaluator.load_tasks() if t.id.endswith("-00"))
    assert evaluator.score(task, good) == 1.0
    assert evaluator.score(task, bad) == 0.0
    assert evaluator.THRESHOLD == 1.0
    assert evaluator.is_hard_fail(task, [{"tool": "anything", "args": {}}]) is False
    assert (kind == "state") == hasattr(evaluator, "setup")


def test_output_schema_types_come_from_the_examples(tmp_path: Path) -> None:
    """`runtime.shallow_check` type-checks OUTPUT_SCHEMA, so a numeric field must say so.

    Typing every field `string` would record a correct numeric answer as a schema error.
    """
    examples = [
        ('Order 1001, 42.50', '{"order": "1001", "total": 42.5}'),
        ('Order 1002, 10.00', '{"order": "1002", "total": 10.0}'),
        ('Order 1003, 99.99', '{"order": "1003", "total": 99.99}'),
    ]
    evaluator = load_domain(generate(tmp_path, name="totals", examples=examples)).eval
    assert evaluator.OUTPUT_SCHEMA["properties"]["total"] == {"type": "number"}
    assert evaluator.OUTPUT_SCHEMA["properties"]["order"] == {"type": "string"}
    task = next(t for t in evaluator.load_tasks() if t.id == "totals-00")
    assert runtime.shallow_check(task.expected, evaluator.OUTPUT_SCHEMA) is None


def test_no_generated_evaluator_ever_calls_a_model(tmp_path: Path) -> None:
    for kind, examples, _good, _bad in EVAL_CASES:
        source = (generate(tmp_path, name=f"judge {kind}", success=kind,
                           examples=examples) / "eval.py").read_text()
        for forbidden in ("anneal.llm", "openai", "get_client", "chat("):
            assert forbidden not in source, f"{kind} template reaches for a model"


# --- goal.md elaboration -------------------------------------------------------------------------


ELABORATION = (
    "- Read the order line in full before answering, and never invent an order number.\n"
    "- Copy the customer name exactly as written, including capitals and punctuation.\n"
    "- If the line is unreadable, answer with the fields you can read and leave the rest empty.\n"
    "- Return the JSON object alone, with no explanation around it."
)


def test_the_model_may_only_add_to_the_goal_template(tmp_path: Path) -> None:
    client = FakeClient(turns=[ELABORATION])
    transport = onboard.ScriptedTransport(answers())
    interview = onboard.run_interview(transport, domains_dir=tmp_path)
    goal = onboard.compose_goal(interview, ("customer", "order"), [], client)
    text, source = goal
    assert source == "elaborated"
    assert "## What done means" in text and "never invent an order number" in text
    assert text.index("never invent") < text.index("## Tools")


def test_an_unusable_reply_leaves_the_template_alone(tmp_path: Path) -> None:
    transport = onboard.ScriptedTransport(answers())
    interview = onboard.run_interview(transport, domains_dir=tmp_path)
    text, source = onboard.compose_goal(interview, ("order",), [], FakeClient(turns=["{}"]))
    assert source == "template"
    assert text == onboard.goal_template(interview, ("order",), [])


# --- end to end: a generated domain really runs ---------------------------------------------------


class GeneratedDomainGateway(FakeClient):
    """Answers every runtime turn with the task's own expected JSON; anything else with prose.

    The architect, diagnose and mutate all talk to this too. They only need a well-formed
    reply, so one piece of prose long enough to pass validation serves all three.
    """

    def __init__(self, tasks: list[Any]) -> None:
        super().__init__(turns=[])
        self.by_request = {t.input["request"]: t for t in tasks}

    def _next(self, **kw: Any) -> FakeRawResponse:
        self.calls.append(kw)
        completion = build_completion(
            self._reply(list(kw.get("messages") or [])), model=str(kw.get("model", "fake"))
        )
        return FakeRawResponse(completion, {"x-tensormux-backend": DEFAULT_BACKEND})

    def _reply(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            payload = str(message.get("content") or "")
            for request, task in self.by_request.items():
                if request in payload:
                    return json.dumps(task.expected)
        return (
            "Work through the order line from left to right. Read the order number first, then"
            " the customer name, and copy both exactly as they are written. Do not round, do"
            " not paraphrase and do not add commentary. When a field is unreadable, leave it"
            " empty rather than guessing at it. Return the JSON object on its own."
        )


def test_a_generated_domain_runs_through_anneal_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acceptance claim: `anneal init` output goes straight into `anneal run`."""
    path = generate(tmp_path / "domains")
    domain = load_domain(path)
    gateway = GeneratedDomainGateway(domain.eval.load_tasks())

    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    monkeypatch.setattr("anneal.prompts.PROMPTS_ROOT", tmp_path / "prompts")
    monkeypatch.setattr("anneal.llm.get_client", lambda tier="mid", path=None: gateway)

    runs_dir = tmp_path / "runs"
    status = cli.main(
        ["run", str(path), "--runs-dir", str(runs_dir), "--iterations", "1",
         "--candidates", "1", "--concurrency", "1", "--budget", "5.00",
         "--ledger", str(tmp_path / "ledger.json"), "--no-memory"]
    )
    assert status == 0

    rows = [
        json.loads(line)
        for file in sorted((runs_dir / "order_desk" / "0").glob("*.search.*.jsonl"))
        for line in file.read_text().splitlines()
        if line.strip()
    ]
    assert rows, "no task rows were written"
    assert any(row["score"] >= domain.eval.THRESHOLD for row in rows)
