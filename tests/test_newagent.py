"""Creating and running an agent from the browser: the chain a person actually walks.

Before these, the product's answer to "how do I make an agent" started with a terminal, and an
agent that had never run did not appear on the index at all. So a person could create one and
watch it vanish. Each test here pins one link of that chain.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from anneal import dashboard, newagent, onboard

FORM = {
    "name": "expense checks",
    "job": "An expense claim arrives as one line. Say whether it can be paid or needs a manager.",
    "tools": "none",
    "tools_target": "",
    "success": "label",
    "given_0": "Taxi to airport, 42 GBP, receipt attached.", "expected_0": "pay",
    "given_1": "Dinner for eight, 310 GBP, no receipt.", "expected_1": "manager",
    "given_2": "Train ticket, 18 GBP, receipt attached.", "expected_2": "pay",
}


def test_the_form_asks_the_same_questions_the_terminal_does() -> None:
    """The form is a transport for the interview, not a second copy of it."""
    body = newagent.form_body()
    for question in onboard.SCRIPT:
        if question.kind == "choice":
            for _value, label in question.choices:
                assert label in body, label
    assert "placeholder=\"\"" in body  # no placeholder-as-label on the example boxes


def test_a_submission_becomes_the_queue_the_script_asks_for() -> None:
    submission = newagent.parse(FORM)
    assert submission.queue() == [
        "expense checks", FORM["job"], "none", "label",
        FORM["given_0"], "pay", FORM["given_1"], "manager", FORM["given_2"], "pay", "", "",
    ]


def test_a_tool_answer_carries_exactly_one_follow_up() -> None:
    """tools_target and tools_module are gated on the tools answer; only one is ever asked."""
    mcp_form = {**FORM, "tools": "mcp", "mcp_choice": "other", "mcp_other": "x"}
    assert newagent.parse(mcp_form).queue()[:4] == ["expense checks", FORM["job"], "mcp", "x"]
    py_form = {**FORM, "tools": "python", "python_choice": "usercode.mine"}
    assert newagent.parse(py_form).queue()[:4] == [
        "expense checks", FORM["job"], "python", "usercode.mine",
    ]
    # nothing chosen means the follow-up is never asked, so the queue moves on to the scorer
    assert newagent.parse({**FORM, "tools": "none"}).queue()[3] == "label"


def test_a_catalogue_pick_becomes_its_launch_command_on_the_server() -> None:
    """The person picks what the agent should be able to do. A command is our problem."""
    picked = newagent.parse({**FORM, "tools": "mcp", "mcp_choice": "files",
                             "files_path": "/tmp/box"})
    assert picked.tools_target == (
        "npx -y @modelcontextprotocol/server-filesystem /tmp/box"
    )
    default = newagent.parse({**FORM, "tools": "mcp", "mcp_choice": "memory"})
    assert default.tools_target == "npx -y @modelcontextprotocol/server-memory"


def test_every_catalogue_entry_names_a_real_published_server() -> None:
    """Each command in the catalogue was checked against the npm registry, not remembered."""
    for key, title, blurb, command in newagent.CATALOGUE:
        assert command.startswith("npx -y @modelcontextprotocol/server-"), command
        assert title and blurb and key
        assert "\u2014" not in blurb


def test_the_picker_never_offers_this_repository_s_own_fixtures(tmp_path: Path) -> None:
    """It used to sweep */*/fixtures/tools.py and show a person our benchmark plumbing.

    Somebody creating an agent was offered domains.airline.fixtures.tools and three like it.
    Those exist to make our own test agents work. They are not the person's code, they are not
    usable by them, and being shown them is how the picker read as random.
    """
    fixtures = tmp_path / "domains" / "airline" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "tools.py").write_text("def get_user_details(x):\n    return x\n", encoding="utf-8")
    (tmp_path / "usercode").mkdir()
    (tmp_path / "usercode" / "mine.py").write_text("def send(x):\n    return x\n", encoding="utf-8")
    assert [m.dotted for m in newagent.python_modules(tmp_path)] == ["usercode.mine"]


def test_an_empty_usercode_folder_says_what_to_put_in_it(tmp_path: Path, monkeypatch) -> None:
    """An empty list is not an answer to "which functions". Say where they go."""
    monkeypatch.setattr(newagent, "ROOT", tmp_path)
    body = newagent.form_body()
    assert "There are none yet" in body
    assert "usercode/" in body
    assert "docstring" in body  # and how to write one
    assert "on this machine" in body  # which machine matters, and it is this one


def test_a_file_name_cannot_be_typed_at_all(tmp_path: Path, monkeypatch) -> None:
    """A typed name is a promise about a file on this machine that nothing can check yet.

    Someone naming a file from a different computer, or misspelling one, used to find out only
    after an agent had been created with nothing to call. Offering exactly what is present
    makes that mistake unmakeable rather than reportable.
    """
    body = newagent.form_body()
    assert "python_other" not in body
    assert "A different file" not in body
    # and a posted value for one is ignored rather than trusted
    smuggled = newagent.parse({**FORM, "tools": "python", "python_other": "usercode.nope"})
    assert smuggled.tools_target == ""


def test_an_agent_is_not_created_with_nothing_to_call(tmp_path: Path, monkeypatch) -> None:
    """Discovery failing is the reason to stop, not a footnote beside a success."""
    monkeypatch.setattr(dashboard, "DOMAINS_DIR", tmp_path / "domains")
    (tmp_path / "domains").mkdir()
    client = TestClient(dashboard.create_app(tmp_path / "runs", tmp_path / "l.json"))
    named_elsewhere = {
        **FORM, "tools": "python", "python_choice": "usercode.a_file_on_another_computer",
    }
    page = client.post("/new", data=named_elsewhere)
    assert page.status_code == 400
    assert "No functions were found" in page.text
    assert "on this machine" in page.text
    assert not (tmp_path / "domains" / "expense_checks").exists()  # nothing was written


def test_a_server_that_will_not_start_stops_the_form(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "DOMAINS_DIR", tmp_path / "domains")
    (tmp_path / "domains").mkdir()
    client = TestClient(dashboard.create_app(tmp_path / "runs", tmp_path / "l.json"))
    unreachable = {**FORM, "tools": "mcp", "mcp_choice": "other",
                   "mcp_other": "http://127.0.0.1:9/nothing-here"}
    page = client.post("/new", data=unreachable)
    assert page.status_code == 400
    assert "gave Anneal no tools" in page.text


def test_the_python_picker_lists_modules_that_are_actually_here(tmp_path: Path) -> None:
    """Typing a dotted path from memory and finding out later is not an experience."""
    (tmp_path / "usercode").mkdir()
    (tmp_path / "usercode" / "mine.py").write_text(
        "def send(x):\n    return x\n\ndef _hidden():\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "usercode" / "_skip.py").write_text("def a():\n    pass\n", encoding="utf-8")
    found = newagent.python_modules(tmp_path)
    assert [m.dotted for m in found] == ["usercode.mine"]
    assert found[0].functions == ["send"]


def test_the_form_offers_choices_instead_of_asking_for_a_command() -> None:
    body = newagent.form_body()
    for _key, title, _blurb, _command in newagent.CATALOGUE:
        assert title in body, title
    assert "Where are they?" not in body  # the question nobody could answer
    assert 'data-when="mcp"' in body and 'data-when="python"' in body


def test_half_filled_example_rows_are_dropped_not_sent_as_blanks() -> None:
    partial = {**FORM, "given_3": "A row with no expected answer", "expected_3": ""}
    assert len(newagent.parse(partial).examples) == 3


def test_the_folder_a_files_agent_needs_is_made_before_the_server_starts(tmp_path: Path) -> None:
    """Picking "work with files in a folder" is the decision. Creating it is not their job.

    Without this the server refuses to start, the interview carries on with no tools, and the
    agent is created with nothing to call.
    """
    folder = tmp_path / "box" / "inner"
    submission = newagent.parse({
        **FORM, "tools": "mcp", "mcp_choice": "files", "files_path": str(folder),
    })
    assert not folder.exists()
    submission.prepare()
    assert folder.is_dir()


def test_prepare_touches_nothing_for_the_other_answers(tmp_path: Path) -> None:
    for form in ({**FORM, "tools": "none"},
                 {**FORM, "tools": "python", "python_choice": "usercode.x"},
                 {**FORM, "tools": "mcp", "mcp_choice": "other",
                  "mcp_other": "https://example.test/mcp"}):
        newagent.parse(form).prepare()  # no directory is created for any of these


def test_what_the_interview_had_to_say_reaches_the_page(tmp_path: Path, monkeypatch) -> None:
    """A server that would not start was recorded and then dropped, so the page said "ready"."""
    monkeypatch.setattr(dashboard, "DOMAINS_DIR", tmp_path / "domains")
    (tmp_path / "domains").mkdir()
    client = TestClient(dashboard.create_app(tmp_path / "runs", tmp_path / "l.json"))
    page = client.post("/new", data=FORM)
    assert page.status_code == 200
    # three examples is enough to run and too few to mean anything, and it says so
    assert "too few for the reserved evaluation split" in page.text


def test_posting_the_form_writes_a_domain_that_loads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "DOMAINS_DIR", tmp_path / "domains")
    (tmp_path / "domains").mkdir()
    client = TestClient(dashboard.create_app(tmp_path / "runs", tmp_path / "l.json"))
    page = client.post("/new", data=FORM)
    assert page.status_code == 200
    assert "agent is ready" in page.text
    written = tmp_path / "domains" / "expense_checks"
    for name in ("goal.md", "tools.yaml", "eval.py", "tasks.jsonl"):
        assert (written / name).is_file(), name


def test_the_interviews_refusal_is_shown_on_the_form_not_as_a_traceback(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(dashboard, "DOMAINS_DIR", tmp_path / "domains")
    (tmp_path / "domains").mkdir()
    client = TestClient(dashboard.create_app(tmp_path / "runs", tmp_path / "l.json"))
    too_few = {k: v for k, v in FORM.items() if not k.startswith(("given_2", "expected_2"))}
    page = client.post("/new", data=too_few)
    assert page.status_code == 400
    assert "at least 3 examples" in page.text  # the reason, not "answers exhausted"
    assert "Traceback" not in page.text


# --- the index, and running from it --------------------------------------------------------


def _with_agent(tmp_path: Path, monkeypatch) -> TestClient:
    domains = tmp_path / "domains"
    monkeypatch.setattr(dashboard, "DOMAINS_DIR", domains)
    domains.mkdir()
    client = TestClient(dashboard.create_app(tmp_path / "runs", tmp_path / "l.json"))
    assert client.post("/new", data=FORM).status_code == 200
    return client


def test_an_agent_that_has_never_run_still_appears_with_a_way_to_run_it(
    tmp_path: Path, monkeypatch
) -> None:
    """The whole point: creating one used to make it invisible until it had produced numbers."""
    body = _with_agent(tmp_path, monkeypatch).get("/agents").text
    assert "An expense claim arrives as one line" in body
    assert "Never run" in body
    assert 'formaction="/agents/expense_checks/run"' in body


def test_running_an_agent_starts_a_process_and_returns_to_the_index(
    tmp_path: Path, monkeypatch
) -> None:
    client = _with_agent(tmp_path, monkeypatch)
    started: list[str] = []
    app_state = client.app.state.anneal

    def fake_start(domain: str, **_: object) -> None:
        started.append(domain)

    monkeypatch.setattr(app_state.launcher, "start", fake_start)
    page = client.post("/agents/expense_checks/run", follow_redirects=False)
    assert page.status_code == 303
    assert page.headers["location"] == "/agents"
    assert started == ["expense_checks"]


def test_a_run_that_cannot_start_does_not_five_hundred(tmp_path: Path, monkeypatch) -> None:
    client = _with_agent(tmp_path, monkeypatch)
    page = client.post("/agents/no-such-agent/run", follow_redirects=False)
    assert page.status_code == 303  # reported in the log, the reader is sent back to the index


def test_an_unknown_page_is_a_page(tmp_path: Path) -> None:
    client = TestClient(dashboard.create_app(tmp_path / "runs", tmp_path / "l.json"))
    page = client.get("/nothing-here")
    assert page.status_code == 404
    assert "No such page" in page.text
    assert 'href="/agents"' in page.text


def test_the_launcher_refuses_to_start_the_same_agent_twice(tmp_path: Path) -> None:
    from anneal import launcher as launcher_mod

    (tmp_path / "domains" / "d").mkdir(parents=True)
    lab = launcher_mod.Launcher(tmp_path / "runs", tmp_path / "domains")

    class Fake:
        returncode = None

        def poll(self) -> None:
            return None

    lab.active["d"] = launcher_mod.Run("d", Fake(), tmp_path / "x.log")
    with pytest.raises(RuntimeError, match="already running"):
        lab.start("d")


def test_the_launcher_picks_the_local_ladder_without_a_key(monkeypatch) -> None:
    """Someone who clicked Run was never asked about model providers."""
    from anneal import config
    from anneal import launcher as launcher_mod

    monkeypatch.setattr(config, "env", lambda name, default=None: "" )
    assert launcher_mod.default_ladder() == launcher_mod.LOCAL_LADDER
    monkeypatch.setattr(config, "env", lambda name, default=None: "sk-real-key")
    assert launcher_mod.default_ladder() == launcher_mod.HOSTED_LADDER


def test_no_em_dash_reaches_a_reader() -> None:
    """The em-dash is our writing tic, and it shows up in product copy as an AI tell.

    Docstrings and comments are ours to write however we like. Anything a person reads on a
    page is not: every question the interview asks, every choice label, every helper line.
    """
    from anneal import vocab

    strings: list[str] = []
    for question in onboard.SCRIPT:
        strings += [question.prompt, question.help, *(label for _, label in question.choices)]
    strings += [name for name, _ in vocab.STEPS.values()]
    strings += list(vocab.TOPOLOGIES.values()) + list(vocab.ROLES.values())
    strings += list(vocab.OPERATORS.values()) + list(vocab.DECISIONS.values())
    strings.append(newagent.form_body())
    # and the console's own rendered pages, which is where two slipped through as &mdash;
    from anneal import dashboard

    strings += [dashboard.CSS, *[b for _, b in vocab.STEPS.values()]]
    strings = [t.replace(dashboard.DASH, "") for t in strings]  # the missing-value glyph stays
    for text in strings:
        assert "\u2014" not in text and "\u2013" not in text, text[:80]
        assert "&mdash;" not in text and "&ndash;" not in text, text[:80]


def test_a_module_is_named_by_what_it_says_it_is_for(tmp_path: Path) -> None:
    """A dotted import path tells nobody what the code does. The file's own first line does."""
    (tmp_path / "usercode").mkdir()
    (tmp_path / "usercode" / "roomdesk_tools.py").write_text(
        '"""Meeting-room desk: the calls a booking assistant may make."""\n'
        "def book_room(room):\n    return room\n",
        encoding="utf-8",
    )
    (tmp_path / "usercode" / "bare.py").write_text("def go():\n    return 1\n", encoding="utf-8")
    by_name = {m.dotted: m for m in newagent.python_modules(tmp_path)}
    assert by_name["usercode.roomdesk_tools"].title == (
        "Meeting-room desk: the calls a booking assistant may make."
    )
    assert by_name["usercode.roomdesk_tools"].summary() == "1 function: book_room"
    # a file with no docstring falls back to its own name turned into words
    assert by_name["usercode.bare"].title == "Bare"


def test_the_page_says_which_agent_already_uses_a_module(tmp_path: Path) -> None:
    """"What is the relation" is a fair question, and the answer is recorded in tools.yaml."""
    (tmp_path / "usercode").mkdir()
    (tmp_path / "usercode" / "desk.py").write_text(
        '"""Desk tools."""\ndef book(x):\n    return x\n', encoding="utf-8"
    )
    agent = tmp_path / "domains" / "roomdesk"
    agent.mkdir(parents=True)
    (agent / "goal.md").write_text("# Goal: Book a meeting room\n", encoding="utf-8")
    (agent / "tools.yaml").write_text(
        "tools:\n- name: book\n  description: d\n  impl: python:usercode.desk.book\n",
        encoding="utf-8",
    )
    found = newagent.python_modules(tmp_path)
    assert found[0].used_by == ["Book a meeting room"]


def test_the_form_never_says_importable(tmp_path: Path, monkeypatch) -> None:
    """Jargon a person cannot act on is not help text."""
    body = newagent.form_body()
    for jargon in ("importable", "import path", "dotted"):
        assert jargon not in body.lower(), jargon


def test_an_answer_of_b_is_an_answer_not_a_back_button(tmp_path: Path, monkeypatch) -> None:
    """The terminal has Back. A form posted in one shot does not.

    "b" and "back" are navigation at a prompt and ordinary answers in a form: a label, an
    expected reply, a name. Reading one as navigation swallowed it and pushed every later
    answer a question out of step, so the person was told a value they never typed there was
    not a valid choice.
    """
    monkeypatch.setattr(dashboard, "DOMAINS_DIR", tmp_path / "domains")
    (tmp_path / "domains").mkdir()
    client = TestClient(dashboard.create_app(tmp_path / "runs", tmp_path / "l.json"))
    literal = {
        "name": "grader", "job": "Grade an answer as a or b.", "tools": "none",
        "success": "label",
        "given_0": "First one", "expected_0": "b",
        "given_1": "Second one", "expected_1": "back",
        "given_2": "Third one", "expected_2": "a",
    }
    page = client.post("/new", data=literal)
    assert page.status_code == 200, page.text[:400]
    tasks = (tmp_path / "domains" / "grader" / "tasks.jsonl").read_text(encoding="utf-8")
    assert '"b"' in tasks and '"back"' in tasks  # both survived as answers


def test_the_terminal_keeps_its_back(tmp_path: Path) -> None:
    """Turning it off for the form must not turn it off everywhere."""
    from anneal import onboard as ob

    assert ob.ScriptedTransport(["b"]).ask(ob.SCRIPT[0]) == ob.BACK
    assert ob.ScriptedTransport(["b"], allow_back=False).ask(ob.SCRIPT[0]) == "b"
