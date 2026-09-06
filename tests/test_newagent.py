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
    for kind in ("mcp", "python"):
        queue = newagent.parse({**FORM, "tools": kind, "tools_target": "x"}).queue()
        assert queue[:4] == ["expense checks", FORM["job"], kind, "x"]
    assert newagent.parse({**FORM, "tools": "none", "tools_target": "x"}).queue()[3] == "label"


def test_half_filled_example_rows_are_dropped_not_sent_as_blanks() -> None:
    partial = {**FORM, "given_3": "A row with no expected answer", "expected_3": ""}
    assert len(newagent.parse(partial).examples) == 3


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
