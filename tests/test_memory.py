"""Offline tests for anneal.memory: reflect, recall, credit/prune, and the runtime wiring.

Every LLM call is a ``tests.fakes.FakeClient``. The headline test uses a client whose reply
depends on the prompt it is given, so "fails with an empty memory, passes with the rule
learned from that failure" is a real causal claim and not a scripted one.

Only the search split is used here; the reserved evaluation split belongs to the gate.
"""

from __future__ import annotations

import json
import sqlite3
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from anneal import cli, llm, runner
from anneal import memory as memory_mod
from anneal.domain import load_domain
from anneal.memory import MEMORY_HEADING, Entry, Memory
from anneal.memory import PROCEDURE as KIND_PROCEDURE
from anneal.memory import RULE as KIND_RULE
from anneal.runtime import run_task
from anneal.spec import HarnessSpec
from tests import test_cli as cli_tests
from tests.fakes import FakeClient, FakeRawResponse, build_completion


@pytest.fixture
def cli_loop(monkeypatch, tmp_path):  # noqa: ANN001, ANN201 - pytest fixtures
    """The whole-loop fixture from tests.test_cli: architect/runner/gate replaced by fakes."""
    return cli_tests.loop.__wrapped__(monkeypatch, tmp_path)

SEARCH = "search"
RULE = (
    "Basic economy reservations cannot be modified; call get_reservation_details and check "
    "the cabin before offering a flight change."
)


EVIDENCE = '{"cabin": "basic_economy"}'


def rules_reply(*rules: dict[str, Any]) -> str:
    """A reflector reply. Rules cite a tool result unless the test says otherwise."""
    return json.dumps([{"evidence": EVIDENCE, **rule} for rule in rules])


def failing_row(task_id: str, **kw: Any) -> dict[str, Any]:
    row = {
        "task_id": task_id,
        "score": 0.0,
        "trace_id": f"tr-{task_id}",
        "output": '{"action": "change_flight"}',
        "trace": [
            {
                "tool": "get_reservation_details",
                "args": {"id": task_id},
                "result": '{"cabin": "basic_economy"}',
            }
        ],
    }
    row.update(kw)
    return row


class StubDomain:
    """The three-file surface ``memory`` needs, without touching a real domain."""

    name = "toy"
    goal = "Serve airline customers within policy."

    class eval:  # noqa: N801 - mirrors the module attribute the core reads
        THRESHOLD = 1.0


# --- reflect -----------------------------------------------------------------------------


def test_reflect_extracts_a_grounded_rule(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "memory.json")
    client = FakeClient(turns=[rules_reply({"text": RULE, "tool": "get_reservation_details"})])
    added = mem.reflect([failing_row("t1")], StubDomain(), client=client, iteration=2)
    assert [e.text for e in added] == [RULE]
    entry = mem.entries[0]
    assert entry.tool == "get_reservation_details" and entry.evidence == EVIDENCE
    assert entry.source_task_ids == ["t1"] and entry.source_trace_ids == ["tr-t1"]
    assert entry.created_iteration == 2 and entry.status == "active"
    # the prompt carries the tool result the rule has to be grounded in
    prompt = client.calls[0]["messages"][1]["content"]
    assert "basic_economy" in prompt and "get_reservation_details" in prompt


def test_reflect_ignores_passing_rows(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "memory.json")
    client = FakeClient(turns=[rules_reply({"text": RULE})])
    assert mem.reflect([failing_row("t1", score=1.0)], StubDomain(), client=client) == []
    # no failures, so no rule pass; one tool call is under MIN_PROCEDURE_STEPS, so no
    # procedure pass either -- a run with nothing to learn from costs nothing.
    assert client.calls == []


def test_reflect_dedupes_against_existing_entries(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "memory.json")
    restated = (
        "Basic economy reservations cannot be modified: check the cabin with "
        "get_reservation_details before offering any flight change."
    )
    client = FakeClient(
        turns=[
            rules_reply({"text": RULE, "tool": "get_reservation_details"}),
            rules_reply(
                {"text": restated, "tool": "get_reservation_details"},
                {"text": "Always confirm the passenger's identity before quoting a refund."},
            ),
        ]
    )
    mem.reflect([failing_row("t1")], StubDomain(), client=client)
    added = mem.reflect([failing_row("t2")], StubDomain(), client=client)
    assert len(mem.entries) == 2, [e.text for e in mem.entries]
    assert [e.text for e in added] == ["Always confirm the passenger's identity before "
                                       "quoting a refund."]
    # the duplicate folded its provenance into the entry that already covers it
    assert mem.entries[0].source_task_ids == ["t1", "t2"]


def test_reflect_survives_an_unreachable_gateway(tmp_path: Path, monkeypatch) -> None:
    mem = Memory(tmp_path / "memory.json")
    monkeypatch.setattr(
        llm, "get_client", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no key"))
    )
    assert mem.reflect([failing_row("t1")], StubDomain()) == []
    assert mem.entries == []


def test_reflect_grounds_on_what_the_runtime_observed(tmp_path: Path) -> None:
    """Runner rows carry no trace; the runtime hands its tool results to the store."""
    mem = Memory(tmp_path / "memory.json")
    mem.observe("t1", [{"tool": "get_reservation_details", "args": {},
                        "result": '{"cabin": "basic_economy"}'}], "changed")
    client = FakeClient(turns=[rules_reply({"text": RULE})])
    row = {"task_id": "t1", "score": 0.0, "output": "changed"}
    assert mem.reflect([row], StubDomain(), client=client)
    assert "basic_economy" in client.calls[0]["messages"][1]["content"]


@pytest.mark.parametrize(
    "reply", ["not json at all", "```json\n[]\n```", '[{"text": "   "}]', "[1, 2, 3]"]
)
def test_reflect_tolerates_junk_replies(tmp_path: Path, reply: str) -> None:
    mem = Memory(tmp_path / "memory.json")
    assert mem.reflect([failing_row("t1")], StubDomain(), client=FakeClient(turns=[reply])) == []


def test_reflect_drops_a_rule_that_cites_no_tool_result(tmp_path: Path) -> None:
    """A rule the model cannot point at a tool result for is a guess, not a lesson."""
    mem = Memory(tmp_path / "memory.json")
    reply = json.dumps(
        [{"text": "Be more careful with reservations.", "evidence": ""},
         {"text": RULE, "tool": "get_reservation_details", "evidence": EVIDENCE}]
    )
    added = mem.reflect([failing_row("t1")], StubDomain(), client=FakeClient(turns=[reply]))
    assert [e.text for e in added] == [RULE]


def test_reflect_keeps_ungrounded_rules_when_there_was_nothing_to_cite(tmp_path: Path) -> None:
    """A run that made no tool call has no result to quote; its lesson still counts."""
    mem = Memory(tmp_path / "memory.json")
    reply = json.dumps([{"text": "Reply with the required JSON object and nothing else."}])
    row = {"task_id": "t1", "score": 0.0, "output": "sorry!", "trace": []}
    assert len(mem.reflect([row], StubDomain(), client=FakeClient(turns=[reply]))) == 1


def test_reflect_caps_the_rules_it_accepts(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "memory.json")
    many = rules_reply(*({"text": f"Rule number {i} about invoices and currency."}
                         for i in range(9)))
    added = mem.reflect([failing_row("t1")], StubDomain(), client=FakeClient(turns=[many]))
    assert len(added) <= memory_mod.MAX_NEW_RULES


# --- procedures ---------------------------------------------------------------------------

BOOKING_TOOLS = ["get_reservation_details", "search_flights", "check_fare_rules",
                 "hold_seat", "confirm_payment"]
RECIPE = (
    "1. get_reservation_details(id) to read the cabin. 2. search_flights(origin, destination, "
    "date). 3. check_fare_rules on the chosen fare. 4. hold_seat before quoting a price. "
    "5. confirm_payment last. Pitfall: never hold a seat before the fare rules are read."
)


def winning_row(task_id: str, tools: list[str] | None = None, **kw: Any) -> dict[str, Any]:
    """A passing row whose trace is a real multi-step tool sequence."""
    row = {
        "task_id": task_id,
        "score": 1.0,
        "trace_id": f"tr-{task_id}",
        "output": '{"action": "rebooked"}',
        "trace": [{"tool": t, "args": {"id": task_id}, "result": '{"ok": true}'}
                  for t in (tools if tools is not None else BOOKING_TOOLS)],
    }
    row.update(kw)
    return row


def procedure_reply(steps: list[str], text: str = RECIPE) -> str:
    return json.dumps([{"text": text, "steps": steps, "tool": steps[0], "evidence": EVIDENCE}])


def test_reflect_writes_a_procedure_from_a_successful_multi_step_run(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "memory.json")
    client = FakeClient(turns=[procedure_reply(BOOKING_TOOLS)])
    added = mem.reflect([winning_row("w1")], StubDomain(), client=client, iteration=1)
    assert [(e.kind, e.steps) for e in added] == [(KIND_PROCEDURE, BOOKING_TOOLS)]
    assert added[0].created_iteration == 1 and added[0].tool == BOOKING_TOOLS[0]
    # the prompt shows the tool calls in the order they happened
    prompt = client.calls[0]["messages"][1]["content"]
    assert "1. get_reservation_details" in prompt and "5. confirm_payment" in prompt


def test_the_routers_pseudo_step_does_not_count_as_a_tool_call(tmp_path: Path) -> None:
    """tool_router records its choice in the trace; it is not a step of any procedure."""
    trace = [{"tool": "route", "args": {}, "result": "booking"},
             *({"tool": t, "args": {}, "result": "{}"} for t in BOOKING_TOOLS)]
    assert memory_mod.tool_steps(trace) == BOOKING_TOOLS
    mem = Memory(tmp_path / "memory.json")
    client = FakeClient(turns=[procedure_reply(["route", *BOOKING_TOOLS])])
    # a "procedure" that starts by calling the router is not grounded in any real tool call
    assert mem.reflect([winning_row("w1", trace=trace)], StubDomain(), client=client) == []


def test_short_successful_runs_are_not_worth_a_procedure(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "memory.json")
    client = FakeClient(turns=[procedure_reply(BOOKING_TOOLS[:2])])
    assert mem.reflect([winning_row("w1", BOOKING_TOOLS[:2])], StubDomain(), client=client) == []
    assert client.calls == []  # under MIN_PROCEDURE_STEPS, so nothing is even asked


def test_a_procedure_naming_a_tool_the_run_never_called_is_dropped(tmp_path: Path) -> None:
    """Grounding for a procedure is checkable, so it is checked rather than trusted."""
    mem = Memory(tmp_path / "memory.json")
    client = FakeClient(turns=[procedure_reply([*BOOKING_TOOLS, "cancel_reservation"])])
    assert mem.reflect([winning_row("w1")], StubDomain(), client=client) == []


def test_a_procedure_with_no_steps_is_dropped(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "memory.json")
    reply = json.dumps([{"text": RECIPE, "steps": [], "evidence": EVIDENCE}])
    assert mem.reflect([winning_row("w1")], StubDomain(), client=FakeClient(turns=[reply])) == []


def test_the_same_recipe_learned_twice_is_merged(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "memory.json")
    client = FakeClient(
        turns=[procedure_reply(BOOKING_TOOLS),
               procedure_reply(BOOKING_TOOLS, "A completely different wording of the recipe.")]
    )
    mem.reflect([winning_row("w1")], StubDomain(), client=client)
    assert mem.reflect([winning_row("w2")], StubDomain(), client=client) == []
    assert len(mem.entries) == 1
    assert mem.entries[0].source_task_ids == ["w1", "w2"]


def test_one_run_yields_both_a_rule_and_a_procedure(tmp_path: Path) -> None:
    """Failures teach constraints, successes teach recipes; a run can do both."""
    mem = Memory(tmp_path / "memory.json")
    client = FakeClient(
        turns=[rules_reply({"text": RULE, "tool": "get_reservation_details"}),
               procedure_reply(BOOKING_TOOLS)]
    )
    added = mem.reflect([failing_row("t1"), winning_row("w1")], StubDomain(), client=client)
    assert [e.kind for e in added] == [KIND_RULE, KIND_PROCEDURE]
    assert mem.by_kind() == {KIND_RULE: 1, KIND_PROCEDURE: 1}
    # the rule pass sees only the failure, the procedure pass only the success
    assert "task t1" in client.calls[0]["messages"][1]["content"]
    assert "task w1" in client.calls[1]["messages"][1]["content"]
    assert "task t1" not in client.calls[1]["messages"][1]["content"]


def test_a_procedure_can_be_recalled_and_rendered(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "memory.json")
    mem.entries = [
        Entry(id="p", text=RECIPE, domain="toy", kind=KIND_PROCEDURE, steps=BOOKING_TOOLS,
              tool="get_reservation_details"),
        Entry(id="r", text=RULE, domain="toy"),
    ]
    got = mem.recall({"id": "t", "input": "Rebook the reservation and confirm payment."}, 2)
    assert {e.id for e in got} == {"p", "r"}
    block = mem.prompt_block(got)
    assert "## Rules" in block and "## Procedures" in block and RECIPE in block


# --- recall ------------------------------------------------------------------------------


@pytest.fixture(params=["fts5", "overlap"])
def backend(request, monkeypatch):  # noqa: ANN001, ANN201 - pytest fixtures
    """Run the recall tests over FTS5 and over the fallback, so neither rots."""
    if request.param == "overlap":
        monkeypatch.setattr(memory_mod, "fts5_available", lambda: False)
    elif not memory_mod.fts5_available():
        pytest.skip("this Python's sqlite3 has no FTS5")
    return request.param


def stocked(tmp_path: Path) -> Memory:
    mem = Memory(tmp_path / "memory.json")
    mem.entries = [
        Entry(id="a", text="Basic economy reservations cannot be modified.", domain="toy"),
        Entry(id="b", text="Escalate an invoice whose purchase order currency differs.",
              domain="toy", tool="get_invoice"),
        Entry(id="c", text="Refund a cancelled reservation to the original payment method.",
              domain="toy"),
    ]
    return mem


def test_recall_ranks_by_overlap_and_is_deterministic(backend: str, tmp_path: Path) -> None:
    mem = stocked(tmp_path)
    task = {"id": "t1", "input": "Modify the basic economy reservation for Ada."}
    first = [e.id for e in mem.recall(task, 2)]
    mem.begin_run()
    # the cabin rule outranks the merely reservation-shaped one, and repeats exactly
    assert first[0] == "a"
    assert first == [e.id for e in mem.recall(task, 2)]


def test_recall_prefers_an_entry_whose_tool_the_task_names(backend: str, tmp_path: Path) -> None:
    mem = stocked(tmp_path)
    task = {"id": "t2", "input": "Post this invoice after calling get_invoice."}
    assert [e.id for e in mem.recall(task, 1)] == ["b"]


def test_recall_returns_nothing_for_an_unrelated_task(backend: str, tmp_path: Path) -> None:
    assert stocked(tmp_path).recall({"id": "t3", "input": "Reboot the printer."}, 3) == []


def test_recall_honours_k_and_skips_retired_entries(backend: str, tmp_path: Path) -> None:
    mem = stocked(tmp_path)
    assert mem.recall({"id": "t4", "input": "reservation invoice refund"}, 0) == []
    mem.entries[0].status = "retired"
    task = {"id": "t4", "input": "reservation invoice refund"}
    got = [e.id for e in mem.recall(task, 5)]
    mem.begin_run()
    assert "a" not in got and got == [e.id for e in mem.recall(task, 5)]


def test_recall_records_what_it_injected(backend: str, tmp_path: Path) -> None:
    mem = stocked(tmp_path)
    injected = [e.id for e in mem.recall({"id": "t1", "input": "basic economy reservation"}, 2)]
    assert injected[0] == "a"
    assert mem.injections["t1"] == injected == mem.injected_ids()
    mem.begin_run()
    assert mem.injections == {} and mem.injected_ids() == []


# --- credit and prune --------------------------------------------------------------------


def test_credit_and_prune_retire_a_losing_entry(tmp_path: Path) -> None:
    mem = stocked(tmp_path)
    loser, keeper = mem.entries[0], mem.entries[1]
    for passed in (False, False, True):
        mem.credit([loser], passed)
    for passed in (True, True, False):
        mem.credit([keeper], passed)
    assert (loser.hits, loser.wins, loser.losses) == (3, 1, 2)
    assert [e.id for e in mem.prune()] == ["a"]
    assert loser.status == "retired" and keeper.status == "active"
    assert [e.id for e in mem.active] == ["b", "c"]


def test_prune_waits_for_a_minimum_sample(tmp_path: Path) -> None:
    mem = stocked(tmp_path)
    mem.credit([mem.entries[0]], False)
    assert mem.prune() == [] and mem.entries[0].status == "active"


def test_credit_run_uses_the_injection_log(tmp_path: Path) -> None:
    mem = stocked(tmp_path)
    mem.recall({"id": "t1", "input": "basic economy reservation"}, 1)
    assert mem.credit_run([{"task_id": "t1", "score": 1.0}, {"task_id": "other", "score": 0.0}],
                          1.0) == 1
    assert (mem.entries[0].hits, mem.entries[0].wins) == (1, 1)


# --- persistence -------------------------------------------------------------------------


def test_store_round_trips_through_sqlite(tmp_path: Path) -> None:
    mem = stocked(tmp_path)
    mem.entries[0].wins = 4
    mem.entries.append(Entry(id="p", text=RECIPE, domain="toy", kind=KIND_PROCEDURE,
                             steps=BOOKING_TOOLS, evidence=EVIDENCE))
    path = mem.save()
    assert path == tmp_path / "memory.db" and path.is_file()
    reloaded = Memory.load(path)
    assert [e.id for e in reloaded.entries] == ["a", "b", "c", "p"]
    assert reloaded.entries[0].wins == 4 and reloaded.entries[1].tool == "get_invoice"
    procedure = reloaded.entries[3]
    assert procedure.kind == KIND_PROCEDURE and procedure.steps == BOOKING_TOOLS


def test_the_json_export_is_written_beside_the_db(tmp_path: Path) -> None:
    """The dashboard and a human read the export; SQLite is only the store of record."""
    stocked(tmp_path).save()
    exported = json.loads((tmp_path / "memory.json").read_text())
    assert [e["id"] for e in exported["entries"]] == ["a", "b", "c"]
    assert exported["entries"][0]["kind"] == KIND_RULE


def test_the_db_is_queryable_without_this_module(tmp_path: Path) -> None:
    stocked(tmp_path).save()
    with sqlite3.connect(tmp_path / "memory.db") as conn:
        rows = conn.execute("SELECT id, kind, status FROM entries ORDER BY id").fetchall()
    assert rows == [("a", KIND_RULE, "active"), ("b", KIND_RULE, "active"),
                    ("c", KIND_RULE, "active")]


def test_a_json_only_store_still_loads(tmp_path: Path) -> None:
    """Stores written before the SQLite index existed are read from the export."""
    (tmp_path / "memory.json").write_text(
        json.dumps({"entries": [{"id": "old", "text": RULE, "domain": "toy"}]})
    )
    loaded = Memory.load(tmp_path / "memory.db")
    assert [e.id for e in loaded.entries] == ["old"] and loaded.entries[0].kind == KIND_RULE


def test_missing_or_corrupt_store_loads_empty(tmp_path: Path) -> None:
    assert Memory.load(tmp_path / "nope.json").entries == []
    (tmp_path / "bad.json").write_text("{ not json")
    assert Memory.load(tmp_path / "bad.json").entries == []
    (tmp_path / "bad.db").write_bytes(b"this is not a database")
    assert Memory.load(tmp_path / "bad.db").entries == []


def test_recall_stays_fast_on_a_large_store(tmp_path: Path) -> None:
    """A measured floor, not a claim: 1k entries must not make recall unusable."""
    mem = Memory(tmp_path / "memory.json", [
        Entry(id=f"e{i:04d}", domain="toy",
              text=f"Escalate a {i} dollar refund when the reservation fare class is restricted.")
        for i in range(1000)
    ])
    task = {"id": "t", "input": "Refund a restricted fare reservation for 42 dollars."}
    mem.recall(task, 3)  # first call pays for building the index
    start = time.perf_counter()
    for _ in range(20):
        mem.begin_run()
        assert len(mem.recall(task, 3)) == 3
    per_call_ms = (time.perf_counter() - start) / 20 * 1000
    assert per_call_ms < 100, per_call_ms


# --- a domain the runtime can actually run -----------------------------------------------

CABIN_TOOLS = textwrap.dedent(
    """
    tools:
    - name: get_reservation_details
      description: Look up one reservation.
      args: {type: object, properties: {id: {type: string}}, required: [id]}
      impl: python:tests.test_memory.get_reservation_details
    """
)

CABIN_EVAL = textwrap.dedent(
    '''
    """Toy policy domain: a basic economy reservation may not be changed."""
    THRESHOLD = 1.0

    TASKS = [{"id": "cabin-1", "split": "search",
              "input": "Change the flight on reservation R1 to tomorrow."}]

    def load_tasks(split=None):
        return [t for t in TASKS if split in (None, t["split"])]

    def score(task, output):
        got = output if isinstance(output, dict) else {}
        return 1.0 if got.get("action") == "refuse" else 0.0

    def is_hard_fail(task, trace):
        return False
    '''
)


def get_reservation_details(id: str) -> str:  # noqa: A002 - mirrors the manifest arg name
    return json.dumps({"id": id, "cabin": "basic_economy", "status": "confirmed"})


@pytest.fixture
def cabin_domain(tmp_path: Path) -> Any:
    domain_dir = tmp_path / "cabin"
    domain_dir.mkdir()
    (domain_dir / "goal.md").write_text("Change flights only when policy allows it.")
    (domain_dir / "tools.yaml").write_text(CABIN_TOOLS)
    (domain_dir / "eval.py").write_text(CABIN_EVAL)
    return load_domain(domain_dir)


def cabin_spec(*, memory: bool, top_k: int = 3) -> HarnessSpec:
    return HarnessSpec.model_validate(
        {
            "id": "cand-cabin",
            "topology": "single",
            "step_budget": 6,
            "memory": {"enabled": memory, "kind": "episodic" if memory else "none",
                       "top_k": top_k},
            "nodes": [
                {
                    "name": "executor",
                    "role": "executor",
                    "model_tier": "mid",
                    "system_prompt_ref": "cabin/executor@v1",
                    "tools": ["get_reservation_details"],
                    "max_steps": 4,
                }
            ],
        }
    )


class PolicyClient(FakeClient):
    """A model that only refuses when the prompt tells it basic economy is unchangeable.

    It is not scripted: the reply is a function of the system prompt it is handed, so a
    passing run is caused by the recalled rule and by nothing else.
    """

    def _next(self, **kw: Any) -> FakeRawResponse:
        self.calls.append(kw)
        messages = kw["messages"]
        system = str(messages[0]["content"]).lower()
        if not any(m.get("role") == "tool" for m in messages):
            turn: Any = [{"name": "get_reservation_details",
                          "arguments": json.dumps({"id": "R1"})}]
        elif MEMORY_HEADING.lower() in system and "basic economy" in system:
            turn = json.dumps({"action": "refuse", "reason": "basic economy is not changeable"})
        else:
            turn = json.dumps({"action": "change_flight"})
        return FakeRawResponse(
            build_completion(turn, model=str(kw.get("model", "fake-model"))),
            {"x-tensormux-backend": "fake-backend"},
        )


def run_cabin(domain: Any, spec: HarnessSpec, mem: Memory | None) -> tuple[Any, PolicyClient]:
    client = PolicyClient()
    task = domain.eval.load_tasks(SEARCH)[0]
    with memory_mod.activate(mem):
        result = run_task(spec, task, domain, client_factory=lambda tier: client)
    return result, client


# --- the runtime actually reads memory ---------------------------------------------------


def test_runtime_injects_recalled_rules_into_the_prompt(cabin_domain: Any, tmp_path: Path) -> None:
    mem = Memory(tmp_path / "memory.json", [Entry(id="a", text=RULE, domain="cabin")])
    result, client = run_cabin(cabin_domain, cabin_spec(memory=True), mem)
    system = client.calls[0]["messages"][0]["content"]
    assert MEMORY_HEADING in system and RULE in system
    assert result.memory_ids == ["a"]
    assert mem.injections["cabin-1"] == ["a"]


def test_runtime_ignores_memory_when_the_spec_does_not_ask_for_it(
    cabin_domain: Any, tmp_path: Path
) -> None:
    mem = Memory(tmp_path / "memory.json", [Entry(id="a", text=RULE, domain="cabin")])
    result, client = run_cabin(cabin_domain, cabin_spec(memory=False), mem)
    assert MEMORY_HEADING not in client.calls[0]["messages"][0]["content"]
    assert result.memory_ids == [] and mem.injections == {}


def test_runtime_records_tool_results_for_the_next_reflection(
    cabin_domain: Any, tmp_path: Path
) -> None:
    mem = Memory(tmp_path / "memory.json")
    run_cabin(cabin_domain, cabin_spec(memory=True), mem)
    assert "basic_economy" in str(mem.observed("cabin-1"))


def test_runtime_without_an_active_store_is_unchanged(cabin_domain: Any) -> None:
    result, client = run_cabin(cabin_domain, cabin_spec(memory=True), None)
    assert MEMORY_HEADING not in client.calls[0]["messages"][0]["content"]
    assert result.memory_ids == []


def test_recall_survives_the_runner_thread_pool(
    cabin_domain: Any, tmp_path: Path, monkeypatch  # noqa: ANN001 - pytest fixture
) -> None:
    """The production path is runner.run -> asyncio -> a worker thread; the store must reach it."""
    mem = Memory(tmp_path / "memory.json", [Entry(id="a", text=RULE, domain="cabin")])
    monkeypatch.setattr(llm, "get_client", lambda *a, **kw: PolicyClient())
    with memory_mod.activate(mem):
        rows = runner.run(cabin_spec(memory=True), cabin_domain, SEARCH, runs_dir=tmp_path)
    assert mem.injections["cabin-1"] == ["a"]
    assert [r["score"] for r in rows] == [1.0]
    assert "basic_economy" in str(mem.observed("cabin-1"))


# --- the headline: fail empty, learn, pass -----------------------------------------------


def test_task_fails_with_empty_memory_and_passes_with_the_learned_rule(
    cabin_domain: Any, tmp_path: Path
) -> None:
    mem = Memory(tmp_path / "memory.json")
    spec = cabin_spec(memory=True)
    task = cabin_domain.eval.load_tasks(SEARCH)[0]

    # run 1: nothing learned yet, the agent changes a basic economy flight and fails
    first, _ = run_cabin(cabin_domain, spec, mem)
    assert cabin_domain.eval.score(task, first.output) == 0.0
    assert first.memory_ids == []

    # reflect on that failure, using the tool results the run actually saw
    row = {"task_id": task["id"], "score": 0.0, "trace_id": first.trace_id,
           "output": first.output, "trace": first.trace}
    reflector = FakeClient(turns=[rules_reply({"text": RULE, "tool": "get_reservation_details"})])
    assert mem.reflect([row], cabin_domain, client=reflector, iteration=0)
    mem.save()

    # run 2, a separate store loaded from disk: the same agent now passes
    reloaded = Memory.load(tmp_path / "memory.json")
    reloaded.begin_run()
    second, _ = run_cabin(cabin_domain, spec, reloaded)
    assert cabin_domain.eval.score(task, second.output) == 1.0
    assert second.memory_ids == [reloaded.entries[0].id]

    # and the entry that made the difference is credited with the win
    reloaded.credit_run([{"task_id": task["id"], "score": 1.0}], cabin_domain.eval.THRESHOLD)
    assert (reloaded.entries[0].hits, reloaded.entries[0].wins) == (1, 1)


# --- the loop persists a growing store ---------------------------------------------------


class GrowingReflector(FakeClient):
    """Returns a different grounded rule on every call, so the store grows monotonically."""

    RULES = (
        "Refuse a flight change on a basic economy cabin; quote the fare rules instead.",
        "Escalate any refund request above five hundred dollars to a human supervisor.",
        "Confirm the passenger identity with a membership number before disclosing an itinerary.",
        "Never book a segment whose departure airport differs from the stated origin.",
    )

    def _next(self, **kw: Any) -> FakeRawResponse:
        self.calls.append(kw)
        rule = self.RULES[(len(self.calls) - 1) % len(self.RULES)]
        turn = rules_reply({"text": rule, "tool": "get_reservation_details"})
        return FakeRawResponse(
            build_completion(turn, model=str(kw.get("model", "fake-model"))),
            {"x-tensormux-backend": "fake-backend"},
        )


def test_summary_reports_a_memory_that_grows_across_iterations(
    cli_loop, tmp_path: Path, monkeypatch  # noqa: ANN001 - pytest fixtures
) -> None:
    reflector = GrowingReflector()
    monkeypatch.setattr(llm, "get_client", lambda *a, **kw: reflector)
    assert cli.main(["run", "domains/airline", "--runs-dir", str(tmp_path / "runs"),
                     "--iterations", "2", "--candidates", "1"]) == 0
    root = tmp_path / "runs" / "airline"
    counts = [
        json.loads((root / str(i) / "summary.json").read_text())["memory_entries"]
        for i in (0, 1)
    ]
    assert counts == sorted(counts) and counts[-1] > counts[0], counts
    stored = Memory.load(root / "memory.json")
    assert len(stored.entries) == counts[-1] >= 2


def test_every_candidate_reflects_on_its_own_run(cli_loop, tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """Rules must cite the run that produced them, so reflection follows each search run."""
    seen: list[tuple[str, set[str]]] = []
    monkeypatch.setattr(
        cli, "_learn",
        lambda loop, candidate, rows, iteration: seen.append(
            (candidate.id, {r["candidate_id"] for r in rows})
        ),
    )
    assert cli.main(["run", "domains/airline", "--runs-dir", str(tmp_path / "runs"),
                     "--iterations", "1", "--candidates", "3"]) == 0
    assert [c for c, _ in seen] == ["cand-1", "cand-2", "cand-3", "cand-1-m1"]
    assert all(rows == {candidate} for candidate, rows in seen)


def test_no_memory_flag_leaves_the_store_alone(cli_loop, tmp_path: Path) -> None:  # noqa: ANN001
    assert cli.main(["run", "domains/airline", "--runs-dir", str(tmp_path / "runs"),
                     "--iterations", "1", "--candidates", "1", "--no-memory"]) == 0
    root = tmp_path / "runs" / "airline"
    assert not (root / "memory.json").exists()
    assert json.loads((root / "0" / "summary.json").read_text())["memory_entries"] == 0
