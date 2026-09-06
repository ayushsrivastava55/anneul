"""End-to-end, offline: architect -> runner -> runtime -> diagnose -> mutate -> gate -> report.

Nothing is stubbed out here except the LLM. ``anneal.cli.main`` drives the real loop over the
real ``domains/airline`` domain, and every module writes its real artifacts into a tmp runs
dir; the assertions read those files back. The only fake is a
:class:`~tests.fakes.FakeClient` subclass that routes each request to a scripted reply by
looking at the messages, so the whole run is deterministic and needs no keys.

``--concurrency 1`` is mandatory: ``domains/airline/fixtures/tools.py`` keeps the tau-bench
database in a module-level dict that ``eval.setup`` resets before every task, so parallel
tasks would score each other's database.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from anneal import cli, diagnose, mutate, runner, spec
from anneal.domain import load_domain
from tests.fakes import DEFAULT_BACKEND, FakeClient, FakeRawResponse, build_completion

ROOT = Path(__file__).resolve().parent.parent
DOMAIN_DIR = ROOT / "domains" / "airline"

# Tasks the scripted executor deliberately gives up on: two on search (so diagnose has
# something to classify and mutate something to fix) and two on holdout (so the gate sees a
# non-degenerate pass^3 for both specs).
FAILING_TASK_IDS = frozenset({"airline-46", "airline-03", "airline-29", "airline-06"})

# Marker the scripted architect writes into every node prompt so the runtime request can be
# attributed back to a node even after an operator has rewritten the prompt.
MARKER = "NODE:"

ARCHITECT_SYSTEM = "You design system prompts for nodes of a tool-using agent"
CLASSIFIER_SYSTEM = "You classify why an LLM agent failed a task"
TOOL_DESC_SYSTEM = "You improve tool descriptions for an LLM agent"
FEWSHOT_SYSTEM = "You write worked examples for an LLM agent"


# --- the scripted gateway ------------------------------------------------------------------


class ScriptedGateway(FakeClient):
    """A ``FakeClient`` that answers by request shape instead of by a fixed queue.

    Every conversation the loop starts is recognised from its system prompt (architect,
    diagnose classifier, mutate operator) or, for runtime nodes, from the ``NODE:`` marker the
    scripted architect put in the node prompt. Executor turns replay the task's expected
    tau-bench actions -- so a task passes ``eval.score`` for real -- unless its id is in
    ``failing``, in which case the executor answers without touching a tool and scores 0.
    """

    def __init__(self, tasks: list[Any], failing: frozenset[str]) -> None:
        super().__init__(turns=[])
        self.by_input = {json.dumps(t.input, ensure_ascii=False): t for t in tasks}
        self.failing = failing
        self.unmatched: list[list[dict[str, Any]]] = []

    # FakeClient routes both create() and with_raw_response.create() through _next.
    def _next(self, **kw: Any) -> FakeRawResponse:
        self.calls.append(kw)
        turn = self._reply(list(kw.get("messages") or []))
        completion = build_completion(turn, model=str(kw.get("model", "fake-model")))
        return FakeRawResponse(completion, {"x-tensormux-backend": DEFAULT_BACKEND})

    # --- routing ---

    def _reply(self, messages: list[dict[str, Any]]) -> Any:
        system = str(messages[0].get("content") or "") if messages else ""
        if ARCHITECT_SYSTEM in system:
            return self._architect_prompt(messages)
        if CLASSIFIER_SYSTEM in system:
            return json.dumps({"class": "wrong_tool", "node": "executor"})
        if TOOL_DESC_SYSTEM in system:
            return (
                "Look up or change a reservation.\n"
                "Use when: the user named a reservation id you have already read.\n"
                "Do not use when: the reservation has not been fetched yet."
            )
        if FEWSHOT_SYSTEM in system:
            return "### Example\n\nRead the reservation first, then apply the change.\n"
        return self._runtime_reply(system, messages)

    @staticmethod
    def _architect_prompt(messages: list[dict[str, Any]]) -> str:
        user = str(messages[-1].get("content") or "")
        for role in ("planner", "executor", "critic"):
            if f"The {role.upper()} " in user:
                return f"{MARKER}{role}\nFollow the goal and use the tools you are given."
        return f"{MARKER}executor\nFollow the goal and use the tools you are given."

    def _runtime_reply(self, system: str, messages: list[dict[str, Any]]) -> Any:
        role = _role_of(system)
        task = self._task_of(messages)
        if task is None:
            self.unmatched.append(messages)
            raise AssertionError(f"scripted gateway saw an unroutable request: {messages!r}")
        if role == "planner":
            last = str(_last_user(messages))
            return "DONE" if "Reply DONE" in last else "1. Read the record.\n2. Apply the change."
        if role == "critic":
            return "PASS the answer matches the goal."
        return self._executor_turn(task, messages)

    def _executor_turn(self, task: Any, messages: list[dict[str, Any]]) -> Any:
        already_acted = any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages)
        actions = list(task.expected.get("actions") or [])
        if already_acted or not actions or task.id in self.failing:
            if task.id in self.failing and not already_acted:
                return "I was not able to complete this request."
            return _final_text(task)
        return [{"name": a["name"], "arguments": json.dumps(a["kwargs"])} for a in actions]

    def _task_of(self, messages: list[dict[str, Any]]) -> Any:
        for message in messages:
            if message.get("role") == "user":
                task = self.by_input.get(str(message.get("content") or ""))
                if task is not None:
                    return task
        return None


def _role_of(system: str) -> str:
    for role in ("planner", "critic"):
        if f"{MARKER}{role}" in system or f"You are the {role}." in system:
            return role
    return "executor"


def _last_user(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _final_text(task: Any) -> str:
    """The executor's closing message: it must contain every expected output string."""
    return " ".join(task.expected.get("outputs") or []) or "Done."


# --- fixtures ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def offline_loop(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """Run the real ``anneal run`` loop over domains/airline with only the LLM faked.

    Module-scoped: the loop runs the whole domain three times over (search for every
    architect candidate, then holdout for the incumbent and the mutant), so every assertion
    below reads the artifacts of one shared run rather than paying for its own.
    """
    tmp_path = tmp_path_factory.mktemp("e2e")
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in ("NEATLOGS_API_KEY", "ANNEAL_HOLDOUT_RUNS", "ANNEAL_CONCURRENCY"):
            monkeypatch.delenv(name, raising=False)
        prompts_root = tmp_path / "prompts"
        monkeypatch.setattr("anneal.prompts.PROMPTS_ROOT", prompts_root)
        monkeypatch.setattr(mutate, "PROMPT_STORE", mutate.LocalPromptStore(prompts_root))

        domain = load_domain(DOMAIN_DIR)
        gateway = ScriptedGateway(list(domain.eval.load_tasks()), FAILING_TASK_IDS)
        monkeypatch.setattr("anneal.llm.get_client", lambda tier="mid", path=None: gateway)

        runs_dir = tmp_path / "runs"
        status = cli.main(
            [
                "run",
                str(DOMAIN_DIR),
                "--runs-dir",
                str(runs_dir),
                "--iterations",
                "1",
                "--concurrency",
                "1",
                "--budget",
                "5.00",
            ]
        )
        iteration_dir = runs_dir / "airline" / "0"
        assert status == 0, f"anneal run exited {status}"
        yield {
            "runs_dir": runs_dir,
            "dir": iteration_dir,
            "summary": json.loads((iteration_dir / "summary.json").read_text()),
            "gateway": gateway,
            "domain": domain,
        }


# --- the end-to-end assertions ---------------------------------------------------------------


def test_architect_specs_are_written_and_validate(offline_loop: dict[str, Any]) -> None:
    """Iteration 0 leaves one validated candidate yaml per architect topology on disk."""
    summary = offline_loop["summary"]
    yamls = sorted(offline_loop["dir"].glob("cand-*.yaml"))
    assert len(yamls) >= 3, [p.name for p in yamls]
    topologies = set()
    for path in yamls:
        loaded = spec.load_spec(path)  # validates
        assert loaded.id == path.stem
        assert summary["specs"][loaded.id] == str(path)
        topologies.add(loaded.topology)
    assert {"single", "planner_executor", "critic_loop"} <= topologies


def test_jsonl_rows_match_the_contract_schema(offline_loop: dict[str, Any]) -> None:
    """Every row of every jsonl the run wrote has exactly the CONTRACTS.md keys and types."""
    files = sorted(offline_loop["dir"].glob("*.jsonl"))
    assert files, "the run wrote no per-task jsonl"
    seen = 0
    for path in files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert tuple(row) == runner.ROW_KEYS, (path.name, list(row))
            assert isinstance(row["task_id"], str) and row["task_id"]
            assert row["candidate_id"] == runner.parse_run_name(path)[0]
            assert isinstance(row["iteration"], int)
            assert isinstance(row["score"], float)
            for flag in ("hard_fail", "hit_step_budget", "schema_error"):
                assert isinstance(row[flag], bool)
            assert isinstance(row["tokens_in"], int) and row["tokens_in"] > 0
            assert isinstance(row["tokens_out"], int)
            assert isinstance(row["latency_ms"], float)
            assert row["trace_id"] is None or isinstance(row["trace_id"], str)
            for node in row["per_node"].values():
                assert tuple(node) == runner.NODE_KEYS
            seen += 1
    assert seen >= 30


def test_scored_tasks_really_pass_and_really_fail(offline_loop: dict[str, Any]) -> None:
    """The domain evaluator -- not the fake -- decided these, via the real tau-bench DB."""
    rows = _search_rows(offline_loop)
    by_id = {r["task_id"]: r for r in rows}
    failed_search = FAILING_TASK_IDS & set(by_id)
    assert failed_search, "no scripted failure landed on the search split"
    for task_id in failed_search:
        assert by_id[task_id]["score"] == 0.0
    assert any(r["score"] == 1.0 for r in rows), "no task passed eval.score"


def test_ledger_holds_a_taxonomy_issue(offline_loop: dict[str, Any]) -> None:
    """diagnose wrote a real ledger whose classes all exist in the taxonomy."""
    ledger_path = offline_loop["runs_dir"] / "airline" / "ledger.json"
    ledger = diagnose.load_ledger(ledger_path)
    assert ledger, "diagnose produced no ledger entries"
    taxonomy = diagnose.load_taxonomy()
    specs = [spec.load_spec(p) for p in offline_loop["dir"].glob("cand-*.yaml")]
    nodes = {node.name for candidate in specs for node in candidate.nodes}
    for issue in ledger:
        assert issue["class"] in taxonomy
        assert issue["node"] in nodes
        assert issue["count"] >= 1 and issue["evidence"]
        assert set(issue) >= {"id", "class", "node", "count", "evidence", "status"}
    assert any(issue["operators_tried"] for issue in ledger), "no operator was recorded as tried"


def test_mutant_spec_has_lineage_back_to_the_incumbent(offline_loop: dict[str, Any]) -> None:
    """mutate produced a candidate on disk, lineage pointing at the incumbent and the issue."""
    summary = offline_loop["summary"]
    mutant_id = summary["candidate_id"]
    assert mutant_id and mutant_id != summary["incumbent_id"]
    mutant = spec.load_spec(summary["specs"][mutant_id])
    assert mutant.lineage is not None
    assert mutant.lineage.parent == summary["incumbent_id"]
    assert mutant.lineage.operator == summary["operator"]
    assert mutant.lineage.operator in mutate.OPERATORS
    ledger_ids = {
        i["id"] for i in diagnose.load_ledger(offline_loop["runs_dir"] / "airline" / "ledger.json")
    }
    assert mutant.lineage.ledger_issue in ledger_ids
    incumbent = spec.load_spec(summary["specs"][summary["incumbent_id"]])
    assert mutant.model_dump() != incumbent.model_dump(), "the mutation changed nothing"


def test_gate_json_records_pass3_hard_fails_p_and_a_decision(offline_loop: dict[str, Any]) -> None:
    gate_json = json.loads((offline_loop["dir"] / "gate.json").read_text())
    assert gate_json["decision"] in ("promote", "reject")
    assert 0.0 <= gate_json["p"] <= 1.0
    assert gate_json["reason"]
    for side in ("incumbent", "candidate"):
        block = gate_json[side]
        assert 0.0 <= block["pass3_rate"] <= 1.0
        assert isinstance(block["hard_fails"], int)
        assert block["n_runs"] == 3 and block["n_tasks"] > 0
        assert set(block["pass3"]) and all(isinstance(v, bool) for v in block["pass3"].values())
    assert gate_json["candidate"]["gen_gap"] is not None


def test_summary_has_every_field_report_needs(offline_loop: dict[str, Any]) -> None:
    summary = offline_loop["summary"]
    required = {
        "domain",
        "domain_path",
        "iteration",
        "incumbent_id",
        "candidate_id",
        "winner_id",
        "operator",
        "issue",
        "mean_score",
        "pass3_rate",
        "hard_fails",
        "gen_gap",
        "p_value",
        "cost_usd",
        "p95_latency_ms",
        "decision",
        "reason",
        "spend_usd",
        "budget_usd",
        "search",
        "specs",
        "incumbent",
        "candidate",
    }
    assert required <= set(summary)
    assert summary["winner_id"] in (summary["incumbent_id"], summary["candidate_id"])
    assert summary["issue"]["class"] in diagnose.load_taxonomy()
    for block in (summary["incumbent"], summary["candidate"]):
        assert {"mean_score", "pass3_rate", "hard_fails", "candidate_id"} <= set(block)
    for candidate_id, metrics in summary["search"].items():
        assert candidate_id in summary["specs"]
        assert {"mean_score", "n_tasks", "cost_per_task", "p95_latency_ms"} <= set(metrics)


def test_report_renders_a_markdown_row_from_the_summary(
    offline_loop: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["report", str(offline_loop["runs_dir"])]) == 0
    out = capsys.readouterr().out
    lines = [
        line for line in out.splitlines()
        if any(line.startswith(f"| airline | {stage} |") for stage in ("iteration 0", "final"))
    ]
    assert len(lines) == 2
    for line, stage in zip(lines, ("iteration 0", "final"), strict=True):
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert len(cells) == 9
        assert cells[0] == "airline" and cells[1] == stage
        assert cells[2] != cli.DASH  # accuracy came from the summary, not a placeholder


def test_incumbent_search_jsonl_survives_the_gate(offline_loop: dict[str, Any]) -> None:
    """The incumbent's iteration-0 jsonl should still hold its search rows after the gate."""
    summary = offline_loop["summary"]
    path = runner.run_path(
        offline_loop["runs_dir"], "airline", 0, summary["incumbent_id"], cli.SEARCH_SPLIT, 0
    )
    written = {json.loads(line)["task_id"] for line in path.read_text().splitlines()}
    expected = {t.id for t in offline_loop["domain"].eval.load_tasks(cli.SEARCH_SPLIT)}
    assert written == expected


def _search_rows(offline_loop: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows of any candidate's search-split jsonl."""
    search_ids = {t.id for t in offline_loop["domain"].eval.load_tasks(cli.SEARCH_SPLIT)}
    for path in runner.find_runs(offline_loop["runs_dir"], "airline", 0, split=cli.SEARCH_SPLIT):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if {r["task_id"] for r in rows} == search_ids:
            return rows
    raise AssertionError("no untouched search jsonl left in the iteration directory")
