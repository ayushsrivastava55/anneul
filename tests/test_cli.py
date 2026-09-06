"""Offline tests for anneal.cli. Every module the loop calls is replaced by a fake."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anneal import cli, diagnose, gate, llm
from anneal import spec as spec_mod
from anneal.spec import HarnessSpec, Node

SEARCH = "search"


# --- fixtures ----------------------------------------------------------------------------


def make_spec(spec_id: str, *, parent: str | None = None, iteration: int = 0) -> HarnessSpec:
    data: dict[str, Any] = {
        "id": spec_id,
        "topology": "single",
        "step_budget": 8,
        "nodes": [
            Node(
                name="executor",
                role="executor",
                model_tier="mid",
                system_prompt_ref="airline-executor@1",
                tools=["get_user_details"],
                max_steps=8,
            )
        ],
    }
    if parent is not None:
        data["lineage"] = {"parent": parent, "iteration": iteration, "operator": "add_fewshots"}
    return HarnessSpec.model_validate(data)


def make_row(task_id: str, candidate_id: str, score: float, **kw: Any) -> dict[str, Any]:
    row = {
        "task_id": task_id,
        "candidate_id": candidate_id,
        "iteration": 0,
        "score": score,
        "hard_fail": False,
        "hit_step_budget": False,
        "schema_error": False,
        "tokens_in": 10,
        "tokens_out": 5,
        "per_node": {},
        "latency_ms": 100.0,
        "trace_id": None,
        "output": f"answer for {task_id}",
    }
    row.update(kw)
    return row


class Recorder:
    """Records the order of loop steps so tests can assert the data flow."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def note(self, name: str) -> None:
        self.calls.append(name)


@pytest.fixture
def loop(monkeypatch, tmp_path):
    """Patch architect/runner/diagnose/mutate/gate with fakes; return the recorder."""
    rec = Recorder()
    state = {
        "scores": {},
        "promote": True,
        "classes": ["wrong_tool", "output_format", "unsupported_claim"],
    }

    def fake_propose(domain, n=3, **kw):
        rec.note("architect.propose")
        return [make_spec(f"cand-{i}") for i in range(1, n + 1)]

    def fake_run(spec, domain, split, *, iteration=0, seed=0, concurrency=None,
                 on_row=None, **kw):
        """Stands in for runner.run, including its per-row ``on_row`` budget callback.

        The real runner fires ``on_row`` as each task lands, which is how the loop meters
        spend; a fake that ignored it would report every run as free.
        """
        rec.note(f"runner.run:{spec.id}:{split}")
        base = state["scores"].get(spec.id, 0.5)
        rows = []
        for t in domain.eval.load_tasks(split):
            row = make_row(t.id, spec.id, base, iteration=iteration)
            if on_row is not None:
                on_row(row)
            rows.append(row)
        return rows

    def fake_diagnose(rows, domain, spec, *, ledger_path="ledger.json", **kw):
        """One issue per call, cycling classes so operators do not run out immediately."""
        classes = state["classes"]
        cls = classes[rec.calls.count("diagnose.diagnose") % len(classes)]
        rec.note("diagnose.diagnose")
        ledger = diagnose.load_ledger(ledger_path)
        issue = diagnose.upsert(ledger, cls, "executor", [r["task_id"] for r in rows[:2]])
        diagnose.save_ledger(ledger_path, ledger)
        return [issue]

    def fake_apply(spec, issue, evidence, domain, *, client=None):
        rec.note("mutate.apply")
        assert evidence, "cli must hand the operator concrete evidence"
        assert all(e.get("split") == SEARCH for e in evidence)
        it = (spec.lineage.iteration if spec.lineage else 0) + 1
        return make_spec(f"{spec.id}-m{it}", parent=spec.id, iteration=it)

    def fake_gate(incumbent, candidate, domain, iteration, runs_dir, *, search_rows=None, **kw):
        rec.note("gate.gate")
        inc = gate.SpecMetrics(incumbent.id, 0.5, 0.5, 0, None, {"a": True}, 1, 3)
        promoted = state["promote"]
        cand = gate.SpecMetrics(
            candidate.id, 0.9 if promoted else 0.1, 0.9 if promoted else 0.1, 0, 0.02,
            {"a": promoted}, 1, 3,
        )
        return gate.GateResult(
            domain=domain.name, iteration=iteration, incumbent=inc, candidate=cand,
            wins=3, losses=0, p=0.01 if promoted else 0.9, promoted=promoted,
            reason="promoted" if promoted else "p 0.900 >= alpha 0.1",
            path=Path(runs_dir) / domain.name / str(iteration) / "gate.json",
        )

    monkeypatch.setattr(cli.architect, "propose", fake_propose)
    monkeypatch.setattr(cli.runner, "run", fake_run)
    monkeypatch.setattr(cli.diagnose, "diagnose", fake_diagnose)
    monkeypatch.setattr(cli.mutate, "apply", fake_apply)
    monkeypatch.setattr(cli.gate, "gate", fake_gate)
    rec.state = state  # type: ignore[attr-defined]
    return rec


def run_cli(tmp_path, *extra: str) -> int:
    return cli.main(["run", "domains/airline", "--runs-dir", str(tmp_path / "runs"), *extra])


def summaries(tmp_path) -> list[dict[str, Any]]:
    root = tmp_path / "runs" / "airline"
    return [
        json.loads((root / str(i) / "summary.json").read_text())
        for i in sorted(int(p.name) for p in root.iterdir() if p.name.isdigit())
    ]


# --- loop order --------------------------------------------------------------------------


def test_loop_order_iteration_zero(loop, tmp_path):
    assert run_cli(tmp_path, "--iterations", "1", "--candidates", "2") == 0
    assert loop.calls[0] == "architect.propose"
    assert loop.calls[1:3] == ["runner.run:cand-1:search", "runner.run:cand-2:search"]
    assert loop.calls[3] == "diagnose.diagnose"
    assert loop.calls[4] == "mutate.apply"
    # the mutant is scored on search (for gen_gap and cost) before the gate sees it
    assert loop.calls[5] == "runner.run:cand-1-m1:search"
    assert loop.calls[6] == "gate.gate"


def test_architect_runs_only_on_iteration_zero(loop, tmp_path):
    run_cli(tmp_path, "--iterations", "3")
    assert loop.calls.count("architect.propose") == 1


def test_best_candidate_by_mean_score_becomes_incumbent(loop, tmp_path):
    loop.state["scores"] = {"cand-1": 0.2, "cand-2": 0.8, "cand-3": 0.4}
    run_cli(tmp_path, "--iterations", "1")
    assert summaries(tmp_path)[0]["incumbent_id"] == "cand-2"


def test_ledger_defaults_to_one_file_per_domain(loop, tmp_path):
    """diagnose keys issues on (class, node) and node names repeat across domains."""
    run_cli(tmp_path, "--iterations", "1")
    assert (tmp_path / "runs" / "airline" / "ledger.json").is_file()
    assert not (tmp_path / "runs" / "ledger.json").exists()


def test_cli_never_names_the_reserved_split(loop, tmp_path):
    run_cli(tmp_path, "--iterations", "1")
    assert gate.HOLDOUT_SPLIT not in Path(cli.__file__).read_text()
    assert all(gate.HOLDOUT_SPLIT not in c for c in loop.calls)


# --- summary.json ------------------------------------------------------------------------

REQUIRED = (
    "mean_score", "pass3_rate", "hard_fails", "gen_gap", "p_value",
    "cost_usd", "p95_latency_ms", "decision", "reason",
)


def test_summary_has_every_required_field(loop, tmp_path):
    run_cli(tmp_path, "--iterations", "1")
    summary = summaries(tmp_path)[0]
    for field in REQUIRED:
        assert field in summary, field
    assert summary["decision"] == "promote"
    assert summary["candidate_id"] == "cand-1-m1"
    assert summary["winner_id"] == "cand-1-m1"
    assert summary["incumbent"]["mean_score"] == 0.5
    assert summary["search"]["cand-1"]["n_tasks"] == 10


def test_reject_keeps_the_incumbent_as_winner(loop, tmp_path):
    loop.state["promote"] = False
    run_cli(tmp_path, "--iterations", "1")
    summary = summaries(tmp_path)[0]
    assert summary["decision"] == "reject"
    assert summary["winner_id"] == "cand-1"
    assert summary["reason"].startswith("p ")


def test_promoted_spec_yaml_is_written_for_the_gate_subcommand(loop, tmp_path):
    run_cli(tmp_path, "--iterations", "1")
    specs = summaries(tmp_path)[0]["specs"]
    assert Path(specs["cand-1-m1"]).is_file()


# --- stopping conditions -----------------------------------------------------------------


def test_plateau_stops_after_two_consecutive_rejects(loop, tmp_path):
    loop.state["promote"] = False
    assert run_cli(tmp_path, "--iterations", "6") == 0
    assert len(summaries(tmp_path)) == 2
    assert summaries(tmp_path)[-1]["stop_reason"] == "plateau"


def test_promote_resets_the_plateau_counter(loop, tmp_path):
    run_cli(tmp_path, "--iterations", "4")
    assert len(summaries(tmp_path)) == 4
    assert [s["decision"] for s in summaries(tmp_path)] == ["promote"] * 4


def test_exhausted_operators_stop_the_loop(loop, tmp_path):
    """Only two operators exist for wrong_tool; the third attempt has nothing left to try."""
    loop.state["classes"] = ["wrong_tool"]
    assert run_cli(tmp_path, "--iterations", "5") == 0
    last = summaries(tmp_path)[-1]
    assert last["decision"] == "no_candidate"
    assert last["stop_reason"] == "no_candidate"


def test_budget_halts_the_loop(loop, tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli.runner, "summarize",
        lambda rows, threshold, **kw: {
            "mean_score": 0.5, "pass_rate": 0.5, "hard_fails": 0, "tokens_in": 0,
            "tokens_out": 0, "p95_latency_ms": 1.0, "cost_usd": 9.0,
        },
    )
    assert run_cli(tmp_path, "--iterations", "5", "--budget", "10.00") == 1
    assert summaries(tmp_path)[-1]["stop_reason"] == "budget"
    assert summaries(tmp_path)[-1]["spend_usd"] >= 10.0


def test_budget_stops_mid_split_instead_of_after_it(priced, tmp_path):
    """The cap interrupts a split in progress rather than charging the whole thing.

    Regression: spend used to be charged once per finished split, so a $0.50 cap on a
    10-task airline split at ~$0.28/task ran to $2.85 before anything checked. At the
    fixture's $2.00/task, a $3.00 cap must stop after ~2 tasks, not after all 10.
    """
    assert run_cli(tmp_path, "--iterations", "1", "--budget", "3.00", "--models", priced) == 1
    last = summaries(tmp_path)[-1]
    assert last["stop_reason"] == "budget"
    assert 3.0 <= last["spend_usd"] < 20.0


def test_budget_message_is_explicit(loop, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli.runner, "summarize",
        lambda rows, threshold, **kw: {
            "mean_score": 0.5, "pass_rate": 0.5, "hard_fails": 0, "tokens_in": 0,
            "tokens_out": 0, "p95_latency_ms": 1.0, "cost_usd": 40.0,
        },
    )
    run_cli(tmp_path, "--iterations", "5", "--budget", "1.00")
    assert "budget" in capsys.readouterr().out.lower()


# --- report ------------------------------------------------------------------------------


def write_summary(root: Path, iteration: int, **kw: Any) -> None:
    body: dict[str, Any] = {
        "domain": "airline", "iteration": iteration,
        "incumbent_id": "cand-1", "candidate_id": "cand-1-m1", "winner_id": "cand-1-m1",
        "mean_score": 0.9, "pass3_rate": 0.8, "gen_gap": 0.05, "hard_fails": 0,
        "p_value": 0.01, "cost_usd": 0.42, "p95_latency_ms": 1200.0,
        "decision": "promote", "reason": "promoted", "operator": "add_fewshots",
        "incumbent": {
            "candidate_id": "cand-1", "mean_score": 0.5, "pass3_rate": 0.4,
            "gen_gap": None, "hard_fails": 2,
        },
        "candidate": {
            "candidate_id": "cand-1-m1", "mean_score": 0.9, "pass3_rate": 0.8,
            "gen_gap": 0.05, "hard_fails": 0,
        },
        "search": {
            "cand-1": {"cost_per_task": 0.01, "p95_latency_ms": 900.0, "n_tasks": 10},
            "cand-1-m1": {"cost_per_task": 0.02, "p95_latency_ms": 1200.0, "n_tasks": 10},
        },
        "specs": {},
    }
    body.update(kw)
    path = root / "airline" / str(iteration) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2))


def test_report_renders_iteration_zero_and_final_rows(tmp_path, capsys):
    root = tmp_path / "runs"
    write_summary(root, 0)
    write_summary(root, 1)
    assert cli.main(["report", str(root)]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("|")]
    assert len(lines) == 2
    assert lines[0].split("|")[1:4] == [" airline ", " iteration 0 ", " 0.500 "]
    assert lines[1].split("|")[2].strip() == "final"
    assert " 0.900 " in lines[1]
    assert " 0.010 " in lines[0]  # $/task from the search block
    assert lines[0].split("|")[9].strip() == "—"  # no gate p-value for iteration 0


def write_pareto(root: Path, **point: Any) -> None:
    body = {
        "config_id": "cand-1-m1-anneal-2", "node_tiers": {"executor": "cheap"},
        "score": 0.88, "pass3": 0.8, "cost_per_task": 0.004, "p95_latency_ms": 700.0,
        "kept": True, "hard_fails": 0,
    }
    body.update(point)
    path = root / "airline" / "anneal" / "pareto.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"winner": body["config_id"], "points": [body]}))


def test_report_adds_an_annealed_row_when_the_downshift_has_run(tmp_path, capsys):
    """The third README stage comes from pareto.json, not from a summary."""
    root = tmp_path / "runs"
    write_summary(root, 0)
    write_pareto(root)
    assert cli.main(["report", str(root)]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("|")]
    assert [ln.split("|")[2].strip() for ln in lines] == ["iteration 0", "final", "annealed"]
    annealed = lines[2].split("|")
    assert annealed[3].strip() == "0.880"
    assert annealed[7].strip() == "0.004"  # cheaper $/task than the search rows above
    assert annealed[8].strip() == "700"


def test_report_omits_the_annealed_row_before_the_downshift_runs(tmp_path, capsys):
    root = tmp_path / "runs"
    write_summary(root, 0)
    assert cli.main(["report", str(root)]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("|")]
    assert [ln.split("|")[2].strip() for ln in lines] == ["iteration 0", "final"]


def test_report_final_row_is_the_last_winner(tmp_path, capsys):
    root = tmp_path / "runs"
    write_summary(root, 0)
    write_summary(root, 1, decision="reject", winner_id="cand-1", reason="p 0.9 >= alpha 0.1")
    cli.main(["report", str(root)])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("|")]
    assert lines[1].split("|")[3].strip() == "0.500"  # winner fell back to the incumbent


def test_report_renders_rows_generated_by_a_real_run(loop, tmp_path, capsys):
    run_cli(tmp_path, "--iterations", "1")
    assert cli.main(["report", str(tmp_path / "runs")]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("|")]
    assert [ln.split("|")[2].strip() for ln in lines] == ["iteration 0", "final"]


def test_report_final_row_survives_a_halted_last_iteration(loop, tmp_path, capsys):
    """The run ends on no_candidate, whose summary has no gate blocks; the row is still real."""
    loop.state["classes"] = ["wrong_tool"]
    run_cli(tmp_path, "--iterations", "5")
    assert summaries(tmp_path)[-1]["decision"] == "no_candidate"
    cli.main(["report", str(tmp_path / "runs")])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("|")]
    assert lines[1].split("|")[3].strip() == "0.900"


def test_report_ignores_non_numeric_dirs(tmp_path, capsys):
    root = tmp_path / "runs"
    write_summary(root, 0)
    junk = root / "airline" / "latest"
    junk.mkdir()
    (junk / "summary.json").write_text("not json")
    assert cli.main(["report", str(root)]) == 0


def test_report_on_an_empty_dir_is_an_error(tmp_path):
    assert cli.main(["report", str(tmp_path)]) == 1


# --- cost attribution and iteration tagging (regressions) -------------------------------

MODELS_YAML = """\
tiers:
  frontier: {provider: tensormux, model: test-frontier, price_in: 1000.0, price_out: 1000.0}
  mid: {provider: tensormux, model: test-mid, price_in: 1000.0, price_out: 1000.0}
  cheap: {provider: tensormux, model: test-cheap, price_in: 1000.0, price_out: 1000.0}
downshift_order: [frontier, mid, cheap]
providers:
  tensormux: {base_url_env: TENSORMUX_BASE_URL, api_key_env: TENSORMUX_API_KEY}
"""


@pytest.fixture
def priced(loop, tmp_path, monkeypatch):
    """A price table plus rows whose backend is unknown, so only the tier fallback can price."""
    models = tmp_path / "models.yaml"
    models.write_text(MODELS_YAML)
    llm.load_models.cache_clear()
    base = cli.runner.run

    # backend None is what every offline run records; pricing must fall back to the model
    # the node's tier requested, not silently charge 0.
    usage = {"executor": {"tokens_in": 1000, "tokens_out": 1000, "backend": None, "ms": 1.0}}

    def run_with_usage(spec_, domain, split, *, on_row=None, **kw):
        # The real runner fills per_node before firing on_row, so the budget callback sees a
        # priced row. Decorating after the split returned would charge every task as free.
        def decorate(row):
            row["per_node"] = dict(usage)
            if on_row is not None:
                on_row(row)

        rows = base(spec_, domain, split, on_row=decorate, **kw)
        for row in rows:
            row.setdefault("per_node", dict(usage))
        return rows

    monkeypatch.setattr(cli.runner, "run", run_with_usage)
    yield str(models)
    llm.load_models.cache_clear()


def test_cost_falls_back_to_the_requested_model_when_the_backend_is_unpriced(priced, tmp_path):
    """Regression: summarize was called without models_path/node_models, so cost was always 0."""
    run_cli(tmp_path, "--iterations", "1", "--budget", "1000", "--models", priced)
    summary = summaries(tmp_path)[0]
    # 1000 in + 1000 out at $1000/1M each = $2.00 per task, 10 search tasks per spec
    assert summary["search"]["cand-1"]["cost_usd"] == pytest.approx(20.0)
    assert summary["search"]["cand-1"]["cost_per_task"] == pytest.approx(2.0)
    assert summary["cost_usd"] == pytest.approx(20.0)
    assert summary["spend_usd"] > 0


def test_budget_halts_on_real_prices(priced, tmp_path):
    """With cost attributed, --budget actually bites; before the fix nothing ever halted."""
    assert run_cli(tmp_path, "--iterations", "3", "--budget", "25.00", "--models", priced) == 1
    assert summaries(tmp_path)[-1]["stop_reason"] == "budget"


def test_row_iteration_matches_the_span_iteration(loop, tmp_path, monkeypatch):
    """Regression: a mutant's rows said 0 while runtime opened its spans at lineage 1."""
    seen: list[tuple[str, int, int]] = []
    base = cli.runner.run

    def recording_run(spec_, domain, split, *, iteration=0, **kw):
        lineage = spec_.lineage.iteration if spec_.lineage else 0
        seen.append((spec_.id, iteration, lineage))
        return base(spec_, domain, split, iteration=iteration, **kw)

    monkeypatch.setattr(cli.runner, "run", recording_run)
    run_cli(tmp_path, "--iterations", "2")
    assert seen, "no runs recorded"
    mutants = [s for s in seen if "-m" in s[0]]
    assert mutants, "the loop never ran a mutated spec"
    for spec_id, iteration, lineage in seen:
        assert iteration == lineage, f"{spec_id}: rows say {iteration}, spans say {lineage}"


def test_persisted_spec_carries_the_loop_iteration(loop, tmp_path):
    """The yaml the gate subcommand reloads is stamped with the iteration it ran under."""
    run_cli(tmp_path, "--iterations", "1")
    summary = summaries(tmp_path)[0]
    mutant = spec_mod.load_spec(summary["specs"][summary["candidate_id"]])
    assert mutant.lineage.iteration == 0
    assert Path(summary["specs"][summary["candidate_id"]]).parent.name == "0"


# --- surface -----------------------------------------------------------------------------


def test_dashboard_serves_instead_of_stubbing(monkeypatch, tmp_path):
    """`anneal dashboard` reaches uvicorn with the runs dir and port it was given."""
    served: dict[str, Any] = {}

    def fake_run(app, host, port):
        served.update(app=app, host=host, port=port)

    monkeypatch.setattr("uvicorn.run", fake_run)
    code = cli.main(
        ["dashboard", "--runs-dir", str(tmp_path), "--port", "8123", "--host", "127.0.0.1"]
    )
    assert code == 0
    assert (served["host"], served["port"]) == ("127.0.0.1", 8123)
    assert served["app"].state.anneal.runs_dir == tmp_path


def test_anneal_subcommand_needs_a_gated_run(capsys, tmp_path):
    """`anneal anneal` on an empty runs dir explains itself rather than crashing."""
    assert cli.main(["anneal", str(tmp_path)]) == 1
    assert "no summary.json" in capsys.readouterr().out


def test_anneal_subcommand_downshifts_the_winner(loop, monkeypatch, tmp_path):
    """A gated run reaches `anneal.downshift` with the winner spec and its gated score."""
    run_cli(tmp_path, "--iterations", "1")
    seen: dict[str, Any] = {}

    def fake_downshift(spec, domain, *, peak_score, runs_dir, iteration, **kw):
        seen.update(spec_id=spec.id, peak=peak_score, iteration=iteration)
        point = cli.anneal_stage.ParetoPoint(
            config_id=f"{spec.id}-anneal-0", node_tiers={"executor": "mid"}, score=peak_score,
            pass3=1.0, cost_per_task=0.01, p95_latency_ms=100.0, kept=True,
        )
        return cli.anneal_stage.AnnealResult(
            spec=spec, points=[point], front=[point],
            pareto_path=Path(runs_dir) / "pareto.json", spec_path=Path(runs_dir) / "w.yaml",
        )

    monkeypatch.setattr(cli.anneal_stage, "downshift", fake_downshift)
    assert cli.main(["anneal", str(tmp_path / "runs")]) == 0
    summary = summaries(tmp_path)[-1]
    assert seen["spec_id"] == summary["winner_id"]
    assert seen["peak"] is not None


def test_no_args_prints_usage(capsys):
    assert cli.main([]) == 0
    assert "usage" in capsys.readouterr().out
