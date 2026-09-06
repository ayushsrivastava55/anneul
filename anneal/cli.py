"""`anneal` command line entry point: the architect -> run -> diagnose -> mutate -> gate loop.

The loop here is pure wiring. Every decision is made by the module that owns it
(``architect``, ``runner``, ``diagnose``, ``mutate``, ``gate``); this file only sequences
them, tracks spend against ``--budget`` and writes ``runs/<domain>/<iter>/summary.json``.
The reserved evaluation split is never named here -- only ``anneal.gate`` may touch it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from anneal import __version__, architect, diagnose, gate, llm, mutate, runner, spec
from anneal.domain import load_domain
from anneal.spec import HarnessSpec

logger = logging.getLogger("anneal.cli")

COMMANDS: dict[str, str] = {
    "run": "generate, run, diagnose, mutate and gate candidates for a domain",
    "gate": "re-run the held-out gate on the incumbent",
    "report": "print the markdown results row(s) from runs/<domain>/<iter>/summary.json",
    "anneal": "downshift node models along the cost/latency Pareto front",
    "dashboard": "serve the runs/ dashboard on http://localhost:8000",
}
STUBS = ("anneal", "dashboard")

SEARCH_SPLIT = diagnose.SEARCH_SPLIT
DEFAULT_BUDGET = 5.00
DEFAULT_CANDIDATES = len(architect.MENU)
PLATEAU = 2
DASH = "—"


class BudgetExceeded(RuntimeError):
    """Raised by the budgeted run wrapper once cumulative spend reaches ``--budget``."""


@dataclass
class _NullSpec:
    """Stand-in id when the budget runs out before any incumbent is chosen."""

    id: str = "none"


@dataclass
class Loop:
    """Mutable state carried across iterations of one ``anneal run``."""

    domain: Any
    runs_dir: Path
    ledger: Path
    budget: float
    concurrency: int | None
    seed: int
    candidates: int = DEFAULT_CANDIDATES
    models_path: str | None = None
    node_models: dict[str, dict[str, str]] = field(default_factory=dict)
    spend: float = 0.0
    rejects: int = 0
    search: dict[str, dict[str, Any]] = field(default_factory=dict)
    specs: dict[str, str] = field(default_factory=dict)


# --- running with a budget ---------------------------------------------------------------


def _threshold(domain: Any) -> float:
    return float(domain.eval.THRESHOLD)


def _node_models(loop: Loop, candidate: Any) -> dict[str, str]:
    """node name -> the model id the node requested, cached per spec.

    ``runner.cost_by_node`` prices a node by the backend that served it and falls back to
    this map when that backend has no entry in ``models.yaml`` -- which is every offline run
    and any backend TensorMux routes to that we do not list. Without it, cost is always 0.
    """
    if candidate.id not in loop.node_models:
        try:
            loop.node_models[candidate.id] = {
                n.name: llm.resolve_model(n.model_tier, loop.models_path) for n in candidate.nodes
            }
        except (KeyError, FileNotFoundError, ValueError, TypeError) as exc:
            logger.warning(
                json.dumps({"event": "unresolved_tiers", "spec": candidate.id, "error": str(exc)})
            )
            loop.node_models[candidate.id] = {}
    return loop.node_models[candidate.id]


def _summarize(loop: Loop, candidate: Any, rows: list[dict]) -> dict[str, Any]:
    """``runner.summarize`` with this run's price table and per-node model fallback."""
    return runner.summarize(
        rows, _threshold(loop.domain),
        models_path=loop.models_path, node_models=_node_models(loop, candidate),
    )


def _charge(loop: Loop, candidate: Any, rows: list[dict]) -> float:
    """Add the cost of ``rows`` to cumulative spend and raise once the budget is reached."""
    cost = float(_summarize(loop, candidate, rows)["cost_usd"])
    loop.spend += cost
    if loop.spend >= loop.budget:
        raise BudgetExceeded(
            f"budget exhausted: spent ${loop.spend:.4f} of ${loop.budget:.2f}"
        )
    return cost


def _budgeted_run(loop: Loop) -> Any:
    """A ``runner.run`` with ``--concurrency``/``--runs-dir`` bound and spend metered."""

    def run(spec_, domain, split, *, iteration=0, seed=0, **kw):
        if loop.spend >= loop.budget:
            raise BudgetExceeded(
                f"budget exhausted before running {spec_.id}: "
                f"spent ${loop.spend:.4f} of ${loop.budget:.2f}"
            )
        rows = runner.run(
            spec_, domain, split, iteration=iteration, seed=seed,
            concurrency=loop.concurrency, runs_dir=loop.runs_dir,
        )
        _charge(loop, spec_, rows)
        return rows

    return run


def _stamp(candidate: HarnessSpec, iteration: int) -> HarnessSpec:
    """Return ``candidate`` with ``lineage.iteration`` set to the loop iteration.

    ``runtime.run_task`` opens its span context with ``spec.lineage.iteration`` while
    ``runner.run`` stamps rows with the iteration it is handed, so rows and spans only join
    on ``(candidate_id, iteration)`` if the two agree. The loop index is authoritative here:
    it keys ``runs/<domain>/<iter>/`` and every number in summary.json. A mutant arrives from
    ``mutate.apply`` numbered parent+1, which is only the loop index when every iteration
    promotes, so the loop re-stamps whatever it is about to run.
    """
    if candidate.lineage is None or candidate.lineage.iteration == iteration:
        return candidate
    data = candidate.model_dump()
    data["lineage"]["iteration"] = iteration
    return HarnessSpec.model_validate(data)


def _persist(loop: Loop, candidate: HarnessSpec, iteration: int) -> HarnessSpec:
    """Stamp ``candidate`` with the loop iteration and write its yaml under that iteration."""
    candidate = _stamp(candidate, iteration)
    path = loop.runs_dir / loop.domain.name / str(iteration) / f"{candidate.id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    spec.dump_spec(candidate, path)
    loop.specs[candidate.id] = str(path)
    return candidate


def _run_search(
    loop: Loop, candidate: HarnessSpec, iteration: int
) -> tuple[HarnessSpec, list[dict]]:
    """Run one spec on the search split, persist its yaml and record its search metrics."""
    candidate = _persist(loop, candidate, iteration)
    rows = _budgeted_run(loop)(
        candidate, loop.domain, SEARCH_SPLIT, iteration=iteration, seed=loop.seed
    )
    metrics = _summarize(loop, candidate, rows)
    metrics["n_tasks"] = len(rows)
    loop.search[candidate.id] = metrics
    return candidate, rows


# --- one iteration -----------------------------------------------------------------------


def _evidence(issue: dict[str, Any], rows: list[dict]) -> list[dict[str, Any]]:
    """Search rows named by the issue's evidence ids, shaped for the mutation operators."""
    wanted = set(issue.get("evidence") or ())
    hits = [r for r in rows if str(r.get("trace_id") or r["task_id"]) in wanted]
    return [
        {"task_id": r["task_id"], "output": r.get("output"), "split": SEARCH_SPLIT}
        for r in (hits or rows[: mutate.MAX_EVIDENCE])
    ]


def _propose_mutation(
    loop: Loop, incumbent: Any, rows: list[dict]
) -> tuple[Any, dict[str, Any]] | None:
    """Diagnose failures, then mutate the highest-ranked issue an operator can still fix."""
    diagnose.diagnose(rows, loop.domain, incumbent, ledger_path=loop.ledger)
    ledger = diagnose.load_ledger(loop.ledger)
    for issue in diagnose.rank(ledger):
        try:
            operator = mutate.select_operator(issue)
            candidate = mutate.apply(incumbent, issue, _evidence(issue, rows), loop.domain)
        except (mutate.NoOperatorAvailable, KeyError):
            continue
        issue.setdefault("operators_tried", []).append(operator)
        diagnose.save_ledger(loop.ledger, ledger)
        return candidate, issue
    return None


def _metrics_block(metrics: Any) -> dict[str, Any]:
    block = dict(metrics)
    block.pop("pass3", None)
    return block


def _summary(loop: Loop, iteration: int, inc: Any, **kw: Any) -> dict[str, Any]:
    """The common summary skeleton; the gate fills in the decision fields when it ran."""
    body: dict[str, Any] = {
        "domain": loop.domain.name,
        "domain_path": str(loop.domain.path),
        "iteration": iteration,
        "incumbent_id": inc.id,
        "candidate_id": None,
        "winner_id": inc.id,
        "operator": None,
        "issue": None,
        "mean_score": None,
        "pass3_rate": None,
        "hard_fails": None,
        "gen_gap": None,
        "p_value": None,
        "cost_usd": None,
        "p95_latency_ms": None,
        "decision": "halted",
        "reason": "",
        "stop_reason": None,
        "spend_usd": round(loop.spend, 6),
        "budget_usd": loop.budget,
        "search": dict(loop.search),
        "specs": dict(loop.specs),
    }
    body.update(kw)
    return body


def _write_summary(loop: Loop, body: dict[str, Any]) -> Path:
    path = loop.runs_dir / loop.domain.name / str(body["iteration"]) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _gate_summary(
    loop: Loop, iteration: int, incumbent: Any, candidate: Any, issue: dict[str, Any],
    result: Any, cand_search: dict[str, Any],
) -> dict[str, Any]:
    data = result.to_dict()
    cand = _metrics_block(data["candidate"])
    return _summary(
        loop, iteration, incumbent,
        candidate_id=candidate.id,
        winner_id=candidate.id if result.promoted else incumbent.id,
        operator=(candidate.lineage.operator if candidate.lineage else None),
        issue={"id": issue["id"], "class": issue["class"], "node": issue["node"]},
        mean_score=cand["mean_score"],
        pass3_rate=cand["pass3_rate"],
        hard_fails=cand["hard_fails"],
        gen_gap=cand["gen_gap"],
        p_value=data["p"],
        cost_usd=cand_search["cost_usd"],
        p95_latency_ms=cand_search["p95_latency_ms"],
        decision=data["decision"],
        reason=data["reason"],
        incumbent=_metrics_block(data["incumbent"]),
        candidate=cand,
        gate_path=data["path"],
    )


def _pick_incumbent(loop: Loop, console: Console) -> tuple[Any, list[dict]]:
    """Iteration 0: propose candidates, run each on search, keep the best mean score."""
    candidates = architect.propose(loop.domain, n=loop.candidates)
    best: tuple[float, Any, list[dict]] | None = None
    for candidate in candidates:
        candidate, rows = _run_search(loop, candidate, 0)
        score = loop.search[candidate.id]["mean_score"]
        console.print(f"  [dim]{candidate.id} ({candidate.topology}) mean={score:.3f}[/dim]")
        if best is None or score > best[0]:
            best = (score, candidate, rows)
    assert best is not None
    return best[1], best[2]


def _iteration(
    loop: Loop, i: int, incumbent: Any, rows: list[dict], console: Console
) -> tuple[Any, list[dict], dict[str, Any]]:
    """Diagnose -> mutate -> score the mutant on search -> gate. Returns the next incumbent."""
    proposed = _propose_mutation(loop, incumbent, rows)
    if proposed is None:
        return incumbent, rows, _summary(
            loop, i, incumbent, decision="no_candidate", stop_reason="no_candidate",
            reason="every ranked issue has exhausted its implemented operators",
        )
    candidate, issue = proposed
    candidate, cand_rows = _run_search(loop, candidate, i)
    # the gate runs the incumbent again at this iteration, so it needs the same stamp
    result = gate.gate(
        _persist(loop, incumbent, i), candidate, loop.domain, i, loop.runs_dir,
        search_rows=cand_rows, run=_budgeted_run(loop),
    )
    body = _gate_summary(loop, i, incumbent, candidate, issue, result, loop.search[candidate.id])
    console.print(
        f"  [bold]{result.reason}[/bold] "
        f"({candidate.id} via {body['operator']}, p={result.p:.3f})"
    )
    if result.promoted:
        loop.rejects = 0
        return candidate, cand_rows, body
    loop.rejects += 1
    if loop.rejects >= PLATEAU:
        body["stop_reason"] = "plateau"
    return incumbent, rows, body


# --- `anneal run` ------------------------------------------------------------------------


def _progress(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(title="anneal", show_edge=False)
    for column in ("iter", "incumbent", "candidate", "operator", "acc", "p", "$", "decision"):
        table.add_column(column)
    for r in rows:
        table.add_row(
            str(r["iteration"]), str(r["incumbent_id"]), str(r["candidate_id"] or DASH),
            str(r["operator"] or DASH), _num(r["mean_score"]), _num(r["p_value"]),
            f"{r['spend_usd']:.4f}", str(r["decision"]),
        )
    console.print(table)


def cmd_run(args: argparse.Namespace, console: Console) -> int:
    domain = load_domain(args.domain_dir)
    runs_dir = Path(args.runs_dir)
    loop = Loop(
        domain=domain,
        runs_dir=runs_dir,
        # per domain: diagnose keys issues on (class, node) and node names repeat across
        # domains, so one shared ledger would merge unrelated issues and their operators_tried.
        ledger=Path(args.ledger) if args.ledger else runs_dir / domain.name / "ledger.json",
        budget=float(args.budget),
        concurrency=args.concurrency,
        seed=args.seed,
        candidates=args.candidates,
        models_path=args.models,
    )
    loop.ledger.parent.mkdir(parents=True, exist_ok=True)
    gate.clear_cache()
    console.print(f"[bold]anneal run[/bold] {loop.domain.name} "
                  f"iterations={args.iterations} budget=${loop.budget:.2f}")
    written: list[dict[str, Any]] = []
    incumbent: Any = None
    status = 0
    try:
        incumbent, rows = _pick_incumbent(loop, console)
        for i in range(args.iterations):
            incumbent, rows, body = _iteration(loop, i, incumbent, rows, console)
            body["spend_usd"] = round(loop.spend, 6)
            written.append(body)
            _write_summary(loop, body)
            if body["stop_reason"]:
                break
    except BudgetExceeded as exc:
        status = 1
        body = _summary(
            loop, len(written), incumbent or _NullSpec(), stop_reason="budget", reason=str(exc)
        )
        written.append(body)
        _write_summary(loop, body)
        console.print(f"[red]{exc}[/red]")
    _progress(console, written)
    console.print(f"summaries in {loop.runs_dir / loop.domain.name}")
    return status


# --- `anneal gate` and `anneal report` ---------------------------------------------------


def _summaries(runs_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(
        (p for p in runs_dir.glob("*/*/summary.json") if p.parent.name.isdigit()),
        key=lambda p: (p.parent.parent.name, int(p.parent.name)),
    )
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def cmd_gate(args: argparse.Namespace, console: Console) -> int:
    """Re-run the gate for the last iteration of every domain under ``runs_dir``."""
    summaries = _summaries(Path(args.runs_dir))
    if not summaries:
        console.print(f"[red]no summary.json under {args.runs_dir}[/red]")
        return 1
    latest: dict[str, dict[str, Any]] = {}
    for body in summaries:
        latest[body["domain"]] = body
    gate.clear_cache()
    try:
        _regate(latest.values(), args, console)
    except BudgetExceeded as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    return 0


def _regate(bodies: Any, args: argparse.Namespace, console: Console) -> None:
    for body in bodies:
        if not body.get("candidate_id"):
            console.print(f"[yellow]{body['domain']}: no gated candidate to re-run[/yellow]")
            continue
        domain = load_domain(body["domain_path"])
        specs = body["specs"]
        loop = Loop(
            domain=domain, runs_dir=Path(args.runs_dir),
            ledger=Path(args.runs_dir) / domain.name / "ledger.json",
            budget=float(args.budget), concurrency=args.concurrency, seed=0,
            models_path=args.models,
        )
        result = gate.gate(
            spec.load_spec(specs[body["incumbent_id"]]),
            spec.load_spec(specs[body["candidate_id"]]),
            domain, body["iteration"], Path(args.runs_dir), run=_budgeted_run(loop),
        )
        console.print(f"{body['domain']} i{body['iteration']}: "
                      f"{result.reason} (p={result.p:.3f})")


def _num(value: Any, digits: int = 3) -> str:
    return DASH if value is None else f"{float(value):.{digits}f}"


def _row(domain: str, stage: str, block: dict[str, Any], search: dict[str, Any], p: Any) -> str:
    cells = [
        domain, stage, _num(block.get("mean_score")), _num(block.get("pass3_rate")),
        _num(block.get("gen_gap")),
        DASH if block.get("hard_fails") is None else str(block["hard_fails"]),
        _num(search.get("cost_per_task")), _num(search.get("p95_latency_ms"), 0), _num(p),
    ]
    return "| " + " | ".join(cells) + " |"


def _winner_block(summaries: list[dict[str, Any]]) -> tuple[dict[str, Any], Any]:
    """Metrics of the spec that survived the last iteration, plus its gate p-value.

    The final iteration may have halted before the gate ran (budget, no operator left), so
    walk backwards to the most recent summary that actually scored the winning spec.
    """
    winner = summaries[-1]["winner_id"]
    for body in reversed(summaries):
        for key in ("candidate", "incumbent"):
            block = body.get(key) or {}
            if block.get("candidate_id") == winner:
                return block, (body.get("p_value") if key == "candidate" else None)
    return {}, None


def _report_rows(summaries: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for name in dict.fromkeys(b["domain"] for b in summaries):
        got = [b for b in summaries if b["domain"] == name]
        first, last = got[0], got[-1]
        rows.append(
            _row(name, "iteration 0", first.get("incumbent") or {},
                 first["search"].get(first["incumbent_id"], {}), None)
        )
        block, p = _winner_block(got)
        rows.append(_row(name, "final", block, last["search"].get(last["winner_id"], {}), p))
    return rows


def cmd_report(args: argparse.Namespace, console: Console) -> int:
    summaries = _summaries(Path(args.runs_dir))
    if not summaries:
        console.print(f"[red]no summary.json under {args.runs_dir}[/red]")
        return 1
    for line in _report_rows(summaries):
        print(line)
    return 0


# --- argument parsing --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anneal", description="Anneal agent harnesses.")
    parser.add_argument("--version", action="version", version=f"anneal {__version__}")
    sub = parser.add_subparsers(dest="command")
    for name, help_text in COMMANDS.items():
        sub.add_parser(name, help=help_text)

    run = sub.choices["run"]
    run.add_argument("domain_dir", help="path to domains/<name>")
    run.add_argument("--iterations", type=int, default=4)
    run.add_argument("--budget", type=float, default=DEFAULT_BUDGET, metavar="USD")
    run.add_argument("--concurrency", type=int, default=None)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    run.add_argument("--runs-dir", default=str(runner.RUNS_DIR))
    run.add_argument("--ledger", default=None, help="issue ledger path")
    run.add_argument("--models", default=None, help="price table (default specs/models.yaml)")

    for name in ("gate", "report"):
        sub.choices[name].add_argument("runs_dir", help="runs/ directory to read")
    sub.choices["gate"].add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                                     metavar="USD")
    sub.choices["gate"].add_argument("--concurrency", type=int, default=None)
    sub.choices["gate"].add_argument("--models", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    console = Console()
    args = build_parser().parse_args(argv)
    if args.command is None:
        console.print(f"[bold]anneal[/bold] {__version__}")
        console.print(r"usage: anneal <command> \[options]", end="\n\n")
        for name, help_text in COMMANDS.items():
            console.print(f"  [cyan]{name:<10}[/cyan] {help_text}")
        return 0
    if args.command in STUBS:
        console.print(f"[yellow]anneal {args.command}[/yellow]: not implemented yet")
        return 2
    handler = {"run": cmd_run, "gate": cmd_gate, "report": cmd_report}[args.command]
    return handler(args, console)


if __name__ == "__main__":
    sys.exit(main())
