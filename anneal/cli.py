"""`anneal` command line entry point: the architect -> run -> diagnose -> mutate -> gate loop.

The loop here is pure wiring. Every decision is made by the module that owns it
(``architect``, ``runner``, ``diagnose``, ``mutate``, ``gate``); this file only sequences
them, tracks spend against ``--budget`` and writes ``runs/<domain>/<iter>/summary.json``.
The reserved evaluation split is never named here -- only ``anneal.gate`` may touch it.

Episodic memory (``anneal.memory``) is opened once per ``anneal run`` and stays active for
the whole loop, so ``anneal.runtime`` can recall at task time. Around each search run the
loop clears the injection log, credits the entries the finished rows used, reflects on that
run (rules from what failed, procedures from what succeeded), retires the losing entries and
saves the store. The counts land in summary.json as ``memory_entries``, ``memory_by_kind``
and ``memory_injected``.
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

from anneal import __version__, architect, diagnose, gate, llm, mutate, runner, spec, tracing
from anneal import anneal as anneal_stage
from anneal import memory as memory_mod
from anneal.domain import load_domain
from anneal.spec import HarnessSpec

logger = logging.getLogger("anneal.cli")

COMMANDS: dict[str, str] = {
    "init": "interview the user and generate a runnable domains/<name>/",
    "run": "generate, run, diagnose, mutate and gate candidates for a domain",
    "gate": "re-run the held-out gate on the incumbent",
    "report": "render the README results block from runs/<domain>/<iter>/summary.json",
    "anneal": "downshift node models along the cost/latency Pareto front",
    "dashboard": "serve the runs/ dashboard on http://localhost:8000",
}
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
    # episodic memory store shared by every run of this loop (None disables it entirely)
    memory: Any | None = None
    # client for the reflection call; tests inject a fake, live runs use the gateway
    memory_client: Any | None = None
    # distinct memory entries injected during the most recent search run
    injected: int = 0


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
    """A ``runner.run`` with ``--concurrency``/``--runs-dir`` bound and spend metered.

    Spend is charged per completed task, not per finished split. Charging a whole split at
    once meant the cap could not interrupt work already running: a 10-task airline split at
    ~$0.28/task blew a $0.50 budget out to $2.85 before anyone looked. Overshoot is now
    bounded by whatever is in flight, i.e. ``--concurrency`` tasks.
    """

    def run(spec_, domain, split, *, iteration=0, seed=0, **kw):
        if loop.spend >= loop.budget:
            raise BudgetExceeded(
                f"budget exhausted before running {spec_.id}: "
                f"spent ${loop.spend:.4f} of ${loop.budget:.2f}"
            )
        return runner.run(
            spec_, domain, split, iteration=iteration, seed=seed,
            concurrency=loop.concurrency, runs_dir=loop.runs_dir,
            on_row=lambda row: _charge(loop, spec_, [row]),
        )

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
    if loop.memory is not None:
        loop.memory.begin_run()
    rows = _budgeted_run(loop)(
        candidate, loop.domain, SEARCH_SPLIT, iteration=iteration, seed=loop.seed
    )
    if loop.memory is not None:
        loop.memory.credit_run(rows, _threshold(loop.domain))
        loop.injected = len(loop.memory.injected_ids())
        _learn(loop, candidate, rows, iteration)
    metrics = _summarize(loop, candidate, rows)
    metrics["n_tasks"] = len(rows)
    loop.search[candidate.id] = metrics
    return candidate, rows


def _learn(loop: Loop, candidate: Any, rows: list[dict], iteration: int) -> None:
    """Reflect on this run's failures, retire losing rules and persist the store.

    Called immediately after the run that produced ``rows``, so the tool results the store
    recorded still belong to ``candidate`` and the rules it learns cite that candidate's own
    traces. Reflection reads only search rows, is tagged with the same run context as those
    rows so Neatlogs joins the failing trace to the rule learned from it, and never raises:
    an unreachable gateway costs new rules, not the run.
    """
    if loop.memory is None:
        return
    with tracing.run_context(
        candidate_id=candidate.id, iteration=iteration,
        domain=loop.domain.name, split=SEARCH_SPLIT,
    ):
        loop.memory.reflect(
            rows, loop.domain, client=loop.memory_client, iteration=iteration,
            threshold=_threshold(loop.domain),
        )
        loop.memory.prune()
        loop.memory.save()


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
        # Refuse to fix what we do not believe. Attribution accuracy for this class of
        # classifier is 14-48%, so acting on every diagnosis spends gate cycles on
        # hypotheses -- and a fix that passes the gate for a misdiagnosed reason is worse
        # than one that fails. The issue stays open and ranked for when more evidence
        # arrives; we just move to the next one we do believe.
        if not diagnose.actionable(issue):
            logger.info(json.dumps({
                "event": "issue_skipped_low_confidence", "issue": issue["id"],
                "class": issue["class"], "confidence": round(diagnose.confidence_of(issue), 3),
                "floor": diagnose.CONFIDENCE_FLOOR,
            }))
            continue
        if not diagnose.confidence_known(issue):
            logger.info(json.dumps({
                "event": "issue_confidence_unreported", "issue": issue["id"],
                "class": issue["class"],
            }))
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
        # the ladder this run resolved against, so a report can state it rather than assume it
        "models_path": str(loop.models_path or llm.MODELS_PATH),
        "search": dict(loop.search),
        "specs": dict(loop.specs),
        # what the optimiser has learned so far: the judges' "memory growing" series
        "ledger": diagnose.ledger_stats(diagnose.load_ledger(loop.ledger)),
        # episodic memory: what the agent has learned, and how much of it this run used
        "memory_entries": len(loop.memory.active) if loop.memory is not None else 0,
        "memory_by_kind": loop.memory.by_kind() if loop.memory is not None else {},
        "memory_injected": loop.injected,
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
    # A plateau is meant to mean "we tried things and they genuinely do not help". A
    # rejection the gate itself flags as underpowered means only "too few discordant tasks
    # for any verdict", so counting it conceded the loop on no evidence. That is not
    # hypothetical: PLATEAU is 2, and both airline and bugfix stopped at iteration 1 on two
    # such rejections -- one iteration before rewrite_tool_desc, the third operator listed
    # for their top-ranked issue, would have been tried at all.
    if gate.is_underpowered(result.wins, result.losses):
        body["underpowered"] = True
        console.print(
            f"    [yellow]not counted toward plateau: only {result.wins + result.losses} "
            f"discordant task(s), need {gate._min_discordant_to_promote()} for any verdict"
            "[/yellow]"
        )
        return incumbent, rows, body
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


def cmd_init(args: argparse.Namespace, console: Console) -> int:
    """`anneal init`: five questions in, a runnable domain directory out.

    All the work belongs to ``anneal.onboard``; this only picks the terminal transport (which
    reads piped stdin just as happily as a keyboard) and prints the command to run next.
    """
    from anneal import onboard

    domains_dir = Path(args.domains_dir)
    domains_dir.mkdir(parents=True, exist_ok=True)
    transport = onboard.RichTransport(console)
    interview = None
    try:
        interview = onboard.run_interview(transport, domains_dir=domains_dir, console=console)
        path = onboard.generate_domain(interview, domains_dir=domains_dir)
    except onboard.OnboardError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    for note in interview.notes:
        console.print(f"[yellow]{note}[/yellow]")
    console.print(f"\n[green]wrote[/green] {path}")
    for name in sorted(p.name for p in path.iterdir()):
        console.print(f"  {name}")
    console.print(f"\nNext: [cyan]uv run anneal run {path}[/cyan]")
    return 0


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
        memory=_open_memory(runs_dir, domain.name, enabled=not args.no_memory),
    )
    loop.ledger.parent.mkdir(parents=True, exist_ok=True)
    gate.clear_cache()
    console.print(f"[bold]anneal run[/bold] {loop.domain.name} "
                  f"iterations={args.iterations} budget=${loop.budget:.2f}")
    written: list[dict[str, Any]] = []
    with memory_mod.activate(loop.memory):
        status = _run_loop(loop, args, console, written)
    _progress(console, written)
    console.print(f"summaries in {loop.runs_dir / loop.domain.name}")
    if loop.memory is not None:
        kinds = ", ".join(f"{n} {kind}s" for kind, n in loop.memory.by_kind().items())
        console.print(f"memory: {kinds} in {loop.memory.db_path}")
    return status


def _open_memory(runs_dir: Path, domain_name: str, *, enabled: bool) -> Any | None:
    """The domain's episodic store, carried over from every previous run, or None."""
    if not enabled:
        return None
    return memory_mod.Memory.load(memory_mod.memory_path(runs_dir, domain_name))


def _run_loop(
    loop: Loop, args: argparse.Namespace, console: Console, written: list[dict[str, Any]]
) -> int:
    """The iteration loop itself, run with the episodic memory store active."""
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
        loop.memory = _open_memory(loop.runs_dir, domain.name, enabled=True)
        with memory_mod.activate(loop.memory):
            result = _regate_one(loop, body, specs, args)
        console.print(f"{body['domain']} i{body['iteration']}: "
                      f"{result.reason} (p={result.p:.3f})")


def _regate_one(loop: Loop, body: dict[str, Any], specs: dict[str, str],
                args: argparse.Namespace) -> Any:
    """One gate re-run for ``body``; memory is read here and never written."""
    return gate.gate(
        spec.load_spec(specs[body["incumbent_id"]]),
        spec.load_spec(specs[body["candidate_id"]]),
        loop.domain, body["iteration"], Path(args.runs_dir), run=_budgeted_run(loop),
    )


# --- `anneal anneal` and `anneal dashboard` ----------------------------------------------


def _peak_score(runs_dir: Path, summaries: list[dict[str, Any]]) -> float | None:
    """The winner's held-out mean score, which is the bar the downshift has to hold 95% of."""
    block = _winner_block(runs_dir, summaries)
    score = block.get("mean_score")
    return None if score is None else float(score)


def _anneal_domain(summaries: list[dict[str, Any]], args: argparse.Namespace,
                   console: Console) -> int:
    """Downshift one domain's winning spec. Returns 0 when it ran, 1 when it could not."""
    last = summaries[-1]
    name, winner_id = last["domain"], last["winner_id"]
    spec_path = last["specs"].get(winner_id)
    if spec_path is None:
        console.print(f"[yellow]{name}: no spec was recorded for {winner_id}[/yellow]")
        return 1
    # None when the optimiser never gated this domain (it saturated at iteration 0 with no
    # failures to diagnose). downshift then measures its own baseline to hold against.
    peak = _peak_score(Path(args.runs_dir), summaries)
    if peak is None:
        console.print(
            f"[dim]{name}: no gate result for {winner_id}; "
            f"holding against its own measured baseline[/dim]"
        )
    domain = load_domain(last["domain_path"])
    runs_dir = Path(args.runs_dir)
    loop = Loop(
        domain=domain, runs_dir=runs_dir, ledger=runs_dir / domain.name / "ledger.json",
        budget=float(args.budget), concurrency=args.concurrency, seed=0,
        models_path=args.models,
    )
    result = anneal_stage.downshift(
        spec.load_spec(spec_path), domain,
        peak_score=peak, runs_dir=runs_dir, iteration=int(last["iteration"]),
        run=_budgeted_run(loop), models_path=args.models,
    )
    _anneal_table(console, name, result.points[0].score if peak is None else peak, result)
    console.print(f"  pareto: {result.pareto_path}\n  winner: {result.spec_path}")
    return 0


def _anneal_table(console: Console, domain: str, peak: float, result: Any) -> None:
    front = {p.config_id for p in result.front}
    table = Table(title=f"anneal {domain} (peak {peak:.3f})", show_edge=False)
    for column in ("config", "tiers", "acc", "pass^3", "$/task", "p95 ms", "kept", "front"):
        table.add_column(column)
    for p in result.points:
        table.add_row(
            p.config_id, ",".join(f"{n}={t}" for n, t in sorted(p.node_tiers.items())),
            _num(p.score), _num(p.pass3), _num(p.cost_per_task, 4),
            _num(p.p95_latency_ms, 0), "yes" if p.kept else "no",
            "*" if p.config_id in front else "",
        )
    console.print(table)


def cmd_anneal(args: argparse.Namespace, console: Console) -> int:
    """Walk each domain's winning spec down the model tiers while its gated score holds."""
    summaries = _summaries(Path(args.runs_dir))
    if not summaries:
        console.print(f"[red]no summary.json under {args.runs_dir}[/red]")
        return 1
    gate.clear_cache()
    wanted = set(args.domain or ())
    names = [n for n in dict.fromkeys(b["domain"] for b in summaries) if not wanted or n in wanted]
    if wanted - set(names):
        console.print(f"[red]no summaries for {sorted(wanted - set(names))}[/red]")
        return 1
    status = 0
    for name in names:
        got = [b for b in summaries if b["domain"] == name]
        try:
            status |= _anneal_domain(got, args, console)
        except BudgetExceeded as exc:
            console.print(f"[red]{name}: {exc}[/red]")
            status = 1
    return status


def cmd_dashboard(args: argparse.Namespace, console: Console) -> int:
    """Serve the runs/ dashboard. Imported lazily so the CLI does not pay for FastAPI."""
    from anneal import dashboard

    return dashboard.main(
        ["--runs-dir", args.runs_dir, "--ledger", args.ledger,
         "--host", args.host, "--port", str(args.port)]
    )


def _num(value: Any, digits: int = 3) -> str:
    return DASH if value is None else f"{float(value):.{digits}f}"


def _usd(value: Any) -> str:
    """A cost in dollars at three significant figures -- never a non-zero cost as ``0.000``.

    Local runs cost fractions of a cent per task ($0.000169 is a real figure from the first
    invoices run), so fixed decimals silently report them as free. ``%g`` keeps three real
    digits wherever the magnitude lands and switches to exponent notation below 1e-4.
    """
    if value is None:
        return DASH
    number = float(value)
    return "0" if number == 0 else f"{number:.3g}"


def _secs(ms: Any) -> str:
    """A latency in seconds with one decimal; 148281 ms is not a number anyone can read."""
    return DASH if ms is None else f"{float(ms) / 1000:.1f}"


def _gate_json(runs_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    """The gate.json this iteration wrote, or {} when no gate ran.

    Resolved under ``runs_dir`` rather than from ``body["gate_path"]``: that path is absolute
    on the machine that produced the run and means nothing anywhere else.
    """
    path = Path(runs_dir) / str(body["domain"]) / str(body["iteration"]) / "gate.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _p_value(body: dict[str, Any], gate_json: dict[str, Any]) -> Any:
    """The exact binomial p of the gate run at this iteration, from summary or gate.json."""
    p = body.get("p_value")
    return gate_json.get("p") if p is None else p


def _gen_gap(block: dict[str, Any], search: dict[str, Any]) -> Any:
    """search mean - reserved-split mean, exactly as ``gate.spec_metrics`` defines it.

    The gate only computes a gen_gap for the candidate, so the incumbent's is null in both
    summary.json and gate.json. Both terms are still on record, so recompute rather than
    print an em-dash: the gated mean from the gate block, the search mean from summary.
    """
    if block.get("gen_gap") is not None:
        return block["gen_gap"]
    gated, searched = block.get("mean_score"), search.get("mean_score")
    if gated is None or searched is None:
        return None
    return float(searched) - float(gated)


def _row(domain: str, stage: str, block: dict[str, Any], search: dict[str, Any], p: Any) -> str:
    cells = [
        domain, stage, _num(block.get("mean_score")), _num(block.get("pass3_rate")),
        _num(_gen_gap(block, search)),
        DASH if block.get("hard_fails") is None else str(block["hard_fails"]),
        _usd(search.get("cost_per_task")), _secs(search.get("p95_latency_ms")), _num(p),
    ]
    return "| " + " | ".join(cells) + " |"


def _winner_block(runs_dir: Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics of the spec that survived the last iteration.

    The final iteration may have halted before the gate ran (budget, no operator left), so
    walk backwards to the most recent iteration that actually scored the winning spec, taking
    the metrics from that summary or, when it recorded none, from the gate.json beside it.
    """
    winner = summaries[-1]["winner_id"]
    for body in reversed(summaries):
        gate_json = _gate_json(runs_dir, body)
        for key in ("candidate", "incumbent"):
            block = _blocks(body, gate_json, key)
            if block.get("candidate_id") == winner:
                return block
    return {}


def _annealed_row(runs_dir: Path, domain: str) -> str | None:
    """The `annealed` row from the downshift stage, or None if it has not run for ``domain``.

    ``anneal anneal`` scores its configurations on the reserved split through the gate, so the
    winning Pareto point carries the same accuracy/pass^3/hard-fail meaning as the rows above
    it. Its cost and latency come from that same measurement rather than from the search split.
    """
    path = runs_dir / domain / "anneal" / "pareto.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    winner = data.get("winner")
    point = next((p for p in data.get("points", []) if p.get("config_id") == winner), None)
    if point is None:
        return None
    block = {
        "mean_score": point.get("score"), "pass3_rate": point.get("pass3"),
        "gen_gap": None, "hard_fails": point.get("hard_fails"),
    }
    search = {
        "cost_per_task": point.get("cost_per_task"),
        "p95_latency_ms": point.get("p95_latency_ms"),
    }
    return _row(domain, "annealed", block, search, None)


def _blocks(body: dict[str, Any], gate_json: dict[str, Any], key: str) -> dict[str, Any]:
    """The summary's ``incumbent``/``candidate`` metrics, falling back to the gate's own copy."""
    return body.get(key) or gate_json.get(key) or {}


def _last_p(runs_dir: Path, bodies: list[dict[str, Any]]) -> Any:
    """The p-value of the most recent iteration whose gate produced one, else None."""
    for body in reversed(bodies):
        p = _p_value(body, _gate_json(runs_dir, body))
        if p is not None:
            return p
    return None


def _report_rows(runs_dir: Path, summaries: list[dict[str, Any]]) -> list[str]:
    """Per domain: the iteration-0 baseline, whatever survived the last gate, and - when the
    downshift stage has run - the annealed configuration that holds the score for less."""
    rows: list[str] = []
    for name in dict.fromkeys(b["domain"] for b in summaries):
        got = [b for b in summaries if b["domain"] == name]
        first, last = got[0], got[-1]
        first_gate = _gate_json(runs_dir, first)
        rows.append(_row(
            name, "iteration 0", _blocks(first, first_gate, "incumbent"),
            first["search"].get(first["incumbent_id"], {}), _p_value(first, first_gate),
        ))
        rows.append(_row(
            name, "final", _winner_block(runs_dir, got),
            last["search"].get(last["winner_id"], {}), _last_p(runs_dir, got),
        ))
        annealed = _annealed_row(runs_dir, name)
        if annealed is not None:
            rows.append(annealed)
    return rows


def _rejected_rows(summaries: list[dict[str, Any]], runs_dir: Path) -> list[str]:
    """One row per mutation the gate refused, naming the condition that failed."""
    rows = []
    for body in summaries:
        gate_json = _gate_json(runs_dir, body)
        decision = body.get("decision") or gate_json.get("decision")
        if decision != "reject":
            continue
        reason = body.get("reason") or gate_json.get("reason") or DASH
        rows.append(
            f"| {body['domain']} | {body['iteration']} | "
            f"{body.get('operator') or DASH} | {reason} |"
        )
    return rows


HEADER = (
    "| Domain | Stage | Holdout acc | pass^3 | Gen gap | Hard fails | $/task | p95 s | p (gate) |"
    "\n|---|---|---|---|---|---|---|---|---|"
)
REJECT_HEADER = (
    "**Rejected mutations** — the gate refusing to promote, and which condition failed.\n\n"
    "| Domain | Iteration | Operator | Gate condition that failed |\n|---|---|---|---|"
)
NO_REJECTS = "**Rejected mutations** — no rejected mutations in these runs."
FOOTNOTE_COLUMNS = f"""\
`{DASH}` means the value does not exist in the runs (no gate ran at that iteration, or the
spec was never scored on that split) — it is never a zero and never a rounded-away number.
Holdout accuracy, pass^3, hard fails and p come from the gate's `gate.json`; `$/task` and p95
are measured on the search split. Gen gap is the search mean minus the gated mean, recomputed
from those two recorded means when the gate stored it only for the candidate."""

# Provider facts are computed from the ladder, never written here. A previous version of this
# footnote asserted "inference is local (Ollama) ... $0 in real money" and kept saying it after
# the ladder moved to hosted models, directly above a $/task column that showed otherwise.
LADDER_NOTE = (
    "Tokens and latency are measured. USD is those tokens priced at the rates the run's ladder "
    "declares; that ladder is listed below, never assumed. `price_source` is what each tier "
    "recorded: `published` is the provider's list price, `scaled` is derived from a published "
    "rate for a different model size, and `unrecorded` means the tier carries no provenance. "
    "A local provider costs nothing in money; the column is still what those tokens would cost "
    "at the listed rate, so rows stay comparable across ladders."
)
LADDER_HEADER = (
    "| Tier | Provider | Model | $/1M in | $/1M out | price_source |\n|---|---|---|---|---|---|"
)


def _ladder_paths(summaries: list[dict[str, Any]]) -> list[Path]:
    """Distinct ladders the summaries recorded, else the ladder in force at report time.

    Runs record ``models_path`` (see ``_summary``); older runs did not, and for those the only
    honest statement is the ladder the reporting process itself resolves.
    """
    seen: list[Path] = []
    for body in summaries:
        recorded = body.get("models_path")
        if recorded and Path(recorded) not in seen:
            seen.append(Path(recorded))
    return seen or [Path(llm.MODELS_PATH)]


def _display_path(path: Path) -> str:
    """A recorded path may be absolute on the machine that ran it; show it relative when we can."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _ladder_table(path: Path) -> str:
    """The ladder at ``path`` as a markdown table, or a plain statement that it was unreadable."""
    try:
        ladder = llm.describe_ladder(path)
    except (OSError, ValueError, KeyError) as exc:
        return f"Ladder `{_display_path(path)}` could not be read at report time ({exc})."
    rows = [
        f"| {r['tier']} | {r['provider']} | `{r['model']}` | {r['price_in']} | "
        f"{r['price_out']} | {r['price_source']} |"
        for r in ladder
    ]
    return f"Ladder `{_display_path(path)}`:\n\n" + "\n".join([LADDER_HEADER, *rows])


def render_block(runs_dir: Path | str, summaries: list[dict[str, Any]]) -> str:
    """The markdown that goes between the README result markers. Every cell comes from runs/."""
    runs_dir = Path(runs_dir)
    rejected = _rejected_rows(summaries, runs_dir)
    parts = [
        HEADER + "\n" + "\n".join(_report_rows(runs_dir, summaries)),
        ("\n".join([REJECT_HEADER, *rejected]) if rejected else NO_REJECTS),
        FOOTNOTE_COLUMNS,
        LADDER_NOTE,
        *[_ladder_table(path) for path in _ladder_paths(summaries)],
    ]
    return "\n\n".join(parts) + "\n"


RESULTS_START = "<!-- results:start -->"
RESULTS_END = "<!-- results:end -->"


def write_readme(path: Path, block: str) -> None:
    """Replace the marked results block in ``path`` and touch nothing else."""
    text = path.read_text(encoding="utf-8")
    head, marker, rest = text.partition(RESULTS_START)
    body, end, tail = rest.partition(RESULTS_END)
    if not marker or not end:
        raise ValueError(f"{path} has no {RESULTS_START} / {RESULTS_END} block")
    path.write_text(f"{head}{marker}\n{block}{end}{tail}", encoding="utf-8")


def cmd_report(args: argparse.Namespace, console: Console) -> int:
    runs_dir = Path(args.runs_dir)
    summaries = _summaries(runs_dir)
    if not summaries:
        console.print(f"[red]no summary.json under {args.runs_dir}[/red]")
        return 1
    block = render_block(runs_dir, summaries)
    if args.write_readme:
        try:
            write_readme(Path(args.write_readme), block)
        except (OSError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
        # stdout stays exactly the block, so `anneal report --write-readme` stays pipeable
        Console(stderr=True).print(f"[green]wrote[/green] {args.write_readme}")
    print(block, end="")
    return 0


# --- argument parsing --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anneal", description="Anneal agent harnesses.")
    parser.add_argument("--version", action="version", version=f"anneal {__version__}")
    sub = parser.add_subparsers(dest="command")
    for name, help_text in COMMANDS.items():
        sub.add_parser(name, help=help_text)

    init = sub.choices["init"]
    init.add_argument("--domains-dir", default="domains",
                      help="where to write the generated domain (default domains/)")
    init.add_argument("--models", default=None, help="price table (default specs/models.yaml)")

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
    run.add_argument("--no-memory", action="store_true",
                     help="disable episodic memory (no recall, no reflection)")

    for name in ("gate", "report", "anneal"):
        sub.choices[name].add_argument("runs_dir", help="runs/ directory to read")
    for name in ("gate", "anneal"):
        sub.choices[name].add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                                       metavar="USD")
        sub.choices[name].add_argument("--concurrency", type=int, default=None)
        sub.choices[name].add_argument("--models", default=None)
    # without this every `anneal anneal` re-anneals every domain in runs/, which both wastes
    # spend on finished domains and collides with one that is still mid-loop.
    sub.choices["anneal"].add_argument(
        "--domain", action="append", metavar="NAME",
        help="only anneal this domain (repeatable); default is every domain in runs_dir",
    )

    report = sub.choices["report"]
    report.add_argument("--markdown", action="store_true",
                        help="print the README results block (the default output)")
    report.add_argument("--write-readme", nargs="?", const="README.md", default=None,
                        metavar="PATH",
                        help="replace the <!-- results --> block in PATH (default README.md)")

    dash = sub.choices["dashboard"]
    dash.add_argument("--runs-dir", default=str(runner.RUNS_DIR))
    dash.add_argument("--ledger", default="ledger.json")
    dash.add_argument("--host", default="127.0.0.1")
    dash.add_argument("--port", type=int, default=8000)
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
    if getattr(args, "models", None):
        # One ladder per process. The flag drives the same default that ANNEAL_MODELS_PATH
        # seeds, so architect, diagnose and mutate can never resolve a different file than
        # runner does (docs/ARCHITECTURE.md, "Configuration").
        llm.set_default_models_path(args.models)
    handler = {
        "init": cmd_init, "run": cmd_run, "gate": cmd_gate, "report": cmd_report,
        "anneal": cmd_anneal, "dashboard": cmd_dashboard,
    }[args.command]
    # Nothing on the run path used to call this -- only mutate.py did, for the prompt
    # registry -- so a fully configured Neatlogs project still received zero traces, and
    # Diagnose would ask the MCP for spans that were never sent. No-op without a key.
    if args.command in ("run", "gate", "anneal"):
        from anneal.tracing import init_tracing, shutdown

        init_tracing()
        try:
            return handler(args, console)
        finally:
            # Traces are batched; without this the last iteration's spans die with the process.
            shutdown()
    return handler(args, console)


if __name__ == "__main__":
    sys.exit(main())
