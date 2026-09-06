"""Anneal: walk the winning architecture down to the cheapest model mix that still passes.

For each node, in descending order of token share measured on prior runs, we try the next
cheaper tier from ``downshift_order`` in ``specs/models.yaml`` and re-evaluate the whole spec
on the held-out split ``n_runs`` times. A downshift is kept when the mean score holds at
``>= 0.95 * peak_score`` and hard failures do not increase; otherwise the node reverts to its
previous tier and we move on. A node keeps stepping down while it keeps passing.

Every configuration tried -- kept or reverted -- is recorded as a Pareto point
``(score, pass3, $/task, p95 ms)`` and written to ``runs/<domain>/anneal/pareto.json``
alongside the winning spec. Evaluation goes through ``gate.run_holdout``/``gate.spec_metrics``
so the name of the held-out split stays confined to ``anneal/gate.py``.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anneal import gate, llm, runner
from anneal import spec as spec_mod
from anneal.spec import HarnessSpec, Lineage

log = logging.getLogger("anneal.anneal")

KEEP_RATIO = 0.95
DEFAULT_RUNS = 3

Rows = list[dict[str, Any]]
RunFn = Callable[..., Rows]


@dataclass(frozen=True)
class ParetoPoint:
    """One configuration we measured: what it cost, what it scored, whether we kept it."""

    config_id: str
    node_tiers: dict[str, str]
    score: float
    pass3: float
    cost_per_task: float
    p95_latency_ms: float
    kept: bool
    hard_fails: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class AnnealResult:
    """The cheapest spec that survived, plus every point on the way and where they were written."""

    spec: HarnessSpec
    points: list[ParetoPoint]
    front: list[ParetoPoint]
    pareto_path: Path
    spec_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.points[-1].config_id if self.points else None,
            "winning_spec": str(self.spec_path),
            "node_tiers": node_tiers(self.spec),
            "points": [p.to_dict() for p in self.points],
            "front": [p.config_id for p in self.front],
        }


# --- tiers ------------------------------------------------------------------------------


def downshift_order(models_path: Path | str | None = None) -> list[str]:
    """Tier names most to least expensive, from ``specs/models.yaml``."""
    return [str(t) for t in llm.load_models(models_path).get("downshift_order", [])]


def next_tier(tier: str, order: list[str]) -> str | None:
    """The next cheaper tier after ``tier``, or None if it is already the cheapest/unknown."""
    if tier not in order:
        return None
    index = order.index(tier) + 1
    return order[index] if index < len(order) else None


def node_tiers(spec: HarnessSpec) -> dict[str, str]:
    """node name -> model tier."""
    return {node.name: str(node.model_tier) for node in spec.nodes}


def token_share(rows: Rows) -> dict[str, float]:
    """Fraction of all tokens each node accounted for in ``rows`` (empty when no tokens)."""
    totals: dict[str, float] = {}
    for row in rows:
        for name, node in row.get("per_node", {}).items():
            tokens = int(node.get("tokens_in") or 0) + int(node.get("tokens_out") or 0)
            totals[name] = totals.get(name, 0.0) + tokens
    grand = sum(totals.values())
    return {name: value / grand for name, value in totals.items()} if grand else {}


def node_order(spec: HarnessSpec, prior_rows: Rows) -> list[str]:
    """Node names by descending token share; nodes absent from ``prior_rows`` come last."""
    share = token_share(prior_rows)
    names = [node.name for node in spec.nodes]
    return sorted(names, key=lambda name: (-share.get(name, 0.0), names.index(name)))


def _relabel(
    base: HarnessSpec, config_id: str, node_name: str = "", tier: str = ""
) -> HarnessSpec:
    """Copy ``base`` under a new id/lineage, optionally moving one node to ``tier``."""
    candidate = base.model_copy(deep=True)
    for node in candidate.nodes:
        if node.name == node_name:
            node.model_tier = tier  # type: ignore[assignment]
    candidate.id = config_id
    candidate.lineage = Lineage(
        parent=base.id,
        iteration=base.lineage.iteration if base.lineage else 0,
        operator="downshift",
    )
    return HarnessSpec.model_validate(candidate.model_dump(mode="json"))


# --- Pareto -----------------------------------------------------------------------------


def _dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """``a`` dominates ``b``: no worse on every objective and strictly better on one."""
    no_worse = (
        a.score >= b.score
        and a.cost_per_task <= b.cost_per_task
        and a.p95_latency_ms <= b.p95_latency_ms
    )
    strictly_better = (
        a.score > b.score
        or a.cost_per_task < b.cost_per_task
        or a.p95_latency_ms < b.p95_latency_ms
    )
    return no_worse and strictly_better


def pareto_front(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """The non-dominated points: maximise score, minimise $/task and p95 latency."""
    return [
        p for p in points if not any(_dominates(q, p) for q in points if q is not p)
    ]


# --- evaluation -------------------------------------------------------------------------


def _evaluate(
    candidate: HarnessSpec,
    domain: Any,
    *,
    iteration: int,
    run: RunFn,
    n_runs: int,
    models_path: Path | str | None,
    kept: bool = False,
) -> ParetoPoint:
    """Run ``candidate`` on the held-out split via the gate and turn it into a Pareto point."""
    threshold = float(domain.eval.THRESHOLD)
    runs = gate.run_holdout(candidate, domain, iteration, run, n_runs)
    metrics = gate.spec_metrics(runs, threshold)
    flat = [row for rows in runs for row in rows]
    models = {n.name: llm.resolve_model(n.model_tier, models_path) for n in candidate.nodes}
    summary = runner.summarize(flat, threshold, models_path=models_path, node_models=models)
    return ParetoPoint(
        config_id=candidate.id,
        node_tiers=node_tiers(candidate),
        score=metrics.mean_score,
        pass3=metrics.pass3_rate,
        cost_per_task=float(summary["cost_per_task"]),
        p95_latency_ms=float(summary["p95_latency_ms"]),
        kept=kept,
        hard_fails=metrics.hard_fails,
    )


def _load_prior_rows(runs_dir: Path | str, domain: Any, iteration: int, spec_id: str) -> Rows:
    """Read the incumbent's rows for this iteration, if the runner already wrote them."""
    path = Path(runs_dir) / domain.name / str(iteration) / f"{spec_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write(
    result_dir: Path, winner: HarnessSpec, points: list[ParetoPoint], domain: Any, iteration: int
) -> tuple[Path, Path]:
    result_dir.mkdir(parents=True, exist_ok=True)
    spec_path = result_dir / f"{winner.id}.yaml"
    spec_mod.dump_spec(winner, spec_path)
    pareto_path = result_dir / "pareto.json"
    payload = {
        "domain": domain.name,
        "iteration": iteration,
        "points": [p.to_dict() for p in points],
        "front": [p.config_id for p in pareto_front(points)],
        "winner": winner.id,
        "winning_spec": str(spec_path),
        "node_tiers": node_tiers(winner),
    }
    pareto_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return pareto_path, spec_path


# --- the loop ---------------------------------------------------------------------------


def downshift(
    spec: HarnessSpec,
    domain: Any,
    *,
    peak_score: float,
    runs_dir: Path | str,
    iteration: int,
    run: RunFn | None = None,
    n_runs: int = DEFAULT_RUNS,
    models_path: Path | str | None = None,
    prior_rows: Rows | None = None,
) -> AnnealResult:
    """Downshift ``spec`` node by node while the score holds; write pareto.json + the winner.

    A configuration is kept when ``mean_score >= 0.95 * peak_score`` and its hard-fail count
    does not exceed that of the configuration it would replace. ``peak_score`` is the score
    the incumbent earned before annealing (the caller's gate result).
    """
    run = run if run is not None else functools.partial(runner.run, runs_dir=runs_dir)
    order = downshift_order(models_path)
    if prior_rows is None:
        prior_rows = _load_prior_rows(runs_dir, domain, iteration, spec.id)
    ev = functools.partial(
        _evaluate, domain=domain, iteration=iteration, run=run,
        n_runs=n_runs, models_path=models_path,
    )

    winner = _relabel(spec, f"{spec.id}-anneal-0")
    base = ev(winner, kept=True)
    points = [base]
    best = base

    for name in node_order(spec, prior_rows):
        while (tier := next_tier(node_tiers(winner)[name], order)) is not None:
            candidate = _relabel(winner, f"{spec.id}-anneal-{len(points)}", name, tier)
            point = ev(candidate)
            keep = point.score >= KEEP_RATIO * peak_score and point.hard_fails <= best.hard_fails
            points.append(dataclasses.replace(point, kept=keep))
            log.info(json.dumps({"event": "downshift", "node": name, "tier": tier, "kept": keep}))
            if not keep:
                break
            winner, best = candidate, points[-1]

    pareto_path, spec_path = _write(
        Path(runs_dir) / domain.name / "anneal", winner, points, domain, iteration
    )
    return AnnealResult(winner, points, pareto_front(points), pareto_path, spec_path)
