"""Holdout gate: the only module allowed to read the ``holdout`` split.

Runs incumbent and candidate N times (default 3) on holdout, computes per-task pass^3,
mean score, hard-fail count and generalisation gap, runs an exact two-sided binomial test
on the discordant pairs (McNemar without continuity correction) and promotes the candidate
iff pass^3 >= incumbent, hard_fails <= incumbent and p < ALPHA. Writes
``runs/<domain>/<iteration>/gate.json`` with everything it computed and why it decided.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from scipy.stats import binomtest

from anneal import tracing
from anneal.config import env
from anneal.spec import HarnessSpec

log = logging.getLogger("anneal.gate")

HOLDOUT_SPLIT = "holdout"
ALPHA = 0.1
DEFAULT_RUNS = 3

Rows = list[dict[str, Any]]
RunFn = Callable[..., Rows]


@dataclass(frozen=True)
class SpecMetrics:
    """Holdout metrics for one spec, aggregated over all runs."""

    candidate_id: str
    mean_score: float
    pass3_rate: float
    hard_fails: int
    gen_gap: float | None
    pass3: dict[str, bool]
    n_tasks: int
    n_runs: int


@dataclass(frozen=True)
class PairedResult:
    """Discordant-pair counts and exact binomial p-value."""

    wins: int
    losses: int
    p: float


@dataclass(frozen=True)
class GateResult:
    """Everything the gate computed plus its decision."""

    domain: str
    iteration: int
    incumbent: SpecMetrics
    candidate: SpecMetrics
    wins: int
    losses: int
    p: float
    promoted: bool
    reason: str
    path: Path

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["path"] = str(self.path)
        data["decision"] = "promote" if self.promoted else "reject"
        data["alpha"] = ALPHA
        return data


# --- math -------------------------------------------------------------------------------


def pass3_by_task(runs: list[Rows], threshold: float) -> dict[str, bool]:
    """``pass3[task] = all(score >= threshold)`` across every run the task appears in."""
    scores: dict[str, list[float]] = {}
    for rows in runs:
        for row in rows:
            scores.setdefault(str(row["task_id"]), []).append(float(row["score"]))
    return {task: all(s >= threshold for s in ss) for task, ss in scores.items()}


def spec_metrics(
    runs: list[Rows], threshold: float, search_rows: Rows | None = None
) -> SpecMetrics:
    """Aggregate holdout runs of one spec. ``gen_gap`` is None when no search rows are given."""
    flat = [row for rows in runs for row in rows]
    if not flat:
        raise ValueError("no holdout rows to score")
    pass3 = pass3_by_task(runs, threshold)
    holdout_mean = mean(float(r["score"]) for r in flat)
    gen_gap = mean(float(r["score"]) for r in search_rows) - holdout_mean if search_rows else None
    return SpecMetrics(
        candidate_id=str(flat[0].get("candidate_id", "")),
        mean_score=holdout_mean,
        pass3_rate=mean(pass3.values()),
        hard_fails=sum(1 for r in flat if r.get("hard_fail")),
        gen_gap=gen_gap,
        pass3=pass3,
        n_tasks=len(pass3),
        n_runs=len(runs),
    )


def paired_test(candidate: dict[str, bool], incumbent: dict[str, bool]) -> PairedResult:
    """Exact two-sided binomial test on discordant per-task pass3 pairs (McNemar exact)."""
    tasks = set(candidate) | set(incumbent)
    wins = sum(1 for t in tasks if candidate.get(t, False) and not incumbent.get(t, False))
    losses = sum(1 for t in tasks if incumbent.get(t, False) and not candidate.get(t, False))
    if wins + losses == 0:
        return PairedResult(wins, losses, 1.0)
    p = float(binomtest(wins, wins + losses, 0.5, alternative="two-sided").pvalue)
    return PairedResult(wins, losses, p)


def decide(candidate: SpecMetrics, incumbent: SpecMetrics, p: float) -> tuple[bool, str]:
    """Promote iff pass3 >= incumbent, hard_fails <= incumbent, p < ALPHA.

    The reason names the first condition that failed.
    """
    if candidate.pass3_rate < incumbent.pass3_rate:
        return False, (
            f"pass3_rate {candidate.pass3_rate:.3f} < incumbent {incumbent.pass3_rate:.3f}"
        )
    if candidate.hard_fails > incumbent.hard_fails:
        return False, f"hard_fails {candidate.hard_fails} > incumbent {incumbent.hard_fails}"
    if not p < ALPHA:
        return False, f"p {p:.3f} >= alpha {ALPHA}"
    return True, "promoted"


# --- running ----------------------------------------------------------------------------

_incumbent_cache: dict[tuple[str, int, str], list[Rows]] = {}


def clear_cache() -> None:
    """Forget cached incumbent holdout rows (tests)."""
    _incumbent_cache.clear()


def _default_run() -> RunFn:
    from anneal import runner  # runner is owned by another worker; resolve lazily

    return runner.run


def run_holdout(
    spec: HarnessSpec, domain: Any, iteration: int, run: RunFn, n_runs: int = DEFAULT_RUNS
) -> list[Rows]:
    """Run ``spec`` ``n_runs`` times on holdout with seeds 0..n-1; one row list per run."""
    runs: list[Rows] = []
    with tracing.run_context(
        candidate_id=spec.id, iteration=iteration, domain=domain.name, split=HOLDOUT_SPLIT
    ):
        for seed in range(n_runs):
            runs.append(run(spec, domain, HOLDOUT_SPLIT, seed=seed, iteration=iteration))
    return runs


def _incumbent_holdout(
    spec: HarnessSpec, domain: Any, iteration: int, run: RunFn, n_runs: int
) -> list[Rows]:
    key = (domain.name, iteration, spec.id)
    if key not in _incumbent_cache:
        _incumbent_cache[key] = run_holdout(spec, domain, iteration, run, n_runs)
    return _incumbent_cache[key]


def promote_prompts(spec: HarnessSpec) -> list[str]:
    """Flip the candidate's prompt labels staging -> production. No-op without NEATLOGS_API_KEY."""
    if not env("NEATLOGS_API_KEY"):
        return []
    refs = sorted({node.system_prompt_ref for node in spec.nodes})
    try:
        from anneal import prompts  # owned by architect; may not exist yet
    except ImportError:
        log.warning(json.dumps({"event": "promote_prompts_skipped", "refs": refs}))
        return []
    set_label = getattr(prompts, "set_label", None)
    if set_label is None:
        log.warning(json.dumps({"event": "promote_prompts_no_set_label", "refs": refs}))
        return []
    for ref in refs:
        set_label(ref, "production")
    return refs


def _n_runs() -> int:
    raw = env("ANNEAL_HOLDOUT_RUNS")
    return int(raw) if raw else DEFAULT_RUNS


def gate(
    incumbent: HarnessSpec,
    candidate: HarnessSpec,
    domain: Any,
    iteration: int,
    runs_dir: Path | str,
    *,
    search_rows: Rows | None = None,
    run: RunFn | None = None,
    n_runs: int | None = None,
) -> GateResult:
    """Holdout-gate ``candidate`` against ``incumbent``; write gate.json; flip labels on promote."""
    run = run or _default_run()
    n = n_runs or _n_runs()
    threshold = float(domain.eval.THRESHOLD)
    inc_runs = _incumbent_holdout(incumbent, domain, iteration, run, n)
    cand_runs = run_holdout(candidate, domain, iteration, run, n)
    inc = spec_metrics(inc_runs, threshold)
    cand = spec_metrics(cand_runs, threshold, search_rows)
    pair = paired_test(cand.pass3, inc.pass3)
    promoted, reason = decide(cand, inc, pair.p)
    path = Path(runs_dir) / domain.name / str(iteration) / "gate.json"
    result = GateResult(
        domain=domain.name, iteration=iteration, incumbent=inc, candidate=cand,
        wins=pair.wins, losses=pair.losses, p=pair.p, promoted=promoted, reason=reason, path=path,
    )
    if promoted:
        promote_prompts(candidate)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    log.info(json.dumps({"event": "gate", "promoted": promoted, "reason": reason}))
    return result
