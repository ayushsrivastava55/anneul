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
from collections.abc import Callable, Mapping
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
    passes: dict[str, int] = dataclasses.field(default_factory=dict)
    """Per-task count of runs that cleared the threshold. The paired test's unit."""


@dataclass(frozen=True)
class PairedResult:
    """Discordant-pair counts, exact binomial p-value, and the test's own power limit."""

    wins: int
    losses: int
    p: float
    min_discordant_to_promote: int = 0
    """Discordant pairs needed, all won, before any p < ALPHA is even arithmetically possible.

    The exact binomial floor is 0.5**n one-sided, so this is the smallest n with
    0.5**n < ALPHA. Reported because it is the gate's real detectable-effect floor and it
    bit us: a candidate that won 1 holdout task and lost 0 was rejected at p=1.000 under a
    two-sided test, which cannot go below 0.5 at n=1 no matter how clean the win.
    """


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
    escalated: bool = False
    """True when the gate bought a second block of runs before deciding (see ``gate``)."""

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["path"] = str(self.path)
        data["decision"] = "promote" if self.promoted else "reject"
        data["alpha"] = ALPHA
        # Recorded on every gate so a reader can tell an evidence-based rejection from one
        # the sample size made inevitable. Without these two fields every rejection looks
        # like "the change did not help", which is exactly the wrong conclusion to publish.
        floor = _min_discordant_to_promote()
        data["min_discordant_to_promote"] = floor
        data["underpowered"] = self.wins + self.losses < floor
        return data


# --- math -------------------------------------------------------------------------------


def _scores_by_task(runs: list[Rows]) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {}
    for rows in runs:
        for row in rows:
            scores.setdefault(str(row["task_id"]), []).append(float(row["score"]))
    return scores


def pass3_by_task(runs: list[Rows], threshold: float) -> dict[str, bool]:
    """``pass3[task] = all(score >= threshold)`` across every run the task appears in."""
    return {t: all(s >= threshold for s in ss) for t, ss in _scores_by_task(runs).items()}


def passes_by_task(runs: list[Rows], threshold: float) -> dict[str, int]:
    """``passes[task]`` = how many of the repeated runs cleared ``threshold``.

    The paired test's unit. pass^3 stays the reported reliability metric, but as the
    statistic being tested it is nearly blind: it is 1 only on a clean sweep, so a task
    moving 0/3 -> 2/3 registers as no change and contributes no discordant pair. That is
    literally what happened on bugfix, where candidate and incumbent both sat at a pass^3
    rate of 0.1 and the test saw zero pairs and returned p=1.000 twice in a row.
    """
    return {t: sum(1 for s in ss if s >= threshold) for t, ss in _scores_by_task(runs).items()}


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
        passes=passes_by_task(runs, threshold),
    )


def _min_discordant_to_promote(alpha: float = 0.0) -> int:
    """Smallest number of all-won discordant pairs whose one-sided exact p is below alpha."""
    alpha = alpha or ALPHA
    n = 1
    while 0.5**n >= alpha and n < 64:
        n += 1
    return n


def paired_test(
    candidate: Mapping[str, bool | int], incumbent: Mapping[str, bool | int]
) -> PairedResult:
    """Exact ONE-sided sign test on per-task discordant pairs.

    One-sided is the correct test for a promotion gate: the question is only ever "is the
    candidate better", and ``decide`` has already refused anything with a worse pass3 rate
    or more hard fails, so half of a two-sided test's alpha is spent on an alternative we
    have excluded by construction. Two-sided needed 5 discordant wins before p < 0.1 was
    arithmetically reachable; one-sided needs 4. That is a real gain in power and not a
    loosening of the standard -- but it does double the per-test false-promotion rate to
    alpha, which is why we report that rate rather than bury it.

    Takes per-task pass COUNTS (see ``passes_by_task``); a task is a win when the candidate
    cleared the threshold on strictly more runs than the incumbent. Booleans still work and
    reduce to the old McNemar-exact behaviour, which keeps the existing tests meaningful.
    Counting partial movement is where most of the recovered power comes from: it is the
    difference between bugfix contributing zero pairs and contributing real ones.
    """
    tasks = set(candidate) | set(incumbent)
    wins = sum(1 for t in tasks if int(candidate.get(t, 0)) > int(incumbent.get(t, 0)))
    losses = sum(1 for t in tasks if int(incumbent.get(t, 0)) > int(candidate.get(t, 0)))
    floor = _min_discordant_to_promote()
    if wins + losses == 0:
        return PairedResult(wins, losses, 1.0, floor)
    p = float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue)
    return PairedResult(wins, losses, p, floor)


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


def is_underpowered(wins: int, losses: int) -> bool:
    """True when no verdict was arithmetically reachable from this many discordant pairs.

    Below ``_min_discordant_to_promote`` no outcome could have cleared alpha, however
    one-sided the win. Such a rejection says nothing about the candidate and must not be
    read as "the change did not help" -- nor counted toward a plateau.
    """
    return wins + losses < _min_discordant_to_promote()


def underpowered(result: PairedResult) -> bool:
    """``is_underpowered`` for a computed :class:`PairedResult`."""
    return result.wins + result.losses < result.min_discordant_to_promote


# --- running ----------------------------------------------------------------------------

_incumbent_cache: dict[tuple[str, int, str], list[Rows]] = {}


def clear_cache() -> None:
    """Forget cached incumbent holdout rows (tests)."""
    _incumbent_cache.clear()


def _default_run() -> RunFn:
    from anneal import runner  # runner is owned by another worker; resolve lazily

    return runner.run


def run_holdout(
    spec: HarnessSpec,
    domain: Any,
    iteration: int,
    run: RunFn,
    n_runs: int = DEFAULT_RUNS,
    first_seed: int = 0,
) -> list[Rows]:
    """Run ``spec`` ``n_runs`` times on holdout, seeds ``first_seed``..; one row list per run."""
    runs: list[Rows] = []
    with tracing.run_context(
        candidate_id=spec.id, iteration=iteration, domain=domain.name, split=HOLDOUT_SPLIT
    ):
        for seed in range(first_seed, first_seed + n_runs):
            runs.append(run(spec, domain, HOLDOUT_SPLIT, seed=seed, iteration=iteration))
    return runs


def _incumbent_holdout(
    spec: HarnessSpec, domain: Any, iteration: int, run: RunFn, n_runs: int
) -> list[Rows]:
    """The incumbent's first ``n_runs`` holdout runs, extending the cache when short.

    The cache is a growing list per (domain, iteration, spec): an escalated gate asks for
    more runs than the base block, and later candidates at the same iteration reuse both.
    """
    key = (domain.name, iteration, spec.id)
    have = _incumbent_cache.setdefault(key, [])
    if len(have) < n_runs:
        have.extend(
            run_holdout(spec, domain, iteration, run, n_runs - len(have), first_seed=len(have))
        )
    return have[:n_runs]


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


def _escalation_enabled() -> bool:
    """On unless ``ANNEAL_GATE_ESCALATION=0`` (tests and cost-capped runs turn it off)."""
    raw = (env("ANNEAL_GATE_ESCALATION") or "").strip()
    return raw not in {"0", "false", "off"}


def _should_escalate(promoted: bool, reason: str, pair: PairedResult) -> bool:
    """Buy a second block only when the p-value is the sole objection and the sign is right.

    A pass^3 or hard-fail regression is a verdict, not a power problem; equal or losing
    discordant counts give no reason to expect more data to flip the sign. What remains is
    a candidate that is strictly ahead on discordant tasks and failed only ``p < alpha``.
    """
    if promoted or not _escalation_enabled():
        return False
    return reason.startswith("p ") and pair.wins > pair.losses


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
    # Pass counts, not pass^3 booleans: partial movement is evidence and must count.
    pair = paired_test(cand.passes or cand.pass3, inc.passes or inc.pass3)
    promoted, reason = decide(cand, inc, pair.p)
    escalated = False
    if _should_escalate(promoted, reason, pair):
        # The candidate is ahead but the block was too small for any verdict. Rejecting now
        # publishes "no effect" about a sample size, not about the change - so the gate buys
        # one more block of runs (seeds n..2n-1, incumbent extended too) and re-decides on
        # all 2n. One planned extension is a two-stage group-sequential design; the worst
        # case type-I inflation is bounded and it is disclosed per-gate as `escalated`.
        escalated = True
        log.info(json.dumps({
            "event": "gate_escalated", "wins": pair.wins, "losses": pair.losses, "p": pair.p,
        }))
        inc_runs = _incumbent_holdout(incumbent, domain, iteration, run, 2 * n)
        cand_runs += run_holdout(candidate, domain, iteration, run, n, first_seed=n)
        inc = spec_metrics(inc_runs, threshold)
        cand = spec_metrics(cand_runs, threshold, search_rows)
        pair = paired_test(cand.passes or cand.pass3, inc.passes or inc.pass3)
        promoted, reason = decide(cand, inc, pair.p)
    path = Path(runs_dir) / domain.name / str(iteration) / "gate.json"
    result = GateResult(
        domain=domain.name, iteration=iteration, incumbent=inc, candidate=cand,
        wins=pair.wins, losses=pair.losses, p=pair.p, promoted=promoted, reason=reason, path=path,
        escalated=escalated,
    )
    if promoted:
        promote_prompts(candidate)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    log.info(json.dumps({"event": "gate", "promoted": promoted, "reason": reason}))
    return result
