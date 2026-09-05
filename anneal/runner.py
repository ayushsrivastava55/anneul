"""Run one candidate spec over a task split, in parallel, and write ``runs/*.jsonl``.

The runner is the only place that scores: it calls ``runtime.run_task`` for every task in
``domain.eval.load_tasks(split)``, scores the output with ``domain.eval.score``, flags hard
failures with ``domain.eval.is_hard_fail`` and writes one JSON line per task to
``runs/<domain>/<iteration>/<candidate_id>.jsonl`` (keys: ``ROW_KEYS``, in that order).

``split`` is an opaque parameter here. Which split is safe to read is decided by callers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from anneal import config, llm, tracing

logger = logging.getLogger("anneal.runner")

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
DEFAULT_CONCURRENCY = 6

ROW_KEYS: tuple[str, ...] = (
    "task_id",
    "candidate_id",
    "iteration",
    "score",
    "hard_fail",
    "hit_step_budget",
    "schema_error",
    "tokens_in",
    "tokens_out",
    "per_node",
    "latency_ms",
    "trace_id",
    "output",
)
NODE_KEYS: tuple[str, ...] = ("tokens_in", "tokens_out", "backend", "ms")

RunTask = Callable[..., Any]


def default_concurrency() -> int:
    """``ANNEAL_CONCURRENCY`` from the environment, else 6."""
    return int(config.env("ANNEAL_CONCURRENCY", str(DEFAULT_CONCURRENCY)) or DEFAULT_CONCURRENCY)


def _task_id(task: Any) -> str:
    if hasattr(task, "id"):
        return str(task.id)
    return str(task["id"])


def _resolve_run_task(run_task: RunTask | None) -> RunTask:
    if run_task is not None:
        return run_task
    from anneal import runtime  # deferred: runtime is a sibling task and may land later

    return runtime.run_task


def _node_row(node: dict[str, Any]) -> dict[str, Any]:
    return {key: node.get(key) for key in NODE_KEYS}


def _row_from_result(task: Any, spec: Any, iteration: int, domain: Any, result: Any) -> dict:
    per_node = {name: _node_row(node) for name, node in dict(result.per_node).items()}
    return {
        "task_id": _task_id(task),
        "candidate_id": spec.id,
        "iteration": iteration,
        "score": float(domain.eval.score(task, result.output)),
        "hard_fail": bool(domain.eval.is_hard_fail(task, result.trace)),
        "hit_step_budget": bool(result.hit_step_budget),
        "schema_error": bool(result.schema_error),
        "tokens_in": sum(int(n["tokens_in"] or 0) for n in per_node.values()),
        "tokens_out": sum(int(n["tokens_out"] or 0) for n in per_node.values()),
        "per_node": per_node,
        "latency_ms": float(result.latency_ms),
        "trace_id": result.trace_id,
        "output": result.output,
    }


def _row_from_error(task: Any, spec: Any, iteration: int, exc: BaseException) -> dict:
    """A crashed task counts as a failed task: score 0, no trace, error text as output."""
    return {
        "task_id": _task_id(task),
        "candidate_id": spec.id,
        "iteration": iteration,
        "score": 0.0,
        "hard_fail": False,
        "hit_step_budget": False,
        "schema_error": False,
        "tokens_in": 0,
        "tokens_out": 0,
        "per_node": {},
        "latency_ms": 0.0,
        "trace_id": None,
        "output": f"error: {exc!r}",
    }


def _run_one(
    run_task: RunTask, spec: Any, task: Any, domain: Any, split: str, iteration: int, seed: int
) -> dict:
    """Execute and score one task inside the tagged run context (runs in a worker thread)."""
    tags = {"candidate_id": spec.id, "iteration": iteration, "domain": domain.name, "split": split}
    with tracing.run_context(**tags):
        try:
            result = run_task(spec, task, domain, seed=seed)
            return _row_from_result(task, spec, iteration, domain, result)
        except Exception as exc:  # one crash must not sink the whole split
            logger.warning(
                json.dumps(
                    {"event": "task_error", "task_id": _task_id(task), **tags, "error": repr(exc)}
                )
            )
            return _row_from_error(task, spec, iteration, exc)


async def _run_all(
    run_task: RunTask,
    spec: Any,
    tasks: list[Any],
    domain: Any,
    split: str,
    iteration: int,
    seed: int,
    concurrency: int,
) -> list[dict]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def guarded(task: Any) -> dict:
        async with sem:
            return await asyncio.to_thread(
                _run_one, run_task, spec, task, domain, split, iteration, seed
            )

    return list(await asyncio.gather(*(guarded(t) for t in tasks)))


def write_rows(rows: list[dict], path: Path) -> Path:
    """Write ``rows`` as JSON lines to ``path`` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    return path


def run(
    spec: Any,
    domain: Any,
    split: str,
    *,
    iteration: int = 0,
    seed: int = 0,
    concurrency: int | None = None,
    run_task: RunTask | None = None,
    runs_dir: Path | str | None = None,
) -> list[dict]:
    """Run ``spec`` over every task in ``split``; write and return the contract rows.

    Rows come back in ``load_tasks`` order regardless of completion order. ``run_task``
    defaults to ``anneal.runtime.run_task``; tests inject a stub. ``runs_dir`` defaults to
    ``<repo>/runs``.
    """
    tasks = list(domain.eval.load_tasks(split))
    rows = asyncio.run(
        _run_all(
            _resolve_run_task(run_task),
            spec,
            tasks,
            domain,
            split,
            iteration,
            seed,
            concurrency if concurrency is not None else default_concurrency(),
        )
    )
    path = Path(runs_dir or RUNS_DIR) / domain.name / str(iteration) / f"{spec.id}.jsonl"
    write_rows(rows, path)
    logger.info(json.dumps({"event": "run_complete", "path": str(path), "tasks": len(rows)}))
    return rows


def _p95(values: list[float]) -> float:
    """Nearest-rank 95th percentile; 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def _node_cost(node: dict[str, Any], models_path: Path | str | None) -> float:
    """USD for one node's usage; unpriced backends (e.g. offline ``None``) cost 0 and warn."""
    backend = node.get("backend")
    usage = llm.Usage(
        tokens_in=int(node.get("tokens_in") or 0),
        tokens_out=int(node.get("tokens_out") or 0),
        backend=backend,
        model=str(backend or ""),
    )
    try:
        return llm.cost(usage, models_path)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        logger.warning(
            json.dumps({"event": "unpriced_node", "backend": backend, "error": str(exc)})
        )
        return 0.0


def summarize(
    rows: list[dict], threshold: float, *, models_path: Path | str | None = None
) -> dict[str, Any]:
    """Aggregate contract rows: scores, pass rate (``score >= threshold``), tokens, p95, cost.

    Cost is priced per node via ``llm.cost`` keyed on the node's backend. Nodes whose backend
    has no price in ``models.yaml`` contribute 0 and log a warning, so ``cost_usd`` is a lower
    bound when backends are unknown.
    """
    n = len(rows)
    cost_usd = sum(
        _node_cost(node, models_path) for row in rows for node in row["per_node"].values()
    )
    return {
        "mean_score": sum(r["score"] for r in rows) / n if n else 0.0,
        "pass_rate": sum(1 for r in rows if r["score"] >= threshold) / n if n else 0.0,
        "hard_fails": sum(1 for r in rows if r["hard_fail"]),
        "tokens_in": sum(int(r["tokens_in"]) for r in rows),
        "tokens_out": sum(int(r["tokens_out"]) for r in rows),
        "p95_latency_ms": _p95([float(r["latency_ms"]) for r in rows]),
        "cost_usd": round(cost_usd, 6),
    }
