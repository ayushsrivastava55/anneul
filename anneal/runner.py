"""Run one candidate spec over a task split, in parallel, and write ``runs/*.jsonl``.

The runner is the only place that scores: it calls ``runtime.run_task`` for every task in
``domain.eval.load_tasks(split)``, scores the output with ``domain.eval.score``, flags hard
failures with ``domain.eval.is_hard_fail`` and writes one JSON line per task to
``runs/<domain>/<iteration>/<candidate_id>.<split>.s<seed>.jsonl`` (keys: ``ROW_KEYS``, in
that order). Split and seed are part of the name so repeated runs of one candidate at the
same iteration (e.g. the gate's seeded re-runs on another split) never overwrite each other.

``split`` is an opaque parameter here. Which split is safe to read is decided by callers.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

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
    """Fan tasks out over a thread pool of exactly ``concurrency`` workers, in task order."""
    loop = asyncio.get_running_loop()

    def submit(task: Any) -> asyncio.Future[dict]:
        # copy_context() carries the caller's contextvars (run context) into the pool thread.
        call = functools.partial(
            contextvars.copy_context().run,
            _run_one,
            run_task,
            spec,
            task,
            domain,
            split,
            iteration,
            seed,
        )
        return loop.run_in_executor(pool, call)

    with ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="anneal") as pool:
        return list(await asyncio.gather(*(submit(t) for t in tasks)))


def run_path(
    runs_dir: Path | str | None,
    domain_name: str,
    iteration: int,
    candidate_id: str,
    split: str,
    seed: int,
) -> Path:
    """``<runs_dir>/<domain>/<iteration>/<candidate_id>.<split>.s<seed>.jsonl``."""
    name = f"{candidate_id}.{split}.s{seed}.jsonl"
    return Path(runs_dir or RUNS_DIR) / domain_name / str(iteration) / name


def find_runs(
    runs_dir: Path | str | None,
    domain_name: str,
    iteration: int,
    candidate_id: str = "*",
    split: str = "*",
    seed: int | str = "*",
) -> list[Path]:
    """Sorted jsonl files matching the pattern; ``*`` wildcards any component."""
    pattern = run_path(runs_dir, domain_name, iteration, candidate_id, split, seed)  # type: ignore[arg-type]
    return sorted(pattern.parent.glob(pattern.name))


def parse_run_name(path: Path | str) -> tuple[str, str, int]:
    """Inverse of ``run_path``: ``(candidate_id, split, seed)`` from a run file name."""
    name = Path(path).name
    if not name.endswith(".jsonl"):
        raise ValueError(f"not a run file: {name!r}")
    candidate_id, split, seed = name[: -len(".jsonl")].rsplit(".", 2)
    if not seed.startswith("s") or not seed[1:].isdigit():
        raise ValueError(f"not a run file: {name!r}")
    return candidate_id, split, int(seed[1:])


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
    path = run_path(runs_dir, domain.name, iteration, spec.id, split, seed)
    write_rows(rows, path)
    logger.info(json.dumps({"event": "run_complete", "path": str(path), "tasks": len(rows)}))
    return rows


def _p95(values: list[float]) -> float:
    """Nearest-rank 95th percentile; 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def _node_cost(
    node: dict[str, Any], models_path: Path | str | None, requested_model: str = ""
) -> float:
    """USD for one node's usage.

    Priced by the backend that actually served the call (``x-tensormux-backend``, recorded
    per node by the runtime) when that backend is in ``models.yaml``, else by the model the
    node requested (``requested_model``, from the node's tier). Anything still unpriced --
    offline runs record ``backend=None`` -- costs 0 and logs a warning.
    """
    backend = node.get("backend")
    usage = llm.Usage(
        tokens_in=int(node.get("tokens_in") or 0),
        tokens_out=int(node.get("tokens_out") or 0),
        backend=backend,
        model=str(requested_model or backend or ""),
    )
    try:
        return llm.cost(usage, models_path)
    except (KeyError, FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        logger.warning(
            json.dumps({"event": "unpriced_node", "backend": backend, "error": str(exc)})
        )
        return 0.0


def cost_by_node(
    rows: list[dict],
    *,
    models_path: Path | str | None = None,
    node_models: dict[str, str] | None = None,
) -> dict[str, float]:
    """Total USD per node name across ``rows``.

    ``node_models`` maps node name -> the model id the node requested (callers get it from
    ``llm.resolve_model(node.model_tier)``); it is only consulted when the recorded backend
    has no price of its own.
    """
    totals: dict[str, float] = {}
    for row in rows:
        for name, node in row["per_node"].items():
            requested = (node_models or {}).get(name, "")
            totals[name] = totals.get(name, 0.0) + _node_cost(node, models_path, requested)
    return {name: round(usd, 6) for name, usd in sorted(totals.items())}


def summarize(
    rows: list[dict],
    threshold: float,
    *,
    models_path: Path | str | None = None,
    node_models: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate contract rows: scores, pass rate (``score >= threshold``), tokens, p95, cost.

    Cost is attributed per node by ``cost_by_node`` (backend header first, requested model
    second) and reported both as the total ``cost_usd`` and as ``cost_per_task``; the split
    itself is under ``cost_by_node``. Nodes that stay unpriced contribute 0 and log a warning,
    so the cost figures are a lower bound when backends are unknown.
    """
    n = len(rows)
    by_node = cost_by_node(rows, models_path=models_path, node_models=node_models)
    cost_usd = sum(by_node.values())
    return {
        "mean_score": sum(r["score"] for r in rows) / n if n else 0.0,
        "pass_rate": sum(1 for r in rows if r["score"] >= threshold) / n if n else 0.0,
        "hard_fails": sum(1 for r in rows if r["hard_fail"]),
        "tokens_in": sum(int(r["tokens_in"]) for r in rows),
        "tokens_out": sum(int(r["tokens_out"]) for r in rows),
        "p95_latency_ms": _p95([float(r["latency_ms"]) for r in rows]),
        "cost_usd": round(cost_usd, 6),
        "cost_per_task": round(cost_usd / n, 6) if n else 0.0,
        "cost_by_node": by_node,
    }
