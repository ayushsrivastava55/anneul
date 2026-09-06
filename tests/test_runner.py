"""runner.run writes one contract-shaped jsonl row per task; summarize aggregates rows.

Runs fully offline: a stub ``run_task`` returns canned results for a fake in-memory domain.
Nothing here imports ``anneal.runtime`` or ``domains/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from anneal import runner
from anneal.spec import HarnessSpec

SPLIT = "search"

SPEC = HarnessSpec.model_validate(
    {
        "id": "cand-1",
        "topology": "single",
        "step_budget": 5,
        "nodes": [
            {
                "name": "solver",
                "role": "executor",
                "model_tier": "mid",
                "system_prompt_ref": "solver@v1",
            }
        ],
    }
)


@dataclass(frozen=True)
class FakeTask:
    id: str
    split: str = SPLIT
    expected: str = "ok"


@dataclass
class FakeResult:
    """Mirror of the pinned runtime.TaskResult contract."""

    output: Any
    trace: list[dict[str, Any]] = field(default_factory=list)
    per_node: dict[str, dict[str, Any]] = field(default_factory=dict)
    steps: int = 1
    hit_step_budget: bool = False
    schema_error: bool = False
    trace_id: str | None = "trace-1"
    latency_ms: float = 10.0


def _per_node(tokens_in: int, tokens_out: int, backend: str | None = "fake-backend") -> dict:
    return {
        "solver": {"tokens_in": tokens_in, "tokens_out": tokens_out, "backend": backend, "ms": 8.0}
    }


def make_domain(tasks: list[FakeTask], calls: dict[str, Any]) -> SimpleNamespace:
    def load_tasks(split: str | None = None) -> list[FakeTask]:
        calls["split"] = split
        return [t for t in tasks if split is None or t.split == split]

    def score(task: FakeTask, output: Any) -> float:
        return 1.0 if output == task.expected else 0.0

    def is_hard_fail(task: FakeTask, trace: list[dict[str, Any]]) -> bool:
        return any(step["tool"] == "cancel" for step in trace)

    ev = SimpleNamespace(
        load_tasks=load_tasks, score=score, is_hard_fail=is_hard_fail, THRESHOLD=1.0
    )
    return SimpleNamespace(name="fake", eval=ev)


def make_run_task(calls: dict[str, Any]):
    canned = {
        "t1": FakeResult(output="ok", per_node=_per_node(100, 20), trace_id="tr-1", latency_ms=5.0),
        "t2": FakeResult(
            output="wrong",
            trace=[{"tool": "cancel", "args": {}, "result": None}],
            per_node=_per_node(300, 40),
            trace_id="tr-2",
            latency_ms=50.0,
            hit_step_budget=True,
        ),
        "t3": FakeResult(output="ok", per_node=_per_node(10, 5), trace_id="tr-3", latency_ms=20.0),
    }

    def run_task(spec: HarnessSpec, task: FakeTask, domain: Any, *, seed: int = 0) -> FakeResult:
        calls.setdefault("seeds", []).append(seed)
        calls.setdefault("spec_ids", []).append(spec.id)
        return canned[task.id]

    return run_task


@pytest.fixture
def tasks() -> list[FakeTask]:
    return [FakeTask("t1"), FakeTask("t2"), FakeTask("t3"), FakeTask("other", split="train")]


def test_run_writes_contract_rows(tmp_path: Path, tasks: list[FakeTask]) -> None:
    calls: dict[str, Any] = {}
    domain = make_domain(tasks, calls)
    rows = runner.run(
        SPEC,
        domain,
        SPLIT,
        iteration=2,
        seed=7,
        concurrency=2,
        run_task=make_run_task(calls),
        runs_dir=tmp_path,
    )

    assert calls["split"] == SPLIT
    assert calls["seeds"] == [7, 7, 7]
    assert calls["spec_ids"] == ["cand-1"] * 3
    assert [r["task_id"] for r in rows] == ["t1", "t2", "t3"]

    path = tmp_path / "fake" / "2" / "cand-1.jsonl"
    assert path.exists()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == rows
    for row in lines:
        assert set(row) == set(runner.ROW_KEYS)
        assert list(row) == list(runner.ROW_KEYS)
        for node in row["per_node"].values():
            assert set(node) == {"tokens_in", "tokens_out", "backend", "ms"}
        assert row["candidate_id"] == "cand-1"
        assert row["iteration"] == 2

    t1, t2, _ = lines
    assert t1["score"] == 1.0 and t1["hard_fail"] is False
    assert t1["tokens_in"] == 100 and t1["tokens_out"] == 20
    assert t1["trace_id"] == "tr-1" and t1["latency_ms"] == 5.0
    assert t2["score"] == 0.0 and t2["hard_fail"] is True
    assert t2["hit_step_budget"] is True and t2["schema_error"] is False
    assert t2["output"] == "wrong"


def test_run_concurrency_defaults_from_env(monkeypatch, tmp_path: Path, tasks) -> None:
    monkeypatch.setenv("ANNEAL_CONCURRENCY", "3")
    assert runner.default_concurrency() == 3
    monkeypatch.delenv("ANNEAL_CONCURRENCY")
    assert runner.default_concurrency() == 6
    calls: dict[str, Any] = {}
    rows = runner.run(
        SPEC, make_domain(tasks, calls), SPLIT, run_task=make_run_task(calls), runs_dir=tmp_path
    )
    assert len(rows) == 3
    assert (tmp_path / "fake" / "0" / "cand-1.jsonl").exists()


def test_run_records_task_exception_as_failure(tmp_path: Path, tasks) -> None:
    def boom(spec, task, domain, *, seed=0):
        if task.id == "t2":
            raise RuntimeError("runtime exploded")
        return FakeResult(output="ok", per_node=_per_node(1, 1))

    rows = runner.run(SPEC, make_domain(tasks, {}), SPLIT, run_task=boom, runs_dir=tmp_path)
    failed = rows[1]
    assert failed["task_id"] == "t2"
    assert failed["score"] == 0.0 and failed["hard_fail"] is False
    assert failed["trace_id"] is None and failed["per_node"] == {}
    assert "runtime exploded" in failed["output"]
    assert rows[0]["score"] == 1.0 and rows[2]["score"] == 1.0


def _models_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "providers": {"p": {"base_url_env": "X_URL", "api_key_env": "X_KEY"}},
                "tiers": {
                    "mid": {
                        "provider": "p",
                        "model": "fake-backend",
                        "price_in": 1.0,
                        "price_out": 2.0,
                    }
                },
            }
        )
    )
    return path


def _row(score: float, hard_fail: bool, latency_ms: float, backend: str | None = "fake-backend"):
    return {
        "score": score,
        "hard_fail": hard_fail,
        "latency_ms": latency_ms,
        "per_node": {
            "a": {"tokens_in": 1_000_000, "tokens_out": 500_000, "backend": backend, "ms": 1.0},
            "b": {"tokens_in": 0, "tokens_out": 500_000, "backend": backend, "ms": 1.0},
        },
        "tokens_in": 1_000_000,
        "tokens_out": 1_000_000,
    }


def test_summarize_math(tmp_path: Path) -> None:
    rows = [
        _row(1.0, False, 10.0),
        _row(0.5, True, 20.0),
        _row(1.0, False, 30.0),
        _row(0.0, False, 100.0),
    ]
    summary = runner.summarize(rows, 1.0, models_path=_models_yaml(tmp_path))
    assert summary == {
        "mean_score": 0.625,
        "pass_rate": 0.5,
        "hard_fails": 1,
        "tokens_in": 4_000_000,
        "tokens_out": 4_000_000,
        "p95_latency_ms": 100.0,
        "cost_usd": pytest.approx(4 * (1.0 * 1.0 + 1.0 * 2.0)),
        "cost_per_task": pytest.approx(1.0 * 1.0 + 1.0 * 2.0),
        "cost_by_node": {
            "a": pytest.approx(4 * (1.0 * 1.0 + 0.5 * 2.0)),
            "b": pytest.approx(4 * 0.5 * 2.0),
        },
    }


def test_summarize_threshold_is_inclusive_and_p95_nearest_rank(tmp_path: Path) -> None:
    rows = [_row(0.7, False, float(i)) for i in range(1, 21)]
    summary = runner.summarize(rows, 0.7, models_path=_models_yaml(tmp_path))
    assert summary["pass_rate"] == 1.0
    assert summary["p95_latency_ms"] == 19.0


def test_summarize_skips_unpriced_nodes(tmp_path: Path) -> None:
    rows = [_row(1.0, False, 1.0, backend=None)]
    summary = runner.summarize(rows, 1.0, models_path=_models_yaml(tmp_path))
    assert summary["cost_usd"] == 0.0


def test_summarize_empty() -> None:
    assert runner.summarize([], 1.0) == {
        "mean_score": 0.0,
        "pass_rate": 0.0,
        "hard_fails": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "p95_latency_ms": 0.0,
        "cost_usd": 0.0,
        "cost_per_task": 0.0,
        "cost_by_node": {},
    }


def test_summarize_tolerates_malformed_models_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "models.yaml"
    bad.write_text("tiers: [unclosed\n")
    summary = runner.summarize([_row(1.0, False, 1.0)], 1.0, models_path=bad)
    assert summary["cost_usd"] == 0.0


def test_run_uses_requested_number_of_threads(tmp_path: Path, tasks) -> None:
    import threading
    import time

    seen: set[str] = set()
    started = threading.Barrier(3, timeout=5)

    def slow(spec, task, domain, *, seed=0):
        seen.add(threading.current_thread().name)
        started.wait()  # all three tasks must be in flight together
        time.sleep(0.01)
        return FakeResult(output="ok", per_node=_per_node(1, 1))

    rows = runner.run(
        SPEC, make_domain(tasks, {}), SPLIT, concurrency=3, run_task=slow, runs_dir=tmp_path
    )
    assert len(rows) == 3 and len(seen) == 3
    assert all(name.startswith("anneal") for name in seen)
