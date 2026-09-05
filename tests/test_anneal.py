"""Anneal downshift, Pareto front and per-node cost attribution.

Fully offline: a fake domain, an injected ``run`` that fabricates contract-shaped rows from a
scripted score/tokens table, and a temporary ``models.yaml``. Nothing here calls an LLM, and
the held-out split is only ever named via ``gate.HOLDOUT_SPLIT``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from anneal import gate, runner
from anneal.anneal import (
    KEEP_RATIO,
    ParetoPoint,
    downshift,
    next_tier,
    node_order,
    pareto_front,
    token_share,
)
from anneal.spec import HarnessSpec

THRESHOLD = 0.5

BASE_SPEC = HarnessSpec.model_validate(
    {
        "id": "cand-1",
        "topology": "planner_executor",
        "step_budget": 5,
        "nodes": [
            {
                "name": "planner",
                "role": "planner",
                "model_tier": "frontier",
                "system_prompt_ref": "planner@v1",
            },
            {
                "name": "executor",
                "role": "executor",
                "model_tier": "frontier",
                "system_prompt_ref": "executor@v1",
            },
        ],
    }
)


@pytest.fixture(autouse=True)
def _clear_gate_cache() -> None:
    gate.clear_cache()


def _models_yaml(tmp_path: Path) -> Path:
    """Three tiers priced 10x apart, so a downshift visibly moves $/task."""
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "providers": {"p": {"base_url_env": "X_URL", "api_key_env": "X_KEY"}},
                "tiers": {
                    "frontier": {
                        "provider": "p",
                        "model": "big",
                        "price_in": 100.0,
                        "price_out": 100.0,
                    },
                    "mid": {"provider": "p", "model": "med", "price_in": 10.0, "price_out": 10.0},
                    "cheap": {
                        "provider": "p",
                        "model": "small",
                        "price_in": 1.0,
                        "price_out": 1.0,
                    },
                },
                "downshift_order": ["frontier", "mid", "cheap"],
            }
        )
    )
    return path


DOMAIN = SimpleNamespace(
    name="fakedom",
    eval=SimpleNamespace(THRESHOLD=THRESHOLD),
)

TASK_IDS = ("t1", "t2", "t3", "t4")
# node name -> tokens per task, so the executor dominates token share
TOKENS = {"planner": 100_000, "executor": 900_000}


class FakeRun:
    """Injected ``runner.run``: scores come from ``scores(node_tiers)``, tokens are fixed.

    Records every (candidate_id, split, seed) it was asked for so tests can assert the
    downshift really re-evaluated on the split the gate owns.
    """

    def __init__(
        self,
        scores: Any,
        hard_fails: Any = None,
        backends: dict[str, str | None] | None = None,
    ) -> None:
        self.scores = scores
        self.hard_fails = hard_fails or (lambda tiers: 0)
        self.backends = backends or {"planner": None, "executor": "small"}
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, spec: Any, domain: Any, split: str, *, seed: int, iteration: int) -> list:
        assert split == gate.HOLDOUT_SPLIT
        tiers = {n.name: n.model_tier for n in spec.nodes}
        self.calls.append((spec.id, split, seed))
        score = float(self.scores(tiers))
        n_hard = int(self.hard_fails(tiers))
        return [
            {
                "task_id": task_id,
                "candidate_id": spec.id,
                "iteration": iteration,
                "score": score,
                "hard_fail": i < n_hard,
                "hit_step_budget": False,
                "schema_error": False,
                "tokens_in": sum(TOKENS.values()),
                "tokens_out": 0,
                "per_node": {
                    name: {
                        "tokens_in": TOKENS[name],
                        "tokens_out": 0,
                        "backend": self.backends.get(name),
                        "ms": 1.0,
                    }
                    for name in TOKENS
                },
                "latency_ms": 10.0,
                "trace_id": f"tr-{task_id}",
                "output": "ok",
            }
            for i, task_id in enumerate(TASK_IDS)
        ]


def _prior_rows() -> list[dict[str, Any]]:
    """One search-split run of the incumbent, the token-share input to node ordering."""
    return FakeRun(lambda tiers: 1.0)(BASE_SPEC, DOMAIN, gate.HOLDOUT_SPLIT, seed=0, iteration=1)


def _anneal(tmp_path: Path, fake: FakeRun, peak_score: float = 1.0, **kw: Any):
    kw.setdefault("prior_rows", _prior_rows())
    return downshift(
        BASE_SPEC,
        DOMAIN,
        peak_score=peak_score,
        runs_dir=tmp_path / "runs",
        iteration=2,
        run=fake,
        n_runs=3,
        models_path=_models_yaml(tmp_path),
        **kw,
    )


# --- tier helpers -----------------------------------------------------------------------


def test_next_tier_walks_down_then_stops() -> None:
    order = ["frontier", "mid", "cheap"]
    assert next_tier("frontier", order) == "mid"
    assert next_tier("mid", order) == "cheap"
    assert next_tier("cheap", order) is None
    assert next_tier("nonesuch", order) is None


def test_node_order_is_by_descending_token_share() -> None:
    rows = FakeRun(lambda tiers: 1.0)(BASE_SPEC, DOMAIN, gate.HOLDOUT_SPLIT, seed=0, iteration=0)
    assert token_share(rows) == pytest.approx({"planner": 0.1, "executor": 0.9})
    assert node_order(BASE_SPEC, rows) == ["executor", "planner"]
    # no prior rows: fall back to spec order rather than crashing
    assert node_order(BASE_SPEC, []) == ["planner", "executor"]


# --- downshift --------------------------------------------------------------------------


def test_downshift_keeps_cheaper_tier_when_score_holds(tmp_path: Path) -> None:
    """Score is flat across tiers, so every node anneals all the way to ``cheap``."""
    result = _anneal(tmp_path, FakeRun(lambda tiers: 1.0))
    assert {n.name: n.model_tier for n in result.spec.nodes} == {
        "planner": "cheap",
        "executor": "cheap",
    }
    assert all(p.kept for p in result.points)
    # highest token share goes first
    assert result.points[1].node_tiers == {"planner": "frontier", "executor": "mid"}


def test_downshift_reverts_the_node_whose_score_drops(tmp_path: Path) -> None:
    """The executor is score-critical below ``mid``; the planner is not."""

    def scores(tiers: dict[str, str]) -> float:
        return 0.4 if tiers["executor"] == "cheap" else 1.0

    result = _anneal(tmp_path, FakeRun(scores))
    tiers = {n.name: n.model_tier for n in result.spec.nodes}
    assert tiers == {"planner": "cheap", "executor": "mid"}
    rejected = [p for p in result.points if not p.kept]
    assert [p.node_tiers["executor"] for p in rejected] == ["cheap"]


def test_downshift_keeps_a_score_within_the_five_percent_band(tmp_path: Path) -> None:
    """0.96 of peak survives; 0.94 does not. KEEP_RATIO is the only knob."""
    just_inside = _anneal(tmp_path, FakeRun(lambda tiers: 0.96), peak_score=1.0)
    assert all(p.kept for p in just_inside.points)
    assert KEEP_RATIO == 0.95

    just_outside = _anneal(tmp_path, FakeRun(lambda tiers: 0.94), peak_score=1.0)
    assert {n.name: n.model_tier for n in just_outside.spec.nodes} == {
        "planner": "frontier",
        "executor": "frontier",
    }


def test_downshift_reverts_when_hard_fails_increase(tmp_path: Path) -> None:
    """Score holds, but the cheap executor introduces a hard failure: reject anyway."""

    def hard_fails(tiers: dict[str, str]) -> int:
        return 1 if tiers["executor"] != "frontier" else 0

    result = _anneal(tmp_path, FakeRun(lambda tiers: 1.0, hard_fails))
    tiers = {n.name: n.model_tier for n in result.spec.nodes}
    assert tiers["executor"] == "frontier"
    assert tiers["planner"] == "cheap"  # a node that adds no hard fails still anneals


def test_downshift_evaluates_on_the_gate_owned_split_only(tmp_path: Path) -> None:
    fake = FakeRun(lambda tiers: 1.0)
    _anneal(tmp_path, fake)
    assert {split for _, split, _ in fake.calls} == {gate.HOLDOUT_SPLIT}
    assert {seed for _, _, seed in fake.calls} == {0, 1, 2}
    # every configuration gets its own candidate id, so run files never collide
    ids = [cid for cid, _, _ in fake.calls]
    assert len(set(ids)) == len(ids) / 3


def test_downshift_reads_prior_rows_from_runs_dir_when_not_given(tmp_path: Path) -> None:
    """With no rows on disk the node order falls back to spec order (planner first)."""
    runs_dir = tmp_path / "runs"
    no_rows = _anneal(tmp_path, FakeRun(lambda tiers: 1.0), prior_rows=None)
    assert no_rows.points[1].node_tiers == {"planner": "mid", "executor": "frontier"}

    runner.write_rows(_prior_rows(), runs_dir / "fakedom" / "2" / "cand-1.jsonl")
    from_disk = _anneal(tmp_path, FakeRun(lambda tiers: 1.0), prior_rows=None)
    assert from_disk.points[1].node_tiers == {"planner": "frontier", "executor": "mid"}


def test_downshift_lowers_cost_per_task(tmp_path: Path) -> None:
    result = _anneal(tmp_path, FakeRun(lambda tiers: 1.0))
    assert result.points[-1].cost_per_task < result.points[0].cost_per_task


# --- outputs ----------------------------------------------------------------------------


def test_pareto_json_has_at_least_three_points_and_the_winning_spec(tmp_path: Path) -> None:
    result = _anneal(tmp_path, FakeRun(lambda tiers: 1.0))
    assert result.pareto_path == tmp_path / "runs" / "fakedom" / "anneal" / "pareto.json"
    payload = json.loads(result.pareto_path.read_text())
    assert len(payload["points"]) >= 3
    assert set(payload["points"][0]) == {
        "config_id",
        "node_tiers",
        "score",
        "pass3",
        "cost_per_task",
        "p95_latency_ms",
        "kept",
        "hard_fails",
    }
    assert payload["winner"] == result.spec.id
    assert payload["front"] == [p.config_id for p in result.front]

    written = HarnessSpec.model_validate(yaml.safe_load(Path(payload["winning_spec"]).read_text()))
    assert written.id == result.spec.id
    assert written.lineage is not None and written.lineage.operator == "downshift"


# --- pareto front -----------------------------------------------------------------------


def _point(config_id: str, score: float, cost: float, p95: float) -> ParetoPoint:
    return ParetoPoint(config_id, {}, score, score, cost, p95, kept=True)


def test_pareto_front_drops_dominated_points() -> None:
    best_score = _point("a", 1.0, 10.0, 100.0)
    cheapest = _point("b", 0.8, 1.0, 100.0)
    dominated = _point("c", 0.7, 5.0, 200.0)  # worse than a and b on every axis
    fastest = _point("d", 0.8, 1.0, 50.0)  # dominates b on latency
    front = pareto_front([best_score, cheapest, dominated, fastest])
    assert [p.config_id for p in front] == ["a", "d"]


def test_pareto_front_keeps_everything_when_nothing_dominates() -> None:
    points = [_point("a", 1.0, 10.0, 10.0), _point("b", 0.5, 1.0, 10.0)]
    assert pareto_front(points) == points
    assert pareto_front([]) == []


# --- cost attribution -------------------------------------------------------------------


def test_cost_by_node_prefers_the_backend_header_over_the_requested_model(
    tmp_path: Path,
) -> None:
    """The planner asked for ``big`` but TensorMux served ``small``: price what actually ran."""
    fake = FakeRun(lambda tiers: 1.0, backends={"planner": "small", "executor": None})
    rows = fake(BASE_SPEC, DOMAIN, gate.HOLDOUT_SPLIT, seed=0, iteration=0)
    by_node = runner.cost_by_node(
        rows,
        models_path=_models_yaml(tmp_path),
        node_models={"planner": "big", "executor": "big"},
    )
    # planner: 4 tasks x 100k tokens at $1/1M (header wins) = 0.4
    assert by_node["planner"] == pytest.approx(4 * 100_000 / 1e6 * 1.0)
    # executor: no header, falls back to the requested model at $100/1M
    assert by_node["executor"] == pytest.approx(4 * 900_000 / 1e6 * 100.0)


def test_cost_by_node_zero_and_warns_when_nothing_is_priced(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeRun(lambda tiers: 1.0, backends={"planner": None, "executor": None})
    rows = fake(BASE_SPEC, DOMAIN, gate.HOLDOUT_SPLIT, seed=0, iteration=0)
    with caplog.at_level("WARNING", logger="anneal.runner"):
        by_node = runner.cost_by_node(rows, models_path=_models_yaml(tmp_path))
    assert by_node == {"executor": 0.0, "planner": 0.0}
    assert "unpriced_node" in caplog.text


def test_summary_reports_cost_per_task_and_the_per_node_split(tmp_path: Path) -> None:
    fake = FakeRun(lambda tiers: 1.0, backends={"planner": "med", "executor": "small"})
    rows = fake(BASE_SPEC, DOMAIN, gate.HOLDOUT_SPLIT, seed=0, iteration=0)
    summary = runner.summarize(rows, THRESHOLD, models_path=_models_yaml(tmp_path))
    assert summary["cost_by_node"] == {
        "planner": pytest.approx(4 * 100_000 / 1e6 * 10.0),
        "executor": pytest.approx(4 * 900_000 / 1e6 * 1.0),
    }
    assert summary["cost_usd"] == pytest.approx(sum(summary["cost_by_node"].values()))
    assert summary["cost_per_task"] == pytest.approx(summary["cost_usd"] / len(TASK_IDS))
