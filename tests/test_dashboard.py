"""Dashboard tests: every rendered number comes from fixtures on disk, nothing crashes."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from anneal import dashboard

SUMMARIES = [
    {"iteration": 0, "score": 0.412, "pass3": 0.2, "cost_per_task": 0.0131, "p95_latency_ms": 9100},
    {"iteration": 1, "score": 0.658, "pass3": 0.45, "cost_per_task": 0.0177,
     "p95_latency_ms": 8400},
    {"iteration": 2, "score": 0.734, "pass3": 0.55, "cost_per_task": 0.0092,
     "p95_latency_ms": 6250},
]

LEDGER = [
    {
        "id": "L-0001",
        "class": "tool_misuse",
        "node": "executor",
        "count": 7,
        "status": "open",
        "operators_tried": ["tighten_tool_schema"],
        "evidence": ["t-3"],
    },
    {
        "id": "L-0002",
        "class": "premature_stop",
        "node": "planner",
        "count": 3,
        "status": "fixed",
        "operators_tried": [],
        "evidence": [],
    },
]

PARETO = [
    {"candidate_id": "c-base", "score": 0.734, "cost_per_task": 0.0092, "p95_latency_ms": 6250},
    {"candidate_id": "c-mid", "score": 0.721, "cost_per_task": 0.0041, "p95_latency_ms": 5100},
    {"candidate_id": "c-cheap", "score": 0.688, "cost_per_task": 0.0016, "p95_latency_ms": 4300},
]


def write_fixture(tmp_path: Path, *, domain: str = "airline") -> tuple[Path, Path]:
    """Write runs/<domain>/<iter>/summary.json, the pareto front and a ledger."""
    runs = tmp_path / "runs"
    for summary in SUMMARIES:
        out = runs / domain / str(summary["iteration"])
        out.mkdir(parents=True)
        (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    anneal_dir = runs / domain / "anneal"
    anneal_dir.mkdir(parents=True)
    (anneal_dir / "pareto.json").write_text(json.dumps(PARETO), encoding="utf-8")
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(LEDGER), encoding="utf-8")
    return runs, ledger


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    runs, ledger = write_fixture(tmp_path)
    return TestClient(dashboard.create_app(runs, ledger))


def test_page_renders_fixture_numbers(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "airline" in body
    for text in ("0.734", "0.550", "$0.0092", "6250"):
        assert text in body, text


def test_ledger_fragment_has_class_node_count_status_and_operators(client: TestClient) -> None:
    body = client.get("/fragments/ledger").text
    assert client.get("/fragments/ledger").status_code == 200
    for text in ("tool_misuse", "executor", "premature_stop", "planner", "fixed",
                 "tighten_tool_schema", ">7<", ">3<"):
        assert text in body, text


def test_pareto_fragment_plots_every_candidate(client: TestClient) -> None:
    body = client.get("/fragments/pareto").text
    assert body.count("<circle") == len(PARETO)
    for text in ("c-cheap", "$0.0016", "4300ms"):
        assert text in body, text


def test_pareto_radius_tracks_p95(tmp_path: Path) -> None:
    runs, _ = write_fixture(tmp_path)
    points = dashboard.load_pareto(runs)["airline"]
    slowest = max(points, key=lambda p: p.p95_latency_ms or 0)
    fastest = min(points, key=lambda p: p.p95_latency_ms or 0)
    svg = dashboard.pareto_chart(points)
    radii = {
        label: float(svg.split('r="')[i + 1].split('"')[0])
        for i, label in enumerate([p.label for p in points])
    }
    assert radii[slowest.label] > radii[fastest.label]


def test_curves_have_one_chart_per_metric(client: TestClient) -> None:
    body = client.get("/fragments/curves").text
    for _, title, _ in dashboard.METRICS:
        assert title in body, title
    assert body.count("<polyline") == len(dashboard.METRICS)


def test_empty_runs_directory_renders_friendly_empty_state(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    client = TestClient(dashboard.create_app(runs, tmp_path / "ledger.json"))
    response = client.get("/")
    assert response.status_code == 200
    assert "No runs yet" in response.text
    assert "Ledger is empty" in response.text
    assert "No pareto.json yet" in response.text


def test_missing_runs_directory_does_not_traceback(tmp_path: Path) -> None:
    client = TestClient(dashboard.create_app(tmp_path / "nope", tmp_path / "nope.json"))
    assert client.get("/").status_code == 200
    assert client.get("/fragments/curves").status_code == 200


def test_malformed_json_is_skipped_not_fatal(tmp_path: Path) -> None:
    runs, ledger = write_fixture(tmp_path)
    broken = runs / "airline" / "9"
    broken.mkdir()
    (broken / "summary.json").write_text("{not json", encoding="utf-8")
    (runs / "airline" / "anneal" / "pareto.json").write_text("[[[", encoding="utf-8")
    ledger.write_text("nope", encoding="utf-8")
    client = TestClient(dashboard.create_app(runs, ledger))
    response = client.get("/")
    assert response.status_code == 200
    assert "0.734" in response.text
    assert "Ledger is empty" in response.text
    assert "No pareto.json yet" in response.text
    assert len(dashboard.load_summaries(runs)["airline"]) == len(SUMMARIES)


def test_summary_field_aliases_are_read(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "invoices" / "0"
    runs.mkdir(parents=True)
    (runs / "summary.json").write_text(
        json.dumps({"mean_score": 0.5, "pass3_rate": 0.25, "usd_per_task": 0.02, "p95_ms": 1200}),
        encoding="utf-8",
    )
    point = dashboard.load_summaries(tmp_path / "runs")["invoices"][0]
    assert (point.score, point.pass3, point.cost_per_task, point.p95_latency_ms) == (
        0.5, 0.25, 0.02, 1200.0,
    )


def test_balance_is_a_dash_without_billing(client: TestClient, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "anneal.billing", None)
    assert "—" in client.get("/fragments/balance").text


def test_balance_comes_from_billing_when_importable(client: TestClient, monkeypatch) -> None:
    fake = types.ModuleType("anneal.billing")
    fake.balance = lambda: 4.25  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anneal.billing", fake)
    body = client.get("/fragments/balance").text
    assert "$4.25" in body


def test_ledger_accepts_dict_wrapper(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"issues": LEDGER}), encoding="utf-8")
    assert [row["id"] for row in dashboard.load_ledger(path)] == ["L-0001", "L-0002"]


REAL_SUMMARY = {
    "domain": "airline",
    "iteration": 1,
    "incumbent_id": "c-000",
    "candidate_id": "c-001",
    "winner_id": "c-001",
    "mean_score": 0.6125,
    "pass3_rate": 0.5,
    "hard_fails": 1,
    "p95_latency_ms": 9900.0,
    "cost_usd": 0.5216,
    "decision": "promote",
    "search": {
        "c-000": {"cost_per_task": 0.0210, "p95_latency_ms": 7700.0, "n_tasks": 40},
        "c-001": {"cost_per_task": 0.0130, "p95_latency_ms": 8100.0, "n_tasks": 40},
    },
}


def test_reads_cli_summary_shape_and_prefers_winner_search_metrics(tmp_path: Path) -> None:
    """cli.py writes $/task under summary["search"][winner_id]; the top-level cost_usd is a
    split total and must never be shown as $/task."""
    out = tmp_path / "runs" / "airline" / "1"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(REAL_SUMMARY), encoding="utf-8")
    point = dashboard.load_summaries(tmp_path / "runs")["airline"][0]
    assert point.score == 0.6125
    assert point.pass3 == 0.5
    assert point.cost_per_task == 0.0130
    assert point.p95_latency_ms == 8100.0
    body = TestClient(dashboard.create_app(tmp_path / "runs", tmp_path / "l.json")).get("/").text
    assert "$0.0130" in body
    assert "0.5216" not in body


REAL_PARETO = {
    "domain": "airline",
    "front": ["cfg-2", "cfg-0"],
    "points": [
        {"config_id": "cfg-0", "score": 0.73, "pass3": 0.55, "cost_per_task": 0.0092,
         "p95_latency_ms": 6250.0, "kept": True, "node_tiers": {"planner": "mid"}},
        {"config_id": "cfg-1", "score": 0.61, "pass3": 0.40, "cost_per_task": 0.0071,
         "p95_latency_ms": 5900.0, "kept": False, "node_tiers": {"planner": "small"}},
        {"config_id": "cfg-2", "score": 0.72, "pass3": 0.50, "cost_per_task": 0.0041,
         "p95_latency_ms": 5100.0, "kept": True, "node_tiers": {"planner": "small"}},
    ],
    "winner": "cfg-2",
}


def test_reads_anneal_pareto_payload_and_marks_the_front(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "airline" / "anneal"
    out.mkdir(parents=True)
    (out / "pareto.json").write_text(json.dumps(REAL_PARETO), encoding="utf-8")
    points = dashboard.load_pareto(tmp_path / "runs")["airline"]
    assert [p.label for p in points] == ["cfg-0", "cfg-1", "cfg-2"]
    assert [bool(p.extra["on_front"]) for p in points] == [True, False, True]
    svg = dashboard.pareto_chart(points)
    assert svg.count('class="pt front"') == 2
    assert "cfg-2 (front): score 0.720, $0.0041/task, p95 5100ms" in svg
