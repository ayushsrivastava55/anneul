"""Dashboard tests: every rendered number comes from fixtures on disk, nothing crashes."""

from __future__ import annotations

import json
import sys
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
    # anneal.billing is a real module now, so `from anneal import billing` resolves the
    # package attribute and ignores a sys.modules stub. Patch the real accessor instead.
    from anneal import billing

    monkeypatch.setattr(billing, "balance", lambda: 4.25)
    body = client.get("/fragments/balance").text
    assert "$4.25" in body


def test_balance_reads_the_cache_billing_writes(tmp_path: Path) -> None:
    """The real integration: billing caches balance.json, the dashboard reads it."""
    from anneal import billing

    cache = tmp_path / "balance.json"
    cache.write_text(json.dumps({"balance_tokens": 1234.0, "unit": "tokens"}), encoding="utf-8")
    assert billing.balance(cache) == 1234.0
    assert billing.balance(tmp_path / "missing.json") is None


def test_ledger_accepts_dict_wrapper(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"issues": LEDGER}), encoding="utf-8")
    assert [row["id"] for row in dashboard.load_ledger(path)] == ["L-0001", "L-0002"]


def test_ledgers_aggregate_per_domain_files_when_no_explicit_ledger_exists(
    tmp_path: Path,
) -> None:
    """The run loop writes runs/<domain>/ledger.json; the dashboard must find those."""
    runs = tmp_path / "runs"
    for domain, issues in (("airline", LEDGER[:1]), ("bugfix", LEDGER[1:])):
        (runs / domain).mkdir(parents=True)
        (runs / domain / "ledger.json").write_text(json.dumps(issues), encoding="utf-8")
    rows = dashboard.load_ledgers(runs, tmp_path / "nonexistent-ledger.json")
    assert [(r["domain"], r["id"]) for r in rows] == [
        ("airline", "L-0001"),
        ("bugfix", "L-0002"),
    ]
    # an explicit file that exists still wins
    explicit = tmp_path / "ledger.json"
    explicit.write_text(json.dumps(LEDGER), encoding="utf-8")
    rows = dashboard.load_ledgers(runs, explicit)
    assert [r["id"] for r in rows] == ["L-0001", "L-0002"]
    assert all(r["domain"] == tmp_path.name for r in rows)


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


def test_landing_page_is_served_and_names_the_product(client: TestClient) -> None:
    """/landing serves the static product page; the file is the single source of truth."""
    response = client.get("/landing")
    assert response.status_code == 200
    assert "Anneal" in response.text
    assert "the agent that engineers agents" in response.text.lower()


# --- console regions -------------------------------------------------------------------
#
# The console renders six regions from the artefacts of the latest iteration. These tests
# pin the two rules that matter: stage state is derived from which artefacts exist, and a
# value that is not on disk renders as an em-dash rather than a zero.

DASH = "—"

CONSOLE_SUMMARY = {
    "domain": "airline",
    "iteration": 1,
    "incumbent_id": "cand-01",
    "candidate_id": "cand-01-add_validator_node-i1",
    "winner_id": "cand-01",
    "decision": "reject",
    "operator": "add_validator_node",
    "spend_usd": 7.3615,
    "budget_usd": 50.0,
    "mean_score": 0.667,
    "pass3_rate": 0.6,
    "search": {
        "cand-01": {"mean_score": 0.5, "cost_per_task": 0.042989,
                    "p95_latency_ms": 20779.3, "hard_fails": 3},
        # no cost_per_task and no p95 on the challenger: both must render as em-dashes
        "cand-01-add_validator_node-i1": {"mean_score": 0.6, "hard_fails": 3},
    },
}

CONSOLE_GATE = {
    "decision": "reject",
    "promoted": False,
    "p": 1.0,
    "alpha": 0.1,
    "wins": 0,
    "losses": 1,
    "min_discordant_to_promote": 4,
    "underpowered": True,
    "reason": "pass3_rate 0.600 < incumbent 0.700",
    "candidate": {"candidate_id": "cand-01-add_validator_node-i1", "mean_score": 0.6667,
                  "pass3_rate": 0.6, "hard_fails": 5, "gen_gap": -0.0667},
    "incumbent": {"candidate_id": "cand-01", "mean_score": 0.7, "pass3_rate": 0.7,
                  "hard_fails": 2, "gen_gap": None},
}

CONSOLE_SPEC = """
id: cand-01-add_validator_node-i1
topology: single
nodes:
- name: executor
  role: executor
  model_tier: cheap
- name: validator
  role: validator
  model_tier: mid
"""

LIVE_ROWS = [
    {"task_id": "airline-10", "score": 0.0, "hard_fail": True, "latency_ms": 20851.0,
     "trace_id": "7430a268", "output": "x" * 50, "trace": [{"tool": "get_user_details"}]},
    {"task_id": "airline-13", "score": 1.0, "hard_fail": False, "hit_step_budget": True,
     "latency_ms": 9100.0},
    # no score and no latency at all: the row must show em-dashes, not zeros
    {"task_id": "airline-18"},
]

CONSOLE_LEDGER = [
    {"id": "L-0001", "class": "unsafe_action", "node": "executor", "count": 15,
     "status": "open", "operators_tried": ["add_escalation_node"]},
    # a class the taxonomy does not know: severity and next operator are unknown, not zero
    {"id": "L-0002", "class": "not_a_real_class", "node": "planner", "count": 2,
     "status": "open", "operators_tried": []},
]


def write_console_fixture(
    tmp_path: Path,
    *,
    domain: str = "airline",
    gate: bool = True,
    spec: bool = True,
    live: bool = True,
) -> Path:
    """A realistic iteration directory; each artefact can be withheld to test stage state."""
    runs = tmp_path / "runs"
    out = runs / domain / "1"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(CONSOLE_SUMMARY), encoding="utf-8")
    if spec:
        (out / "cand-01.yaml").write_text(CONSOLE_SPEC, encoding="utf-8")
        (out / "cand-01-add_validator_node-i1.yaml").write_text(CONSOLE_SPEC, encoding="utf-8")
    if gate:
        (out / "gate.json").write_text(json.dumps(CONSOLE_GATE), encoding="utf-8")
    if live:
        body = "\n".join(json.dumps(row) for row in LIVE_ROWS) + "\n{ broken line\n"
        (out / "cand-01-add_validator_node-i1.search.s0.jsonl").write_text(body, encoding="utf-8")
    (runs / domain / "ledger.json").write_text(json.dumps(CONSOLE_LEDGER), encoding="utf-8")
    return runs


def console(tmp_path: Path, **kwargs) -> TestClient:
    runs = write_console_fixture(tmp_path, **kwargs)
    return TestClient(dashboard.create_app(runs, tmp_path / "no-explicit-ledger.json"))


def stage_states(body: str) -> dict[str, str]:
    """{stage name: data-state} parsed out of the rendered rail."""
    states = {}
    for chunk in body.split('class="stage" data-state="')[1:]:
        state = chunk.split('"')[0]
        name = chunk.split('class="sname">')[1].split("<")[0]
        states[name] = state
    return states


def test_rail_marks_stages_done_from_the_artefacts_on_disk(tmp_path: Path) -> None:
    body = console(tmp_path).get("/fragments/rail?domain=airline").text
    assert stage_states(body) == {
        "Architect": "done",   # cand-*.yaml exist
        "Run": "done",         # summary["search"] has candidates
        "Diagnose": "done",    # runs/airline/ledger.json has issues
        "Mutate": "done",      # summary["operator"]
        "Gate": "done",        # gate.json exists
        "Anneal": "active",    # no pareto.json yet — this is where the loop is
    }


def test_rail_stops_at_the_first_missing_artefact(tmp_path: Path) -> None:
    """Withhold gate.json and the rail must show Gate as the active stage, Anneal as future."""
    body = console(tmp_path, gate=False).get("/fragments/rail?domain=airline").text
    states = stage_states(body)
    assert states["Mutate"] == "done"
    assert states["Gate"] == "active"
    assert states["Anneal"] == "todo"


def test_rail_on_a_bare_runs_dir_makes_architect_the_active_stage(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    client = TestClient(dashboard.create_app(runs, tmp_path / "ledger.json"))
    states = stage_states(client.get("/fragments/rail").text)
    assert states["Architect"] == "active"
    assert set(states.values()) == {"active", "todo"}


def test_absent_values_render_an_em_dash_not_a_zero(tmp_path: Path) -> None:
    """The challenger has no cost or p95, one task row has no score, one issue has no severity."""
    client = console(tmp_path)
    candidates = client.get("/fragments/candidates?domain=airline").text
    # the incumbent's real numbers are there ...
    assert "$0.0430" in candidates
    assert "20779ms" in candidates
    # ... and the challenger's missing ones are em-dashes, never 0.0000
    assert candidates.count(f'<td class="mono num">{DASH}</td>') == 2
    assert "$0.0000" not in candidates
    assert ">0ms<" not in candidates

    live = client.get("/fragments/live?domain=airline").text
    assert "airline-18" in live
    assert live.count(f'<td class="mono num">{DASH}</td>') == 2  # airline-18 score and latency
    assert ">0ms<" not in live

    ledger = client.get("/fragments/ledger?domain=airline").text
    assert "not_a_real_class" in ledger
    assert f'<td class="mono num">{DASH}</td>' in ledger  # unknown severity
    assert f"&rarr; {DASH}" in ledger                     # no operator ladder to draw from
    # a known class with an untried operator points at it; an exhausted ladder says so
    assert "&rarr; add_validator_node" in ledger


def test_gate_panel_renders_the_verdict_stats_and_reason(tmp_path: Path) -> None:
    body = console(tmp_path).get("/fragments/gate?domain=airline").text
    assert "REJECT" in body
    assert "pass3_rate 0.600 &lt; incumbent 0.700" in body
    for text in ("0.667", "0.700", "1.000", "0.10", "underpowered"):
        assert text in body, text


def test_gate_panel_without_gate_json_says_so_and_invents_nothing(tmp_path: Path) -> None:
    body = console(tmp_path, gate=False).get("/fragments/gate?domain=airline").text
    assert "No gate.json" in body
    assert "REJECT" not in body
    assert "0.000" not in body


def test_candidates_show_topology_node_pills_and_tier_badges(tmp_path: Path) -> None:
    body = console(tmp_path).get("/fragments/candidates?domain=airline").text
    assert "cand-01-add_validator_node-i1" in body
    for text in ("single", "executor", "validator", "cheap", "mid", "challenger", "reject"):
        assert text in body, text
    assert 'class="link"' in body  # the pills are connected


def test_candidates_without_a_spec_yaml_still_render(tmp_path: Path) -> None:
    body = console(tmp_path, spec=False).get("/fragments/candidates?domain=airline").text
    assert "cand-01" in body
    assert DASH in body


def test_live_run_reads_the_search_jsonl_and_flags_hard_fails(tmp_path: Path) -> None:
    body = console(tmp_path).get("/fragments/live?domain=airline").text
    for text in ("airline-10", "airline-13", "hard fail", "step budget", "20851ms"):
        assert text in body, text
    assert 'class="lrow bad"' in body       # the hard-fail row is flagged
    assert "get_user_details" not in body   # traces are dropped, not rendered


def test_live_run_never_opens_the_reserved_split(tmp_path: Path) -> None:
    """Only *.search.s*.jsonl is read here; the gate owns the other split."""
    runs = write_console_fixture(tmp_path)
    reserved = runs / "airline" / "1" / "cand-01-add_validator_node-i1.holdout.s0.jsonl"
    reserved.write_text(json.dumps({"task_id": "reserved-task", "score": 1.0}), encoding="utf-8")
    body = TestClient(dashboard.create_app(runs, tmp_path / "l.json")).get("/").text
    assert "reserved-task" not in body
    assert "holdout" not in dashboard.__file__ or True  # the module never names that split
    assert "holdout" not in Path(dashboard.__file__).read_text(encoding="utf-8")


def test_topbar_lists_the_domains_present_in_runs_and_shows_the_budget(tmp_path: Path) -> None:
    runs = write_console_fixture(tmp_path)
    (runs / "invoices" / "0").mkdir(parents=True)
    (runs / "invoices" / "0" / "summary.json").write_text(json.dumps({"iteration": 0}))
    body = TestClient(dashboard.create_app(runs, tmp_path / "l.json")).get("/").text
    assert '?domain=airline' in body and '?domain=invoices' in body
    assert "$7.36 / $50.00" in body


def test_malformed_spec_yaml_and_jsonl_lines_are_tolerated(tmp_path: Path) -> None:
    runs = write_console_fixture(tmp_path)
    (runs / "airline" / "1" / "cand-01.yaml").write_text("id: [unclosed\n", encoding="utf-8")
    client = TestClient(dashboard.create_app(runs, tmp_path / "l.json"))
    response = client.get("/?domain=airline")
    assert response.status_code == 200
    assert "cand-01" in response.text
    # the broken jsonl line is skipped, the three good rows survive
    assert len(dashboard.load_iteration(runs, "airline").live) == len(LIVE_ROWS)


def test_page_renders_every_region_and_no_cdn_script(tmp_path: Path) -> None:
    body = console(tmp_path).get("/?domain=airline").text
    for region in ("topbar", "rail", "candidates", "live", "ledger", "gate", "pareto", "curves"):
        assert f'id="{region}"' in body, region
    assert "http://" not in body and "https://" not in body


def test_rail_never_borrows_another_domains_ledger(tmp_path: Path) -> None:
    """bugfix has no ledger of its own, so its Diagnose stage is not done."""
    runs = write_console_fixture(tmp_path)
    bare = runs / "bugfix" / "0"
    bare.mkdir(parents=True)
    (bare / "summary.json").write_text(
        json.dumps({"iteration": 0, "search": {"cand-01": {"mean_score": 0.4}}}), encoding="utf-8"
    )
    (bare / "cand-01.yaml").write_text(CONSOLE_SPEC, encoding="utf-8")
    client = TestClient(dashboard.create_app(runs, tmp_path / "l.json"))
    assert stage_states(client.get("/fragments/rail?domain=airline").text)["Diagnose"] == "done"
    bugfix = stage_states(client.get("/fragments/rail?domain=bugfix").text)
    assert bugfix["Run"] == "done"
    assert bugfix["Diagnose"] == "active"
    assert bugfix["Gate"] == "todo"
