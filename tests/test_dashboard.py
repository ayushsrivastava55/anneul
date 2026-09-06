"""Dashboard tests: every rendered number comes from fixtures on disk, nothing crashes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from anneal import dashboard, vocab

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
    response = client.get("/console")
    assert response.status_code == 200
    body = response.text
    assert "airline" in body
    for text in ("0.734", "0.550", "$0.0092", "6250"):
        assert text in body, text


def test_ledger_fragment_has_class_node_count_status_and_operators(client: TestClient) -> None:
    body = client.get("/fragments/ledger").text
    assert client.get("/fragments/ledger").status_code == 200
    # the reader sees the phrase; the class id survives only as small secondary text
    for text in (vocab.failure("tool_misuse"), "Doer", vocab.failure("premature_stop"),
                 "Planner", "fixed", ">7<", ">3<"):
        assert text in body, text
    assert "tool_misuse" in body  # the id is still there to type on the CLI
    assert "executor" not in body.replace("tool_misuse", "")  # but the role name is not


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
    response = client.get("/console")
    assert response.status_code == 200
    assert "No runs yet" in response.text
    # the console opens one step at a time, so the ledger and pareto empty states live on
    # their own fragments rather than all on the page at once
    assert "nothing to fix" in client.get("/fragments/ledger").text
    assert "Nothing made cheaper yet" in client.get("/fragments/pareto").text


def test_missing_runs_directory_does_not_traceback(tmp_path: Path) -> None:
    client = TestClient(dashboard.create_app(tmp_path / "nope", tmp_path / "nope.json"))
    assert client.get("/console").status_code == 200
    assert client.get("/fragments/curves").status_code == 200


def test_malformed_json_is_skipped_not_fatal(tmp_path: Path) -> None:
    runs, ledger = write_fixture(tmp_path)
    broken = runs / "airline" / "9"
    broken.mkdir()
    (broken / "summary.json").write_text("{not json", encoding="utf-8")
    (runs / "airline" / "anneal" / "pareto.json").write_text("[[[", encoding="utf-8")
    ledger.write_text("nope", encoding="utf-8")
    client = TestClient(dashboard.create_app(runs, ledger))
    response = client.get("/console")
    assert response.status_code == 200
    assert "0.734" in response.text
    assert "nothing to fix" in client.get("/fragments/ledger").text
    assert "Nothing made cheaper yet" in client.get("/fragments/pareto").text
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
    app = dashboard.create_app(tmp_path / "runs", tmp_path / "l.json")
    body = TestClient(app).get("/console").text
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
    # the page leads with what it does for the reader, not with a description of itself
    assert "anneal builds it" in response.text.lower()
    # the front door leads to the thing the reader came to do
    assert 'href="/new"' in response.text
    # and it is a front door: both other surfaces are reachable from it
    assert 'href="/agents"' in response.text
    assert 'href="/console"' in response.text


def test_the_three_surfaces_are_each_reachable_from_the_others(tmp_path: Path) -> None:
    """Anneal is a landing page, an agents index and a console -- not one screen.

    "/" served the console until this, so the console *was* the product: there was no way to
    reach the landing page or to see which agents existed without typing a URL.
    """
    runs, ledger = write_fixture(tmp_path)
    client = TestClient(dashboard.create_app(runs, ledger))
    for path in ("/", "/agents", "/console"):
        page = client.get(path)
        assert page.status_code == 200, path
        for href, _ in dashboard.NAV:
            assert f'href="{href}"' in page.text, f"{path} cannot reach {href}"
    assert client.get("/landing").status_code == 200  # the old link still works


def test_the_agents_index_lists_every_agent_with_a_way_into_it(tmp_path: Path) -> None:
    runs, ledger = write_fixture(tmp_path)
    body = TestClient(dashboard.create_app(runs, ledger)).get("/agents").text
    assert 'href="/console?domain=airline"' in body
    assert "0.734" in body           # its latest measured score
    assert "$0.0092" in body         # and its latest cost per task
    assert "Design" in body  # the step this fixture's loop is on, named as the flow names it


def test_the_agents_index_with_nothing_at_all_says_how_to_make_one(
    tmp_path: Path, monkeypatch
) -> None:
    """Empty means no agents on disk, not merely none that have produced numbers yet."""
    monkeypatch.setattr(dashboard, "DOMAINS_DIR", tmp_path / "domains")
    runs = tmp_path / "runs"
    runs.mkdir()
    body = TestClient(dashboard.create_app(runs, tmp_path / "l.json")).get("/agents").text
    assert "No agents yet" in body
    assert 'href="/new"' in body  # the way out is a page, not a command
    assert "anneal init" in body  # the terminal is still offered


# --- the step timeline ------------------------------------------------------------------
#
# The console is a flow: six steps, exactly one expanded, its artefact beside it. These tests
# pin the three rules that matter — step state is derived from which artefacts exist on disk,
# only the open step's detail is in the document, and an absent value renders as an em-dash.

DASH = "—"

CONSOLE_SUMMARY = {
    "domain": "airline",
    "iteration": 1,
    "incumbent_id": "cand-01",
    "candidate_id": "cand-01-add_validator_node-i1",
    "winner_id": "cand-01",
    "decision": "reject",
    "operator": "add_validator_node",
    "issue": {"id": "L-0001", "class": "unsafe_action", "node": "executor"},
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
    # the same verdict as fields, which is what the console renders; the sentence above stays
    # because it is what the report quotes and what a pre-reason_parts run has on disk
    "reason_parts": {
        "metric": "pass3_rate", "value": 0.6, "comparator": "<", "against": 0.7,
        "against_label": "incumbent", "fmt": "{:.3f}",
    },
    "candidate": {"candidate_id": "cand-01-add_validator_node-i1", "mean_score": 0.6667,
                  "pass3_rate": 0.6, "hard_fails": 5, "gen_gap": -0.0667},
    "incumbent": {"candidate_id": "cand-01", "mean_score": 0.7, "pass3_rate": 0.7,
                  "hard_fails": 2, "gen_gap": None},
}

PARENT_SPEC = """
id: cand-01
topology: single
step_budget: 12
nodes:
- name: executor
  role: executor
  model_tier: cheap
"""

CHILD_SPEC = """
id: cand-01-add_validator_node-i1
topology: single
step_budget: 13
nodes:
- name: executor
  role: executor
  model_tier: cheap
- name: validator
  role: validator
  model_tier: mid
lineage:
  parent: cand-01
  operator: add_validator_node
  ledger_issue: L-0001
"""

LIVE_ROWS = [
    {"task_id": "airline-10", "score": 0.0, "hard_fail": True, "latency_ms": 20851.0,
     "trace_id": "7430a268", "output": "x" * 50, "trace": [{"tool": "trace_payload_marker"}]},
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
    """A realistic iteration directory; each artefact can be withheld to test step state."""
    runs = tmp_path / "runs"
    out = runs / domain / "1"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(CONSOLE_SUMMARY), encoding="utf-8")
    if spec:
        (out / "cand-01.yaml").write_text(PARENT_SPEC, encoding="utf-8")
        (out / "cand-01-add_validator_node-i1.yaml").write_text(CHILD_SPEC, encoding="utf-8")
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


def step_states(body: str) -> dict[str, str]:
    """{step name: data-state} parsed out of the rendered timeline."""
    names = {name: key for key, (name, _) in vocab.STEPS.items()}
    states = {}
    for chunk in body.split('<li class="step" data-state="')[1:]:
        state = chunk.split('"')[0]
        name = chunk.split('class="sname mono">')[1].split("<")[0]
        # keyed by the stage key so that rewording a step in vocab.py is not a test change
        states[names.get(name, name)] = state
    return states


def open_step(body: str) -> str:
    return body.split('class="detail" data-step="')[1].split('"')[0]


def test_timeline_marks_steps_done_from_the_artefacts_on_disk(tmp_path: Path) -> None:
    body = console(tmp_path).get("/fragments/timeline?domain=airline").text
    assert step_states(body) == {
        "architect": "done",   # cand-*.yaml exist
        "run": "done",         # summary["search"] has candidates
        "diagnose": "done",    # runs/airline/ledger.json has issues
        "mutate": "done",      # summary["operator"]
        "gate": "done",        # gate.json exists
        "anneal": "active",    # no pareto.json yet — this is where the loop is
    }


def test_timeline_stops_at_the_first_missing_artefact(tmp_path: Path) -> None:
    """Withhold gate.json and Gate becomes the active step, Anneal a future one."""
    body = console(tmp_path, gate=False).get("/fragments/timeline?domain=airline").text
    states = step_states(body)
    assert states["mutate"] == "done"
    assert states["gate"] == "active"
    assert states["anneal"] == "todo"


def test_timeline_on_a_bare_runs_dir_makes_architect_the_active_step(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    client = TestClient(dashboard.create_app(runs, tmp_path / "ledger.json"))
    states = step_states(client.get("/console").text)
    assert states["architect"] == "active"
    assert set(states.values()) == {"active", "todo"}


def test_timeline_never_borrows_another_domains_ledger(tmp_path: Path) -> None:
    """bugfix has no ledger of its own, so its Diagnose step is not done."""
    runs = write_console_fixture(tmp_path)
    bare = runs / "bugfix" / "0"
    bare.mkdir(parents=True)
    (bare / "summary.json").write_text(
        json.dumps({"iteration": 0, "search": {"cand-01": {"mean_score": 0.4}}}), encoding="utf-8"
    )
    (bare / "cand-01.yaml").write_text(PARENT_SPEC, encoding="utf-8")
    client = TestClient(dashboard.create_app(runs, tmp_path / "l.json"))
    assert step_states(client.get("/console?domain=airline").text)["diagnose"] == "done"
    bugfix = step_states(client.get("/console?domain=bugfix").text)
    assert bugfix["run"] == "done"
    assert bugfix["diagnose"] == "active"
    assert bugfix["gate"] == "todo"


# one marker that only appears in that step's artefact, and nowhere else
STEP_MARKERS = {
    "architect": ">step limit<",
    "run": "airline-10",
    "diagnose": "Not a real class",
    "mutate": "NODES ADDED",
    "gate": "wins needed to pass",
    "anneal": "Nothing made cheaper yet",
}


def test_only_the_open_steps_detail_is_in_the_document(tmp_path: Path) -> None:
    client = console(tmp_path)
    for step, marker in STEP_MARKERS.items():
        body = client.get(f"/console?domain=airline&step={step}").text
        assert body.count('class="detail"') == 1, step
        assert open_step(body) == step
        assert marker in body, step
        for other, other_marker in STEP_MARKERS.items():
            if other != step:
                assert other_marker not in body, f"{other_marker} leaked into {step}"


def test_the_open_step_defaults_to_the_step_the_loop_is_on(tmp_path: Path) -> None:
    """No ?step: the console opens the active step, and an unknown step falls back to it."""
    client = console(tmp_path)
    assert open_step(client.get("/console?domain=airline").text) == "anneal"
    assert open_step(client.get("/console?domain=airline&step=nonsense").text) == "anneal"
    body = console(tmp_path / "b", gate=False).get("/console?domain=airline").text
    assert open_step(body) == "gate"


def test_every_step_is_a_plain_link_not_a_script(tmp_path: Path) -> None:
    body = console(tmp_path).get("/console?domain=airline").text
    for step in STEP_MARKERS:
        # The trailing #flow is what keeps a click from reloading to the top of the page and
        # losing the reader's place; the flow sits below the contract rows.
        assert f'href="/console?step={step}&amp;domain=airline#flow"' in body, step
    assert 'id="flow"' in body  # the anchor those links point at must exist


def test_the_curves_fragment_shows_the_same_domain_the_page_does(tmp_path: Path) -> None:
    """A refresh must not widen the selection the page made.

    Opening "/" resolves a default domain, but the ten-second refresh fetches
    /fragments/curves with no query string. That fragment used to filter only when a domain
    was passed, so the charts for one domain were silently replaced by every domain stacked.
    """
    client = console(tmp_path)
    page = client.get("/console").text
    fragment = client.get("/fragments/curves").text
    for domain in ("airline", "invoices"):
        assert (f"<h3 class=\"mono\">{domain}</h3>" in page) == (
            f"<h3 class=\"mono\">{domain}</h3>" in fragment
        ), domain


def test_absent_values_render_an_em_dash_not_a_zero(tmp_path: Path) -> None:
    """The challenger has no cost or p95, one task row has no score, one issue has no severity."""
    client = console(tmp_path)
    run = client.get("/console?domain=airline&step=run").text
    # the incumbent's real numbers are there ...
    assert "$0.0430" in run
    assert "20.8s" in run  # milliseconds are shown as seconds
    # ... and the challenger's missing ones are em-dashes, never 0.0000
    assert run.count(f'<td class="mono num">{DASH}</td>') == 4  # challenger cost/p95, task 18
    assert "$0.0000" not in run
    assert ">0ms<" not in run
    assert "airline-18" in run

    ledger = client.get("/console?domain=airline&step=diagnose").text
    assert "not_a_real_class" in ledger                    # the id, as secondary text
    assert "Not a real class" in ledger                    # de-slugged, since it has no label
    assert f'<td class="mono num">{DASH}</td>' in ledger   # unknown severity
    assert f'<td class="op">{DASH}' in ledger              # no operator ladder to draw from
    # a known class names its next repair in words, never by the function that performs it
    assert vocab.operator("add_validator_node") in ledger
    assert "add_validator_node" not in ledger.split('class="op"')[1][:200]


def test_gate_step_renders_the_verdict_stats_and_reason(tmp_path: Path) -> None:
    body = console(tmp_path).get("/console?domain=airline&step=gate").text
    assert vocab.decision("reject") in body
    # the gate's own reason line, with its field names read out as words and its numbers intact
    assert "right 3 times running 0.600 is below the version in use&#x27;s 0.700" in body
    assert "pass3_rate" not in body  # the field name is never shown to a reader
    for text in ("0.667", "0.700", "1.000", "0.10", "underpowered"):
        assert text in body, text


def test_gate_step_without_gate_json_says_so_and_invents_nothing(tmp_path: Path) -> None:
    body = console(tmp_path, gate=False).get("/console?domain=airline&step=gate").text
    assert "Nothing to judge this round" in body
    # scoped to the panel: the header strip reports the summary's own decision on every page,
    # and the point here is that the gate panel invents no verdict of its own.
    panel = body.split('class="detail"')[1]
    assert vocab.decision("reject") not in panel
    assert "0.000" not in body


def test_architect_step_shows_the_specs_with_node_pills_and_tiers(tmp_path: Path) -> None:
    body = console(tmp_path).get("/console?domain=airline&step=architect").text
    assert "cand-01-add_validator_node-i1" in body  # the id stays, as small secondary text
    for text in (vocab.topology("single"), vocab.role("executor"), vocab.role("validator"),
                 vocab.tier("cheap"), vocab.tier("mid"), vocab.decision("reject"),
                 vocab.design_name("cand-01-add_validator_node-i1")):
        assert text in body, text
    assert "round's try" in body  # the challenger, named for what it is
    assert 'class="link"' in body  # the pills are connected


def test_architect_step_without_a_spec_yaml_still_lists_the_candidates(tmp_path: Path) -> None:
    body = console(tmp_path, spec=False).get("/console?domain=airline&step=architect").text
    assert "cand-01" in body
    assert DASH in body


def test_mutate_step_shows_the_operator_the_issue_and_the_node_change(tmp_path: Path) -> None:
    body = console(tmp_path).get("/console?domain=airline&step=mutate").text
    for text in ("add_validator_node", "unsafe_action", "L-0001", "cand-01", "validator",
                 "12", "13", "PROMPTS"):
        assert text in body, text


def test_run_step_reads_the_search_jsonl_and_flags_hard_fails(tmp_path: Path) -> None:
    body = console(tmp_path).get("/console?domain=airline&step=run").text
    for text in ("airline-10", "airline-13", "hard fail", "step budget", "20.9s"):
        assert text in body, text
    assert 'class="lrow bad"' in body       # the hard-fail row is flagged
    assert "trace_payload_marker" not in body  # traces are dropped on the way in
    # The answer is not dropped with them. It used to be, which left the one column a reader
    # who will never open a trace file can actually read permanently empty. It is clipped
    # instead, so the size problem that justified dropping it is still solved.
    assert 'class="said"' in body
    assert body.count("x") < 5000


def test_run_step_never_opens_the_reserved_split(tmp_path: Path) -> None:
    """Only *.search.s*.jsonl is read here; the gate owns the other split."""
    runs = write_console_fixture(tmp_path)
    reserved = runs / "airline" / "1" / "cand-01-add_validator_node-i1.holdout.s0.jsonl"
    reserved.write_text(json.dumps({"task_id": "reserved-task", "score": 1.0}), encoding="utf-8")
    client = TestClient(dashboard.create_app(runs, tmp_path / "l.json"))
    assert "reserved-task" not in client.get("/console?domain=airline&step=run").text
    assert "holdout" not in Path(dashboard.__file__).read_text(encoding="utf-8")


def test_the_contract_rows_are_always_visible_and_read_the_domain_files(tmp_path: Path) -> None:
    """GOAL / TOOLS / SCORER come from domains/<name>/; a domain we do not have em-dashes."""
    client = console(tmp_path)
    for step in STEP_MARKERS:
        body = client.get(f"/console?domain=airline&step={step}").text
        for label in ("GOAL", "TOOLS", "SCORER"):
            assert f">{label}</span>" in body, (step, label)
    contract = dashboard.load_contract(tmp_path / "runs", "airline", CONSOLE_SUMMARY)
    assert "tools" in contract and "goal" in contract  # domains/airline is in this repo
    assert dashboard.load_contract(tmp_path / "runs", "no-such-domain", {}) == {}
    body = dashboard.render_contract({}, "no-such-domain")
    assert body.count(f'class="dv">{DASH}</span>') == 3


def test_topbar_lists_the_domains_present_in_runs_and_shows_the_budget(tmp_path: Path) -> None:
    runs = write_console_fixture(tmp_path)
    (runs / "invoices" / "0").mkdir(parents=True)
    (runs / "invoices" / "0" / "summary.json").write_text(json.dumps({"iteration": 0}))
    body = TestClient(dashboard.create_app(runs, tmp_path / "l.json")).get("/console").text
    assert '?domain=airline' in body and '?domain=invoices' in body
    assert "$7.36 / $50.00" in body


def test_unmeasured_iterations_are_hollow_on_a_dotted_line(tmp_path: Path) -> None:
    """An iteration directory with no readable summary is never drawn as a measurement."""
    runs, _ = write_fixture(tmp_path)
    started = runs / "airline" / "3"
    started.mkdir()
    assert dashboard.pending_iterations(runs, "airline") == ("3",)
    body = TestClient(dashboard.create_app(runs, tmp_path / "l.json")).get("/console").text
    assert 'class="pending"' in body
    assert body.count('class="hollow"') == len(dashboard.METRICS)
    assert "not measured yet" in body


def test_malformed_spec_yaml_and_jsonl_lines_are_tolerated(tmp_path: Path) -> None:
    runs = write_console_fixture(tmp_path)
    (runs / "airline" / "1" / "cand-01.yaml").write_text("id: [unclosed\n", encoding="utf-8")
    client = TestClient(dashboard.create_app(runs, tmp_path / "l.json"))
    response = client.get("/console?domain=airline&step=architect")
    assert response.status_code == 200
    assert "cand-01" in response.text
    # the broken jsonl line is skipped, the three good rows survive
    assert len(dashboard.load_iteration(runs, "airline").live) == len(LIVE_ROWS)


def test_page_has_the_standing_regions_and_no_cdn_script(tmp_path: Path) -> None:
    body = console(tmp_path).get("/console?domain=airline").text
    for region in ("topbar", "contract", "curves", "timeline"):
        assert f'id="{region}"' in body, region
    assert "http://" not in body and "https://" not in body


def test_a_gate_written_before_reason_parts_is_still_read_out_in_words(tmp_path: Path) -> None:
    """Old runs carry only the sentence, and every number needed to say it properly.

    The metric name and comparator are the two tokens gate.py writes in a format it owns; the
    values come from the structured incumbent/candidate metrics the same file records. A gate
    whose line does not match that shape falls back to showing the sentence as written.
    """
    runs = write_console_fixture(tmp_path)
    path = runs / "airline" / "1" / "gate.json"
    old_style = {k: v for k, v in CONSOLE_GATE.items() if k != "reason_parts"}
    path.write_text(json.dumps(old_style), encoding="utf-8")
    client = TestClient(dashboard.create_app(runs, tmp_path / "no-ledger.json"))
    panel = client.get("/console?domain=airline&step=gate").text
    # the fields are rebuilt from the metrics that run did record, so even an old gate.json
    # reads as words; the field name is never shown
    assert "right 3 times running 0.600 is below the version in use&#x27;s 0.700" in panel
    assert "pass3_rate" not in panel


# --- the two reading levels ---------------------------------------------------------------


def test_the_plain_view_is_the_default_and_the_technical_one_is_asked_for(
    tmp_path: Path,
) -> None:
    """A person who wanted an agent that triages invoices has no use for run directories."""
    runs, ledger = write_fixture(tmp_path)
    client = TestClient(dashboard.create_app(runs, ledger))
    plain = client.get("/console?domain=airline").text
    assert 'class="plain"' in plain
    assert "Show technical detail" in plain
    technical = client.get("/console?domain=airline&detail=technical").text
    assert 'class="technical"' in technical
    assert "Hide technical detail" in technical
    # nonsense is not a third level
    assert 'class="plain"' in client.get("/console?domain=airline&detail=wat").text


def test_the_switch_returns_you_to_the_view_you_were_looking_at(tmp_path: Path) -> None:
    runs, ledger = write_fixture(tmp_path)
    body = TestClient(dashboard.create_app(runs, ledger)).get(
        "/console?domain=airline&step=gate"
    ).text
    assert 'href="?domain=airline&amp;step=gate&amp;detail=technical"' in body


def test_the_machinery_is_hidden_by_css_not_removed_from_the_page(tmp_path: Path) -> None:
    """Hiding by class keeps the switch instant and keeps one page, not two.

    It also means these assertions check the *rule*, since the identifiers are in both
    documents either way.
    """
    runs, ledger = write_fixture(tmp_path)
    body = TestClient(dashboard.create_app(runs, ledger)).get("/console?domain=airline").text
    for selector in ("body.plain .dslug", "body.plain .cid", "body.plain .strip"):
        assert selector in body, selector


def test_the_contract_rows_say_what_they_mean_before_naming_the_file(tmp_path: Path) -> None:
    runs, ledger = write_fixture(tmp_path)
    body = TestClient(dashboard.create_app(runs, ledger)).get("/console?domain=airline").text
    for plain in ("What it should do", "What it can use", "How we mark it"):
        assert plain in body, plain
    # the loop's own names for the three files survive, as secondary text
    for technical in ("GOAL", "TOOLS", "SCORER"):
        assert f'class="dslug mono">{technical}<' in body, technical


def test_no_em_dash_is_used_as_punctuation_on_any_page(tmp_path: Path) -> None:
    """Two reached readers as &mdash; inside sentences, which the source-level guard missed.

    ``DASH`` itself stays. A lone em-dash standing in for a value that does not exist is a
    table convention, it is documented in the README's results block, and it is the opposite
    of the thing being guarded against here: an em-dash used as prose punctuation, which is
    the writing tic that makes copy read as machine-written.
    """
    import re

    runs, ledger = write_fixture(tmp_path)
    client = TestClient(dashboard.create_app(runs, ledger))
    pages = ["/agents", "/new", "/console?domain=airline"]
    pages += [f"/console?domain=airline&step={key}" for key in dashboard.vocab.STEPS]
    for path in pages:
        body = client.get(path).text
        assert "&mdash;" not in body and "&ndash;" not in body, path
        # a dash with a word on either side of it is punctuation, not a missing value
        assert not re.search(r"\w\s*[\u2013\u2014]\s*\w", body), path


def test_the_run_panel_shows_what_the_agent_actually_answered(tmp_path: Path) -> None:
    """Every other column measures the answer. This column is the answer.

    It was permanently empty at first: load_live_rows dropped ``output`` alongside ``trace``,
    which is right for a trace payload and wrong for one line of text.
    """
    body = console(tmp_path).get("/console?domain=airline&step=run").text
    assert "what it answered" in body
    assert 'class="said"' in body


def test_an_answer_is_shown_without_the_container_the_evaluator_wanted_it_in() -> None:
    """A structured domain records {"label": "manager"}. The reader wants "manager"."""
    assert dashboard._plain_answer('{"label": "manager"}') == "manager"
    assert dashboard._plain_answer({"label": "pay"}) == "pay"
    assert dashboard._plain_answer({"team": "network", "why": "vpn"}) == "team: network, why: vpn"
    assert dashboard._plain_answer("just words") == "just words"
    assert dashboard._plain_answer("{not json") == "{not json"
    assert dashboard._plain_answer(None) == ""


def test_a_long_answer_is_clipped_in_memory_not_only_on_screen(tmp_path: Path) -> None:
    """The reason output was dropped was size; clipping is what makes keeping it safe."""
    runs = tmp_path / "runs" / "d" / "0"
    runs.mkdir(parents=True)
    row = {"task_id": "t-1", "score": 1.0, "output": "x" * 5000, "trace": ["huge"] * 1000}
    (runs / "cand-01.search.s0.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    _, rows = dashboard.load_live_rows(runs, "cand-01")
    assert len(rows[0]["output"]) == dashboard.OUTPUT_CLIP
    assert "trace" not in rows[0]


def test_an_agent_with_no_runs_is_shown_not_swapped_for_another(
    tmp_path: Path, monkeypatch
) -> None:
    """The worst thing found by walking it: right heading, somebody else's numbers.

    Opening an agent whose first round was still going fell through to domains[0], so the page
    showed a different agent entirely and said nothing about it. Being wrong is bad; being
    wrong under the name you asked for is worse.
    """
    runs, ledger = write_fixture(tmp_path)          # airline has runs
    domains = tmp_path / "domains"
    (domains / "refund_checks").mkdir(parents=True)
    (domains / "refund_checks" / "goal.md").write_text(
        "# Goal: Decide whether a refund needs a manager\n", encoding="utf-8"
    )
    monkeypatch.setattr(dashboard, "DOMAINS_DIR", domains)
    body = TestClient(dashboard.create_app(runs, ledger)).get(
        "/console?domain=refund_checks"
    ).text
    assert "Decide whether a refund needs a manager" in body
    assert "0.734" not in body          # airline's score is not shown under this name
    assert "No runs yet" in body        # it says it has none instead


def test_the_console_switcher_offers_agents_that_have_never_run(
    tmp_path: Path, monkeypatch
) -> None:
    runs, ledger = write_fixture(tmp_path)
    domains = tmp_path / "domains"
    (domains / "brand_new").mkdir(parents=True)
    (domains / "brand_new" / "goal.md").write_text("# Goal: Something new\n", encoding="utf-8")
    monkeypatch.setattr(dashboard, "DOMAINS_DIR", domains)
    body = TestClient(dashboard.create_app(runs, ledger)).get("/console").text
    assert 'href="/console?domain=brand_new"' in body
