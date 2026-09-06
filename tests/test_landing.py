"""The landing page's figures are generated from the runs, never typed into the HTML.

CLAUDE.md's fifth non-negotiable: numbers come from ``runs/``, never by hand. The README obeys
it through ``anneal report --write-readme`` and ``tests/test_report.py`` guards that. The
landing page did not, and drifted exactly as far as you would expect: it advertised "29x
cheaper", "$0.0014 per task" and "5 -> 3 hard fails on airline", none of which appear anywhere
in ``runs/final``. These tests are the same guard for the same rule.
"""

from __future__ import annotations

import json
from pathlib import Path

from anneal import cli, landing

REPO = Path(__file__).resolve().parents[1]
LANDING = REPO / "landing" / "index.html"
REPORTED_RUNS = REPO / "runs" / "final"


def _generated() -> str:
    summaries = cli._summaries(REPORTED_RUNS)
    assert summaries, "runs/final has no summary.json; the landing page cannot be generated"
    return landing.render_stats(
        REPORTED_RUNS, summaries, lambda body: cli._gate_json(REPORTED_RUNS, body)
    )


def test_the_committed_page_is_what_the_tool_emits() -> None:
    """Byte-identical, so a figure cannot be edited into the HTML and survive the build."""
    text = LANDING.read_text(encoding="utf-8")
    body = text.split(landing.LANDING_START)[1].split(landing.LANDING_END)[0]
    assert body.strip() == _generated().strip()


def test_the_old_invented_figures_cannot_come_back() -> None:
    page = LANDING.read_text(encoding="utf-8")
    for invented in ("29&times;", "29x", "$0.0014", "$0.07", "5 &rarr; 3"):
        assert invented not in page, invented


def test_every_figure_shown_also_appears_in_the_report_table() -> None:
    """The two generated surfaces must agree; both read the same runs."""
    report = cli.render_block(REPORTED_RUNS, cli._summaries(REPORTED_RUNS))
    for move in landing.movements(REPORTED_RUNS, cli._summaries(REPORTED_RUNS))[:landing.MAX_CARDS]:
        for value in (move.before, move.after):
            assert landing._fmt(move.metric, value).lstrip("$").rstrip("s") in report, value


def test_a_domain_that_got_worse_contributes_no_headline(tmp_path: Path) -> None:
    """Only improvements are shown; a regression stays in the report table, not on the page."""
    runs = tmp_path / "runs"
    (runs / "worse" / "0").mkdir(parents=True)
    body = {
        "domain": "worse", "iteration": 0, "incumbent_id": "cand-01", "winner_id": "cand-01",
        "incumbent": {"hard_fails": 1},
        "search": {"cand-01": {"cost_per_task": 0.001, "p95_latency_ms": 1000}},
    }
    (runs / "worse" / "0" / "summary.json").write_text(json.dumps(body), encoding="utf-8")
    anneal_dir = runs / "worse" / "anneal"
    anneal_dir.mkdir()
    # the downshift ended up dearer and slower than the baseline: neither is an improvement
    (anneal_dir / "pareto.json").write_text(json.dumps({
        "winner": "c1",
        "points": [{"config_id": "c1", "cost_per_task": 0.002, "p95_latency_ms": 2000}],
    }), encoding="utf-8")
    assert landing.movements(runs, [body]) == []
    block = landing.render_stats(runs, [body], lambda _: {})
    assert "changes refused" in block  # the refusal card is always there
    assert "cost per task" not in block


def test_a_missing_pareto_file_is_not_fatal(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    (runs / "d" / "0").mkdir(parents=True)
    body = {"domain": "d", "iteration": 0, "incumbent_id": "c", "search": {"c": {}}}
    assert landing.movements(runs, [body]) == []
