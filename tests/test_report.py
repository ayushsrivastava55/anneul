"""`anneal report`: the publication-quality results block that README.md is generated from.

Every number in the README comes from here, so these tests pin the *rendering* -- a golden
string for the whole block, plus the three ways a cell can go wrong: a tiny cost rounded to
zero, a value that exists only in gate.json printed as an em-dash, and a genuinely missing
value printed as anything but an em-dash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anneal import cli

# --- fixtures ----------------------------------------------------------------------------

SEARCH = {
    "cand-01": {
        "mean_score": 0.4, "cost_per_task": 0.000169, "p95_latency_ms": 148280.6,
        "n_tasks": 15, "hard_fails": 1,
    },
    "cand-01-m0": {
        "mean_score": 0.03333333333333333, "cost_per_task": 0.000108,
        "p95_latency_ms": 39262.1, "n_tasks": 15, "hard_fails": 0,
    },
    "cand-01-m1": {
        "mean_score": 0.26666666666666666, "cost_per_task": 0.000168,
        "p95_latency_ms": 63368.1, "n_tasks": 15, "hard_fails": 2,
    },
}


def summary(iteration: int, cand_id: str, operator: str, reason: str, **kw: Any) -> dict:
    """One iteration of the real invoices run: a rejected mutation, incumbent still winning."""
    body: dict[str, Any] = {
        "domain": "invoices",
        "domain_path": "/nowhere/domains/invoices",
        "iteration": iteration,
        "incumbent_id": "cand-01",
        "candidate_id": cand_id,
        "winner_id": "cand-01",
        "operator": operator,
        "decision": "reject",
        "reason": reason,
        # the gate's own holdout metrics; the incumbent never carries a gen_gap
        "incumbent": {
            "candidate_id": "cand-01", "mean_score": 0.18888888888888888,
            "pass3_rate": 0.0, "gen_gap": None, "hard_fails": 3,
        },
        "candidate": {
            "candidate_id": cand_id, "mean_score": 0.05555555555555555,
            "pass3_rate": 0.0, "gen_gap": -0.02222222222222222, "hard_fails": 0,
        },
        "mean_score": 0.05555555555555555,
        "pass3_rate": 0.0,
        "gen_gap": -0.02222222222222222,
        "hard_fails": 0,
        "p_value": 1.0,
        "cost_usd": 0.001624,
        "p95_latency_ms": 39262.1,
        "search": SEARCH,
        "specs": {},
        "spend_usd": 0.020218,
        "budget_usd": 5.0,
        "stop_reason": None,
    }
    body.update(kw)
    return body


def write(root: Path, body: dict, gate: dict | None = None) -> Path:
    """Write one iteration directory, optionally with the gate.json that iteration wrote."""
    path = root / body["domain"] / str(body["iteration"])
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    if gate is not None:
        (path / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return path


def real_run(root: Path) -> Path:
    """Two iterations, both rejected, exactly as the first real invoices run came out."""
    write(root, summary(0, "cand-01-m0", "rewrite_tool_desc", "p 1.000 >= alpha 0.1"))
    write(
        root,
        summary(
            1, "cand-01-m1", "add_fewshots", "hard_fails 2 > incumbent 1",
            incumbent={
                "candidate_id": "cand-01", "mean_score": 0.17777777777777778,
                "pass3_rate": 0.0, "gen_gap": None, "hard_fails": 1,
            },
            candidate={
                "candidate_id": "cand-01-m1", "mean_score": 0.03333333333333333,
                "pass3_rate": 0.0, "gen_gap": 0.23333333333333334, "hard_fails": 2,
            },
            stop_reason="plateau",
        ),
    )
    return root


GOLDEN = """\
| Domain | Stage | Holdout acc | pass^3 | Gen gap | Hard fails | $/task | p95 s | p (gate) |
|---|---|---|---|---|---|---|---|---|
| invoices | iteration 0 | 0.189 | 0.000 | 0.211 | 3 | 0.000169 | 148.3 | 1.000 |
| invoices | final | 0.178 | 0.000 | 0.222 | 1 | 0.000169 | 148.3 | 1.000 |

**Rejected mutations** — the gate refusing to promote, and which condition failed.

| Domain | Iteration | Operator | Gate condition that failed |
|---|---|---|---|
| invoices | 0 | rewrite_tool_desc | p 1.000 >= alpha 0.1 |
| invoices | 1 | add_fewshots | hard_fails 2 > incumbent 1 |
"""


# --- the table ---------------------------------------------------------------------------


def block(root: Path) -> str:
    return cli.render_block(root, cli._summaries(root))


def test_rendered_block_matches_the_golden_table(tmp_path):
    got = block(real_run(tmp_path / "runs"))
    assert got.startswith(GOLDEN)


def test_a_tiny_cost_is_never_rounded_to_zero(tmp_path):
    """$0.000169/task is real money at scale; three significant figures, not three decimals."""
    cells = [ln for ln in block(real_run(tmp_path / "runs")).splitlines() if "iteration 0" in ln]
    assert cells[0].split("|")[7].strip() == "0.000169"
    assert cli._usd(0.0000000169) == "1.69e-08"
    assert cli._usd(0.0) == "0"
    assert cli._usd(None) == cli.DASH


def test_p95_latency_is_seconds_with_one_decimal(tmp_path):
    row = [ln for ln in block(real_run(tmp_path / "runs")).splitlines() if "iteration 0" in ln]
    assert row[0].split("|")[8].strip() == "148.3"


def test_gen_gap_and_p_are_read_from_gate_json_when_the_summary_has_nulls(tmp_path):
    """The summary can carry nulls the gate itself computed; prefer gate.json over an em-dash."""
    root = tmp_path / "runs"
    body = summary(0, "cand-01-m0", "rewrite_tool_desc", "p 1.000 >= alpha 0.1")
    body["p_value"] = None
    body["incumbent"] = None
    write(root, body, gate={
        "domain": "invoices", "iteration": 0, "p": 0.031, "promoted": False,
        "reason": "p 1.000 >= alpha 0.1", "decision": "reject",
        "incumbent": {
            "candidate_id": "cand-01", "mean_score": 0.18888888888888888,
            "pass3_rate": 0.0, "gen_gap": None, "hard_fails": 3,
        },
        "candidate": body["candidate"],
    })
    row = [ln for ln in block(root).splitlines() if "iteration 0" in ln][0].split("|")
    assert row[3].strip() == "0.189"   # holdout accuracy recovered from gate.json
    assert row[5].strip() == "0.211"   # search mean 0.400 - holdout 0.189, as spec_metrics does
    assert row[9].strip() == "0.031"   # the gate's own p, not an em-dash


def test_a_genuinely_missing_value_renders_as_an_em_dash(tmp_path):
    """No gate ran and no search metrics exist: every derived cell must say so, not guess."""
    root = tmp_path / "runs"
    body = summary(0, None, None, "", decision="no_candidate", candidate_id=None,
                   winner_id="cand-01", p_value=None, incumbent=None, candidate=None,
                   search={}, stop_reason="no_candidate")
    write(root, body)
    row = [ln for ln in block(root).splitlines() if "iteration 0" in ln][0].split("|")
    assert [c.strip() for c in row[3:10]] == [cli.DASH] * 7


def test_the_footnote_names_local_inference_and_the_em_dash(tmp_path):
    text = block(real_run(tmp_path / "runs"))
    assert "Ollama" in text and "qwen2.5" in text
    assert "specs/models.yaml" in text and "price_source" in text
    assert f"`{cli.DASH}`" in text  # what an em-dash means is stated, not assumed


def test_a_run_with_no_rejections_says_so_instead_of_an_empty_table(tmp_path):
    root = tmp_path / "runs"
    write(root, summary(0, "cand-01-m0", "add_fewshots", "promoted",
                        decision="promote", winner_id="cand-01-m0"))
    text = block(root)
    assert "Gate condition that failed" not in text
    assert "no rejected mutations" in text


# --- README generation -------------------------------------------------------------------


README = """\
# Anneal

## Results

<!-- results:start -->
| Domain | Stage |
|---|---|
| invoices | — |
<!-- results:end -->

Numbers are generated from `runs/`.
"""


def test_write_readme_replaces_only_the_marked_block(tmp_path, capsys):
    root = real_run(tmp_path / "runs")
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")
    assert cli.main(["report", str(root), "--write-readme", str(readme)]) == 0
    text = readme.read_text(encoding="utf-8")
    before, _, rest = text.partition(cli.RESULTS_START)
    body, _, after = rest.partition(cli.RESULTS_END)
    assert before == "# Anneal\n\n## Results\n\n"
    assert after == "\n\nNumbers are generated from `runs/`.\n"
    assert body.strip() == block(root).strip()
    assert "| invoices | — |" not in text


def test_write_readme_is_idempotent(tmp_path):
    root = real_run(tmp_path / "runs")
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")
    cli.main(["report", str(root), "--write-readme", str(readme)])
    once = readme.read_text(encoding="utf-8")
    cli.main(["report", str(root), "--write-readme", str(readme)])
    assert readme.read_text(encoding="utf-8") == once


def test_write_readme_without_markers_is_an_error(tmp_path):
    root = real_run(tmp_path / "runs")
    readme = tmp_path / "README.md"
    readme.write_text("# Anneal\n\nno markers here\n", encoding="utf-8")
    assert cli.main(["report", str(root), "--write-readme", str(readme)]) == 1
    assert readme.read_text(encoding="utf-8") == "# Anneal\n\nno markers here\n"


def test_markdown_flag_emits_exactly_what_write_readme_writes(tmp_path, capsys):
    root = real_run(tmp_path / "runs")
    assert cli.main(["report", str(root), "--markdown"]) == 0
    printed = capsys.readouterr().out
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")
    cli.main(["report", str(root), "--write-readme", str(readme)])
    written = readme.read_text(encoding="utf-8").split(cli.RESULTS_START)[1]
    assert written.split(cli.RESULTS_END)[0].strip() == printed.strip()


def test_the_repo_readme_block_is_what_the_tool_emits():
    """The committed README block must be byte-identical to the tool's output for runs/.

    Skipped on a checkout with no runs (they are gitignored); wherever the run data lives,
    this is what proves no cell in the README was typed by hand.
    """
    repo = Path(__file__).resolve().parents[1]
    summaries = cli._summaries(repo / "runs")
    if not summaries:
        pytest.skip("runs/ has no summaries on this checkout")
    text = (repo / "README.md").read_text(encoding="utf-8")
    body = text.split(cli.RESULTS_START)[1].split(cli.RESULTS_END)[0]
    assert body.strip() == cli.render_block(repo / "runs", summaries).strip()
