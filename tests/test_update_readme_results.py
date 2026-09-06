"""The README results splicer rewrites only the data rows of the marked table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from anneal import cli
from scripts import update_readme_results as script

ROWS = [
    "| invoices | iteration 0 | 0.600 | 0.200 | 0.050 | 3 | 0.004 | 900 | — |",
    "| invoices | final | 0.850 | 0.700 | 0.010 | 0 | 0.005 | 1100 | 0.021 |",
]

README = """# Anneal

## Results

<!-- results:start -->
| Domain | Stage | Gated acc | pass^3 | Gen gap | Hard fails | $/task | p95 ms | p (gate) |
|---|---|---|---|---|---|---|---|---|
| stale | iteration 0 | — | — | — | — | — | — | — |
| stale | final | — | — | — | — | — | — | — |

Rejected mutations (the gate working): _none yet_
<!-- results:end -->

Numbers are generated, never typed by hand.
"""


@pytest.fixture()
def canned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_summaries", lambda runs_dir: [{"domain": "invoices"}])
    monkeypatch.setattr(cli, "_report_rows", lambda summaries: list(ROWS))


def _write(tmp_path: Path, body: str) -> Path:
    readme = tmp_path / "README.md"
    readme.write_text(body, encoding="utf-8")
    return readme


def test_splice_replaces_only_the_data_rows(tmp_path: Path, canned: Any) -> None:
    readme = _write(tmp_path, README)

    assert script.main(["--readme", str(readme), "--runs-dir", str(tmp_path / "runs")]) == 0

    lines = readme.read_text(encoding="utf-8").splitlines()
    start, end = lines.index(script.START), lines.index(script.END)
    assert lines[start + 1].startswith("| Domain | Stage |")
    assert lines[start + 2] == "|---|---|---|---|---|---|---|---|---|"
    assert lines[start + 3 : start + 5] == ROWS
    assert not any(line.startswith("| stale ") for line in lines)
    assert lines[start + 5 : end] == ["", "Rejected mutations (the gate working): _none yet_"]
    assert lines[:start] == README.splitlines()[:start]
    assert lines[end:] == [script.END, "", "Numbers are generated, never typed by hand."]


def test_missing_markers_exit_nonzero(tmp_path: Path, canned: Any) -> None:
    readme = _write(tmp_path, "# Anneal\n\nNo markers here.\n")
    with pytest.raises(SystemExit):
        script.main(["--readme", str(readme)])


def test_no_summaries_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_summaries", lambda runs_dir: [])
    with pytest.raises(SystemExit):
        script.main(["--readme", str(_write(tmp_path, README)), "--runs-dir", str(tmp_path)])
