"""Regenerate the Results table in README.md from the run summaries under ``runs/``.

The rows are produced by ``anneal.cli`` (the same code behind ``anneal report``), so the
README can never drift from what the gated runs actually recorded. Only the data rows inside
the ``<!-- results:start -->`` / ``<!-- results:end -->`` block are rewritten: the markers,
the table header, the ``|---|`` separator and any prose below the table are left alone.

    python scripts/update_readme_results.py [runs_dir] [--readme PATH] [--runs-dir PATH]
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

from anneal import cli

REPO_ROOT = Path(__file__).resolve().parent.parent
START = "<!-- results:start -->"
END = "<!-- results:end -->"
HEADER_PREFIX = "| Domain | Stage |"


def report_rows(runs_dir: Path) -> list[str]:
    """Markdown data rows for every domain found under ``runs_dir``."""
    summaries = cli._summaries(runs_dir)
    if not summaries:
        raise SystemExit(f"no summary.json under {runs_dir}")
    if len(inspect.signature(cli._report_rows).parameters) > 1:
        return cli._report_rows(summaries, runs_dir)
    return cli._report_rows(summaries)


def splice(readme: Path, rows: list[str]) -> None:
    """Replace the data rows of the marked results table in ``readme`` with ``rows``."""
    lines = readme.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(START)
        end = lines.index(END, start)
    except ValueError as exc:
        raise SystemExit(f"{readme}: missing {START} / {END} markers") from exc

    block = lines[start + 1 : end]
    head = next((i for i, ln in enumerate(block) if ln.startswith(HEADER_PREFIX)), None)
    if head is None or not block[head + 1 : head + 2] or not block[head + 1].startswith("|-"):
        raise SystemExit(f"{readme}: results block has no '{HEADER_PREFIX} ...' header row")

    first = head + 2
    last = first
    while last < len(block) and block[last].startswith("|"):
        last += 1
    kept = block[:first] + rows + block[last:]
    readme.write_text("\n".join(lines[: start + 1] + kept + lines[end:]) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_dir_pos", nargs="?", default=None, metavar="runs_dir",
                        help="run directory (same as --runs-dir)")
    parser.add_argument("--readme", default=str(REPO_ROOT / "README.md"), type=Path)
    parser.add_argument("--runs-dir", default=None, type=Path)
    args = parser.parse_args(argv)
    args.runs_dir = args.runs_dir or args.runs_dir_pos or REPO_ROOT / "runs"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = report_rows(Path(args.runs_dir))
    splice(Path(args.readme), rows)
    print(f"{args.readme}: wrote {len(rows)} result row(s) from {args.runs_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
