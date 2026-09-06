"""Generator for the bugfix dataset. Committed so the split and the bugs can be audited.

Each case is authored once under `sources/<id>/` as the CORRECT module plus its pytest file
and a `bug.json` describing the single defect to inject:

    sources/bugfix_01/chunk_list.py          the known-good module (never shown to an agent)
    sources/bugfix_01/tests/test_bugfix_01.py
    sources/bugfix_01/bug.json               {"module", "bug_type", "old", "new", "note"}

`build()` injects the bug by replacing `old` with `new` exactly once and writes the agent's
view of the task to `cases/<id>/` (buggy module + tests + an empty conftest.py that puts the
case directory on sys.path). It then assigns splits and writes ../tasks.jsonl.

Split: 20 train / 10 search / 10 holdout, seeded with Random(0). Cases are grouped by bug
type, shuffled inside each group and dealt round-robin as train, train, search, holdout, so
every bug type is spread evenly across the three splits instead of a plain shuffle stranding
a whole type in the holdout.

    uv run python -m domains.bugfix.fixtures.make_tasks           # rebuild cases + tasks.jsonl
    uv run python -m domains.bugfix.fixtures.make_tasks --verify  # every pair, buggy vs fixed
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOMAIN = HERE.parent
SOURCES = HERE / "sources"
CASES = HERE / "cases"
TASKS_PATH = DOMAIN / "tasks.jsonl"
SEED = 0
DEAL = ("train", "train", "search", "holdout")
CONFTEST = (
    "# Empty on purpose: its presence puts this directory on sys.path, so the test file\n"
    "# under tests/ can import the module being fixed.\n"
)
INSTRUCTION = (
    "The module `{module}` in this directory has exactly one bug. `{test_file}` is the "
    "specification: it describes the behaviour the module is supposed to have. Read both "
    "files, then write `{module}` back with the single defect fixed. Do not change anything "
    "under tests/."
)


def case_ids() -> list[str]:
    return sorted(p.name for p in SOURCES.iterdir() if p.is_dir())


def read_bug(case_id: str) -> dict[str, str]:
    return json.loads((SOURCES / case_id / "bug.json").read_text())


def buggy_source(case_id: str) -> str:
    """The fixed module with its single bug injected. Raises if `old` is not unique."""
    bug = read_bug(case_id)
    fixed = (SOURCES / case_id / bug["module"]).read_text()
    if fixed.count(bug["old"]) != 1:
        raise ValueError(f"{case_id}: bug.old occurs {fixed.count(bug['old'])} times, want 1")
    return fixed.replace(bug["old"], bug["new"], 1)


def build_case(case_id: str) -> None:
    """Write the agent's view of the task to cases/<id>/."""
    bug = read_bug(case_id)
    out = CASES / case_id
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    (out / bug["module"]).write_text(buggy_source(case_id))
    shutil.copytree(SOURCES / case_id / "tests", out / "tests")
    (out / "conftest.py").write_text(CONFTEST)


def assign_splits(ids: list[str]) -> dict[str, str]:
    """Deterministic, bug-type-stratified 20/10/10 split."""
    by_type: dict[str, list[str]] = {}
    for case_id in ids:
        by_type.setdefault(read_bug(case_id)["bug_type"], []).append(case_id)
    rng = random.Random(SEED)
    ordered: list[str] = []
    for bug_type in sorted(by_type):
        group = sorted(by_type[bug_type])
        rng.shuffle(group)
        ordered.extend(group)
    return {case_id: DEAL[i % len(DEAL)] for i, case_id in enumerate(ordered)}


def write_tasks() -> list[dict]:
    ids = case_ids()
    splits = assign_splits(ids)
    rows = []
    for case_id in ids:
        bug = read_bug(case_id)
        test_file = f"tests/test_{case_id}.py"
        rows.append(
            {
                "id": case_id,
                "input": {
                    "instruction": INSTRUCTION.format(module=bug["module"], test_file=test_file),
                    "module": bug["module"],
                    "test_file": test_file,
                },
                "expected": {"bug_type": bug["bug_type"], "note": bug["note"]},
                "split": splits[case_id],
                "tags": [bug["bug_type"]],
            }
        )
    TASKS_PATH.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def _pytest_passes(work_dir: Path, test_file: str) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test_file],
        cwd=work_dir,
        capture_output=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return proc.returncode == 0


def verify_case(case_id: str) -> None:
    """Assert the pair really is a pair: tests fail on the bug and pass on the fix."""
    bug = read_bug(case_id)
    test_file = f"tests/test_{case_id}.py"
    fixed = (SOURCES / case_id / bug["module"]).read_text()
    body = [line for line in fixed.splitlines() if line.strip()]
    if not 15 <= len(body) <= 40:
        raise AssertionError(f"{case_id}: module is {len(body)} non-blank lines, want 15-40")
    work = Path(tempfile.mkdtemp(prefix=f"verify-{case_id}-"))
    try:
        shutil.copytree(CASES / case_id, work, dirs_exist_ok=True)
        if _pytest_passes(work, test_file):
            raise AssertionError(f"{case_id}: tests PASS on the buggy module")
        (work / bug["module"]).write_text(fixed)
        if not _pytest_passes(work, test_file):
            raise AssertionError(f"{case_id}: tests FAIL on the fixed module")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv: list[str]) -> int:
    ids = case_ids()
    for case_id in ids:
        build_case(case_id)
    rows = write_tasks()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    print(f"built {len(ids)} cases -> {TASKS_PATH.name} {counts}")
    if "--verify" in argv:
        for case_id in ids:
            verify_case(case_id)
            print(f"verified {case_id}")
        print(f"verified {len(ids)} pairs (tests fail on the bug, pass on the fix)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
