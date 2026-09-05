"""Tests for anneal.domain.load_domain (offline, never touches holdout)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from anneal.domain import Domain, load_domain

ROOT = Path(__file__).resolve().parents[1]
AIRLINE = ROOT / "domains" / "airline"

EVAL_CONTRACT = ("load_tasks", "score", "is_hard_fail", "THRESHOLD")


def test_load_airline_domain() -> None:
    domain = load_domain(AIRLINE)
    assert isinstance(domain, Domain)
    assert domain.name == "airline"
    assert "airline" in domain.goal.lower()
    assert "get_user_details" in {t.name for t in domain.tools.tools}
    for attr in EVAL_CONTRACT:
        assert hasattr(domain.eval, attr), attr
    assert callable(getattr(domain.eval, "setup", None))


def test_load_domain_accepts_relative_string_path() -> None:
    domain = load_domain("domains/airline")
    assert domain.name == "airline"
    assert domain.path == AIRLINE


def test_eval_is_imported_as_package_module_inside_repo() -> None:
    domain = load_domain(AIRLINE)
    assert domain.eval.__name__ == "domains.airline.eval"


def test_missing_input_file_raises(tmp_path: Path) -> None:
    (tmp_path / "goal.md").write_text("goal")
    (tmp_path / "tools.yaml").write_text("tools: []\n")
    with pytest.raises(FileNotFoundError, match="eval.py"):
        load_domain(tmp_path)


def test_domain_outside_repo_is_loaded_by_file_path(tmp_path: Path) -> None:
    domain_dir = tmp_path / "toy"
    domain_dir.mkdir()
    (domain_dir / "goal.md").write_text("Answer with the number 42.")
    (domain_dir / "tools.yaml").write_text("tools: []\n")
    (domain_dir / "eval.py").write_text(
        textwrap.dedent(
            """
            THRESHOLD = 1.0

            def load_tasks(split=None):
                return []

            def score(task, output):
                return 1.0 if output == 42 else 0.0

            def is_hard_fail(task, trace):
                return False
            """
        )
    )
    domain = load_domain(domain_dir)
    assert domain.name == "toy"
    assert domain.tools.tools == []
    assert domain.eval.score(None, 42) == 1.0
