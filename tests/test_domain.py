"""Tests for anneal.domain.load_domain (offline, never touches holdout)."""

from __future__ import annotations

import logging
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


# --- tools.generated.yaml merge -----------------------------------------------------------

TOY_EVAL = textwrap.dedent(
    """
    THRESHOLD = 1.0

    def load_tasks(split=None):
        return []

    def score(task, output):
        return 1.0

    def is_hard_fail(task, trace):
        return False
    """
)

HUMAN_TOOLS = textwrap.dedent(
    """
    tools:
    - name: lookup
      description: human-written lookup
      args: {type: object, properties: {id: {type: string}}, required: [id]}
      impl: python:tests.test_domain.human_lookup
    """
)


def human_lookup(id: str) -> str:  # noqa: A002 - mirrors the manifest arg name
    return f"human:{id}"


def generated_double(n: int) -> str:
    return str(n * 2)


def make_toy_domain(tmp_path: Path, generated: str | None = None) -> Path:
    domain_dir = tmp_path / "toy"
    domain_dir.mkdir()
    (domain_dir / "goal.md").write_text("Toy goal.")
    (domain_dir / "tools.yaml").write_text(HUMAN_TOOLS)
    (domain_dir / "eval.py").write_text(TOY_EVAL)
    if generated is not None:
        (domain_dir / "tools.generated.yaml").write_text(generated)
    return domain_dir


GENERATED_TOOLS = textwrap.dedent(
    """
    # written by synthesize_tool
    tools:
    - name: double
      description: generated doubling tool
      args: {type: object, properties: {n: {type: integer}}, required: [n]}
      impl: python:tests.test_domain.generated_double
      mutates: false
    """
)


def test_generated_tools_are_appended_after_human_tools(tmp_path: Path) -> None:
    domain = load_domain(make_toy_domain(tmp_path, GENERATED_TOOLS))
    assert [t.name for t in domain.tools.tools] == ["lookup", "double"]
    assert domain.tools.tools[1].impl == "python:tests.test_domain.generated_double"
    # tools.yaml itself is untouched
    assert (domain.path / "tools.yaml").read_text() == HUMAN_TOOLS


def test_without_generated_file_manifest_is_unchanged(tmp_path: Path) -> None:
    domain = load_domain(make_toy_domain(tmp_path))
    assert [t.name for t in domain.tools.tools] == ["lookup"]


def test_generated_tool_never_overrides_human_tool(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    clash = textwrap.dedent(
        """
        tools:
        - name: lookup
          description: machine rewrite of lookup
          args: {type: object, properties: {}}
          impl: python:tests.test_domain.generated_double
        - name: double
          description: generated doubling tool
          args: {type: object, properties: {n: {type: integer}}, required: [n]}
          impl: python:tests.test_domain.generated_double
        """
    )
    with caplog.at_level(logging.WARNING, logger="anneal.domain"):
        domain = load_domain(make_toy_domain(tmp_path, clash))
    names = [t.name for t in domain.tools.tools]
    assert names == ["lookup", "double"]
    assert domain.tools.tools[0].description == "human-written lookup"
    assert any("collides" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "broken",
    [
        "tools: [\n  - name: {{{{",  # invalid YAML
        "tools:\n- name: x\n  impl: shell:rm\n",  # fails ToolSpec validation
        "not: a manifest\n",  # wrong shape
    ],
)
def test_malformed_generated_yaml_does_not_crash(
    tmp_path: Path, broken: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="anneal.domain"):
        domain = load_domain(make_toy_domain(tmp_path, broken))
    assert [t.name for t in domain.tools.tools] == ["lookup"]
    assert any("malformed" in r.message for r in caplog.records)
