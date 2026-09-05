"""Load a domain directory (goal.md + tools.yaml + eval.py) into a ``Domain``.

This is the only bridge between the core and a domain: the core never imports from
``domains/``; it receives the three input files through ``load_domain`` at runtime.
``eval`` is the domain's evaluator module. Inside the repo it is imported as a package
module (``domains.<name>.eval``) so its relative fixture imports resolve; a directory
outside the repo is loaded by file path.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from anneal.spec import ToolsManifest, load_tools

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = ("goal.md", "tools.yaml", "eval.py")


@dataclass(frozen=True)
class Domain:
    """The three input files of one domain, parsed and ready for the runtime."""

    name: str
    goal: str
    tools: ToolsManifest
    eval: ModuleType
    path: Path


def ensure_repo_root_on_path() -> None:
    """Make ``domains.<name>...`` and ``python:`` tool impls importable."""
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _import_eval(domain_dir: Path) -> ModuleType:
    """Import eval.py as ``domains.<name>.eval`` when inside the repo, else by file path."""
    try:
        relative = domain_dir.relative_to(ROOT)
    except ValueError:
        relative = None
    if relative is not None:
        return importlib.import_module(".".join([*relative.parts, "eval"]))
    module_name = f"anneal_domain_{domain_dir.name}_eval"
    spec = importlib.util.spec_from_file_location(module_name, domain_dir / "eval.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {domain_dir / 'eval.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_domain(path: str | Path) -> Domain:
    """Parse ``<path>/{goal.md,tools.yaml,eval.py}`` into a ``Domain``."""
    domain_dir = Path(path).resolve()
    for name in REQUIRED_FILES:
        if not (domain_dir / name).is_file():
            raise FileNotFoundError(f"domain {domain_dir} is missing {name}")
    ensure_repo_root_on_path()
    return Domain(
        name=domain_dir.name,
        goal=(domain_dir / "goal.md").read_text(encoding="utf-8"),
        tools=load_tools(domain_dir / "tools.yaml"),
        eval=_import_eval(domain_dir),
        path=domain_dir,
    )
