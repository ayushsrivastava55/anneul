"""Load a domain directory (goal.md + tools.yaml + eval.py) into a ``Domain``.

This is the only bridge between the core and a domain: the core never imports from
``domains/``; it receives the three input files through ``load_domain`` at runtime.
``tools`` is the human-written tools.yaml plus, when present, ``tools.generated.yaml``
(written by ``mutate.synthesize_tool``); generated entries are appended after the
human-written ones and never override a human-written tool of the same name (such an entry
is skipped with a warning, as is a generated file that fails to parse).
``eval`` is the domain's evaluator module. Inside the repo it is imported as a package
module (``domains.<name>.eval``) so its relative fixture imports resolve; a directory
outside the repo is loaded by file path.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from anneal import mcp
from anneal.spec import ToolsManifest, ToolSpec, load_tools

logger = logging.getLogger("anneal.domain")

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = ("goal.md", "tools.yaml", "eval.py")
GENERATED_TOOLS_FILE = "tools.generated.yaml"  # written by mutate.synthesize_tool, never by hand


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


def _discovered_tools(manifest: ToolsManifest) -> list[ToolSpec]:
    """Every tool published by each server named in ``discover:``.

    Declaring a server is enough: this asks it what it can do rather than making the domain
    restate a list the server already publishes. Names are how the runtime routes a call, so
    they must be known before the architect assigns tools to nodes -- which is why this runs at
    load time rather than lazily. The descriptions and schemas fetched here are the same ones
    the runtime later uses, so a discovered tool needs no offline fallback text.

    A server that will not start is logged and skipped, never fatal: the domain still loads
    with whatever it listed explicitly, exactly as an unreachable server degrades to a readable
    tool error during a run.
    """
    if not manifest.discover:
        return []
    pool = mcp.get_pool(mcp.parse_servers(manifest.servers))
    found: list[ToolSpec] = []
    for server in manifest.discover:
        if server not in manifest.servers:
            raise ValueError(f"discover names server {server!r}, which servers: does not define")
        try:
            listing = pool.list_tools(server)
        except mcp.MCPError as exc:
            logger.warning("discovery skipped for server %r: %s", server, exc)
            continue
        for name, info in listing.items():
            found.append(
                ToolSpec(
                    name=name,
                    description=info.description,
                    args=info.input_schema or {"type": "object", "properties": {}},
                    impl=f"{mcp.IMPL_PREFIX}{server}/{name}",
                )
            )
        logger.info("discovered %d tools from server %r", len(listing), server)
    return found


def load_tools_manifest(domain_dir: Path) -> ToolsManifest:
    """tools.yaml merged with tools.generated.yaml (if any). tools.yaml is never modified."""
    manifest = load_tools(domain_dir / "tools.yaml")
    discovered = _discovered_tools(manifest)
    if discovered:
        listed = {t.name for t in manifest.tools}
        manifest = ToolsManifest(
            tools=[*manifest.tools, *(t for t in discovered if t.name not in listed)],
            servers=manifest.servers,
        )
    generated_path = domain_dir / GENERATED_TOOLS_FILE
    if not generated_path.is_file():
        return manifest
    try:
        generated = load_tools(generated_path)
    except Exception as exc:  # noqa: BLE001 - a bad generated file must not break the run
        logger.warning("skipping malformed %s: %s", generated_path, exc)
        return manifest
    human_names = {t.name for t in manifest.tools}
    merged = list(manifest.tools)
    for tool in generated.tools:
        if tool.name in human_names:
            logger.warning(
                "generated tool %r in %s collides with tools.yaml; keeping the human-written one",
                tool.name,
                generated_path,
            )
            continue
        merged.append(tool)
    # servers must survive the merge: dropping it silently unroutes every mcp: impl the moment
    # a domain grows a tools.generated.yaml.
    return ToolsManifest(tools=merged, servers=manifest.servers)


def load_domain(path: str | Path) -> Domain:
    """Parse ``<path>/{goal.md,tools.yaml,eval.py}`` (+ tools.generated.yaml) into a ``Domain``."""
    domain_dir = Path(path).resolve()
    for name in REQUIRED_FILES:
        if not (domain_dir / name).is_file():
            raise FileNotFoundError(f"domain {domain_dir} is missing {name}")
    ensure_repo_root_on_path()
    return Domain(
        name=domain_dir.name,
        goal=(domain_dir / "goal.md").read_text(encoding="utf-8"),
        tools=load_tools_manifest(domain_dir),
        eval=_import_eval(domain_dir),
        path=domain_dir,
    )
