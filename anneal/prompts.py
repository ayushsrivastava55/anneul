"""Local prompt registry mirrored to Neatlogs.

Prompts live in ``prompts/<name>/vN.md`` and are referenced from harness specs as
``system_prompt_ref: <name>@vN``. ``save_version`` writes the next local version and, only
when ``NEATLOGS_API_KEY`` is set, mirrors it to the Neatlogs prompt registry with a label
(``staging`` on creation/mutation, ``production`` on promote). Sync failures are logged and
never raised: the local file is the source of truth for the runtime.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("anneal.prompts")

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_ROOT = ROOT / "prompts"

_REF_RE = re.compile(r"^(?P<name>[^@\s]+)@v(?P<version>\d+)$")
_VERSION_RE = re.compile(r"^v(\d+)\.md$")


def parse_ref(ref: str) -> tuple[str, int]:
    """Split ``name@vN`` into ``(name, N)``."""
    match = _REF_RE.match(ref)
    if match is None:
        raise ValueError(f"bad prompt ref {ref!r}; expected '<name>@v<N>'")
    return match.group("name"), int(match.group("version"))


def make_ref(name: str, version: int) -> str:
    """Inverse of :func:`parse_ref`."""
    return f"{name}@v{version}"


def prompt_path(ref: str, *, root: Path | None = None) -> Path:
    """Local file backing ``ref``."""
    name, version = parse_ref(ref)
    return (root or PROMPTS_ROOT) / name / f"v{version}.md"


def get_prompt(ref: str, *, root: Path | None = None) -> str:
    """Read the prompt text for ``name@vN`` from the local registry."""
    path = prompt_path(ref, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"prompt {ref!r} not found at {path}")
    return path.read_text(encoding="utf-8")


def latest_version(name: str, *, root: Path | None = None) -> int:
    """Highest local version of ``name``, or 0 when none exists."""
    directory = (root or PROMPTS_ROOT) / name
    if not directory.is_dir():
        return 0
    versions = [int(m.group(1)) for p in directory.iterdir() if (m := _VERSION_RE.match(p.name))]
    return max(versions, default=0)


def save_version(name: str, text: str, label: str = "staging", *, root: Path | None = None) -> str:
    """Write ``text`` as the next version of ``name`` and return its ref ``name@vN``.

    Mirrors to Neatlogs only when ``NEATLOGS_API_KEY`` is set.
    """
    version = latest_version(name, root=root) + 1
    ref = make_ref(name, version)
    path = prompt_path(ref, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _sync_to_neatlogs(name, version, text, label)
    return ref


def _neatlogs_enabled() -> bool:
    return bool(os.getenv("NEATLOGS_API_KEY", "").strip())


def _sync_to_neatlogs(name: str, version: int, text: str, label: str) -> None:
    """Push ``text`` as the next registry version of ``name``; never raises.

    ``create_prompt`` (POST /api/managed-prompts) creates a *version* -- calling it again
    for an existing name appends v2, v3, ... (verified live, 2026-09-06). The nominally
    correct ``save_as_version`` endpoint (POST /api/prompt-playground/save-as-version)
    401s under an SDK API key even though the same key can create prompts, so it is
    deliberately not used.
    """
    if not _neatlogs_enabled():
        return
    try:
        import neatlogs

        neatlogs.create_prompt(
            name=name,
            prompt=text,
            type="text",
            labels=[label],
            commit_message=f"anneal: {name} v{version}",
        )
    except Exception as exc:  # noqa: BLE001 - registry sync is best-effort
        logger.warning("neatlogs prompt sync failed for %s v%s: %s", name, version, exc)


def set_label(ref: str, label: str) -> None:
    """Move ``label`` (e.g. ``production``) onto ``ref`` in Neatlogs; no-op offline."""
    if not _neatlogs_enabled():
        return
    name, version = parse_ref(ref)
    try:
        import neatlogs

        neatlogs.update_prompt(name=name, version=version, new_labels=[label])
    except Exception as exc:  # noqa: BLE001 - registry sync is best-effort
        logger.warning("neatlogs label update failed for %s: %s", ref, exc)
