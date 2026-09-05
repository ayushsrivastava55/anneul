# Copyright Sierra (MIT). Hashing helpers lifted from tau_bench.envs.base; the Env class,
# user simulator and litellm plumbing were dropped on purpose.

from hashlib import sha256
from typing import Any

from .data import load_data
from .tool import Tool
from .tools import ALL_TOOLS

TOOLS_BY_NAME: dict[str, type[Tool]] = {
    tool.get_info()["function"]["name"]: tool for tool in ALL_TOOLS
}


def to_hashable(item: Any) -> Any:
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    if isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    if isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    return item


def consistent_hash(value: Any) -> str:
    return sha256(str(value).encode("utf-8")).hexdigest()


def data_hash(data: dict[str, Any]) -> str:
    return consistent_hash(to_hashable(data))


def invoke(data: dict[str, Any], name: str, kwargs: dict[str, Any]) -> str:
    """Run one tool against `data` in place, tau-bench style (errors become strings)."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return f"Unknown action {name}"
    try:
        return tool.invoke(data=data, **kwargs)
    except Exception as exc:  # noqa: BLE001 - mirrors tau_bench Env.step
        return f"Error: {exc}"


__all__ = ["TOOLS_BY_NAME", "consistent_hash", "data_hash", "invoke", "load_data", "to_hashable"]
