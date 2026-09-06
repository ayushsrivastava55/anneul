"""Pydantic models for the harness spec (specs/harness.example.yaml) and tools.yaml.

The spec is the only thing the runtime needs to execute a candidate. Nothing in
here is domain-specific: it describes topology, nodes, model tiers and lineage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Topology = Literal["single", "planner_executor", "critic_loop", "tool_router"]
Role = Literal["planner", "executor", "critic", "router", "validator", "escalate"]
ModelTier = Literal["frontier", "mid", "cheap"]
MemoryKind = Literal["none", "episodic"]

IMPL_PREFIXES = ("python:", "mcp:")


class MemoryCfg(BaseModel):
    """Episodic memory of prior escalations / outcomes."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    kind: MemoryKind = "none"
    top_k: int = Field(default=3, ge=0)


class Node(BaseModel):
    """One LLM node in the harness graph."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    role: Role
    model_tier: ModelTier
    system_prompt_ref: str = Field(min_length=1)
    tools: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=1, ge=1)
    schema_ref: str | None = None


class Lineage(BaseModel):
    """Where this candidate came from: parent + operator + ledger issue."""

    model_config = ConfigDict(extra="forbid")

    parent: str
    iteration: int = Field(ge=0)
    operator: str
    ledger_issue: str | None = None


class HarnessSpec(BaseModel):
    """A complete, executable agent architecture."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    topology: Topology
    step_budget: int = Field(ge=1)
    memory: MemoryCfg = Field(default_factory=MemoryCfg)
    nodes: list[Node] = Field(min_length=1)
    lineage: Lineage | None = None
    # tool name -> description override written by mutate.rewrite_tool_desc. The runtime
    # prefers these over tools.yaml descriptions when building the tool schema for a node.
    tool_overrides: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_nodes(self) -> HarnessSpec:
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            raise ValueError(f"node names must be unique, got {names}")
        roles = [n.role for n in self.nodes]
        if self.topology == "single" and roles.count("executor") != 1:
            raise ValueError("topology 'single' requires exactly one executor node")
        if self.topology == "planner_executor":
            for required in ("planner", "executor"):
                if required not in roles:
                    raise ValueError(
                        f"topology 'planner_executor' requires a {required} node"
                    )
        return self

    def to_yaml(self) -> str:
        """Serialise to YAML in the same key order as the example spec."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


class ToolSpec(BaseModel):
    """One tool from tools.yaml. `args` is a JSON schema for the call arguments."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    impl: str
    mutates: bool = False  # True for write actions (post, book, cancel)

    @field_validator("impl")
    @classmethod
    def _check_impl(cls, value: str) -> str:
        if not value.startswith(IMPL_PREFIXES):
            raise ValueError(
                f"impl must start with one of {IMPL_PREFIXES!r}, got {value!r}"
            )
        return value


class ToolsManifest(BaseModel):
    """The whole tools.yaml file: the tool list plus any MCP servers the tools live on."""

    model_config = ConfigDict(extra="forbid")

    tools: list[ToolSpec] = Field(default_factory=list)
    # MCP server definitions for `impl: mcp:<server>/<tool>` entries. Opaque here; parsed by
    # anneal.mcp.parse_servers into stdio ({command,args,env,cwd}) or http ({url,api_key_env}).
    servers: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_unique(self) -> ToolsManifest:
        names = [t.name for t in self.tools]
        if len(names) != len(set(names)):
            raise ValueError(f"tool names must be unique, got {names}")
        return self


def load_spec(path: str | Path) -> HarnessSpec:
    """Load and validate a harness spec from a YAML file."""
    data = yaml.safe_load(Path(path).read_text())
    return HarnessSpec.model_validate(data)


def dump_spec(spec: HarnessSpec, path: str | Path) -> None:
    """Write a harness spec to a YAML file."""
    Path(path).write_text(spec.to_yaml())


def load_tools(path: str | Path) -> ToolsManifest:
    """Load and validate a tools.yaml manifest."""
    data = yaml.safe_load(Path(path).read_text())
    return ToolsManifest.model_validate(data)
