"""Minimal MCP client so domains can use tools from real third-party MCP servers.

Two transports, one client:

- :class:`StdioTransport` spawns a command (``npx -y @modelcontextprotocol/server-filesystem
  <dir>``, ``uvx ...``) and speaks newline-delimited JSON-RPC over its stdin/stdout.
- :class:`HttpTransport` posts JSON-RPC to a streamable-HTTP endpoint with a bearer token,
  echoing ``Mcp-Session-Id`` once the server issues one and decoding either a JSON or an SSE
  (``data:`` lines) body. This is the same handshake ``anneal.diagnose`` uses against the
  Neatlogs MCP endpoint; the two body decoders (:func:`decode_response`, :func:`mcp_content`)
  live here and diagnose imports them. ``NeatlogsMCP`` itself stays in diagnose because it
  carries a rate limiter and a stateless fallback that no domain server needs.

:class:`Client` performs the ``initialize`` / ``notifications/initialized`` handshake lazily,
caches ``tools/list`` for its lifetime (so a run discovers a third-party server's schemas
exactly once) and exposes ``call_tool``. :class:`Pool` maps the ``servers:`` block of a
domain's tools.yaml to live clients, starting each at most once: a server that fails to start
is remembered as broken and every later call returns an ``Error: ...`` string, so a missing
binary degrades to a tool error the model can read instead of crashing the run.

Every ``tools/call`` is a Neatlogs span of kind ``MCP_TOOL`` (``tracing.mcp_span``).

Nothing here is domain-specific: server definitions arrive as data from tools.yaml.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import shlex
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from anneal import tracing

logger = logging.getLogger("anneal.mcp")

IMPL_PREFIX = "mcp:"
PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "anneal", "version": "0.1"}
DEFAULT_TIMEOUT_S = 60.0
# How long to wait for the very first response (npx may still be downloading the package).
DEFAULT_STARTUP_TIMEOUT_S = 180.0
# Bytes of a dead server's stderr quoted back in the error string.
STDERR_TAIL = 800


class MCPError(RuntimeError):
    """A server could not be started, or a JSON-RPC call failed at the protocol level."""


# --- response decoding (shared with anneal.diagnose) ---------------------------------------


def decode_response(response: Any) -> dict[str, Any]:
    """Parse a JSON or SSE (``data:`` lines) MCP body into the JSON-RPC envelope."""
    content_type = str(response.headers.get("content-type", ""))
    if "text/event-stream" not in content_type:
        return response.json()
    last: dict[str, Any] = {}
    for line in response.text.splitlines():
        if line.startswith("data:"):
            last = json.loads(line[5:].strip() or "{}")
    return last


def mcp_content(result: dict[str, Any]) -> Any:
    """Flatten MCP ``content`` blocks; decode the text as JSON when it is JSON."""
    if "structuredContent" in result:
        return result["structuredContent"]
    texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except ValueError:
        return joined


def split_impl(impl: str) -> tuple[str, str]:
    """``mcp:<server>/<tool>`` -> ``(server, tool)``."""
    rest = impl.removeprefix(IMPL_PREFIX)
    server, sep, tool = rest.partition("/")
    if not sep or not server or not tool:
        raise ValueError(f"malformed mcp impl {impl!r}; expected mcp:<server>/<tool>")
    return server, tool


# --- server configuration ------------------------------------------------------------------


def expand(value: str) -> str:
    """Expand ``${VAR}`` / ``$VAR`` in a config string; an unset variable is an error."""
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise MCPError(f"unresolved environment variable in {value!r}")
    return expanded


@dataclass(frozen=True)
class ServerConfig:
    """One entry of the ``servers:`` block in a domain's tools.yaml."""

    name: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    cwd: str | None = None
    url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S

    @property
    def is_stdio(self) -> bool:
        return self.command is not None


def parse_servers(block: dict[str, Any] | None) -> dict[str, ServerConfig]:
    """``{name: {command, args, env, cwd} | {url, api_key_env}}`` -> ``{name: ServerConfig}``."""
    servers: dict[str, ServerConfig] = {}
    for name, raw in (block or {}).items():
        if not isinstance(raw, dict):
            raise ValueError(f"server {name!r}: expected a mapping, got {type(raw).__name__}")
        command, url = raw.get("command"), raw.get("url")
        if bool(command) == bool(url):
            raise ValueError(f"server {name!r}: set exactly one of 'command' or 'url'")
        args = raw.get("args") or []
        if isinstance(args, str):
            args = shlex.split(args)
        servers[str(name)] = ServerConfig(
            name=str(name),
            command=str(command) if command else None,
            args=tuple(str(a) for a in args),
            env=tuple((str(k), str(v)) for k, v in (raw.get("env") or {}).items()),
            cwd=str(raw["cwd"]) if raw.get("cwd") else None,
            url=str(url) if url else None,
            api_key_env=str(raw["api_key_env"]) if raw.get("api_key_env") else None,
            timeout_s=float(raw.get("timeout_s", DEFAULT_TIMEOUT_S)),
            startup_timeout_s=float(raw.get("startup_timeout_s", DEFAULT_STARTUP_TIMEOUT_S)),
        )
    return servers


# --- transports ------------------------------------------------------------------------------


class Transport(Protocol):
    """Moves one JSON-RPC message to a server and brings the matching reply back."""

    def request(self, message: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        """Send a request and return its JSON-RPC envelope."""
        ...

    def notify(self, message: dict[str, Any]) -> None:
        """Send a notification (no ``id``, no reply)."""
        ...

    def close(self) -> None:
        """Release the process / connection."""
        ...


class StdioTransport:
    """Newline-delimited JSON-RPC over a child process's stdin/stdout.

    A reader thread drains stdout into a queue so a dead or silent server times out instead
    of blocking forever, and stderr goes to a temp file (an unread ``PIPE`` fills up and
    deadlocks servers such as npx that chatter on startup).
    """

    def __init__(self, cfg: ServerConfig) -> None:
        env = {**os.environ, **{k: expand(v) for k, v in cfg.env}}
        argv = [expand(cfg.command or ""), *(expand(a) for a in cfg.args)]
        self._stderr = tempfile.TemporaryFile()
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - argv comes from the domain's tools.yaml
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                env=env,
                cwd=cfg.cwd,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._stderr.close()
            raise MCPError(f"cannot start {argv[0]!r}: {exc}") from exc
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        stdout = self._proc.stdout
        assert stdout is not None
        for line in stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._queue.put(json.loads(line))
            except ValueError:
                logger.debug("ignoring non-JSON line from mcp server: %s", line[:200])
        self._queue.put(None)  # EOF sentinel: the server exited

    def stderr_tail(self) -> str:
        """Last :data:`STDERR_TAIL` bytes the server wrote to stderr."""
        try:
            self._stderr.seek(0)
            return self._stderr.read().decode("utf-8", "replace")[-STDERR_TAIL:].strip()
        except (OSError, ValueError):  # pragma: no cover - closed file
            return ""

    def _write(self, message: dict[str, Any]) -> None:
        stdin = self._proc.stdin
        if stdin is None or self._proc.poll() is not None:
            raise MCPError(f"mcp server exited (rc={self._proc.poll()}): {self.stderr_tail()}")
        try:
            stdin.write(json.dumps(message) + "\n")
            stdin.flush()
        except OSError as exc:
            raise MCPError(f"mcp server closed its stdin: {exc}") from exc

    def request(self, message: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        """Write the request, then read until the reply with a matching ``id`` shows up.

        Server-initiated notifications and replies to abandoned ids are skipped.
        """
        self._write(message)
        wanted = message.get("id")
        while True:
            try:
                payload = self._queue.get(timeout=timeout_s)
            except queue.Empty:
                raise MCPError(f"mcp server timed out after {timeout_s}s") from None
            if payload is None:
                raise MCPError(f"mcp server exited: {self.stderr_tail()}")
            if payload.get("id") == wanted:
                return payload
            logger.debug("skipping unmatched mcp message id=%r", payload.get("id"))

    def notify(self, message: dict[str, Any]) -> None:
        self._write(message)

    def close(self) -> None:
        proc = self._proc
        if proc.poll() is None:
            for stream in (proc.stdin,):
                if stream is not None:
                    with _suppress_os_error():
                        stream.close()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
                proc.kill()
        with _suppress_os_error():
            self._stderr.close()


class _suppress_os_error:  # noqa: N801 - tiny context manager, used like contextlib.suppress
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: type | None, *_: Any) -> bool:
        return exc_type is not None and issubclass(exc_type, (OSError, ValueError))


class HttpTransport:
    """Streamable-HTTP JSON-RPC with a bearer token; echoes ``Mcp-Session-Id``."""

    def __init__(self, cfg: ServerConfig, http: Any | None = None) -> None:
        self._url = expand(cfg.url or "")
        self._headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        key = (os.environ.get(cfg.api_key_env or "") or "").strip()
        if key:
            self._headers["Authorization"] = f"Bearer {key}"
        elif cfg.api_key_env:
            logger.warning("%s is empty; calling %s unauthenticated", cfg.api_key_env, self._url)
        self._http = http or httpx.Client(timeout=cfg.timeout_s)

    def request(self, message: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        del timeout_s  # the httpx client carries the timeout
        try:
            response = self._http.post(self._url, json=message, headers=self._headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MCPError(f"mcp http request failed: {exc}") from exc
        session = response.headers.get("mcp-session-id")
        if session:
            self._headers["Mcp-Session-Id"] = str(session)
        return decode_response(response)

    def notify(self, message: dict[str, Any]) -> None:
        """Fire-and-forget: servers answer a notification with 202 and an empty body."""
        try:
            response = self._http.post(self._url, json=message, headers=self._headers)
            session = response.headers.get("mcp-session-id")
        except httpx.HTTPError as exc:
            logger.debug("mcp notification %s failed: %s", message.get("method"), exc)
            return
        if session:
            self._headers["Mcp-Session-Id"] = str(session)

    def close(self) -> None:
        with _suppress_os_error():
            self._http.close()


def build_transport(cfg: ServerConfig) -> Transport:
    """The transport ``cfg`` describes: stdio when it has a command, else streamable HTTP."""
    return StdioTransport(cfg) if cfg.is_stdio else HttpTransport(cfg)


# --- client ----------------------------------------------------------------------------------


@dataclass
class ToolInfo:
    """One entry of a server's ``tools/list``."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


def parse_tools_list(result: dict[str, Any]) -> dict[str, ToolInfo]:
    """``tools/list`` result -> ``{tool name: ToolInfo}``, tolerating missing optional keys."""
    tools: dict[str, ToolInfo] = {}
    for entry in result.get("tools") or []:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        name = str(entry["name"])
        schema = entry.get("inputSchema") or entry.get("input_schema") or {}
        tools[name] = ToolInfo(
            name=name,
            description=str(entry.get("description") or entry.get("title") or name),
            input_schema=schema if isinstance(schema, dict) else {},
        )
    return tools


class Client:
    """One MCP server: lazy handshake, cached ``tools/list``, traced ``tools/call``."""

    def __init__(self, cfg: ServerConfig, transport: Transport | None = None) -> None:
        self.cfg = cfg
        self._transport = transport
        self._lock = threading.Lock()
        self._next_id = 0
        self._ready = False
        self._tools: dict[str, ToolInfo] | None = None

    @property
    def transport(self) -> Transport:
        if self._transport is None:
            self._transport = build_transport(self.cfg)
        return self._transport

    def _rpc(self, method: str, params: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        self._next_id += 1
        body = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params}
        payload = self.transport.request(body, timeout_s)
        if "error" in payload:
            raise MCPError(f"{self.cfg.name} {method}: {payload['error']}")
        result = payload.get("result")
        return result if isinstance(result, dict) else {}

    def _handshake(self) -> None:
        """``initialize`` + the ``notifications/initialized`` the spec requires afterwards."""
        if self._ready:
            return
        self._rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": CLIENT_INFO,
            },
            self.cfg.startup_timeout_s,
        )
        self.transport.notify({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._ready = True

    def list_tools(self) -> dict[str, ToolInfo]:
        """The server's own tool schemas, fetched once and cached for this client's life."""
        with self._lock:
            if self._tools is None:
                self._handshake()
                self._tools = parse_tools_list(
                    self._rpc("tools/list", {}, self.cfg.startup_timeout_s)
                )
                logger.info(
                    "mcp server %s exposes %d tools", self.cfg.name, len(self._tools)
                )
            return self._tools

    def call_tool(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Invoke ``tool``; an ``isError`` result comes back as an ``Error: ...`` string."""
        with self._lock:
            self._handshake()
            result = self._rpc(
                "tools/call", {"name": tool, "arguments": arguments}, self.cfg.timeout_s
            )
        content = mcp_content(result)
        if result.get("isError"):
            return f"Error: {self.cfg.name}/{tool} failed: {_as_text(content)}"
        return content

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        self._ready = False


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


# --- pool ------------------------------------------------------------------------------------


class Pool:
    """Lazily started clients for a run's ``servers:`` block, keyed by server name.

    A server that fails to start (missing binary, bad handshake) is remembered as broken:
    later calls return the same ``Error: ...`` string without respawning anything, so one
    unavailable third-party server degrades that tool rather than the whole run.
    """

    def __init__(self, servers: dict[str, ServerConfig]) -> None:
        self._servers = servers
        self._clients: dict[str, Client] = {}
        self._broken: dict[str, str] = {}
        self._warned: set[str] = set()
        self._lock = threading.Lock()

    def client(self, name: str) -> Client:
        """The client for ``name``, created on first use. Raises when the name is unknown."""
        with self._lock:
            if name not in self._clients:
                cfg = self._servers.get(name)
                if cfg is None:
                    raise MCPError(f"unknown mcp server {name!r}; declare it under servers:")
                self._clients[name] = Client(cfg)
            return self._clients[name]

    def tool_info(self, server: str, tool: str) -> ToolInfo | None:
        """Live schema for one tool, or None when the server is unavailable or lacks it."""
        try:
            return self.list_tools(server).get(tool)
        except MCPError as exc:
            if server not in self._warned:  # one line per broken server, not one per call
                self._warned.add(server)
                logger.warning("mcp listing unavailable for %s: %s", server, exc)
            return None

    def list_tools(self, server: str) -> dict[str, ToolInfo]:
        """Cached ``tools/list`` for ``server``; raises :class:`MCPError` when it is broken."""
        if server in self._broken:
            raise MCPError(self._broken[server])
        try:
            return self.client(server).list_tools()
        except (MCPError, OSError) as exc:
            self._broken[server] = f"mcp server {server!r} unavailable: {exc}"
            raise MCPError(self._broken[server]) from exc

    def call(self, server: str, tool: str, arguments: dict[str, Any]) -> str:
        """Call ``server``/``tool`` inside an ``MCP_TOOL`` span; failures come back as text."""
        try:
            self.list_tools(server)  # ensures the handshake ran and the server is alive
            traced = tracing.mcp_span(f"mcp.{server}.{tool}")(self.client(server).call_tool)
            result = traced(tool, arguments)
        except (MCPError, OSError) as exc:
            return f"Error: {exc}"
        return _as_text(result)

    def close(self) -> None:
        with self._lock:
            for client in self._clients.values():
                client.close()
            self._clients.clear()


_pools: dict[int, Pool] = {}
_pools_lock = threading.Lock()


def get_pool(servers: dict[str, ServerConfig]) -> Pool:
    """A process-wide pool per distinct ``servers:`` block, so one run spawns each server once.

    ``anneal.runner`` fans tasks out over threads inside a single process, so keying on the
    server definitions (rather than per task) is what "pooled per run" means here.
    """
    key = hash(tuple(sorted(servers.items())))
    with _pools_lock:
        if key not in _pools:
            _pools[key] = Pool(servers)
        return _pools[key]


def close_all() -> None:
    """Shut every pooled server down. Registered with ``atexit``; safe to call repeatedly."""
    with _pools_lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        pool.close()


atexit.register(close_all)
