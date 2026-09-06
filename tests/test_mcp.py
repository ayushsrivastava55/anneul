"""Tests for anneal.mcp: JSON-RPC framing, tools/list parsing, pooling and runtime dispatch.

Three layers, cheapest first:

1. a fake in-memory transport for envelope shapes, ``isError`` mapping and pool degradation;
2. a real subprocess speaking newline-delimited JSON-RPC (``_ECHO_SERVER`` below) for the
   stdio framing: notifications interleaved with replies, out-of-order ids, a dying server.
   No third-party package needed, so this always runs;
3. the real third-party server (``npx -y @modelcontextprotocol/server-filesystem``), marked
   ``integration`` and skipped cleanly when npx or the package is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from anneal import mcp
from anneal.domain import load_domain
from anneal.spec import ToolSpec

ROOT = Path(__file__).resolve().parents[1]
FILESYSTEM = ROOT / "domains" / "filesystem"

TOOLS_LIST = {
    "tools": [
        {
            "name": "write_file",
            "description": "Create a new file or overwrite an existing one.",
            "inputSchema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
        {"name": "list_directory", "title": "List Directory"},
        {"name": "", "description": "nameless entries are skipped"},
        "not a dict at all",
    ]
}


# --- fake transport ---------------------------------------------------------------------------


class FakeTransport:
    """Replays canned results by method; records everything the client sent."""

    def __init__(self, results: dict[str, Any] | None = None) -> None:
        self.results = results or {}
        self.sent: list[dict[str, Any]] = []
        self.notifications: list[dict[str, Any]] = []
        self.closed = False

    def request(self, message: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        del timeout_s
        self.sent.append(message)
        outcome = self.results.get(message["method"], {})
        if isinstance(outcome, Exception):
            raise outcome
        if "error" in outcome:
            return {"jsonrpc": "2.0", "id": message["id"], "error": outcome["error"]}
        return {"jsonrpc": "2.0", "id": message["id"], "result": outcome}

    def notify(self, message: dict[str, Any]) -> None:
        self.notifications.append(message)

    def close(self) -> None:
        self.closed = True


def fake_client(results: dict[str, Any]) -> tuple[mcp.Client, FakeTransport]:
    transport = FakeTransport(results)
    cfg = mcp.ServerConfig(name="fake", command="never-run")
    return mcp.Client(cfg, transport=transport), transport


# --- framing / handshake ----------------------------------------------------------------------


def test_handshake_runs_once_and_sends_initialized_notification() -> None:
    client, transport = fake_client({"initialize": {}, "tools/list": TOOLS_LIST})
    client.list_tools()
    client.list_tools()  # cached: no second round trip
    methods = [m["method"] for m in transport.sent]
    assert methods == ["initialize", "tools/list"]
    assert [m["method"] for m in transport.notifications] == ["notifications/initialized"]
    init = transport.sent[0]
    assert init["jsonrpc"] == "2.0"
    assert init["params"]["protocolVersion"] == mcp.PROTOCOL_VERSION
    assert init["params"]["clientInfo"]["name"] == "anneal"


def test_request_ids_are_monotonic_and_unique() -> None:
    client, transport = fake_client({"initialize": {}, "tools/list": TOOLS_LIST, "tools/call": {}})
    client.list_tools()
    client.call_tool("write_file", {"path": "/tmp/x"})
    ids = [m["id"] for m in transport.sent]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_jsonrpc_error_envelope_becomes_mcp_error() -> None:
    client, _ = fake_client({"initialize": {}, "tools/list": {"error": {"code": -32601}}})
    with pytest.raises(mcp.MCPError, match="tools/list"):
        client.list_tools()


# --- tools/list parsing -----------------------------------------------------------------------


def test_parse_tools_list_keeps_schemas_and_skips_junk() -> None:
    tools = mcp.parse_tools_list(TOOLS_LIST)
    assert set(tools) == {"write_file", "list_directory"}
    assert tools["write_file"].input_schema["required"] == ["path", "content"]
    # meta keywords strict tool-calling backends reject never reach the model
    assert "$schema" not in tools["write_file"].input_schema
    assert tools["write_file"].description.startswith("Create a new file")
    # no description -> fall back to the title, never empty
    assert tools["list_directory"].description == "List Directory"
    assert tools["list_directory"].input_schema == {}


def test_parse_tools_list_accepts_snake_case_schema_key() -> None:
    tools = mcp.parse_tools_list({"tools": [{"name": "t", "input_schema": {"type": "object"}}]})
    assert tools["t"].input_schema == {"type": "object"}


# --- tools/call -------------------------------------------------------------------------------


def test_call_tool_flattens_text_content() -> None:
    client, transport = fake_client(
        {
            "initialize": {},
            "tools/list": TOOLS_LIST,
            "tools/call": {"content": [{"type": "text", "text": "wrote it"}]},
        }
    )
    assert client.call_tool("write_file", {"path": "/tmp/x"}) == "wrote it"
    call = transport.sent[-1]
    assert call["method"] == "tools/call"
    assert call["params"] == {"name": "write_file", "arguments": {"path": "/tmp/x"}}


def test_call_tool_prefers_structured_content() -> None:
    client, _ = fake_client(
        {"initialize": {}, "tools/call": {"structuredContent": {"ok": True}}}
    )
    assert client.call_tool("t", {}) == {"ok": True}


def test_is_error_result_becomes_an_error_string() -> None:
    """MCP reports tool-level failures with ``isError``, not a JSON-RPC error."""
    client, _ = fake_client(
        {
            "initialize": {},
            "tools/call": {
                "isError": True,
                "content": [{"type": "text", "text": "Access denied - path outside"}],
            },
        }
    )
    result = client.call_tool("write_file", {"path": "/etc/passwd"})
    assert result.startswith("Error: fake/write_file failed:")
    assert "Access denied" in result


# --- server config ----------------------------------------------------------------------------


def test_parse_servers_reads_stdio_and_http_entries() -> None:
    servers = mcp.parse_servers(
        {
            "fs": {"command": "npx", "args": ["-y", "pkg", "/tmp"], "env": {"A": "1"}},
            "remote": {"url": "https://example.test/mcp", "api_key_env": "SOME_KEY"},
        }
    )
    assert servers["fs"].is_stdio and servers["fs"].args == ("-y", "pkg", "/tmp")
    assert servers["fs"].env == (("A", "1"),)
    assert not servers["remote"].is_stdio
    assert servers["remote"].url == "https://example.test/mcp"


@pytest.mark.parametrize(
    "block",
    [
        {"bad": {"command": "x", "url": "y"}},  # both
        {"bad": {}},  # neither
        {"bad": "not a mapping"},
    ],
)
def test_parse_servers_rejects_malformed_entries(block: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        mcp.parse_servers(block)


def test_split_impl() -> None:
    assert mcp.split_impl("mcp:fs/write_file") == ("fs", "write_file")
    with pytest.raises(ValueError):
        mcp.split_impl("mcp:fs")


def test_expand_rejects_unset_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANNEAL_TEST_DIR", "/tmp/here")
    assert mcp.expand("${ANNEAL_TEST_DIR}/x") == "/tmp/here/x"
    monkeypatch.delenv("ANNEAL_TEST_MISSING", raising=False)
    with pytest.raises(mcp.MCPError, match="unresolved"):
        mcp.expand("${ANNEAL_TEST_MISSING}")


# --- http transport ---------------------------------------------------------------------------


class FakeHttpResponse:
    def __init__(self, body: str, content_type: str, headers: dict[str, str] | None = None) -> None:
        self.text = body
        self.headers = {"content-type": content_type, **(headers or {})}

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        return None


class FakeHttp:
    def __init__(self, responses: list[FakeHttpResponse]) -> None:
        self.responses = responses
        self.posts: list[dict[str, Any]] = []

    def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> FakeHttpResponse:
        self.posts.append({"url": url, "json": json, "headers": dict(headers)})
        return self.responses.pop(0)

    def close(self) -> None:
        return None


def test_http_transport_sends_bearer_and_echoes_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANNEAL_TEST_MCP_KEY", "secret")
    http = FakeHttp(
        [
            FakeHttpResponse(
                '{"jsonrpc":"2.0","id":1,"result":{}}',
                "application/json",
                {"mcp-session-id": "sess-1"},
            ),
            # the notifications/initialized POST: 202, empty body
            FakeHttpResponse("", "application/json"),
            FakeHttpResponse(
                'data: {"jsonrpc":"2.0","id":2,"result":' + json.dumps(TOOLS_LIST) + "}\n",
                "text/event-stream",
            ),
        ]
    )
    cfg = mcp.ServerConfig(
        name="remote", url="https://x.test/mcp", api_key_env="ANNEAL_TEST_MCP_KEY"
    )
    client = mcp.Client(cfg, transport=mcp.HttpTransport(cfg, http=http))
    assert set(client.list_tools()) == {"write_file", "list_directory"}
    assert http.posts[0]["headers"]["Authorization"] == "Bearer secret"
    # the session id issued on the first response is echoed on every later request
    assert http.posts[-1]["headers"]["Mcp-Session-Id"] == "sess-1"


# --- pool -------------------------------------------------------------------------------------


def test_pool_degrades_a_broken_server_to_a_tool_error() -> None:
    pool = mcp.Pool(mcp.parse_servers({"fs": {"command": "definitely-not-a-real-binary-xyz"}}))
    result = pool.call("fs", "write_file", {"path": "/tmp/x"})
    assert result.startswith("Error:") and "unavailable" in result
    # the failure is remembered: no second spawn attempt, and schema lookup degrades to None
    assert pool.tool_info("fs", "write_file") is None
    assert pool.call("fs", "write_file", {}).startswith("Error:")
    pool.close()


def test_pool_rejects_an_undeclared_server() -> None:
    pool = mcp.Pool({})
    assert "unknown mcp server" in pool.call("nope", "t", {})


def test_get_pool_reuses_one_pool_per_server_block() -> None:
    servers = mcp.parse_servers({"fs": {"command": "x", "args": ["a"]}})
    same = mcp.parse_servers({"fs": {"command": "x", "args": ["a"]}})
    other = mcp.parse_servers({"fs": {"command": "x", "args": ["b"]}})
    assert mcp.get_pool(servers) is mcp.get_pool(same)
    assert mcp.get_pool(servers) is not mcp.get_pool(other)
    mcp.close_all()


# --- stdio framing against a real (tiny) subprocess -------------------------------------------

_ECHO_SERVER = "TOOLS_LIST = " + json.dumps(TOOLS_LIST) + textwrap.dedent(
    """
    import json, sys
    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if "id" not in msg:
            continue  # a notification
        # noise the client must skip: a server notification and a stale reply
        send({"jsonrpc": "2.0", "method": "notifications/message", "params": {"m": "hi"}})
        send({"jsonrpc": "2.0", "id": 9999, "result": {"stale": True}})
        if msg["method"] == "initialize":
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {"protocolVersion": "x"}})
        elif msg["method"] == "tools/list":
            send({"jsonrpc": "2.0", "id": msg["id"], "result": TOOLS_LIST})
        elif msg["method"] == "tools/call":
            send({"jsonrpc": "2.0", "id": msg["id"],
                  "result": {"content": [{"type": "text", "text": json.dumps(msg["params"])}]}})
        elif msg["method"] == "die":
            sys.exit(1)
    """
)


@pytest.fixture
def echo_server(tmp_path: Path) -> mcp.ServerConfig:
    script = tmp_path / "echo_server.py"
    script.write_text(_ECHO_SERVER, encoding="utf-8")
    return mcp.ServerConfig(name="echo", command=sys.executable, args=(str(script),))


def test_stdio_skips_notifications_and_unmatched_ids(echo_server: mcp.ServerConfig) -> None:
    client = mcp.Client(echo_server)
    try:
        assert set(client.list_tools()) == {"write_file", "list_directory"}
        # the echo server returns the params as JSON text; mcp_content decodes it for us
        echoed = client.call_tool("write_file", {"path": "/tmp/x", "content": "hi"})
        assert echoed["arguments"] == {"path": "/tmp/x", "content": "hi"}
    finally:
        client.close()


def test_stdio_expands_env_vars_in_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "echo_server.py"
    script.write_text(_ECHO_SERVER, encoding="utf-8")
    monkeypatch.setenv("ANNEAL_TEST_SCRIPT", str(script))
    cfg = mcp.ServerConfig(name="echo", command=sys.executable, args=("${ANNEAL_TEST_SCRIPT}",))
    client = mcp.Client(cfg)
    try:
        assert "write_file" in client.list_tools()
    finally:
        client.close()


def test_stdio_reports_a_server_that_exits(echo_server: mcp.ServerConfig) -> None:
    pool = mcp.Pool({"echo": echo_server})
    try:
        pool.list_tools("echo")
        client = pool.client("echo")
        with pytest.raises(mcp.MCPError):
            client._rpc("die", {}, timeout_s=10.0)
    finally:
        pool.close()


def test_stdio_times_out_instead_of_hanging(tmp_path: Path) -> None:
    script = tmp_path / "silent.py"
    script.write_text("import sys\nfor _ in sys.stdin:\n    pass\n", encoding="utf-8")
    cfg = mcp.ServerConfig(
        name="silent", command=sys.executable, args=(str(script),), startup_timeout_s=1.0
    )
    client = mcp.Client(cfg)
    try:
        with pytest.raises(mcp.MCPError, match="timed out"):
            client.list_tools()
    finally:
        client.close()


# --- the real third-party server ---------------------------------------------------------------

INTEGRATION_PKG = "@modelcontextprotocol/server-filesystem"


def _npx_server_available(tmp_dir: Path) -> bool:
    """True when npx can actually start the filesystem server (it may need a download)."""
    if shutil.which("npx") is None:
        return False
    cfg = mcp.ServerConfig(
        name="probe",
        command="npx",
        args=("-y", INTEGRATION_PKG, str(tmp_dir)),
        startup_timeout_s=180.0,
    )
    client = mcp.Client(cfg)
    try:
        return "write_file" in client.list_tools()
    except (mcp.MCPError, OSError):
        return False
    finally:
        client.close()


@pytest.fixture(scope="module")
def real_fs_server(tmp_path_factory: pytest.TempPathFactory) -> mcp.ServerConfig:
    sandbox = tmp_path_factory.mktemp("real-mcp-fs").resolve()
    if not _npx_server_available(sandbox):
        pytest.skip(f"npx / {INTEGRATION_PKG} unavailable offline")
    return mcp.ServerConfig(
        name="fs",
        command="npx",
        args=("-y", INTEGRATION_PKG, str(sandbox)),
        startup_timeout_s=180.0,
    )


@pytest.mark.integration
def test_real_server_lists_and_calls_tools(real_fs_server: mcp.ServerConfig) -> None:
    """The third party's own tools/list drives us; we hand-write no schema here."""
    sandbox = Path(real_fs_server.args[-1])
    pool = mcp.Pool({"fs": real_fs_server})
    try:
        listing = pool.list_tools("fs")
        assert {"write_file", "read_text_file", "create_directory"} <= set(listing)
        schema = listing["write_file"].input_schema
        assert schema["required"] == ["path", "content"]

        target = sandbox / "hello.txt"
        pool.call("fs", "write_file", {"path": str(target), "content": "from anneal"})
        assert target.read_text() == "from anneal"
        assert "from anneal" in pool.call("fs", "read_text_file", {"path": str(target)})

        # the server enforces its own sandbox; that must surface as a tool error, not a crash
        denied = pool.call("fs", "read_text_file", {"path": "/etc/hosts"})
        assert denied.startswith("Error:")
    finally:
        pool.close()


@pytest.mark.integration
def test_runtime_dispatches_mcp_tools_through_the_pool(
    real_fs_server: mcp.ServerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``invoke_tool`` on an ``mcp:`` impl reaches the real third-party server."""
    from anneal import runtime

    monkeypatch.setattr(mcp, "_pools", {}, raising=False)
    pool = mcp.Pool({"fs": real_fs_server})
    sandbox = Path(real_fs_server.args[-1])
    tool = ToolSpec(
        name="write_file",
        description="fallback text",
        args={"type": "object", "required": ["path", "content"]},
        impl="mcp:fs/write_file",
        mutates=True,
    )
    try:
        info = pool.tool_info("fs", "write_file")
        assert info is not None
        out = runtime.invoke_tool(
            tool,
            {"path": str(sandbox / "dispatched.txt"), "content": "ok"},
            pool=pool,
            schema=info.input_schema,
        )
        assert not out.startswith("Error"), out
        assert (sandbox / "dispatched.txt").read_text() == "ok"
        # the live schema is used for the argument check, so a bad call never leaves the box
        bad = runtime.invoke_tool(tool, {"path": str(sandbox / "x")}, pool=pool)
        assert bad.startswith("Error: invalid arguments")
    finally:
        pool.close()


@pytest.mark.integration
def test_filesystem_domain_loads_with_the_real_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """The demonstration domain's tools.yaml resolves against the live server's listing."""
    domain = load_domain(FILESYSTEM)
    servers = mcp.parse_servers(domain.tools.servers)
    if shutil.which("npx") is None:
        pytest.skip("npx unavailable offline")
    pool = mcp.Pool(servers)
    try:
        Path(domain.eval.sandbox_root()).mkdir(parents=True, exist_ok=True)
        try:
            listing = pool.list_tools("fs")
        except mcp.MCPError as exc:
            pytest.skip(f"{INTEGRATION_PKG} unavailable offline: {exc}")
        declared = {mcp.split_impl(t.impl)[1] for t in domain.tools.tools}
        assert declared <= set(listing), sorted(declared - set(listing))
    finally:
        pool.close()


def test_npx_is_present_or_reported() -> None:
    """Documents on this machine whether the real server could run at all."""
    if shutil.which("npx") is None:
        pytest.skip("npx not installed; the mcp integration tests are inert here")
    assert subprocess.run(["npx", "-v"], capture_output=True, check=False).returncode == 0
