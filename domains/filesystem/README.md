# filesystem — third-party MCP tools, end to end

This domain exists to prove one thing: Anneal can drive an agent whose tools come from a
**real third-party MCP server** we did not write.

Every tool in `tools.yaml` has `impl: mcp:fs/<tool>`, and `servers.fs` launches the official
[`@modelcontextprotocol/server-filesystem`](https://www.npmjs.com/package/@modelcontextprotocol/server-filesystem)
over stdio via `npx`. At run time `anneal/mcp.py` performs the MCP handshake, reads the
server's own `tools/list`, and hands those schemas to the model — the descriptions and
`args` in `tools.yaml` are only the offline fallback. Every call is a Neatlogs `MCP_TOOL`
span.

## Sandbox

`$ANNEAL_FS_SANDBOX` (default `<tempdir>/anneal-filesystem-sandbox`) is the only directory
the server is allowed to touch; the server enforces that itself. Each task works in
`<root>/<task_id>`, which `setup(task)` wipes and re-seeds from the task's `initial` files.
`is_hard_fail` is the tighter rule the server cannot enforce: a path outside the task's own
directory.

## Tasks and scoring

12 tasks, split 6 train / 3 search / 3 holdout. Small create / write / edit / move / mkdir
jobs. Scoring is state-based and deterministic (`THRESHOLD = 1.0`): the final message is
ignored, the end state of the directory is compared to `expected.files` (exact content,
trailing newlines tolerated) and `expected.absent`.

## Running it

```
uv run pytest tests/test_filesystem_eval.py -v   # skips cleanly without npx
uv run anneal run domains/filesystem --iterations 2
```
