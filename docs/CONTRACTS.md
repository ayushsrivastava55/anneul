# CONTRACTS — Phase 1 shared interfaces (orchestrator-owned)

Workers build against these in parallel. Change only via the orchestrator.

Shared contracts (pin these verbatim; other workers build against them in parallel):
- anneal/domain.py (owned by runtime-single): load_domain(path) -> Domain(name, goal: str, tools: ToolsManifest, eval). eval is the imported domains/<name>/eval.py module exposing load_tasks(split), score(task, output), is_hard_fail(task, trace), THRESHOLD, and optional setup(task).
- anneal/runtime.py (owned by runtime-single): run_task(spec, task, domain, *, seed=0) -> TaskResult(output, trace: list[{tool, args, result}], per_node: {node: {tokens_in, tokens_out, backend, ms}}, steps, hit_step_budget, schema_error, trace_id, latency_ms). Runtime calls domain.eval.setup(task) first if present, imports python: impls with importlib.import_module on the dotted module path (never by file path), adds the repo root to sys.path, calls tracing.set_run_context before the first span. Runtime never scores; runner does.
- anneal/runner.py (owned by runner): run(spec, domain, split, *, iteration=0, seed=0, concurrency=None) -> list[dict]; writes runs/<domain>/<iter>/<candidate_id>.jsonl, one line per task: {task_id, candidate_id, iteration, score, hard_fail, hit_step_budget, schema_error, tokens_in, tokens_out, per_node, latency_ms, trace_id, output}. split is a parameter; the literal string "holdout" must not appear anywhere in anneal/ except anneal/gate.py.
- anneal/prompts.py (owned by architect): get_prompt(ref) -> str reading prompts/<name>/<version>.md from the repo (ref format name@version as in system_prompt_ref); save_version(name, text, label) writes the next local version and syncs to Neatlogs only when NEATLOGS_API_KEY is set.
- anneal/llm.py already has get_client(tier), resolve_model(tier), chat(tier, messages, **kw) -> (text, Usage), cost(usage). A tool-calling variant complete(tier, messages, tools=None, **kw) -> (assistant_message_dict, Usage) and an offline fake client in tests/fakes.py (scripted turns returning text or tool_calls plus usage) are being added by the llm-gateway worker; if they are not on main yet, write a minimal local fake inside your own test file and do not block.
- Keys are empty and specs/models.yaml is REPLACE_ME: every test must run offline with fakes; live paths skip cleanly.

Rules:
- Read CLAUDE.md first; obey its non-negotiables (holdout only in gate.py; every LLM call via anneal/llm; every node/tool/LLM call is a tracing span via anneal/tracing decorators; no domain code in anneal/; no hardcoded model names).
- Touch only the files you own plus your test file. Do NOT edit STATUS.md (the orchestrator owns it; this overrides step 8 of /task). Do NOT edit pyproject.toml or uv.lock; no new dependencies (openai, pydantic, pyyaml, scipy, neatlogs, httpx, rich are available). If one is unavoidable, name it in your final message.
- Before declaring done: git fetch && git merge origin/main, then uv run ruff check . && uv run pytest must be green. Commit as "<session display name>: <imperative summary>" (display name, not the session id). End with the exact command that proves acceptance.
