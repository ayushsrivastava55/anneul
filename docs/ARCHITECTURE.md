# ARCHITECTURE — Anneal

```
 goal.md ─┐                          ┌──────────── rejected / improved ────────────┐
 tools.yaml├─▶ Architect ─▶ Run ─▶ Diagnose ─▶ Mutate ─▶ Gate ─┴─▶ Anneal ─▶ Ship
 eval.py  ─┘      │          │         │           │        │        │          │
                  │      TensorMux  Neatlogs MCP  AO workers  holdout  models.yaml  Dodo meter
                  │      Neatlogs   ledger.json  prompt reg.  pass^3   Pareto
                  └─ every run debits Dodo credits; balance_low halts ───────────────┘
```

## Modules and contracts

### spec.py — harness spec
See `specs/harness.example.yaml`. Key types:

```
HarnessSpec(id, topology, nodes: list[Node], memory: MemoryCfg, step_budget: int)
Node(name, role: planner|executor|critic|router|validator|escalate,
     model_tier: frontier|mid|cheap, system_prompt_ref, tools: list[str], max_steps)
MemoryCfg(enabled: bool, kind: episodic|none, top_k)
```
`system_prompt_ref` is a Neatlogs prompt name + version, cached locally in `prompts/`.

### runtime.py — one executor for every topology
- `single`: one executor node, ReAct loop until final answer or step budget.
- `planner_executor`: planner emits a step list; executor runs each with tools; planner may
  replan once.
- `critic_loop`: executor → critic (pass/fail + reason) → executor retry ≤ 2.
- `tool_router`: router picks a tool group, then executor with only that group.
- Optional `validator` node (schema check) and `escalate` node (returns `{"escalate": reason}`
  which evaluators may score as correct for hard cases).
Tool dispatch: `python:` import path or `mcp:` server+tool. Every node/tool/LLM call is a span.

### runner.py
Input: spec, split. Output: `runs/<domain>/<iter>/<candidate>.jsonl`, one line per task:
`{task_id, score, hard_fail, tokens_in, tokens_out, per_node:{node:{tokens, backend, ms}},
latency_ms, trace_id, output}`. Parallelism via asyncio; concurrency from `--concurrency`.

### diagnose.py — failure taxonomy
For each failed task: fetch trace via MCP (`search_traces` by tag, `get_trace_context`),
then classify with a small LLM call constrained to the class list in
`specs/failure_taxonomy.yaml` (plus deterministic pre-checks: schema error → `output_format`,
step budget hit → `loop_or_timeout`, forbidden tool call → `unsafe_action`).
Ledger entry: `{id, class, node, count, evidence:[trace_ids], status: open|attempted|fixed,
operators_tried:[]}`. Rank open issues by count × severity.

### mutate.py — typed operators
Operator signature: `(spec, issue, evidence) -> spec'`. One operator per iteration so the
gate result is attributable. Mapping lives in `specs/failure_taxonomy.yaml`. Prompt-editing
operators call the LLM with the failing traces as context and write a new prompt version to
Neatlogs (`staging`). Code-level operators (`synthesize_tool`, `switch_topology` when it needs
new glue) go through `ao.py`.

### gate.py — promotion math
- Run candidate and incumbent 3× each on holdout (cache incumbent results per iteration).
- `pass3[task] = all(run_scores ≥ threshold)`; `pass^3 = mean(pass3)`.
- Paired comparison on per-task pass3: wins/losses → exact binomial test (McNemar without
  continuity correction). Promote iff pass^3 ≥ incumbent AND hard_fails ≤ incumbent AND p < 0.1.
  With 12–20 holdout tasks this is lenient by design; report p in the table.
- Generalisation gap = mean(search score) − mean(holdout score).

### anneal.py — cost/latency
Tiers from `specs/models.yaml` (frontier → mid → cheap). For each node, in order of token
share descending: try next cheaper tier; run holdout 3×; keep if score ≥ 0.95 × peak and
hard-fails unchanged. Record each configuration as a Pareto point `(score, $/task, p95_ms)`.
Cost = Σ tokens × price per backend. Backend from `x-tensormux-backend` header if present.

### billing.py — Dodo
- Test mode base URL `https://test.dodopayments.com`, bearer key.
- On start: ensure customer + credit entitlement (unit `tokens`); `--budget` maps to a grant.
- Per run: `POST /events/ingest` with `event_name=anneal_run`, `metadata.tokens`, deterministic
  `event_id = sha1(domain, iter, candidate, task)`. Meter auto-deducts credits.
- Halt: poll `get-customer-balance` each iteration (webhook `credit.balance_low` if time).
- Ship: product for the finished agent with a usage meter; print checkout link.

### ao.py — AO as executor
Daemon `http://127.0.0.1:${AO_PORT:-3001}`. Discover the exact body with
`ao spawn --json --name tool-x --project anneal --agent claude-code --kind worker --branch ao/tool-x --prompt "..."`
then mirror it in `POST /api/v1/sessions`. Poll `GET /api/v1/sessions/{id}`; acceptance =
branch contains `domains/<d>/generated_tools/<tool>.py` + passing `pytest`. Names ≤ 20 chars.

### dashboard.py
FastAPI + HTMX. Reads `runs/` and `ledger.json`. Charts: iteration curve per metric, Pareto
scatter (score vs $/task, point size = p95). Credit balance from billing cache.

## Data flow per iteration (domain-agnostic)
1. Load domain (3 files). 2. Architect → candidates (iteration 0 only). 3. Run on search.
4. Diagnose failures → ledger. 5. Pick top issue → operator → mutated spec. 6. Gate on holdout.
7. Promote/reject; write `runs/<domain>/<iter>/summary.json`. 8. Debit credits; check balance.
9. Repeat until `--iterations` or plateau (no promote in 2 iterations). 10. Anneal. 11. Ship.

## Configuration (rules every module follows)

**One loader.** `anneal/config.py` is the only module that reads `.env`, and `config.env(name,
default)` is the only way to read an environment variable. It treats a blank value as unset, so an
exported-but-empty variable can never shadow a filled-in `.env`. Reading `os.environ` directly
anywhere else reintroduces that bug through a side door; a review should reject it.

**One model ladder per process.** `specs/models.yaml` is the only place model ids and prices
live. `anneal/llm.py` owns which file is in force: `ANNEAL_MODELS_PATH` seeds the default at
import, and the CLI's `--models` flag calls `llm.set_default_models_path()` once before
dispatching, so `architect`, `diagnose`, `mutate` and `runner` always resolve the same ladder.
The setter clears the path-keyed caches; nothing else may mutate `MODELS_PATH`.
`specs/models.local.yaml` is the verified all-local Ollama ladder for machines with no API key.

**Reports compute provider facts, never assert them.** `anneal report` derives its footnote from
`llm.describe_ladder()` and each tier's recorded `price_source`; every `summary.json` records the
`models_path` its run used so a report can say which ladder produced which row.

**Environment variables the core reads** (all via `config.env`):

| Variable | Read by | Meaning |
|---|---|---|
| `<PROVIDER>_BASE_URL`, `<PROVIDER>_API_KEY` | `llm` | per `providers:` in the ladder |
| `ANNEAL_MODELS_PATH` | `llm` | ladder file to use when no `--models` is given |
| `ANNEAL_CLASSIFIER_TIER` | `diagnose` | tier for failure classification (default `frontier`) |
| `ANNEAL_CONNECT_TIMEOUT`, `ANNEAL_READ_TIMEOUT`, `ANNEAL_LLM_RETRIES` | `llm` | client transport limits |
| `ANNEAL_CONCURRENCY`, `ANNEAL_HOLDOUT_RUNS` | `runner`, `gate` | run defaults |
| `NEATLOGS_API_KEY`, `NEATLOGS_WORKFLOW` | `tracing`, `diagnose` | tracing on/off, MCP trace source |
| `DODO_API_KEY`, `DODO_ENV`, `DODO_CUSTOMER_ID` | `billing` | live billing vs local ledger |
