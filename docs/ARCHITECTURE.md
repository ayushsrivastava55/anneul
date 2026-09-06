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

## Onboarding (`anneal init` — the interview → generation contract)

Everything above assumes `domains/<name>/{goal.md,tools.yaml,eval.py,tasks.jsonl}` already
exists. `anneal/onboard.py` is how it comes to exist without anyone writing Python: five
questions in, a runnable domain directory out. The three input files are Anneal's *output*,
not the user's homework.

**The interview is data.** A `Question` (id, prompt, kind `choice|text|examples`, choices,
help, an optional `when` gate on an earlier answer, and the rail `step` it belongs to) is a
value; `SCRIPT` is the ordered list of them; `Interview` holds the answers plus whatever tool
discovery found. A `Transport` is one method, `ask(Question) -> str`, which may return the
`BACK` sentinel instead of an answer. `RichTransport` and `ScriptedTransport` (tests) ship
here; a voice or web frontend is a third implementation of that one method, not a rewrite of
the interview.

`RichTransport` is the interview surface `.stitch/DESIGN.md` governs, so it renders to that
system rather than to terminal habit: one question on screen at a time, a five-step progress
rail derived from `SCRIPT` itself (completed dots Ash, the active dot the one orange accent,
future dots Rule), the step name as a mono uppercase Ash label above the control, and choice
questions as stacked rows inside 1px Rule borders — not numbered radio dots. Selection is by
typing a row number or an unambiguous label prefix, which is the one input path that serves a
keyboard and a pipe identically. Back is always available and never destructive: the interview
steps to the previous question that was actually asked, offers the answer given last time, and
forgets whatever that answer had discovered. A transport may also offer `pin(renderable)`, an
optional hook the interview uses to keep the discovered-tools panel on screen through the
later questions; the protocol stays one method.

The five questions: **name** (slugified to a Python-package-safe directory name, refused if it
already exists), **job**, **tools**, **success**, **examples** (≥3 input/expected pairs).
Choosing an MCP server asks for its launch command or URL and then *connects*: `anneal.mcp`
starts it, runs `tools/list`, and the tools it publishes are shown to the user. A user is
never asked to describe a tool the server already describes; a server that will not start is
reported and the interview continues with no tools. Choosing Python functions imports the
module and introspects its public callables the same way.

**Generation.** `goal.md` uses the same floor-plus-elaboration contract as `architect.py`: a
deterministic template built from the job, the success criterion and the discovered tool names
is always written, and a frontier call may only *add* validated rules to it — no key, no
network or an unusable reply leaves the template standing, and `PROVENANCE.md` records which
path was taken. `tools.yaml` carries the `servers:` block plus the server's own schemas (or
`python:` impls) and is validated by `ToolsManifest` before the write and by `spec.load_tools`
after it. `eval.py` is rendered from one of four templates — one per success kind — and is
**deterministic string comparison only; no LLM-as-judge, ever**, which is the whole reason an
Anneal number means something. `tasks.jsonl` is the examples under a seeded 50/25/25 split
with at least one task in every split; too few examples for an honest split is said plainly in
the console, in `PROVENANCE.md` and in the generated evaluator's own docstring.

**What generation cannot know, it says rather than fakes.** `FORBIDDEN_TOOLS` is emitted empty
with a comment (the interview cannot know which action is unforgivable), and the `state`
template's `read_state()` is a labelled stub that believes the agent's own report until someone
replaces it with a real probe. The core still never imports from `domains/`: `onboard.py`
writes files and stops.
