# PRD — Anneal

**One line.** Anneal turns `goal.md + tools.yaml + eval.py` into a production agent harness,
iterating on prompts, tools, memory and orchestration from trace evidence, and reports accuracy,
reliability, cost and latency improvements across three unrelated domains.

**Track.** Syndicate by Maximor, Track 1: Automated Agent Engineering.
**Window.** Sat 5 Sep 2026 9:30 PM IST → Mon 7 Sep 3:30 AM IST. Submit on Devpost.

## 1. Problem

Teams ship agents by hand-tuning one prompt at a time. When an agent fails they change the
prompt, then the model, then the context window, and it still calls the wrong tool. Existing
"self-improving" tooling in 2026 assumes you already wrote the agent, optimises one layer
(usually prompts or skills), scores on accuracy alone, and rarely holds out data, so it
overfits. Nobody starts from the goal, searches the architecture, ties each failure class to a
specific fix, or treats cost and latency as objectives.

## 2. Users

- **Primary (hackathon):** the judges. They need to see the loop run end to end, the numbers
  move, and AO used throughout.
- **Real:** an engineer with a task, a set of tools (MCP servers or functions) and some way to
  score outputs, who wants a reliable, cheap agent without a week of prompt archaeology.

## 3. Inputs (the only contract)

| File | Contents |
|---|---|
| `goal.md` | Plain-language goal, constraints, and what "done" means. |
| `tools.yaml` | List of tools: name, description, JSON schema for args, how to call (Python import path or MCP server + tool). |
| `eval.py` | `load_tasks() -> list[Task]` and `score(task, output) -> float in [0,1]`; optional `is_hard_fail(task, trace) -> bool`. Tasks carry `split: train|search|holdout`. |

## 4. Outputs

- `runs/<domain>/<iteration>/` with per-task results, cost, latency, trace ids.
- `ledger.json`: failure issues (class, evidence traces, status, operator attempted).
- Winning `harness.yaml` + prompt versions (also in the Neatlogs prompt registry).
- Any synthesised tools under `domains/<name>/generated_tools/` with unit tests.
- Dashboard: iteration curve, ledger, score-vs-$ Pareto, live Dodo credit balance.
- README table: iteration 0 → final → annealed.

## 5. Functional requirements

**F1 Architect.** From `goal.md` + `tools.yaml`, propose 3–4 harness specs from a fixed
topology menu: `single` (ReAct), `planner_executor`, `critic_loop`, `tool_router`. Each spec
sets per-node system prompt, tool subset, memory on/off, model tier.

**F2 Runtime.** One generic executor runs any spec. Step budget per node. Tool calls dispatched
to Python or MCP. Every node/tool/LLM call is a Neatlogs span.

**F3 Runner.** Run candidates on the `search` split in parallel; write jsonl with score,
tokens in/out per node, backend, latency, trace id, hard-fail flag.

**F4 Diagnose.** For failures, fetch span trees (Neatlogs MCP `search_traces`,
`get_trace_context`, `list_detections`) and classify into the taxonomy in
`specs/failure_taxonomy.yaml`. Upsert into `ledger.json` with counts and evidence.

**F5 Mutate.** Pick the top open ledger issue; apply its mapped operator(s) to produce one
mutated spec. Operators: `rewrite_tool_desc`, `add_validator_node`, `add_fewshots`,
`add_cite_or_abstain`, `add_step_budget_and_critic`, `synthesize_tool` (via AO worker),
`add_memory`, `add_escalation_node`, `switch_topology`. Prompt edits saved to Neatlogs prompt
registry as a new version with label `staging`.

**F6 Gate.** Run mutated spec 3× on `holdout`. Compute mean score, pass^3, hard-fail count,
generalisation gap. Promote iff pass^3 ≥ incumbent, hard-fails ≤ incumbent, and a paired
test (McNemar / exact binomial on per-task wins) has p < 0.1 at our split size. On promote,
flip prompt label to `production`; on reject, mark ledger issue `attempted`.

**F7 Anneal.** After N iterations or plateau, for each node try the next cheaper tier from
`specs/models.yaml`; keep if holdout score ≥ 0.95 × peak and hard-fails unchanged. Emit
Pareto front (score, $/task, p95).

**F8 Billing.** Each run debits a Dodo credit ledger (`tokens` metadata). Subscribe to
`credit.balance_low` webhook (or poll balance); halt the loop when budget is exhausted. Each
shipped agent gets a Dodo product + usage meter so per-run invocations are billable.

**F9 AO integration.** `synthesize_tool` and `switch_topology` spawn an AO worker session
(`POST /api/v1/sessions`) with a prompt, branch and acceptance test; poll until the branch has a
passing test; import the tool.

**F10 Dashboard.** Single page: per-domain iteration curve (score, pass^3, $/task, p95), ledger
table, Pareto chart, credit balance.

## 6. Non-functional

- Full loop on one domain in ≤ 15 minutes wall-clock at 40 search tasks.
- Deterministic evaluators; no LLM-as-judge in `eval.py` for any domain.
- Cost cap per run enforced (`--budget`).
- Reproducible: `uv run anneal run` from a clean clone with a `.env` reproduces the README table
  within noise.

## 7. Domains (see `domains/README.md`)

A. AP invoice triage (finance). B. Airline customer ops (tau-bench airline subset).
C. Python bug repair (unit-test evaluator). Fallback for C: text-to-SQL on a Spider subset.

## 8. Success metrics (what the README must show)

Per domain: holdout accuracy, pass^3, generalisation gap, $/task, p95 latency, hard-fails, at
iteration 0, final, and annealed. Target: measurable improvement on ≥ 2 of the 4 track metrics
in every domain, and a visible rejected mutation somewhere.

## 9. Out of scope

Another prompt optimiser (call GEPA if useful, do not rebuild). Skill libraries. A generic
agent-builder UI. LLM-as-judge evaluators. Voice domain (stretch only if H20 is on schedule).

## 10. Judging alignment

AO usage 25% · Technical execution & reliability 25% · Track fit & real-world value 25% ·
Demo & usability 15% · Innovation 10%. F9 + build process cover the first; F6 + hard-fail rule
cover the second; F1–F7 across three domains cover the third; F10 + video cover the fourth;
taxonomy→operators + Anneal cover the fifth.
