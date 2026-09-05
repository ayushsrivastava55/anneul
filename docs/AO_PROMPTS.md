# AO_PROMPTS — paste-ready prompts for AO sessions

Use the orchestrator prompt once per project. Use the worker template for every task in
`docs/TASKS.md`. Keep the "Context" block short; the agent reads CLAUDE.md and docs/ itself.

## Orchestrator (kind: orchestrator, agent: claude-code)

```
You are the orchestrator for Anneal (read CLAUDE.md, docs/PRD.md, docs/PLAN.md, docs/TASKS.md).
Your job: split docs/TASKS.md into worker sessions in phase order, one task per worker, each on
branch ao/<session-name> (name ≤ 20 chars). For each worker: give it the task row, the accept
check, and the relevant section of docs/ARCHITECTURE.md. When a worker reports done, verify
the accept check yourself (run the tests), then merge to main. Never merge a branch whose
tests fail. Never let a worker read holdout tasks outside anneal/gate.py. Keep a running
STATUS.md at repo root: phase, done tasks, current blockers, current README numbers. If a
phase is behind PLAN.md by more than 60 minutes, apply the cut list in PLAN.md and say so in
STATUS.md. Prefer fewer, cleaner numbers over more features.
```

## Worker template (kind: worker, agent: claude-code)

```
Task <id> "<session>" from docs/TASKS.md. Read CLAUDE.md first; obey its non-negotiables.
Deliverable: <copy the Deliverable cell>.
Acceptance: <copy the Accept cell>. Write the test before the code where possible.
Constraints: no domain-specific logic in anneal/; every LLM call via anneal/llm.get_client();
every node/tool/LLM call is a neatlogs span; no hardcoded model names (use specs/models.yaml).
Context: <paste the relevant ARCHITECTURE.md section or file paths>.
When done: run `uv run ruff check . && uv run pytest`, commit with message
"<session>: <what changed>", and report the exact command that proves acceptance.
```

## Specific worker prompts worth having ready

**1.6 gate**
```
Implement anneal/gate.py. Inputs: incumbent spec, candidate spec, domain dir. Run each 3x on
the holdout split via runner.run(spec, split="holdout", seed=i). Compute per task pass3 =
all runs score >= domain threshold (default 1.0 for exact-match domains, 0.8 otherwise).
Metrics: mean_score, pass3_rate, hard_fails, gen_gap = mean(search score) - mean(holdout score).
Paired test: wins = tasks where candidate pass3 and incumbent not; losses = reverse; p = exact
binomial two-sided on (wins, losses). Promote iff pass3_rate >= incumbent and hard_fails <=
incumbent and p < 0.1. On promote: neatlogs update_prompt label -> production for candidate
prompts. Write runs/<domain>/<iter>/gate.json with all of the above. Unit-test the math with
synthetic results (no LLM calls in tests).
```

**2.4 ao-spawn**
```
Implement anneal/ao.py. First run `ao spawn --json --name probe --project anneal --agent
claude-code --kind worker --branch ao/probe --prompt "echo hi"` and paste the request/response
shapes into a docstring. Then implement spawn_worker(name, branch, prompt) -> session_id via
POST http://127.0.0.1:${AO_PORT:-3001}/api/v1/sessions, send(session_id, message), and
wait_for_branch(branch, accept_cmd, timeout) which polls GET /api/v1/sessions/{id} and runs
accept_cmd in a checkout of the branch. Names <= 20 chars. Log every call as a neatlogs span
kind=TOOL name=ao.*.
```

**2.5 synth-tool**
```
Implement the synthesize_tool operator in anneal/mutate.py. Input: ledger issue of class
missing_capability with evidence traces. Ask the LLM (via llm.get_client) for a tool spec:
name, description, JSON args schema, and 3 example calls with expected outputs. Then call
ao.spawn_worker with a prompt that asks the worker to implement
domains/<d>/generated_tools/<name>.py plus tests/test_<name>.py, on branch ao/tool-<name>.
Accept when pytest passes on that branch. On success, append the tool to the candidate spec's
tools list and to tools.generated.yaml; do not modify the human-written tools.yaml.
```

**3.1 domain-invoice**
```
Build domains/invoices: goal.md (AP invoice triage), tools.yaml (lookup_po, lookup_receipt,
post_entry, escalate; python: implementations over a JSON fixture), eval.py, tasks.jsonl with
60 invoices, splits 30/15/15 train/search/holdout, 20% messy cases evenly across splits:
duplicate invoice number, currency mismatch vs PO, missing PO, quantity or price outside
tolerance (2%). Score = 0.5 * field extraction exact match + 0.5 * correct decision
(approve|escalate). is_hard_fail = decision approve when the case is messy. Hand-check 10 and
record them in domains/invoices/CHECKED.md.
```

## Commit message convention
`<session>: <imperative summary>` e.g. `gate: add exact binomial paired test`.
