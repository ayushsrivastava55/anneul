# PLAN — 30 hours, IST

Rule of the plan: a real before/after number on one domain by H8. Everything after that is
widening, not proving. If a block runs over, cut scope from the block, never from Gate.

## H0–H2 · Sat 9:30 PM → 11:30 PM · Foundations
- [ ] Create GitHub repo (public), copy this kit in, first commit from an AO orchestrator session.
- [ ] `uv init`, deps, `ruff`, `pytest` skeleton, `.env` from `.env.example`.
- [ ] Keys: TensorMux (Discord), Neatlogs project, Dodo test mode (+ ask fast-track in
      `#syndicate-help`), AI Grants India form (`aigrants.in/form`).
- [ ] `anneal/spec.py`: pydantic models matching `specs/harness.example.yaml`. Tests.
- [ ] `anneal/llm.py`: client via TensorMux base URL; `specs/models.yaml` with tiers + prices.
- [ ] `anneal/tracing.py`: neatlogs init, span decorators. Smoke-test a traced call.
- [ ] Decide domain B (airline) evaluator path: tau-bench install (time-box 90 min) or the
      fallback sim in `domains/airline/README.md`.

**Checkpoint H2:** a traced LLM call through TensorMux shows up in Neatlogs.

## H2–H8 · Sat 11:30 PM → Sun 5:30 AM · Core loop on domain B
- [ ] `runtime.py`: executes `single` and `planner_executor` topologies with tool dispatch.
- [ ] `runner.py`: parallel over search split; jsonl output with tokens/backend/latency/trace id.
- [ ] `architect.py`: 3 candidates from goal+tools. Run all, rank on search split.
- [ ] `diagnose.py`: MCP client for `search_traces` / `get_trace_context`; classify two classes
      first (`wrong_tool`, `output_format`); ledger upsert.
- [ ] `mutate.py`: two operators (`rewrite_tool_desc`, `add_fewshots`).
- [ ] `gate.py`: holdout 3×, pass^3, paired test, promote/reject; prompt label flip.
- [ ] `cli.py`: `anneal run domains/airline --iterations 3`.

**Checkpoint H8:** README table row for domain B, iteration 0 vs iteration 3, from `runs/`.

## H8–H14 · Sun 5:30 AM → 11:30 AM · Full taxonomy, operators, AO-driven synthesis
- [ ] Remaining failure classes and operators per `specs/failure_taxonomy.yaml`.
- [ ] `critic_loop` and `tool_router` topologies; `switch_topology` operator.
- [ ] `ao.py`: spawn worker session with prompt + branch; poll; `synthesize_tool` operator
      with unit-test acceptance.
- [ ] Neatlogs prompt registry: `create_prompt` / `save_as_version` on every mutation.
- [ ] Detections: token spike rule as a reliability signal into diagnose.

**Checkpoint H14:** a tool synthesised by an AO worker is used by a promoted candidate.

## H14–H20 · Sun 11:30 AM → 5:30 PM · Domains A and C, Anneal
- [ ] Domain A dataset (60 invoices, 20% messy) generated + 10 hand-checked; `eval.py` with
      hard-fail rule (auto-approve on mismatch).
- [ ] Domain C dataset (40 buggy functions + tests); `eval.py` runs pytest in subprocess.
- [ ] Run the identical loop on A and C. Fix core bugs only; no domain hacks in `anneal/`.
- [ ] `anneal.py`: per-node downshift through tiers; Pareto output; cost attribution from
      backend header + `specs/models.yaml`.

**Checkpoint H20:** all three domains have iteration 0 / final / annealed rows.

## H20–H24 · Sun 5:30 PM → 9:30 PM · Dodo + dashboard
- [ ] `billing.py`: credit entitlement, ledger debit per run with `tokens`, balance poll or
      `credit.balance_low` webhook → halt; product + usage meter for the shipped agent.
- [ ] `dashboard.py`: iteration curves, ledger table, Pareto chart, credit balance.
- [ ] Final `README.md` from template; "How we used AO" section with session list.

**Checkpoint H24:** code freeze. Only bug fixes after this.

## H24–H28 · Sun 9:30 PM → Mon 1:30 AM · Numbers and video
- [ ] Clean-clone reproduction run for final numbers (3 domains, 3 holdout runs each).
- [ ] Record 3-minute video per `docs/DEMO_SCRIPT.md`. Upload (YouTube unlisted).
- [ ] README table + charts committed under `runs/final/`.

## H28–H30 · Mon 1:30 AM → 3:30 AM · Submit
- [ ] Devpost: track, description, repo, video, AO usage explanation, team names.
- [ ] Post pass + demo clip on X/LinkedIn tagging @aoagents.
- [ ] Buffer. Do not touch code.

## Team split
- **Builder 1 (core):** spec, runtime, architect, diagnose, mutate, gate, anneal.
- **Builder 2 (integrations):** llm/tracing, ao.py, billing.py, dashboard, datasets A and C.
- **Builder 3 (if any):** domains end to end, README, video.

## Cut list (in order, if behind)
1. `tool_router` topology. 2. Detections. 3. Dodo webhook (poll balance instead).
4. Domain C → text-to-SQL fallback. 5. Dashboard charts → tables only. Never cut Gate or holdout.
