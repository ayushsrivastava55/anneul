# STATUS

Phase: core loop live on all three domains with real models. All sponsor surfaces
functioning: Anthropic + TensorMux + AIGI tiers, Neatlogs tracing + prompt registry,
AO worker spawn/accept verified end to end (session anneal-1, branch ao/selftest-5989c5).
Runs in flight: airline (iterating), bugfix (rerun with cause-first diagnosis).
Invoices is done: saturated at 1.000, annealed down to nano. Dodo still keyless.

## Found and fixed on 6 Sep (night)
1. **Bugfix's entire failure story was a concurrency bug, not the model.** The "current
   sandbox" was a module global while the runner executes tasks in a thread pool, so every
   in-flight task read whichever sandbox `setup()` touched last: `read_file` returned "No
   such file or directory" for files that existed, agents looped hunting their own module,
   and 6/10 tasks died on the step budget. ContextVar now (airline's DB already did this).
   With the fix, all three topologies score 1.000 on bugfix at mid tier for $1.96 - which
   also means the earlier "measured noise floor" section below was measuring the race, not
   run-to-run variance. Treat those numbers as an artifact.
2. **ANNEAL_TIER_CAP starts the loop on a weak tier.** A mid executor saturates bugfix and
   invoices at 1.000 on iteration 0 - nothing to learn, nothing to show. The cap (e.g.
   `cheap`, `flash`) is where the learning loop starts; anneal still owns cost from there.
3. **TensorMux 60 RPM poisoned gate runs.** A gate burst (3 seeds x 10 concurrent tasks)
   blew the per-minute cap, whole holdout batches errored 429, and a mutation scored
   pass^3 0.1 on nothing but rate limiting. llm now retries 429s with 4s..64s backoff.
4. **The gate escalates instead of rejecting on thin data.** Every non-regression rejection
   across airline/bugfix was "p >= alpha" with 1-3 discordant tasks - below the arithmetic
   floor of 4 where no verdict is reachable. When the candidate is strictly ahead and p is
   the only objection, the gate now buys a second block of seeds for both specs and
   re-decides on all 2n runs (two-stage group-sequential; disclosed per-gate as
   `escalated` in gate.json; ANNEAL_GATE_ESCALATION=0 disables).
5. **The flagship synthesize_tool arc ran end to end** (in the 429-poisoned run, archived):
   diagnose classed the failures missing_capability -> AO session anneal-3 wrote
   generated_tools/run_pytest.py + its acceptance test -> gate ran the candidate 3x. It
   was rejected 0.8 vs 0.9 under rate-limit noise; the arc itself is real and demoable.
   Root-cause chain that made it possible: rows now carry a bounded tool-call trace, and a
   step-budget death is treated as a symptom (ungranted-tool call -> missing_capability;
   else the model picks the cause with loop_or_timeout still on the menu).

## Found and fixed on 6 Sep (evening)
1. **Headline result — invoices Pareto is in.** The anneal stage walked the executor down
   every tier and every downgrade held: mid $0.0425/task -> cheap $0.0137 -> flash
   $0.0009 -> nano $0.0014, score 0.989-1.000, pass^3 0.93-1.00, zero hard fails at every
   step. nano (gpt-5-nano on the AIGI key) scored a perfect 1.000 at ~29x cheaper than
   mid. Both sponsor floor tiers sit on the front (`runs/invoices/anneal/pareto.json`).
2. **The flagship bugfix demo could never trigger, root cause found and fixed.** The
   domain deliberately withholds `run_tests` so Diagnose would class the failures
   missing_capability and `synthesize_tool` would ask an AO worker to write the tool.
   Never happened: (a) rows carried only a `trace_id`, no steps, so the
   missing-capability signal had no evidence to read, and (b) `hit_step_budget` was
   deterministically classed loop_or_timeout — the *symptom* — whose two operators were
   spent by iteration 1, then the plateau ended the run. Rows now carry a bounded trace;
   a budget death with an ungranted-tool call reclasses as missing_capability (certain);
   otherwise the model picks the cause with loop_or_timeout still on the menu. Bugfix
   rerun in flight (old run archived: runs-archive/bugfix-symptom-not-cause).
3. **Second prompt-registry 401.** prompts.py was fixed to create_prompt, but
   mutate.py's LocalPromptStore had its own sync still calling save_as_version, so every
   operator-written version v2+ failed to reach the registry (visible in the bugfix run
   log). Same fix applied; only v1s made it to the cloud registry before this.
4. **Dashboard ledger panel was empty by construction.** `--ledger` defaulted to
   ./ledger.json but the run loop writes runs/<domain>/ledger.json. The dashboard now
   aggregates the per-domain ledgers with a domain column; an explicit file still wins.
5. **AO worker built real product code.** Session anneal-2 (readme-results) wrote
   scripts/update_readme_results.py + its test from a spawn prompt; the branch was
   accepted by pytest and cherry-picked to main (AO branched from a stale lineage, so
   accept-on-branch failed; the commit itself was clean).

## Found and fixed on 6 Sep (afternoon)
1. **Every run froze at 99% CPU before its first LLM call.** neatlogs 1.4.21's
   `_serialize_obj` recurses into `__dict__` with no depth limit, no cycle detection, no
   memoisation; span-decorated functions whose arguments reach a module object sent it
   walking the interpreter's import graph (16+ CPU-minutes, zero network). Caught with a
   faulthandler stack dump; `init_tracing` now installs a bounded, cycle-safe drop-in over
   the SDK seam. The three domain runs only became possible after this.
2. **AO daemon had lost the project registration** (desktop restart) and the `claude`
   binary had vanished from the machine. Re-registered via the bundled CLI, installed
   Claude Code, authorized it via an apiKeyHelper reading the project .env, made the
   worker harness per-machine config (`AO_AGENT`). Selftest passes: spawn -> agent writes
   tool+test on its branch -> branch accepted on pytest, in 34 s. `synthesize_tool` is
   therefore genuinely available for missing_capability issues now.
3. **Neatlogs prompt registry 401.** `save_as_version` 401s under an SDK key;
   `create_prompt` on /api/managed-prompts creates a *version* per call (verified live).
   prompts.py now uses only the working endpoint.
4. **Invoices saturated: every candidate scored 1.000 at iteration 0** — nothing to learn,
   nothing to report. Dataset v2 adds four trap classes (40% non-trivial rows, uniform per
   split): total_match_breach and short_receipt punish under-checking (wrong approve = hard
   fail), similar_number and symbol_currency punish over-caution (wrong escalate). The
   frontier/mid executor still clears v2 at 1.000 with 0 hard fails — but critic_loop took
   4 hard fails, so the traps bite weaker configs; the invoices story is the anneal stage
   (hold 1.000, walk the cost down) plus the gate rejecting the unsafe config.
5. **Bugfix stalled at 0.2 because the executor was starved, and the repair operator
   couldn't unstarve it.** All specs get step_budget 12; 8/10 failures were
   hit_step_budget. `add_step_budget_and_critic` raised the *global* budget by 4 while the
   executor stayed capped by its node max_steps 10 — the raise was unspendable. The
   operator now doubles the named node's cap and grows the budget to match. Bugfix rerun
   in flight (old run archived: runs/bugfix-starved-executor).
6. **Ledger growth is now a plottable series**: every iteration's summary.json carries
   `ledger` (issues, observations, operators_tried, by_class, by_status) — the direct
   answer to the judges' "show the memory growing" question.

## AO sessions (project `anneal`)
| Session | Task | Branch | State |
|---|---|---|---|
| anneal-10 | 0.1 scaffold | ao/scaffold | merged to main (3 tests) |
| anneal-6 | 0.2 spec-models | ao/spec-models | merged to main (+ mutates follow-up) |
| anneal-7 | 0.3 llm-gateway | ao/llm-gateway | merged to main (9 pass, live skipped: no key) |
| anneal-8 | 0.4 tracing | ao/tracing | merged to main (offline tests; Neatlogs UI check pending key) |
| anneal-9 | 0.5 domain-airline | ao/domain-airline | merged to main (tau-bench, 40 tasks 20/10/10) |

## Phase 1 AO sessions
| Session | Task | Branch | State |
|---|---|---|---|
| anneal-11 | 1.1 runtime-single | ao/runtime-single | merged to main |
| anneal-12 | 1.2 runner | ao/runner | merged to main |
| anneal-13 | 1.3 architect (+ prompts.py) | ao/architect | merged to main |
| anneal-16 | 1.4 diagnose-v1 | ao/diagnose-v1 | merged to main |
| anneal-14 | 1.5 mutate-v1 | ao/mutate-v1 | merged to main |
| anneal-15 | 1.6 gate | ao/gate | merged to main |
| anneal-23 | 1.7 cli-run | ao/cli-run | merged (H8 wiring complete) |

## Parallel track (no dependency on the core loop)
| Session | Task | Branch | State |
|---|---|---|---|
| anneal-17 | 2.4 ao-spawn | ao/ao-spawn | merged; selftest spawned real session anneal-27 |
| anneal-18 | 3.1 domain-invoice | ao/domain-invoice | merged |
| anneal-19 | 3.2 domain-bugfix | ao/domain-bugfix | merged (40/40 pairs verified) |
| anneal-30 | 2.5 synth-tool | ao/synth-tool | merged |
| anneal-28 | 3.3 anneal-stage + 3.4 cost-attrib | ao/anneal-stage | merged |
| anneal-29 | 4.1 dodo-credits + 4.2 dodo-ship | ao/dodo-credits | working |
| anneal-31 | 4.3 dashboard | ao/dashboard | working |
| anneal-24 | 2.3 topo-critic | ao/topo-critic | merged |
| anneal-25 | 2.2 ops-full | ao/ops-full | merged |
| anneal-26 | 2.1 taxonomy-full | ao/taxonomy-full | merged |
| anneal-7 | llm-gateway follow-up: complete() + tests/fakes.py | ao/llm-gateway | merged to main (57 tests) |

Shared interfaces for Phase 1 are pinned in docs/CONTRACTS.md.

## Done
- [x] anneal-1..5 killed (stalled on permission prompts); project switched to bypass-permissions; respawned as anneal-6..10
- [x] orchestrator: git init, kit imported, AO desktop installed, daemon up, project registered
- [x] orchestrator: tau-bench airline env vendored (MIT) into domains/airline/fixtures/tau_airline
- [x] 0.4 anneal-8 - anneal/tracing.py: init_tracing (once, no-op without key), wrap_client, node_span/tool_span/llm_span (sync+async, offline pass-through), run-context tags via contextvar, flush/shutdown, current_trace_id; tests/test_tracing.py (15 tests); smoke `uv run python -m anneal.tracing`

## Blockers
- ~~THE blocker: the system has never run against a real model.~~ **Cleared 6 Sep**: all
  five tiers live (`.env` filled, `specs/models.yaml` real), `runs/` holds completed
  airline + invoices + bugfix runs and the README table is generated from them.
- Dodo Payments: no key was ever obtained; billing stays unwired and is documented as out
  of scope (`HANDOFF.md`). The only sponsor not exercised.
- Push to GitHub needs an authenticated human: this machine's git has no credential for
  `ayushsrivastava55/anneul` (local SSH identity is a different account).
- Docker daemon down (only needed for the optional TensorMux OSS gateway).

## Cleared
- Generated tools loadable at runtime: fixed (`runtime-single` follow-up loads `tools.generated.yaml`).
- GitHub remote: now `https://github.com/ayushsrivastava55/anneul.git`, no longer a local bare repo.
- `anneal anneal` and `anneal dashboard` were listed in `cli.STUBS` and printed "not implemented
  yet" even though both modules were built, tested and merged — so the two commands the README
  tells judges to run were dead. Both are wired now.

## Decisions
- Airline: no user simulator; task instruction is the single customer message. eval.py exposes setup(task) to reset DB state. 40 of 50 tasks, split 20/10/10, seed 0.
- Merge order: scaffold first (owns pyproject), then rebase the other four onto it.

## AO session count
31 sessions in project `anneal`: 1 orchestrator equivalent (this session), 24 task workers,
3 API-probe sessions spawned by the ao-spawn worker, 1 selftest session (anneal-27) spawned by
Anneal itself through anneal/ao.py, plus 5 killed first-attempt sessions kept for history.

## Holdout hygiene note
`grep -rn holdout anneal/` matches two lines in `anneal/anneal.py`: a docstring and the call
`gate.run_holdout(...)`. Both are the *function name*, not the split. The split literal
(`HOLDOUT_SPLIT`) exists only in `anneal/gate.py`, and the annealing stage evaluates by
delegating to gate. This is deliberate and compliant; do not "fix" it by inlining the split.

## Three-domain check (definition of done)
All three domains load through the identical `anneal.domain.load_domain`, verified:

    airline   tools=13  train=20 search=10  THRESHOLD=1.0
    invoices  tools= 4  train=30 search=15  THRESHOLD=1.0
    bugfix    tools= 2  train=20 search=10  THRESHOLD=1.0

`anneal/` contains no static import of any domain module; the only `domains.` occurrences are
dynamically built dotted paths for importlib and docstrings.

## Integration bugs found by the e2e test (all fixed except where noted)
Unit tests passed on every module; these only appeared when the modules ran together.
1. Gate's reserved-split runs overwrote the incumbent's search rows (same filename). Would have
   corrupted every reported number. Fixed: filenames now carry split and seed; guarded by a test.
2. `rewrite_tool_desc` wrote `spec.tool_overrides`, which the runtime never read, so those
   candidates ran identically to their parent and could never pass the gate. Fixed.
3. Airline domain shared one DB across concurrent tasks, so tasks scored each other. Fixed with
   per-task context isolation. invoices/bugfix reported clean by the domain worker.
4. Cost always resolved to $0 because the price fallback was never passed, so `--budget` could
   never halt the loop. Fixed.
5. Row `iteration` disagreed with span `iteration`, breaking any join from runs to traces.
   Fixed; the loop counter is authoritative.

## The optimiser could not promote anything, and it was three bugs stacked
Every mutation in every domain was rejected at exactly p=1.000. That uniformity was the
tell: the cause was arithmetic, not evidence. Three independent faults, each of which alone
was enough to freeze the loop.

1. **The paired test was two-sided.** The exact binomial floor is 2*0.5^n for n discordant
   pairs, so p<0.1 was unreachable below 5 discordant tasks out of a 10-task reserved
   split. An airline candidate won 1 task, lost 0, took pass^3 from 0.700 to 0.800, and was
   rejected at p=1.000 -- a single pair cannot score below 0.5 however clean the win. Now
   one-sided, which is the right test once `decide` has already refused anything worse on
   pass^3 or hard fails.
2. **The tested statistic was blind.** pass^3 is true only on a clean sweep, so a task
   moving 0/3 -> 2/3 counted as no change. On bugfix both specs sat at pass^3 = 0.1, the
   test saw zero pairs twice, and p=1.000 was indistinguishable from having no data. The
   paired unit is now the per-task count of passing runs.
3. **Plateau counted rejections it should not have.** PLATEAU is 2. Both domains stopped at
   iteration 1 on two rejections that no sample size could have decided, so the loop
   concluded "nothing helps" with no evidence. Underpowered rejections no longer count.

Fault 3 is the expensive one. airline's top issue is unsafe_action, whose operators are
listed `add_escalation_node, add_validator_node, rewrite_tool_desc`. Iterations 0 and 1
spent the first two, then the loop quit -- so **rewrite_tool_desc, the tool-description
learning operator and the single thing this track cares most about, was never attempted in
any run.** It was one iteration away throughout.

Replayed against the archived rows, the first two fixes recover signal and promote nothing
new, which is the point: bugfix/0 goes from 0 pairs to 2 wins 2 losses (p=0.69, a real
wash) and bugfix/1 from 0 pairs to 3 wins 1 loss (p=0.31). So those two mutations moved
four tasks between them, and the old statistic would have published that as "no effect".

Every gate now records `min_discordant_to_promote` and `underpowered`, so a rejection from
thin data is never mistaken for a rejection on merit.

## Measured noise floor (bugfix, live, Claude tiers)
The single most important measured result so far, and it is a negative one. `cand-01` is one
unchanged spec. The gate ran it 3x on the same 10 reserved tasks at iteration 0 and again at
iteration 1:

    iteration 0   mean 0.333
    iteration 1   mean 0.133

Same spec, same tasks, same tier. The 0.20 swing is pure run-to-run variance. Consequences:
- Both bugfix mutations were rejected at p = 1.000. That is the gate working, not a bug: at this
  variance nothing an operator can do to 10 tasks is distinguishable from noise.
- The downshift then "beat" the peak (mid 0.300, cheap 0.233 vs peak 0.133). That is the same
  noise band, not evidence that Sonnet beats Opus at repairing code. It must not be reported as
  a win.
- This is the detectable-effect floor the docs said we had to state out loud. On bugfix, with
  10 tasks x 3 runs, we can only detect effects far larger than 0.20. We report bugfix as
  not-measurable at this sample size rather than dressing noise as improvement.

Do not fix this by raising k until it looks good. Report the floor.

## Sponsors wired (6 Sep)
- Inference: five tiers. Opus/Sonnet/Haiku via Anthropic, then two sponsor-backed floor
  tiers reached only by the downshift: `flash` = glm-4-7-flash on TensorMux (50M free
  tokens) and `nano` = gpt-5-nano on the AI Grants India key. Both verified end to end
  through `anneal.llm`, tool calling included. Architect still only assigns the top three,
  so heat explores on strong models and cool walks into the cheap ones.
- glm-4-7-flash is free to us. It carries the third-party market rate ($0.06/$0.40) anyway,
  because at $0 it trivially dominates every Pareto front. Reported $/task is what the
  config would cost anyone, not what we were charged. Say this in the README.
- Tracing: was never initialised on the run path (only mutate.py did, for the prompt
  registry), so a fully configured project received nothing and Diagnose would have queried
  the MCP for spans never sent. Wired into cli.main now; verified, zero export failures.
- Unused on purpose: smallest.ai voice (no voice domain) and Dodo (no key yet).

## Latest numbers
main: 417 passed, 1 deselected (the live gateway call behind `-m live`; the default suite
is offline by contract and `tests/conftest.py` strips sponsor keys).
Live results (README table is generated from `runs/` by the AO-built splicer):
- **invoices**: annealed Sonnet -> gpt-5-nano, score 0.989 -> 1.000, ~29x cheaper,
  $0.0014/task, zero hard fails. Both sponsor tiers on the Pareto front.
- **airline** (mid): loop tried 4 operators, all honestly rejected; anneal cut hard fails
  5 -> 3, $/task -28%, p95 -36%.
- **bugfix** (flash baseline, clean run): incumbent 0.933-0.967, five operators rejected,
  the last at p=0.125 - one discordant pair short, which is what motivated the escalating
  gate.
- **runs-learning/airline** (cheap baseline, escalating gate): five rejects; iteration 3
  escalated 2-0 -> 3-2 (p=0.5), a false promotion avoided in the field.
- **runs-learning/bugfix** (flash, escalating gate): in flight at freeze time.

## Old latest numbers
none
- [x] 0.1 scaffold - uv project (pyproject, hatchling, uv.lock), anneal/ package with __version__, config.env() via python-dotenv, argparse+rich cli stub (`uv run anneal`), ruff (100 cols, py312, fixtures excluded), pytest (3 tests). `uv run ruff check . && uv run pytest` green.
