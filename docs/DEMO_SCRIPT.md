# DEMO_SCRIPT — 3 minutes, screen + voice

**Rule for this script: nothing on screen that we cannot actually open.** The previous version
staged the Neatlogs UI, TensorMux metrics and the Dodo dashboard. We never received keys for any
of them, so those shots are gone. Every number spoken is read off a file in `runs/final/` at
record time — if a cell is still `—`, say the run did not produce it rather than inventing one.

Record at 1080p. Pre-open every window. Pre-warm the model (`ollama run qwen2.5:3b-instruct`)
so nothing waits on a cold load on camera.

## Numbers to fill in before recording

Run these, paste the output into the table below, and read the numbers off it. Do not type any
figure from memory.

```
uv run anneal report runs/final          # the results table + rejected mutations
cat runs/final/<domain>/<iter>/gate.json # the promote/reject decision and p
cat runs/final/<domain>-ledger.json      # the failure classes it found
```

| Placeholder | Where it comes from |
|---|---|
| `<N_SESSIONS>` | `ao session ls --include-terminated` count |
| `<ITER0_ACC>` / `<FINAL_ACC>` | holdout accuracy, first and last row of the report |
| `<HARD0>` / `<HARDF>` | hard fails, same rows |
| `<REJECT_REASON>` | the `reason` field of a rejected `gate.json` |

## Shot list

| Time | Screen | Say |
|---|---|---|
| 0:00 | AO board, sessions including terminated | **"Every line of this was built through AO — `<N_SESSIONS>` sessions, one per task, each on its own branch, each merged only after its acceptance check passed."** |
| 0:15 | `ls domains/invoices/` — goal.md, tools.yaml, eval.py | **"Anneal only ever sees three files. A goal, a tool list, and a way to score an answer. It never sees the held-out tasks — only the gate does, and there's a test that fails the build if that ever changes."** |
| 0:30 | `uv run anneal run domains/invoices --iterations 2` (pre-recorded if slow) | **"It proposes architectures, runs them, and reads its own traces."** |
| 0:45 | `runs/final/<domain>/0/summary.json` and the ledger | **"Iteration zero: `<ITER0_ACC>` accuracy, `<HARD0>` hard failures. It classified those failures by type — not 'score went down', but *wrong tool*, *unsupported claim*, *output format*."** |
| 1:00 | `specs/failure_taxonomy.yaml` side by side with the ledger | **"Each class maps to a specific repair. That mapping is the product."** |
| 1:15 | `prompts/anneal/<domain>/executor/` v1 vs v2 diff | **"The repair is versioned. Then it has to survive the gate."** |
| 1:30 | `gate.json` — pass^3, hard fails, p, decision | **"Three runs on data it has never seen, and a paired test. `<REJECT_REASON>` — so this one was thrown away. It rejects its own work. That's the point."** |
| 1:50 | The rejected-mutations table in the README | **"Both rejections are published next to the results. A self-improving agent that never rejects anything is just overfitting."** |
| 2:05 | `domains/filesystem/tools.yaml` `servers:` block, then the run calling it | **"This domain's tools aren't ours. They come from the official filesystem MCP server, discovered from the server itself. Third-party tools, real protocol."** |
| 2:20 | `uv run python -m anneal.ao --selftest`, then the AO session it spawned | **"And when the fix needs new code, Anneal spawns its own AO worker to write the tool, and only accepts the branch if the tests pass."** |
| 2:35 | `uv run anneal dashboard` — curves, ledger, Pareto | **"Iteration curves, the issue ledger, and the cost-versus-score front, all read from the run files."** |
| 2:50 | README table + repo URL | **"Four domains, one engine, zero domain-specific code. `<ITER0_ACC>` to `<FINAL_ACC>`, hard failures `<HARD0>` to `<HARDF>`. Anneal."** |

## Honesty notes to say out loud

- Inference is **local** (Ollama, qwen2.5 3b/1.5b/0.5b). Say so. Absolute scores are modest;
  the claim is improvement under a gate, not raw capability.
- If nothing was promoted across the runs, **say that**, and show the rejections instead:
  "it refused to ship two regressions" is a reliability result, and reliability is one of the
  four judged metrics.
- Do not show the Neatlogs, TensorMux or Dodo dashboards. Those integrations are built and
  tested but were never run against a live account. The README says so; the video should match.
