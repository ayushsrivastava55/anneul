# Handoff — read this first

Written 6 Sep ~8:00 PM IST. Deadline: **Devpost closes 3:30 AM IST tonight**
(<https://syndicate-by-maximor.devpost.com/>). Everything below is verifiable in the repo;
`STATUS.md` has the full engineering log, `README.md` the product story and results table.

## What this is

Anneal (Track 1, Automated Agent Engineering): given `goal.md` + `tools.py` + `eval.py`,
it proposes agent architectures, runs them traced, diagnoses failures into a typed
taxonomy, applies typed repair operators, promotes only what survives a statistically
gated holdout, then "anneals" the winner down the model-price ladder. Three domains:
`domains/airline` (tau-bench), `domains/invoices` (AP triage, Maximor's world),
`domains/bugfix` (code repair).

## State right now

- **All 417 offline tests pass** (`uv run pytest -q`, no keys needed — the suite strips
  credentials via `tests/conftest.py`).
- **Completed live runs** in `runs/`: airline (mid tier: loop + anneal, hard fails 5→3,
  $/task −28%, p95 −36%) and invoices (annealed to nano: 0.989→1.000 at ~29× cheaper,
  $0.0014/task). `README.md` results table is generated from these by
  `scripts/update_readme_results.py` (AO-built).
- **In flight (may still be running when you pick this up)**: a cheap-baseline airline
  learning loop in `runs-learning/` (`ANNEAL_TIER_CAP=cheap`, budget $50) — the first run
  with the new *escalating gate* (gate buys a second block of seeds when a candidate is
  ahead but underpowered; see `anneal/gate.py::_should_escalate`). Goal: an honest
  **promotion** to show "output getting better over time". Check
  `runs-learning/airline/*/summary.json` and `gate.json` (`"decision"`, `"escalated"`).
- **Flagship arc already proven once** (archived in `runs-archive/bugfix-flash-429s/`):
  diagnose found `missing_capability` → AO worker session `anneal-3` wrote
  `run_pytest` tool + acceptance test → gate evaluated it. Rejected under 429 noise
  (since fixed), but the arc is real and demoable from the archive + AO kanban.
- **AO sessions so far**: anneal (main build), anneal-2 (README splicer worker),
  anneal-3 (tool-synthesis worker), plus the orchestrator session. The daemon selftest is
  `uv run python -m anneal.ao --selftest`.

## Setup from zero

```bash
git clone https://github.com/ayushsrivastava55/anneul && cd anneul
# uv manages python 3.12 + deps; install uv any way you like, then:
uv sync
cp .env.example .env   # if .env.example is missing, see the variable list below
uv run pytest -q       # must pass with NO keys set — proves your env is sane
```

`.env` variables (get values from the teammate privately — **never commit them**; the
AIGI key is auto-revoked if it ever appears on GitHub):

| var | what |
|---|---|
| `FRONTIER_BASE_URL` / `FRONTIER_API_KEY` | Anthropic (frontier/mid/cheap tiers), OpenAI-compat: `https://api.anthropic.com/v1/` |
| `TENSORMUX_BASE_URL` / `TENSORMUX_API_KEY` | flash tier, glm-4-7-flash, `https://api.tensormux.com/v1` (60 RPM — backoff is built in) |
| `AIGI_BASE_URL` / `AIGI_API_KEY` | nano tier, gpt-5-nano, stock `https://api.openai.com/v1` |
| `NEATLOGS_API_KEY` / `NEATLOGS_WORKFLOW=anneal` | tracing + prompt registry |
| `SMALLEST_API_KEY` | unused (no voice domain) |
| `DODO_API_KEY` | **still empty** — never obtained; billing wiring is the one unbuilt sponsor |
| `AO_PORT=3001` / `AO_PROJECT=anneal` / `AO_AGENT=claude-code` | AO daemon (desktop app must be running; project must be registered — `ao` CLI ships with the app) |
| `ANNEAL_CONCURRENCY=6` / `ANNEAL_HOLDOUT_RUNS=3` | run defaults |

Useful run-time switches: `ANNEAL_TIER_CAP=cheap|flash` (weak baseline so the loop has
room to learn), `ANNEAL_GATE_ESCALATION=0` (disable gate escalation).

## How to run things

```bash
uv run anneal run domains/airline --iterations 5 --budget 50        # full loop
uv run anneal anneal --domain airline                               # downshift stage
uv run anneal dashboard                                             # localhost UI: curves, ledger, Pareto
uv run python scripts/update_readme_results.py                      # refresh README table from runs/
```

Live smoke tests (cost pennies): `uv run pytest -m live -q`.

## Remaining work, in priority order

1. **Check the in-flight run** (`runs-learning/airline/`). If a promotion landed, that's
   the demo centerpiece: show `gate.json` with `"escalated": true, "decision": "promote"`
   and the prompt label flip in `prompts/`. If not, the story is the honest gate +
   detectable-effect floor (judges respect a real gate; STATUS.md explains it).
2. **Refresh `README.md`** results table (`scripts/update_readme_results.py`) and the
   "How we used AO" section (session count is checked by judges — 25% of score).
3. **Record the 3-minute demo video.** Script skeleton is in the original plan: AO kanban
   → `anneal run` on invoices → failing Neatlogs trace → ledger entry → mutation + prompt
   v2 in registry → gate promote → anneal Pareto (invoices 29× is the money shot) →
   final table. Don't leave it to the last hour.
4. **Devpost submission** before 3:30 AM IST: public repo link, video link, Track 1,
   "how we used AO" paragraph (sessions: orchestrator + anneal + anneal-2 splicer +
   anneal-3 tool-synthesis worker).
5. Optional if time remains: Dodo test-mode key → `anneal/billing.py` stub exists;
   post a usage event per run. Skip if under 2 hours remain — it's not load-bearing.

## Things that will bite you if you don't know them

- `runs/` is the headline data; `runs-learning/` is the learning-arc rerun;
  `runs-archive/` holds superseded runs (poisoned or pre-fix) — don't mix them.
- The invoices dataset is **v2** (trap cases); regenerate only via
  `domains/invoices/fixtures/make_tasks.py`, never by hand.
- The bugfix domain saturates at 1.0 on mid tier (it was a sandbox race, now fixed) —
  use `ANNEAL_TIER_CAP=flash` there if you want failures to learn from.
- TensorMux caps at 60 requests/min; the client backs off automatically but a big
  concurrent run still slows down. Keep `ANNEAL_CONCURRENCY=6`.
- Neatlogs prompt sync must go through `create_prompt` (not `save_as_version`, which
  401s on SDK keys) — already handled in `anneal/prompts.py` and `anneal/mutate.py`.
- Commits: hackathon rules want AO used through the build; keep making AO sessions for
  meaningful chunks of work.
