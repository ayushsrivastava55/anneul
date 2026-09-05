# STATUS

Phase: 0 (Foundations) — started Sat 5 Sep 2026, workers spawned.

## AO sessions (project `anneal`)
| Session | Task | Branch | State |
|---|---|---|---|
| anneal-1 | 0.1 scaffold | ao/scaffold | blocked: Bash permission prompt (needs_input) |
| anneal-2 | 0.2 spec-models | ao/spec-models | blocked: Bash permission prompt (needs_input) |
| anneal-3 | 0.3 llm-gateway | ao/llm-gateway | blocked: Bash permission prompt (needs_input) |
| anneal-4 | 0.4 tracing | ao/tracing | blocked: Bash permission prompt (needs_input) |
| anneal-5 | 0.5 domain-airline | ao/domain-airline | blocked: Bash permission prompt (needs_input) |

## Done
- [x] orchestrator: git init, kit imported, AO desktop installed, daemon up, project registered
- [x] orchestrator: tau-bench airline env vendored (MIT) into domains/airline/fixtures/tau_airline

## Blockers
- keys: `.env` exists but every value is empty (TensorMux, Neatlogs, Dodo, AIGI). `specs/models.yaml` still REPLACE_ME.
- GitHub remote: `gh repo create` was blocked in the orchestrator session; `origin` is a local bare repo at ~/.ao/data/anneal-origin.git. Swap to GitHub before submission.
- Worker sessions anneal-1..5 are paused on Claude Code Bash permission prompts; AO project permission mode is accept-edits. Orchestrator cannot escalate (classifier blocks bypass-permissions and approval API). Needs the user.
- Docker daemon down (only needed for the optional TensorMux OSS gateway).

## Decisions
- Airline: no user simulator; task instruction is the single customer message. eval.py exposes setup(task) to reset DB state. 40 of 50 tasks, split 20/10/10, seed 0.
- Merge order: scaffold first (owns pyproject), then rebase the other four onto it.

## Latest numbers
none
- [x] 0.3 anneal-7 — anneal/llm.py gateway (get_client/resolve_model/chat/cost via specs/models.yaml + env), tests/test_llm.py (9 offline, 1 live skipped pending TENSORMUX_API_KEY), CLI smoke `uv run python -m anneal.llm --tier mid "hello"`; minimal pyproject.toml pending scaffold merge
