# START_HERE — first 20 minutes

This kit is documentation and configuration only. Product code starts at 9:30 PM IST on
Saturday, inside AO, so the session history proves the build happened in the window.

1. **Before the window (allowed):** read `docs/PRD.md`, `docs/PLAN.md`, `docs/ARCHITECTURE.md`.
   Install `uv`, `ao`, Docker. Create accounts: Neatlogs, Dodo (test mode), TensorMux hosted.
   Submit the AI Grants India form. Join Discord; post your pass on X/LinkedIn tagging @aoagents.
   Confirm eligibility (Devpost says students only, team required).
2. **At 9:30 PM IST:** create the public GitHub repo, copy this kit in, open AO, start the
   orchestrator session with the prompt in `docs/AO_PROMPTS.md`. First commit from that session.
3. Ask in Discord `#syndicate-help` for the TensorMux key and Dodo fast-track. Fill `.env`.
4. Orchestrator spawns `scaffold`, `spec-models`, `llm-gateway`, `tracing`, `domain-airline`
   (Phase 0 in `docs/TASKS.md`). In each worker, run `/task <id>`.
5. Every two hours: `/status`. Before any number goes in the README: `/gate-check`.
6. H8 checkpoint is the one that matters: a real before/after row on one domain.

Files:
- `CLAUDE.md` — rules every Claude Code / AO agent must follow
- `docs/PRD.md` · `docs/PLAN.md` · `docs/ARCHITECTURE.md` · `docs/TASKS.md`
- `docs/SPONSORS.md` — API surfaces and env vars per partner
- `docs/AO_PROMPTS.md` — paste-ready orchestrator and worker prompts
- `docs/DEMO_SCRIPT.md` · `docs/SUBMISSION_CHECKLIST.md`
- `specs/` — harness spec example, failure taxonomy → operators, model tiers
- `domains/` — three domain briefs and evaluator contracts
- `.claude/commands/` — `/task`, `/status`, `/gate-check`, `/readme-table`
- `.claude/settings.json` — safe permission defaults for the agents
