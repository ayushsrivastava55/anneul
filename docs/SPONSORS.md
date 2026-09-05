# SPONSORS — integration cheat sheet

Verified against public docs on 2026-09-05. Items marked (?) need a check in Discord.

## AO — Agent Orchestrator (mandatory, 25% of score)
- Site https://aoagents.dev · Repo https://github.com/Untrivial-ai/agent-orchestrator ·
  CLI→REST map: `docs/cli/README.md` in the repo · Architecture: https://aoagents.dev/docs/architecture/
- Install: `brew install agentwrapper/tap/agent-orchestrator`. Daemon on `127.0.0.1:3001` (`AO_PORT`).
- Build usage: one orchestrator session per project; spawn workers per task in `docs/TASKS.md`.
  Keep terminated sessions visible; the video must show them.
- Product usage: `POST /api/v1/sessions` (= `ao spawn`), `POST /api/v1/sessions/{id}/send`,
  `GET /api/v1/sessions`, `GET /api/v1/events` (SSE). Scratch sessions need no git.
- (?) Exact JSON body for spawn: run `ao spawn --json ...` once and copy.

## Neatlogs (traces, prompt registry, detections)
- Docs https://docs.neatlogs.com/docs · Python SDK https://docs.neatlogs.com/sdk/python ·
  MCP https://docs.neatlogs.com/guides/mcp-integration · Prompts https://docs.neatlogs.com/sdk/prompt-templates
- `pip install -U neatlogs`; `neatlogs.init(api_key=..., workflow_name="anneal", tags=[...])`;
  `neatlogs.wrap(client)` for the OpenAI client; `@neatlogs.span(kind="AGENT"|"TOOL"|"LLM")`;
  `with neatlogs.trace(name, kind="LLM", system_prompt_template=..., version=...)`.
- MCP: `https://ingest.neatlogs.com/mcp` (HTTP streamable), `Authorization: Bearer <key>`,
  tools `search_traces`, `get_trace_context`, `list_detections`, `get_detection_trend`,
  `triage_*`, `log_trace`. Rate limit 60 req/min → batch per iteration.
- Prompts: `create_prompt(name, prompt, type, labels)`, `save_as_version(prompt_name,
  messages, labels, commit_message)`, `update_prompt(name, version, new_labels)`,
  `get_prompt(name, label=)`. Labels: `staging` on mutation, `production` on promote.
- Detections: numeric rule `total_tokens > 8000` → reliability signal.
- (?) No documented REST for traces (MCP only) and no evals API; scoring stays in `eval.py`.

## TensorMux (inference gateway)
- Site https://tensormux.com · OSS https://github.com/KrxGu/Tensormux · Hosted https://app.tensormux.com
- Free hackathon keys announced on LinkedIn 2 Sep; ask in Discord. Hosted has $5 free credit.
- OSS gateway: `docker run -p 8080:8080 -v ./config.yaml:/app/config.yaml:ro krishom70/tensormux:latest`.
  `backends: [{name, url, engine, model, weight, tags}]`; strategies `weighted_round_robin`,
  `least_inflight`, `ewma_latency`, `token_aware`. `/metrics` (Prometheus), `/tensormux/requests`
  (last 100), JSONL per request, headers `x-request-id`, `x-tensormux-backend`.
- Use: OpenAI SDK with `base_url=$TENSORMUX_BASE_URL`, `api_key=$TENSORMUX_API_KEY`.
- (?) Upstream key passthrough for hosted providers. Plan: open models via TensorMux; frontier
  model direct if needed; attribute both in one cost table.

## AI Grants India (cheap tier / voice)
- https://aigrants.in · form https://aigrants.in/form · approvals under 5 min · free production
  keys, "GPT Nano" key, TTS/STT allowances. Register as a backend behind TensorMux OSS or call
  direct as tier `cheap`. (?) Provider/base URL comes with the key.

## Dodo Payments (budget + monetised output)
- Docs https://docs.dodopayments.com · usage events
  https://docs.dodopayments.com/features/usage-based-billing/event-ingestion · credits
  https://docs.dodopayments.com/features/credit-based-billing · MCP
  https://docs.dodopayments.com/developer-resources/mcp-server · skills `npx skills add dodopayments/skills`
- Base `https://test.dodopayments.com` (test) / `https://live.dodopayments.com`; `Authorization: Bearer`.
- `POST /events/ingest` `{events:[{event_id, customer_id, event_name, timestamp?, metadata{numeric}}]}`
  ≤ 1,000 per call; customer must exist; deterministic `event_id`.
- Credit entitlement (custom unit `tokens`), `create-ledger-entry` (Credit/Debit), `get-customer-balance`,
  `list-customer-ledger`; meters auto-deduct FIFO; webhooks `credit.deducted`, `credit.balance_low`.
- Ask for fast-track verification in `#syndicate-help`; test mode works meanwhile.
- (?) Webhook signing scheme; use polling if short on time.

## Maximor (cash sponsor, internships)
- https://www.maximor.ai · language: Audit-Ready Agents, learns → runs → escalates → improves,
  "nothing posts without the right approval", 100% of entries audit-trailed.
- CFO benchmark: 96% want AI on grunt work, 14% trust it end to end, 97% insist on oversight.
- Use in README for domain A and in the pitch: Anneal's ledger + prompt versions = audit trail
  for the agent's own evolution.

## Env vars (see .env.example)
`TENSORMUX_BASE_URL`, `TENSORMUX_API_KEY`, `FRONTIER_BASE_URL`, `FRONTIER_API_KEY`,
`AIGI_BASE_URL`, `AIGI_API_KEY`, `NEATLOGS_API_KEY`, `DODO_API_KEY`, `DODO_ENV=test`,
`DODO_CUSTOMER_ID`, `AO_PORT=3001`.
