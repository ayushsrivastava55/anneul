Task "onboard-interview" (orchestrator-defined, HIGHEST PRIORITY, product-defining, ~4h left).
Problem: Anneal currently demands the user hand-write goal.md + tools.yaml + eval.py. That makes
it a developer tool. Those three files should be our OUTPUT, not the user's homework.
Deliverable: `anneal init` — a conversational onboarding agent that interviews a non-developer
and GENERATES a runnable domain directory, which `anneal run domains/<name>` then optimises.
Acceptance: from a clean checkout, `anneal init` (fed scripted answers) produces
domains/<name>/{goal.md,tools.yaml,eval.py,tasks.jsonl} that `anneal.domain.load_domain` loads
and `anneal run` executes for at least one task. Prove it end to end in your report.

ARCHITECTURE (follow this; it is the design, not a suggestion):
New module `anneal/onboard.py`, owning ONLY the interview and file generation. It must not
duplicate anything: reuse `anneal.llm` for generation, `anneal.mcp` for tool discovery,
`anneal.spec.ToolsManifest` for validation, `anneal.config.env` for any env read.
Model the interview as data, not control flow: a `Question` dataclass (id, prompt, kind:
`choice|text|examples`, choices, help) and an ordered `SCRIPT: list[Question]`, so the same
script can be driven by a terminal, a test, or later a voice frontend. `Interview` holds the
answers dict. A `Transport` protocol with one method `ask(Question) -> str` — ship
`RichTransport` (terminal, single-choice menus by number) and `ScriptedTransport` (tests).

The five questions, in order:
1. name  (text)    — domain directory name, slugified, must not already exist.
2. job   (text)    — "In one or two sentences, what should the agent do?"
3. tools (choice)  — (a) an MCP server I'll name, (b) Python functions in a module, (c) none yet.
   If (a): ask for the launch command or URL, then CONNECT VIA anneal.mcp AND RUN tools/list,
   and show the user the tools it found. Discovery is the point: never ask a user to describe a
   tool a server already describes. If the server fails to start, say so and fall back to (c).
4. success (choice) — how do we know a run was right? (a) specific fields in the answer match,
   (b) a decision/label matches, (c) the final state of the system matches, (d) examples only.
5. examples (examples) — collect N>=3 input/expected pairs, free text in, JSON-ish expected.

GENERATION rules:
- goal.md: LLM-expanded from `job` + the discovered tool names into the same shape our existing
  goals use (what done means, rules, tools). The LLM elaborates; a deterministic template is the
  floor, exactly as anneal/architect.py already does — READ IT and reuse that pattern.
- tools.yaml: for MCP, emit the `servers:` block plus the discovered tools (schemas from
  tools/list). For python, emit `python:` impls. Validate with spec.load_tools before writing.
- eval.py: GENERATED FROM A TEMPLATE, one per `success` kind, parameterised by the fields the
  examples reveal. It must be deterministic. NO LLM-AS-JUDGE — that is a project non-negotiable
  and it is what makes our numbers mean anything. Emit THRESHOLD, load_tasks, score,
  is_hard_fail, and setup if the domain needs state reset.
- tasks.jsonl: the collected examples, split train/search/holdout, seeded, >=1 in each; if the
  user gave too few for a real split, say so plainly in the file and in the console.
- Write a `PROVENANCE.md` in the domain dir recording every question and answer, so a generated
  domain is auditable rather than magic.

CONSTRAINTS: no new dependencies (rich is present). Everything offline-testable with a fake
client and ScriptedTransport. Do NOT edit STATUS.md or pyproject.toml. Read docs/CONTRACTS.md
and docs/ARCHITECTURE.md 'Configuration' first and obey both. Document the module with a
docstring that states the design above, and add an `## Onboarding` section to
docs/ARCHITECTURE.md describing the interview -> generation contract.
Tests: tests/test_onboard.py — scripted interview generates a loadable domain; MCP discovery
path with a fake pool; each eval template scores a known-good 1.0 and known-bad 0.0; refusing
to overwrite an existing domain; too-few-examples warning.
Files you own: anneal/onboard.py, tests/test_onboard.py, the `init` subcommand in anneal/cli.py,
docs/ARCHITECTURE.md (append only).

Commit prefix = your session display name. Merge origin/main before you finish.
