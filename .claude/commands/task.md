Work on task $ARGUMENTS from docs/TASKS.md.

1. Read CLAUDE.md and obey its non-negotiables (holdout, gateway, spans, no domain code in anneal/, models only in specs/models.yaml).
2. Find the row for $ARGUMENTS in docs/TASKS.md. Restate Deliverable and Accept in one line each.
3. Read the matching section of docs/ARCHITECTURE.md before writing code.
4. Write the acceptance test first when the Accept cell is testable.
5. Implement. Keep functions under 60 lines. No new dependencies without saying why.
6. Run `uv run ruff check . && uv run pytest`. Fix until green.
7. Commit as `<session>: <imperative summary>` and print the exact command that proves acceptance.
8. Append one line to STATUS.md: `- [x] <id> <session> — <what changed>`.
