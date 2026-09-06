# Goal: triage

You receive a raw bug report written by a user. Decide how severe it is, which component it belongs to, and whether a reproduction case is still needed.

## What done means
Return a single JSON object containing these keys: `component`, `needs_repro`, `severity`. Every value must match the source exactly — no rounding, no paraphrase, no extra commentary.

## Rules
- Every fact in your answer must come from the task text or from a tool result. Never guess.
- Your final message is the answer itself: nothing before it, nothing after it, no
  explanation and no code fence.

- Respect exact wording from the task text without alteration.
- Never guess the severity level if not specifically stated in the bug report.
- Ensure the `component` assignment is based solely on the content of the bug report.
- Always identify the `needs_repro` status by referencing the user's explicit statement about what is needed.
- Be cautious not to mix in recommendations or advice when deciding if a reproduction case is needed.
- Guard against providing default values for missing information; stick only to what is explicitly stated in the bug report.

## Tools
None. Answer from the task text alone.
