# Goal: ticketdesk

A helpdesk ticket arrives as one sentence. Decide which team should handle it and answer with just the team name.

## What done means
Return a single JSON object with the key `label`. Its value is your decision, and it must be one of: `access`, `billing`, `hardware`, `network`.

## Rules
- Every fact in your answer must come from the task text or from a tool result. Never guess.
- Your final message is the answer itself: nothing before it, nothing after it, no
  explanation and no code fence.

- Respect the rules: Ensure that every fact is directly derived from the tool results or the task text, never guessing.
- Avoid circular reasoning: Do not refer to the answer as the reference to make decisions; every decision must have an independent basis.
- Prevent tool overuse: Do not use `list_teams` or `lookup_team` more than necessary, optimizing for speed and accuracy.
- Guard against tool misuse: Ensure that `lookup_team` is used only with clear, singular keywords to avoid ambiguities.
- Avoid incomplete decisions: Whenever considering a team, use enough evidence through previous tools calls to be confident in the decision.
- Prevent silent failures: Always catch and acknowledge potential tool failures, ensuring no decisions are made based on unavailable information.

## Tools
- `list_teams`: List every team you can route a ticket to, and what each one handles.
- `lookup_team`: Find which team handles a single keyword. Returns the team name, or 'unknown'.

See tools.yaml for the exact argument schemas.
