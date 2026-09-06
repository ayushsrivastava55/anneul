# Goal: An expense claim arrives as one line

An expense claim arrives as one line. Say whether it can be paid or needs a manager.

## What done means
Return a single JSON object with the key `label`. Its value is your decision, and it must be one of: `manager`, `pay`.

## Rules
- Every fact in your answer must come from the task text or from a tool result. Never guess.
- Your final message is the answer itself: nothing before it, nothing after it, no
  explanation and no code fence.

## Tools
None. Answer from the task text alone.
