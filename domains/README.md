# Domains

Each domain is a folder with exactly the three files Anneal accepts, plus fixtures.
Nothing in `anneal/` may import from here.

```
domains/<name>/
  goal.md          # plain-language goal, constraints, definition of done
  tools.yaml       # tools: name, description, args schema, impl (python:module.fn | mcp:server/tool)
  eval.py          # load_tasks() -> list[Task]; score(task, output) -> float; is_hard_fail(task, trace) -> bool
  tasks.jsonl      # {id, input, expected, split: train|search|holdout, tags}
  fixtures/        # data the tools read
  generated_tools/ # written by AO workers via synthesize_tool (gitignored until promoted)
```

## Split rule
train 50% / search 25% / holdout 25%. Messy or edge cases spread evenly. Only `gate.py` reads
holdout. Seed the split once and commit `tasks.jsonl`; never regenerate after H8.

## Threshold
`score >= threshold` counts as a pass for pass^3. Exact-match domains use 1.0; partial-credit
domains use 0.8. Declare it in `eval.py` as `THRESHOLD`.

| Domain | Why | Evaluator | Hard fail |
|---|---|---|---|
| invoices | Maximor's world; escalation logic | field exact match + decision | approve on a messy case |
| airline | public tau-bench reference; tool use | DB state equality | forbidden action (refund outside policy) |
| bugfix | AO's habitat; tool synthesis demo | pytest pass | writes outside sandbox |

Fallback for bugfix: `text2sql` on a Spider subset, evaluator = result-set equality.
