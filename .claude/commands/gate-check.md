Audit holdout hygiene before we report any number.

1. Grep anneal/ for any read of tasks with `split == "holdout"` or `"holdout"` literals. Only anneal/gate.py may contain them. Report violations with file:line.
2. Check that few-shot examples in prompts/ and generated prompt versions come only from tasks tagged train. Report any task id that appears in a prompt and is tagged search or holdout.
3. Confirm every summary.json under runs/ has pass3_rate, hard_fails, gen_gap and p_value fields.
4. Confirm the README table cells match the values in runs/final/*/summary.json exactly.
5. Print PASS or a list of fixes. Do not edit files; report only.
