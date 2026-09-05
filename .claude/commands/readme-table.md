Regenerate the results table in README.md from runs/final/.

1. For each domain dir under runs/final/, read iteration summaries: iteration 0 (best raw candidate), final promoted, and annealed.
2. Build a markdown table with columns: Domain · Stage · Holdout acc · pass^3 · Gen gap · Hard fails · $/task · p95 ms · p (gate).
3. Add a second small table listing every rejected mutation: domain, iteration, operator, why rejected (which condition failed).
4. Replace the content between `<!-- results:start -->` and `<!-- results:end -->` in README.md. Do not touch anything else.
5. Print the table to the terminal.
