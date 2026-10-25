# `runs_fixed/` — corrected teacher labels

This is the authoritative teacher run for the 10-query pilot. It is identical in
method to [`../runs/`](../runs/) except that the letter-to-candidate mapping bug
is fixed.

## Provenance

| | |
|---|---|
| Model | `qwen/qwen3.6-35b-a3b` (MoE, ~3B active parameters) |
| Row-source variants | `both`, `plan`, `table` |
| Prompts sent | 30 (10 queries x 3 variants) |
| API errors | 0 |
| Responses with a real distribution | 9 / 30 |
| Repetitions per plan | 5, plus 1 untimed warm-up pass, plan order rotated |
| Commit | `af11503` |

## Results

Against the planner's cost estimate as baseline:

| | geomean regret | top-1 | Kendall tau-b |
|---|---|---|---|
| planner cost baseline | 1.106x | 7/10 | +0.708 |
| LLM `both` | 1.139x | 6/10 | +0.280 |
| LLM `plan` | 1.148x | 4/10 | +0.111 |
| LLM `table` | 1.156x | 5/10 | -0.500 |

The teacher reaches parity with planner cost on regret and lags it on top-1.

## Caveats — read before drawing conclusions

1. **Only 5 of 10 queries are resolvable.** The bootstrap CI on the median
   difference cannot separate the best two plans for `14439`, `16425`, `16427`,
   `16432` and `e12a`. Their "correct" answer is close to arbitrary, so
   top-1 accuracy on them is largely a coin flip.
2. **Only 9 of 30 responses carried logprobs.** The rest fell back to a bare
   argmax letter. Kendall tau, NLL and top-2 regret are therefore computed over
   a mixture of genuine distributions and one-hot guesses, and are not yet
   trustworthy. This is the first thing to fix.
3. **The comparison is against the planner's cost, not against measured time
   for the baseline.** The baseline picks the lowest `Total Cost`; the true
   ranking everywhere else comes from real execution time.
4. **n = 10.** This is a pilot, not a result.

## Files

- `<query>.json` — one record per query: plan set, arm mapping, statistics,
  per-plan timings, and one label per row-source variant (prompt, response,
  letter logprobs, decoded distribution).
- `../timings.json` — the same timings keyed by full query path, including the
  bootstrap separability statistics used above.
