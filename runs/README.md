# `runs/` — SUPERSEDED, LABELS ARE MIS-ATTRIBUTED

**Do not use the teacher labels in this directory.** They are retained only as a
record of the first end-to-end run.

## What is wrong with them

The prompt rendered the letter for candidate `c` as `LETTERS[order[c]]`, while
the decoder read letter `L` back as `order[L]`. Because `order` is a
permutation, those two are transposes of each other, so **every stored label
points at the wrong candidate**.

Concretely, for query `08268` the prompt displayed:

```
B (arms 0, 2):  SeqScan...HashJoin
C (arms 1):     BitmapScan...NestLoop
A (arms 3, 4):  IndexScan...NestLoop
```

but the decoder mapped `A -> candidate 2`, `B -> candidate 1`, `C -> candidate 0`.
The model answered `"C"`, meaning the plan it actually saw, and it was recorded
as the hash-join plan.

The effect was not subtle. The corrected run gives `08268` a Kendall tau of
**+1.00** (identical order, correct top-1); these files score it at **-1.00**
(a perfect inversion).

## What IS still valid here

- `timings` — the real PostgreSQL execution times, measured with a warm-up pass
  and rotation over plans. These are unaffected by the labelling bug.
- `prompts` — the exact text sent to the model.
- `unique_arm_plans`, `arms`, `arm_to_plan` — the plan set and arm mapping.
- `response` — the raw API responses, including `letter_probs` keyed by letter.

## How to recover the correct labels

`rescore_correct.py` re-derives the distributions from the stored
`response.letter_probs` using the inverse permutation (`order.index(letter)`)
without calling the API:

````bash
.venv/bin/python rescore_correct.py
````

## The corrected run

See [`../runs_fixed/`](../runs_fixed/) for the equivalent run with the bug
fixed, and `../runs_fixed/README.md` for its provenance.
