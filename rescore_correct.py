#!/usr/bin/env python3
"""Re-score stored teacher logprobs with the CORRECT letter -> candidate map.

The prompt printed the letter for candidate ``c`` as ``LETTERS[order[c]]``, but
the decoder read letter ``L`` back as candidate ``order[L]``.  ``order`` is a
permutation, so those two differ -- every stored label was attributed to the
wrong candidate.  The true inverse is ``candidate = order.index(L)``.

This script rescores without calling the API, using the stored letter_probs.
"""

from __future__ import annotations

import json
import pathlib
import statistics

LETTERS = "ABCDEFGHIJ"
ROOT = pathlib.Path(__file__).resolve().parent


def medians(record):
    out = []
    for t in record.get("timings") or []:
        m = (t or {}).get("median_ms")
        out.append(float("inf") if m is None else float(m))
    return out


def tau_b(xs, ys):
    n = len(xs)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            if (dx > 0) == (dy > 0):
                c += 1
            else:
                d += 1
    return (c - d) / (c + d) if (c + d) else 0.0


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return float(statistics.geometric_mean(xs)) if xs else float("nan")


def decode(label, n):
    """Return {candidate: prob} using the corrected inverse permutation."""
    order = label.get("letter_order") or []
    probs = label.get("response", {}).get("letter_probs") or {}
    dist = {}
    for key, p in probs.items():
        letter_index = LETTERS.index(key) if isinstance(key, str) else int(key)
        if letter_index >= len(order):
            continue
        cand = order.index(letter_index)          # <-- corrected
        dist[cand] = dist.get(cand, 0.0) + float(p)
    if not dist:
        chosen = label.get("response", {}).get("chosen")
        if chosen and chosen in LETTERS[: len(order)]:
            dist = {order.index(LETTERS.index(chosen)): 1.0}   # <-- corrected
    return {i: dist.get(i, 0.0) for i in range(n)}


def main() -> None:
    rows = []
    for path in sorted((ROOT / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        qid = path.stem
        ms = medians(record)
        n = len(ms)
        if not n:
            continue
        true_order = sorted(range(n), key=lambda i: (ms[i], i))
        for label in record.get("labels") or []:
            variant = label.get("row_source")
            full = decode(label, n)
            if not any(full.values()):
                continue
            by_llm = sorted(range(n), key=lambda i: (-full[i], i))
            regret = ms[by_llm[0]] / ms[true_order[0]] if ms[true_order[0]] > 0 else float("inf")
            top2 = min(ms[i] for i in by_llm[:2]) / ms[true_order[0]]
            rows.append({
                "variant": variant, "qid": qid, "n": n, "ms": ms,
                "true": true_order, "llm": by_llm, "dist": full,
                "tau": tau_b([-full[i] for i in range(n)], [ms[i] for i in range(n)]),
                "regret": regret, "top2": top2,
                "top1_ok": by_llm[0] == true_order[0],
                "identical": by_llm == true_order,
            })

    print("CORRECTED DECODING -- candidate = order.index(letter)")
    print()
    for r in rows:
        if r["variant"] != "table":
            continue
        ro = " > ".join(f"C{i+1}" for i in r["true"])
        lo = " > ".join(f"C{i+1}" for i in r["llm"])
        flag = "OK" if r["top1_ok"] else "WRONG"
        print(f'{r["qid"]:7s} ({r["n"]} plans)  real {ro:18s}  llm {lo:18s}  tau={r["tau"]:+.2f}  {flag}')
    print()
    for variant in ("both", "plan", "table"):
        sub = [r for r in rows if r["variant"] == variant]
        if not sub:
            continue
        print(
            f'  [{variant:5s}] n={len(sub)}  identical={sum(r["identical"] for r in sub)}'
            f'  top1={sum(r["top1_ok"] for r in sub)}/{len(sub)}'
            f'  mean tau={statistics.mean(r["tau"] for r in sub):+.3f}'
            f'  geomean regret={geomean([r["regret"] for r in sub]):.3f}x'
        )


if __name__ == "__main__":
    main()