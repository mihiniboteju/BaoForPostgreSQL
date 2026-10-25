#!/usr/bin/env python3
"""Print, per query, the ordering of plans by real execution time next to the
ordering of plans by LLM log-probability, so the two can be compared directly.

    .venv/bin/python show_order.py                  # best variant (table)
    .venv/bin/python show_order.py --variant both
    .venv/bin/python show_order.py --all-variants
"""

from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
LETTERS = "ABCDEFGHIJ"


def short_plan(text: str) -> str:
    """Compress a serialised Bao plan into a readable join/scan signature."""
    t = text.split(" Leading")[0]
    sig = []
    if "NestLoop" in t:
        sig.append("NLJ")
    if "HashJoin" in t:
        sig.append("HshJ")
    if "MergeJoin" in t:
        sig.append("MgJ")
    scans = []
    for node in ("IndexOnlyScan", "BitmapScan", "IndexScan", "SeqScan"):
        n = t.count(node)
        if n:
            scans.append(f"{node.replace('Scan','')}x{n}")
    return "+".join(sig) + ":" + "/".join(scans)


def kendall_tau_b(a: list[int], b: list[int]) -> float:
    """Kendall tau-b between two integer rank vectors given as orderings."""
    n = len(a)
    if n < 2:
        return 0.0
    pos_a = {v: i for i, v in enumerate(a)}
    pos_b = {v: i for i, v in enumerate(b)}
    conc = disc = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            u, v = i, j
            s = (pos_a[u] - pos_a[v]) * (pos_b[u] - pos_b[v])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
            elif pos_a[u] == pos_a[v]:
                ta += 1
            else:
                tb += 1
    d = ((conc + disc + ta) * (conc + disc + tb)) ** 0.5
    return (conc - disc) / d if d else 0.0


def llm_distribution(label: dict) -> dict[int, float]:
    dist = {int(k): float(v) for k, v in (label.get("distribution") or {}).items()}
    if dist:
        return dist
    order = label.get("letter_order") or []
    letter = (label.get("response") or {}).get("chosen")
    if letter and order and letter in LETTERS[: len(order)]:
        return {int(order[LETTERS.index(letter)]): 1.0}
    if label.get("chosen_candidate") is not None:
        return {int(label["chosen_candidate"]): 1.0}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="table", choices=["both", "plan", "table"])
    ap.add_argument("--all-variants", action="store_true")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--timings", default="timings.json")
    args = ap.parse_args()

    variants = ["both", "plan", "table"] if args.all_variants else [args.variant]

    timings = json.loads((ROOT / args.timings).read_text())
    records = {}
    for path in sorted((ROOT / args.runs_dir).glob("*.json")):
        rec = json.loads(path.read_text())
        records[path.stem] = rec

    # timings.json is keyed by full query path; run files by bare query id
    tmap_by_id = {pathlib.Path(k).stem: v for k, v in timings.items()}

    agree_rows = []
    for qid in sorted(set(tmap_by_id) & set(records)):
        rec, tm = records[qid], tmap_by_id[qid]
        plans = rec.get("unique_arm_plans") or []
        n = len(plans)
        if not n:
            continue

        # ---- order by real median execution time (fastest first)
        times = []
        for i in range(n):
            t = (rec.get("timings") or [{}] * n)[i] or {}
            med = t.get("median_ms")
            times.append((float("inf") if med is None else float(med), i))
        by_time = [i for _, i in sorted(times, key=lambda p: (p[0], p[1]))]
        tmap = {i: times[i][0] for i in range(n)}

        gap_noise = tm.get("ratio_gap_noise")
        resolvable = tm.get("resolved")
        head = f"{qid}  ({n} plans)"
        if gap_noise is not None:
            head += f"  gap/noise={float(gap_noise):.1f}  resolvable={resolvable}"
        print("=" * 78)
        print("  " + head)
        print("-" * 78)
        for rank, idx in enumerate(by_time, 1):
            ms = tmap[idx]
            shown = "TIMEOUT" if ms == float("inf") else f"{ms:10.3f} ms"
            print(f"  real rank {rank}  C{idx + 1}  {shown:>14}   {short_plan(plans[idx])}")

        for variant in variants:
            label = next(
                (x for x in rec.get("labels", []) if x.get("row_source") == variant),
                None,
            )
            if label is None:
                print(f"  LLM [{variant}]: no label")
                continue
            dist = llm_distribution(label)
            if len(dist) < n:
                # pad with zero probability for candidates the model never scored
                dist = {i: dist.get(i, 0.0) for i in range(n)}
            by_llm = sorted(range(n), key=lambda i: (-dist[i], i))
            by_time_ranks = {c: r + 1 for r, c in enumerate(by_time)}
            llm_ranks = {c: r + 1 for r, c in enumerate(by_llm)}
            same_order = by_time == by_llm
            tau = kendall_tau_b(by_time, by_llm)
            top1 = by_llm[0] == by_time[0]
            print()
            print(f"  LLM  [{variant}]  probs: " + "  ".join(
                f"C{i + 1}={dist[i]:.3f}" for i in range(n)))
            print("  " + "  ".join(
                f"C{i + 1}(p={dist[i]:.2f})" for i in by_llm))
            print(
                f"  => order {'IDENTICAL' if same_order else 'DIFFERENT'}"
                f"   tau-b={tau:+.2f}"
                f"   model top-1 = C{by_llm[0] + 1}"
                f"   true top-1 = C{by_time[0] + 1}"
                f"   {'CORRECT' if top1 else 'WRONG'}"
            )
            print("     rank by time: " + " > ".join(
                f"C{i + 1}({by_time_ranks[i]})" for i in by_time))
            print("     rank by LLM : " + " > ".join(
                f"C{i + 1}({llm_ranks[i]})" for i in by_llm))
            agree_rows.append((variant, qid, same_order, tau, top1))
        print()

    if agree_rows:
        print("=" * 78)
        print("SUMMARY")
        print("=" * 78)
        for variant in variants:
            rows = [r for r in agree_rows if r[0] == variant]
            if not rows:
                continue
            ident = sum(1 for r in rows if r[2])
            t1 = sum(1 for r in rows if r[4])
            mean_tau = sum(r[3] for r in rows) / len(rows)
            print(
                f"  [{variant}]  identical order: {ident}/{len(rows)}"
                f"   top-1 correct: {t1}/{len(rows)}"
                f"   mean tau-b: {mean_tau:+.3f}"
            )


if __name__ == "__main__":
    main()