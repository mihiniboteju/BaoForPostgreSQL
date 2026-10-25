#!/usr/bin/env python3
"""
compare_runs.py

Compare the LLM's logprob distribution against real execution times, for the
records written by run_pipeline.py.

    python compare_runs.py                       # runs/, timings optional
    python compare_runs.py --no-cost             # skip the planner-cost baseline
    python compare_runs.py --out comparison.json

Design notes
------------
The LLM's distribution lives inside one query and so do the latencies, so
every metric is computed per query and then averaged. There is no shared scale
across queries and a global probability/latency correlation would be
meaningless.

Regret is the headline, not accuracy. Six of ten queries are decided inside
the measurement noise, and there accuracy is a coin flip while regret stays
meaningful: picking the second-best plan when the two differ by 1% costs 1.01x,
not a failed test.

The planner-cost baseline is mandatory. A teacher that does not beat
ORDER BY cost is an expensive way to reproduce the planner.

No third-party dependencies: Kendall tau-b and the bootstrap are implemented
here so the script runs anywhere the pipeline runs.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any

DEFAULT_DSN = "dbname=imdb user=mini password=qwerty host=localhost"
EPS = 1e-9


# ---------------------------------------------------------------------------
# small statistics helpers
# ---------------------------------------------------------------------------


def geo_mean(values: list[float]) -> float | None:
    positives = [v for v in values if v > 0]
    if not positives:
        return None
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


def kendall_tau_b(xs: list[float], ys: list[float]) -> float | None:
    """Kendall's tau-b; handles ties on either side."""
    n = len(xs)
    if n < 2:
        return None
    concordant = discordant = 0
    tied_x = tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                tied_x += 1
                tied_y += 1
            elif dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    total = n * (n - 1) / 2
    denominator = math.sqrt((total - tied_x) * (total - tied_y))
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def median_diff_ci(a: list[float], b: list[float],
                   iters: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """Percentile bootstrap CI for median(a) - median(b)."""
    if len(a) < 2 or len(b) < 2:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        left = statistics.median([rng.choice(a) for _ in a])
        right = statistics.median([rng.choice(b) for _ in b])
        diffs.append(left - right)
    diffs.sort()
    low = diffs[int(0.025 * iters)]
    high = diffs[min(int(0.975 * iters) - 1, iters - 1)]
    return low, high


def mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_records(runs_dir: Path) -> list[dict[str, Any]]:
    records = []
    for name in sorted(glob.glob(str(runs_dir / "*.json"))):
        records.append(json.loads(Path(name).read_text(encoding="utf-8")))
    return records


def plan_medians(record: dict[str, Any]) -> dict[int, float]:
    return {
        index: timing["median_ms"]
        for index, timing in enumerate(record["timings"])
        if timing.get("median_ms") is not None
    }


def true_order(record: dict[str, Any]) -> list[int]:
    """Plan indices sorted fastest to slowest."""
    medians = plan_medians(record)
    return sorted(medians, key=lambda i: medians[i])


def separability(record: dict[str, Any]) -> dict[str, Any]:
    """
    How clearly the fastest plan beats the runner-up.

    Two criteria are reported because they answer different questions:
      gap       relative margin between first and second
      noise     relative measurement spread, from the raw samples
      bootstrap whether a 95% CI on the median difference excludes zero
    """
    order = true_order(record)
    if len(order) < 2:
        return {"order": order, "gap": None, "noise": None,
                "ratio_gap_noise": None, "resolved": False}
    first, second = order[0], order[1]
    first_t, second_t = record["timings"][first], record["timings"][second]

    m1 = first_t["median_ms"] or 0.0
    m2 = second_t["median_ms"] or 0.0
    gap = (m2 - m1) / m1 if m1 else None

    spread = ((first_t.get("spread_ms") or 0.0) + (second_t.get("spread_ms") or 0.0)) / 2
    noise = spread / m1 if m1 else None

    ci = median_diff_ci(first_t.get("samples_ms") or [],
                        second_t.get("samples_ms") or [])

    return {
        "order": order,
        "gap": gap,
        "noise": noise,
        "ratio_gap_noise": (gap / noise) if (gap is not None and noise) else None,
        "resolved": bool(ci and ci[1] < 0),
    }


# ---------------------------------------------------------------------------
# prediction sources
# ---------------------------------------------------------------------------


def variant_distribution(label: dict[str, Any]) -> dict[int, float]:
    """
    Candidate -> probability.

    Falls back to the argmax letter when the provider returned no logprobs, so
    a ranking is still recorded. Dropping those instead would silently remove
    exactly the queries where the model struggled.
    """
    distribution = {int(k): float(v) for k, v in (label.get("distribution") or {}).items()}
    if distribution:
        return distribution

    order = label.get("letter_order") or []
    letter = (label.get("response") or {}).get("chosen")
    if letter and order and letter in "ABCDEFGHIJ"[: len(order)]:
        return {int(order["ABCDEFGHIJ".index(letter)]): 1.0}
    if label.get("chosen_candidate") is not None:
        return {int(label["chosen_candidate"]): 1.0}
    return {}


def label_arms(record: dict[str, Any], label: dict[str, Any]) -> list[int]:
    chosen = label.get("chosen_candidate")
    if chosen is None:
        return []
    mapping = {int(a): c for a, c in record["arm_to_plan"].items()}
    return sorted(a for a, c in mapping.items() if c == chosen)


def cost_ranking(record: dict[str, Any], dsn: str) -> dict[int, float] | None:
    """Plan -> planner Total Cost, obtained by re-EXPLAINing under each arm."""
    try:
        import psycopg2
        from bao_arm_plans import ARMS, get_plan
    except Exception:
        return None

    sql_path = Path(record["query_file"])
    if not sql_path.exists():
        return None

    mapping = {int(a): c for a, c in record["arm_to_plan"].items()}
    first_arm: dict[int, int] = {}
    for arm in sorted(mapping):
        first_arm.setdefault(mapping[arm], arm)

    costs: dict[int, float] = {}
    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        with conn.cursor() as cur:
            query = sql_path.read_text(encoding="utf-8").strip().rstrip(";").rstrip()
            for plan_index, arm in first_arm.items():
                node = get_plan(cur, query, ARMS[arm])
                cost = node.get("Total Cost")
                if cost is not None:
                    costs[plan_index] = float(cost)
        conn.close()
    except Exception as error:                              # noqa: BLE001
        print(f"  cost baseline unavailable for {sql_path.name}: "
              f"{type(error).__name__}", flush=True)
        return None
    return costs or None


# ---------------------------------------------------------------------------
# per-query scoring
# ---------------------------------------------------------------------------


def score_prediction(medians: dict[int, float], order: list[int],
                     scores: dict[int, float]) -> dict[str, Any]:
    """Regret, NLL, top-k accuracy and rank correlation for one prediction."""
    if not scores or not order:
        return {}

    ranked = sorted(scores, key=lambda i: -scores[i])
    chosen = ranked[0]
    best = order[0]

    regret = medians[chosen] / medians[best] if chosen in medians else None

    probability = scores.get(best, 0.0)
    nll = -math.log(max(probability, EPS))

    result = {
        "chosen": chosen,
        "best": best,
        "regret": regret,
        "nll": nll,
        "top1": chosen == best,
        "top2": best in ranked[:2],
        "top3": best in ranked[:3],
        "top1_prob": scores[chosen],
    }

    # Rank correlation over the plans that have both a score and a latency.
    common = [i for i in order if i in scores]
    if len(common) >= 2:
        result["tau"] = kendall_tau_b(
            [-scores[i] for i in common],           # higher score = better
            [medians[i] for i in common],           # lower latency = better
        )
    if len(ranked) >= 2:
        result["top2_regret"] = min(medians[i] for i in ranked[:2]) / medians[best]

    total = sum(scores.values())
    result["p_best"] = probability
    return result


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def summarise(rows: list[dict[str, Any]], label: str) -> None:
    if not rows:
        print(f"  {label}: no results")
        return
    regrets = [r["regret"] for r in rows if r.get("regret") is not None]
    gm = geo_mean(regrets)
    print(f"  {label}")
    print(f"    n={len(rows)}   geomean regret="
          f"{gm:.3f}x" if gm else f"    n={len(rows)}   geomean regret=n/a")
    print(f"    mean NLL   = {mean([r.get('nll') for r in rows]):.3f}"
          if mean([r.get('nll') for r in rows]) is not None else "    mean NLL   = n/a")
    print(f"    top-1 acc  = {sum(r['top1'] for r in rows)}/{len(rows)}"
          f"   top-2 = {sum(r['top2'] for r in rows)}/{len(rows)}"
          f"   top-3 = {sum(r['top3'] for r in rows)}/{len(rows)}")
    tau = mean([r.get("tau") for r in rows])
    if tau is not None:
        print(f"    mean tau-b = {tau:+.3f}")
    t2 = geo_mean([r["top2_regret"] for r in rows if r.get("top2_regret")])
    if t2:
        print(f"    geomean top-2 regret = {t2:.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--no-cost", action="store_true",
                        help="skip the planner-cost baseline")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = load_records(args.runs_dir)
    if not records:
        raise SystemExit(f"no records in {args.runs_dir}")
    print(f"{len(records)} records from {args.runs_dir}\n")

    variants = sorted({label["row_source"]
                       for record in records for label in record["labels"]})
    if not variants:
        raise SystemExit("no labels found; run run_pipeline.py without --plan-only")

    # ---- 1. coverage --------------------------------------------------
    total_prompts = sum(len(r["labels"]) for r in records)
    with_logprobs = sum(1 for r in records for l in r["labels"]
                        if l["response"].get("supported"))
    failed = sum(1 for r in records for l in r["labels"]
                 if not l["response"].get("ok"))
    print("=" * 70)
    print("1. COVERAGE")
    print("=" * 70)
    print(f"  prompts sent        : {total_prompts}")
    print(f"  logprobs returned   : {with_logprobs} ({with_logprobs / total_prompts:.0%})")
    print(f"  API errors          : {failed}")
    print(f"  row_source variants : {', '.join(variants)}")

    # ---- truth and separability ---------------------------------------
    truth = {}
    for record in records:
        medians = plan_medians(record)
        order = true_order(record)
        truth[record["query_file"]] = {
            "medians": medians,
            "order": order,
            **separability(record),
        }

    print()
    print("=" * 70)
    print("2. GROUND TRUTH QUALITY  (does the measurement separate the plans?)")
    print("=" * 70)
    print(f"  {'query':14s} {'plans':>5} {'gap':>7} {'noise':>7} {'gap/noise':>9} {'resolved':>8}")
    resolved_count = 0
    for record in records:
        info = truth[record["query_file"]]
        resolved_count += bool(info["resolved"])
        name = Path(record["query_file"]).stem
        gap = f"{info['gap']:.4f}" if info["gap"] is not None else "-"
        noise = f"{info['noise']:.4f}" if info["noise"] else "-"
        ratio = f"{info['ratio_gap_noise']:.1f}" if info["ratio_gap_noise"] else "-"
        print(f"  {name:14s} {len(info['order']):5d} {gap:>7} {noise:>7} "
              f"{ratio:>9} {str(info['resolved']):>8}")
    print(f"\n  resolvable by bootstrap CI : {resolved_count}/{len(records)}")

    # ---- 3. headline ---------------------------------------------------
    print()
    print("=" * 70)
    print("3. HEADLINE  (LLM vs planner cost baseline)")
    print("=" * 70)

    cost_scores = {}
    if not args.no_cost:
        for record in records:
            costs = cost_ranking(record, args.dsn)
            if costs:
                # lower cost is better, so score = 1 / cost
                cost_scores[record["query_file"]] = {
                    i: 1.0 / c for i, c in costs.items()
                }

    for variant in variants:
        rows = []
        for record in records:
            label = next((l for l in record["labels"]
                          if l["row_source"] == variant), None)
            if not label:
                continue
            info = truth[record["query_file"]]
            outcome = score_prediction(info["medians"], info["order"],
                                       variant_distribution(label))
            if outcome:
                outcome["query"] = Path(record["query_file"]).stem
                outcome["resolved"] = info["resolved"]
                rows.append(outcome)
        print()
        summarise(rows, f"LLM  [{variant}]")
        res = [r for r in rows if r["resolved"]]
        if res and len(res) < len(rows):
            summarise(res, f"LLM  [{variant}] -- resolvable queries only")

    if cost_scores:
        rows = []
        for record in records:
            scores = cost_scores.get(record["query_file"])
            if not scores:
                continue
            info = truth[record["query_file"]]
            outcome = score_prediction(info["medians"], info["order"], scores)
            if outcome:
                outcome["query"] = Path(record["query_file"]).stem
                outcome["resolved"] = info["resolved"]
                rows.append(outcome)
        print()
        summarise(rows, "BASELINE  [planner cost]")
        res = [r for r in rows if r["resolved"]]
        if res and len(res) < len(rows):
            summarise(res, "BASELINE  [planner cost] -- resolvable queries only")

    # ---- 4. paired A/B between row_source variants ---------------------
    if len(variants) > 1:
        print()
        print("=" * 70)
        print("4. ROW_SOURCE A/B  (paired, same timings, same letter order)")
        print("=" * 70)
        per_variant = {v: {} for v in variants}
        for record in records:
            info = truth[record["query_file"]]
            for label in record["labels"]:
                outcome = score_prediction(info["medians"], info["order"],
                                           variant_distribution(label))
                if outcome:
                    per_variant[label["row_source"]][record["query_file"]] = outcome

        reference = variants[0]
        for variant in variants:
            wins = losses = ties = 0
            deltas = []
            for query, outcome in per_variant[variant].items():
                base = per_variant[reference].get(query)
                if not base or outcome.get("regret") is None or base.get("regret") is None:
                    continue
                delta = math.log(outcome["regret"]) - math.log(base["regret"])
                deltas.append(delta)
                wins += delta < -1e-9
                losses += delta > 1e-9
                ties += 1
            mean_delta = mean(deltas)
            print(f"  {variant:>6} vs {reference:<6}: "
                  f"wins={wins} losses={losses} ties={ties}"
                  + (f"   mean log-regret delta={mean_delta:+.3f}"
                     if mean_delta is not None else ""))

    # ---- 5. calibration and priors -------------------------------------
    print()
    print("=" * 70)
    print("5. CALIBRATION AND PRIOR CHECK")
    print("=" * 70)
    for variant in variants:
        probabilities = []
        residuals = []
        chosen_arms: dict[int, int] = {}
        true_arms: dict[int, int] = {}
        for record in records:
            label = next((l for l in record["labels"]
                          if l["row_source"] == variant), None)
            if not label:
                continue
            response = label["response"]
            if response.get("residual_mass") is not None:
                residuals.append(response["residual_mass"])
            outcome = score_prediction(truth[record["query_file"]]["medians"],
                                       truth[record["query_file"]]["order"],
                                       variant_distribution(label))
            if outcome:
                probabilities.append(outcome["top1_prob"])
            for arm in label_arms(record, label):
                chosen_arms[arm] = chosen_arms.get(arm, 0) + 1
            best = record.get("best_arm")
            if best is not None:
                true_arms[int(best)] = true_arms.get(int(best), 0) + 1

        print(f"\n  [{variant}]")
        if probabilities:
            print(f"    top-1 probability : min={min(probabilities):.3f} "
                  f"mean={statistics.mean(probabilities):.3f} "
                  f"max={max(probabilities):.3f} "
                  f"stdev={statistics.stdev(probabilities):.3f}"
                  if len(probabilities) > 1 else
                  f"    top-1 probability : {probabilities[0]:.3f}")
            print("    -> a wide spread means the model discriminates; "
                  "a narrow one means it is following a prior")
        if residuals:
            print(f"    residual mass     : mean={statistics.mean(residuals):.4f} "
                  f"max={max(residuals):.4f} "
                  f"(>0.10 on {sum(r > 0.10 for r in residuals)}/{len(residuals)})")
        print(f"    chosen arms       : "
              f"{dict(sorted(chosen_arms.items()))}")
        print(f"    true best arms    : {dict(sorted(true_arms.items()))}")

    # ---- caveats --------------------------------------------------------
    print()
    print("=" * 70)
    print("6. CAVEATS")
    print("=" * 70)
    print(f"  n = {len(records)} queries; treat this as a pilot, not a result.")
    print(f"  {len(records) - resolved_count} queries are not separated by the")
    print("  measurement noise, so their top-1 label is close to arbitrary.")
    print("  Top-1 accuracy on 'resolvable only' has a very small denominator.")


if __name__ == "__main__":
    main()