"""
Step 3: Analysis.
Compares LLM top-1 accuracy and rank correlation against ground truth,
against a random baseline (binomial test) and the free cost-ratio baseline.
Also breaks down agreement by full rank position (1st/2nd/3rd/4th/5th),
not just top-1, since a teacher that reliably orders the middle arms
correctly is still useful for distillation even if it misses 1st place.

Cost-ratio baseline costs are parsed directly out of the saved prompt text
("Estimated cost: X" per arm), since arm_summaries wasn't saved as a
separate field in this run.
"""
import json
import math
import re
from collections import Counter
from scipy.stats import binomtest, spearmanr

GROUND_TRUTH_FILE = "ground_truth.json"
LLM_LABELS_FILE = "llm_labels.json"

COST_RE = re.compile(r"Arm (\d) \[disables:.*?\]\s*\n\s*Estimated cost:\s*([\d.]+)")


def extract_costs_from_prompt(prompt):
    matches = COST_RE.findall(prompt)
    if len(matches) != 5:
        return None
    return {int(arm): float(cost) for arm, cost in matches}


def cost_ratio_distribution(costs):
    neg = [-costs[a] for a in range(5)]
    m = max(neg)
    exps = [math.exp(x - m) for x in neg]
    total = sum(exps)
    return {str(a): exps[a] / total for a in range(5)}


def rank_from_latencies(latencies):
    """Returns {arm: rank}, rank 0 = fastest (1st place), ..., rank 4 = slowest (5th place)."""
    order = sorted(range(5), key=lambda a: latencies[a] if latencies[a] is not None else 1e18)
    return {a: r for r, a in enumerate(order)}


def rank_from_distribution(dist):
    """Returns {arm: rank}, rank 0 = highest probability (predicted 1st place)."""
    order = sorted(range(5), key=lambda a: -dist[str(a)])
    return {a: r for r, a in enumerate(order)}


def positional_accuracy(per_query, label):
    """
    For each real rank position (1st..5th), what fraction of the time did
    the predicted ranking put the SAME arm in that position?
    This directly answers: does the LLM's 2nd-place pick actually match
    the arm that was really 2nd-fastest, not just "is top-1 right".
    """
    position_names = ["1st (fastest)", "2nd", "3rd", "4th", "5th (slowest)"]
    hits = [0] * 5
    n = len(per_query)
    for q in per_query:
        real_rank = q["real_rank"]     # {arm: rank}
        pred_rank = q[f"{label}_rank"]  # {arm: rank}
        # invert: which arm sits at each rank position
        real_by_pos = {r: a for a, r in real_rank.items()}
        pred_by_pos = {r: a for a, r in pred_rank.items()}
        for pos in range(5):
            if real_by_pos[pos] == pred_by_pos[pos]:
                hits[pos] += 1

    print(f"\n=== {label.upper()} — accuracy by rank position ===")
    for pos, name in enumerate(position_names):
        print(f"  {name:<15}: {hits[pos]}/{n} = {hits[pos]/n:.3f}")
    return hits


def confusion_matrix(per_query, label):
    """
    5x5 matrix: rows = real rank position, columns = predicted rank position
    for the arm that actually occupies that real position.
    Shows e.g. "the real 2nd-place arm was predicted as 3rd place X% of the time".
    """
    counts = Counter()
    n = len(per_query)
    for q in per_query:
        real_rank = q["real_rank"]
        pred_rank = q[f"{label}_rank"]
        for arm in range(5):
            counts[(real_rank[arm], pred_rank[arm])] += 1

    print(f"\n=== {label.upper()} — rank confusion matrix (rows=real, cols=predicted, counts) ===")
    header = "        " + "".join(f"pred{p}".rjust(8) for p in range(5))
    print(header)
    for real_r in range(5):
        row = f"real{real_r}   " + "".join(str(counts[(real_r, pred_r)]).rjust(8) for pred_r in range(5))
        print(row)
    return counts


def main():
    gt = {r["query_file"]: r for r in json.load(open(GROUND_TRUTH_FILE))}
    llm = {r["query_file"]: r for r in json.load(open(LLM_LABELS_FILE))}

    common = [qf for qf in gt if qf in llm and llm[qf]["llm_distribution"] is not None
              and gt[qf]["best_arm"] is not None]
    print(f"Comparable queries: {len(common)} / {len(gt)}")

    per_query = []
    for qf in common:
        latencies = {int(a): v for a, v in gt[qf]["arm_latencies_ms"].items()}
        if any(v is None for v in latencies.values()):
            continue
        real_rank = rank_from_latencies(latencies)

        llm_dist = llm[qf]["llm_distribution"]
        llm_rank = rank_from_distribution(llm_dist)

        costs = extract_costs_from_prompt(llm[qf].get("prompt", ""))
        cost_dist = cost_ratio_distribution(costs) if costs else None
        cost_rank = rank_from_distribution(cost_dist) if cost_dist else None

        llm_top1 = min(llm_rank, key=llm_rank.get)
        rho, _ = spearmanr([real_rank[a] for a in range(5)], [llm_rank[a] for a in range(5)])

        entry = {
            "query_file": qf,
            "real_rank": real_rank,
            "llm_rank": llm_rank,
            "llm_top1_correct": real_rank[llm_top1] == 0,
            "spearman_rho": rho,
        }
        if cost_dist:
            cost_top1 = min(cost_rank, key=cost_rank.get)
            cost_rho, _ = spearmanr([real_rank[a] for a in range(5)], [cost_rank[a] for a in range(5)])
            entry["cost_rank"] = cost_rank
            entry["cost_top1_correct"] = real_rank[cost_top1] == 0
            entry["cost_spearman_rho"] = cost_rho

        per_query.append(entry)

    n = len(per_query)
    llm_correct = sum(1 for q in per_query if [a for a, r in q["real_rank"].items() if r == 0][0]
                       == [a for a, r in q["llm_rank"].items() if r == 0][0])
    llm_acc = llm_correct / n
    binom = binomtest(llm_correct, n=n, p=0.2, alternative="greater")
    mean_rho = sum(q["spearman_rho"] for q in per_query if q["spearman_rho"] == q["spearman_rho"]) / n

    print(f"\n=== LLM Teacher — top-1 summary ===")
    print(f"Top-1 accuracy: {llm_acc:.3f} ({llm_correct}/{n})")
    print(f"Binomial test vs random (p=0.2): p-value = {binom.pvalue:.5f}")
    print(f"Mean Spearman rank correlation: {mean_rho:.3f}")

    positional_accuracy(per_query, "llm")
    confusion_matrix(per_query, "llm")

    has_cost = all("cost_rank" in q for q in per_query)
    if has_cost:
        cost_correct = sum(1 for q in per_query if [a for a, r in q["real_rank"].items() if r == 0][0]
                            == [a for a, r in q["cost_rank"].items() if r == 0][0])
        cost_acc = cost_correct / n
        mean_cost_rho = sum(q["cost_spearman_rho"] for q in per_query) / n
        print(f"\n=== Cost-ratio baseline — top-1 summary ===")
        print(f"Top-1 accuracy: {cost_acc:.3f} ({cost_correct}/{n})")
        print(f"Mean Spearman rank correlation: {mean_cost_rho:.3f}")
        positional_accuracy(per_query, "cost")
        confusion_matrix(per_query, "cost")
    else:
        print("\n(Cost-ratio baseline skipped: arm_summaries not found in llm_labels.json --"
              " see note on wiring this through Step 2.)")

    with open("validation_results.json", "w") as f:
        json.dump({"n": n, "llm_top1_accuracy": llm_acc, "binomial_pvalue": binom.pvalue,
                   "mean_spearman_rho": mean_rho, "per_query": per_query}, f, indent=2, default=str)
    print("\nWrote validation_results.json")


if __name__ == "__main__":
    main()