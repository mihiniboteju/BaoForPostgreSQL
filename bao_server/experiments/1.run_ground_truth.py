"""
Step 1: Ground truth generation.
For each of the 100 validation queries, force each of the 5 arms directly
(bypassing Bao entirely) and measure real execution latency with repetition.
Output: ground_truth.json -- one entry per query, with per-arm median latency
and the identified best arm.
"""
import json
import time
import psycopg2

PG_CONN_STR = "dbname=imdb user=mini password=qwerty host=localhost"
QUERIES_FILE = "validation_queries.json"
OUTPUT_FILE = "ground_truth.json"
REPS = 3

ALL_FLAGS = ["enable_hashjoin", "enable_mergejoin", "enable_nestloop",
             "enable_seqscan", "enable_indexscan", "enable_indexonlyscan"]

ARM_HINTS = {
    0: ["enable_hashjoin", "enable_indexonlyscan", "enable_indexscan", "enable_mergejoin", "enable_seqscan"],
    1: ["enable_hashjoin", "enable_indexonlyscan", "enable_nestloop", "enable_seqscan"],
    2: ["enable_hashjoin", "enable_indexonlyscan", "enable_seqscan"],
    3: ["enable_hashjoin", "enable_indexonlyscan", "enable_indexscan", "enable_nestloop", "enable_seqscan"],
    4: ["enable_indexonlyscan", "enable_indexscan", "enable_mergejoin", "enable_nestloop", "enable_seqscan"],
}


def set_arm(cur, arm_idx):
    cur.execute("SET enable_bao TO off;")
    for f in ALL_FLAGS:
        cur.execute(f"SET {f} TO off;")
    for f in ARM_HINTS[arm_idx]:
        cur.execute(f"SET {f} TO on;")


def measure_arm(cur, arm_idx, query, reps=REPS):
    set_arm(cur, arm_idx)
    times = []
    for _ in range(reps):
        cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
        result = cur.fetchone()[0]
        if isinstance(result, str):
            result = json.loads(result)
        plan = result[0]["Plan"] if "Plan" in result[0] else result[-1]["Plan"]
        times.append(plan["Actual Total Time"] if "Actual Total Time" in plan
                     else result[0].get("Execution Time"))
    times = sorted([t for t in times if t is not None])
    return times[len(times) // 2] if times else None


def main():
    query_files = json.load(open(QUERIES_FILE))
    conn = psycopg2.connect(PG_CONN_STR)
    conn.autocommit = True
    cur = conn.cursor()

    results = []
    for qi, qf in enumerate(query_files):
        with open(qf) as f:
            query = f.read().strip().rstrip(";")

        # one throwaway warm-up execution before timed reps, arm 0
        set_arm(cur, 0)
        try:
            cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
            cur.fetchone()
        except Exception as e:
            print(f"[{qi}] WARMUP FAILED {qf}: {e}", flush=True)
            continue

        arm_latencies = {}
        for arm in range(5):
            try:
                lat = measure_arm(cur, arm, query)
            except Exception as e:
                print(f"[{qi}] arm {arm} FAILED {qf}: {e}", flush=True)
                lat = None
            arm_latencies[arm] = lat
            print(f"[{qi}] {qf} arm={arm} median_ms={lat}", flush=True)

        valid = {a: v for a, v in arm_latencies.items() if v is not None}
        best_arm = min(valid, key=valid.get) if valid else None

        results.append({
            "query_file": qf,
            "arm_latencies_ms": arm_latencies,
            "best_arm": best_arm,
        })

        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Done. Wrote {len(results)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()