import psycopg2
import time
import csv
import glob
import random

QUERY_DIR = "/home/mini/bao/mihini/BaoForPostgreSQL/job_unique"
OUTPUT_CSV = "pg_only_full_run.csv"
PG_CONN_STR = "dbname=imdb user=mini password=qwerty host=localhost"
RANDOM_SEED = 42


def run_one_query(cur, query_file):
    with open(query_file) as f:
        query = f.read()
    start = time.time()
    cur.execute(query)
    cur.fetchall()
    return (time.time() - start) * 1000.0


def main():
    run_start = time.time()

    query_files = sorted(glob.glob(f"{QUERY_DIR}/*.sql"))
    random.seed(RANDOM_SEED)
    random.shuffle(query_files)

    print(f"Found {len(query_files)} query files. Fixed random order (seed={RANDOM_SEED}).")

    conn = psycopg2.connect(PG_CONN_STR)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("SET enable_bao TO off;")   # completely disables Bao's hooks

    results = []
    for i, qf in enumerate(query_files):
        latency = run_one_query(cur, qf)
        results.append((i, qf, latency))
        print(i, qf, f"{latency:.2f}ms", flush=True)

        if i % 500 == 0:
            with open(OUTPUT_CSV, "w", newline="") as f:
                csv.writer(f).writerows(results)

    with open(OUTPUT_CSV, "w", newline="") as f:
        csv.writer(f).writerows(results)

    elapsed = time.time() - run_start
    print(f"Done. Wrote {len(results)} rows to {OUTPUT_CSV}")
    print(f"Total run time: {elapsed:.1f} seconds ({elapsed/3600:.2f} hours)")


if __name__ == "__main__":
    main()