import storage
import train
import baoctl
from constants import DEFAULT_MODEL_PATH, OLD_MODEL_PATH, TMP_MODEL_PATH

import psycopg2
import time
import csv
import glob
import random

# ---- experiment parameters ----
WINDOW_SIZE = 2000
RETRAIN_EVERY = 100
INITIAL_PG_ONLY = 100
QUERY_DIR = "/home/mini/bao/mihini/BaoForPostgreSQL/job_unique"
OUTPUT_CSV = "bao_5000_run.csv"
PG_CONN_STR = "dbname=imdb user=mini password=qwerty host=localhost"
RANDOM_SEED = 42
# --------------------------------


def windowed_experience():
    return storage.experience_window(WINDOW_SIZE)


def run_one_query(cur, query_file):
    with open(query_file) as f:
        query = f.read()
    start = time.time()
    cur.execute(query)
    cur.fetchall()
    return (time.time() - start) * 1000.0


def main():
    run_start = time.time()
    # sorted() first for a deterministic starting point (glob order isn't
    # guaranteed consistent across filesystems/runs), then shuffle with a
    # fixed seed so the resulting order is reproducible across repetitions.
    query_files = sorted(glob.glob(f"{QUERY_DIR}/*.sql"))
    random.seed(RANDOM_SEED)
    random.shuffle(query_files)

    print(f"Found {len(query_files)} query files. Fixed random order (seed={RANDOM_SEED}).")

    if len(query_files) < INITIAL_PG_ONLY + RETRAIN_EVERY:
        print("WARNING: fewer queries than one full PG-only phase + one retrain interval. Check QUERY_DIR.")

    conn = psycopg2.connect(PG_CONN_STR)
    conn.autocommit = True
    cur = conn.cursor()

    results = []

    # Phase 1: initial PG-only queries, still logging reward
    cur.execute("SET enable_bao TO on;")
    cur.execute("SET enable_bao_selection TO off;")
    for i, qf in enumerate(query_files[:INITIAL_PG_ONLY]):
        latency = run_one_query(cur, qf)
        results.append((i, qf, latency, "PG-initial"))
        print(i, qf, f"{latency:.2f}ms", "PG-initial", flush=True)

    # Phase 2: Bao-active, with periodic windowed retraining
    cur.execute("SET enable_bao_selection TO on;")
    remaining = query_files[INITIAL_PG_ONLY:]
    for offset, qf in enumerate(remaining):
        i = INITIAL_PG_ONLY + offset

        if offset % RETRAIN_EVERY == 0:
            print(f"--- Retraining at query index {i}, window={WINDOW_SIZE} ---", flush=True)
            train.train_and_swap(
                DEFAULT_MODEL_PATH, OLD_MODEL_PATH, TMP_MODEL_PATH,
                verbose=True, experience_fn=windowed_experience
            )
            baoctl.send_model_load(DEFAULT_MODEL_PATH)

        latency = run_one_query(cur, qf)
        results.append((i, qf, latency, "Bao-active"))
        print(i, qf, f"{latency:.2f}ms", "Bao-active", flush=True)

        if offset % 500 == 0:
            with open(OUTPUT_CSV, "w", newline="") as f:
                csv.writer(f).writerows(results)

    with open(OUTPUT_CSV, "w", newline="") as f:
        csv.writer(f).writerows(results)

    elapsed_seconds = time.time() - run_start
    hours = elapsed_seconds / 3600
    print(f"Total run time: {elapsed_seconds:.1f} seconds ({hours:.2f} hours)")
    print(f"Done. Wrote {len(results)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()