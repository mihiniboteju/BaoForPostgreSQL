"""
Step 2: Gather statistics and query the LLM teacher.

For each validation query:
- warm the cache identically to ground truth
- snapshot buffer state
- get table row counts and indexes
- get per-arm EXPLAIN (cost/rows/plan summary)
- build the prompt
- call the Gemini teacher through Google API
- parse the returned distribution

Output: llm_labels.json

Gemini is called directly through the Google Gemini HTTP API.
"""

import json
import re
import os
import requests
import psycopg2
import time


PG_CONN_STR = "dbname=imdb user=mini password=qwerty host=localhost"

QUERIES_FILE = "validation_queries.json"
OUTPUT_FILE = "llm_labels.json"

# Gemini model
MODEL = "gemini-3.1-flash-lite"

# Wait 10 seconds after each successful Gemini response
WAIT_AFTER_SUCCESS = 10

ALL_FLAGS = [
    "enable_hashjoin",
    "enable_mergejoin",
    "enable_nestloop",
    "enable_seqscan",
    "enable_indexscan",
    "enable_indexonlyscan",
]

ARM_HINTS = {
    0: [
        "enable_hashjoin",
        "enable_indexonlyscan",
        "enable_indexscan",
        "enable_mergejoin",
        "enable_seqscan",
    ],
    1: [
        "enable_hashjoin",
        "enable_indexonlyscan",
        "enable_nestloop",
        "enable_seqscan",
    ],
    2: [
        "enable_hashjoin",
        "enable_indexonlyscan",
        "enable_seqscan",
    ],
    3: [
        "enable_hashjoin",
        "enable_indexonlyscan",
        "enable_indexscan",
        "enable_nestloop",
        "enable_seqscan",
    ],
    4: [
        "enable_indexonlyscan",
        "enable_indexscan",
        "enable_mergejoin",
        "enable_nestloop",
        "enable_seqscan",
    ],
}


def set_arm(cur, arm_idx):
    cur.execute("SET enable_bao TO off;")

    for f in ALL_FLAGS:
        cur.execute(f"SET {f} TO off;")

    for f in ARM_HINTS[arm_idx]:
        cur.execute(f"SET {f} TO on;")


def arm_disables_text(arm_idx):
    disabled = [
        f.replace("enable_", "")
        for f in ALL_FLAGS
        if f not in ARM_HINTS[arm_idx]
    ]

    return (
        ", ".join(disabled)
        if disabled
        else "nothing (all operators enabled)"
    )


def extract_tables(query):
    # Crude but sufficient:
    # PostgreSQL identifiers after FROM/JOIN, comma-separated.
    m = re.search(
        r"FROM\s+(.*?)\s+WHERE",
        query,
        re.IGNORECASE | re.DOTALL,
    )

    if not m:
        return []

    from_clause = m.group(1)

    tables = re.findall(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\s+"
        r"(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s*"
        r"(?:,|$)",
        from_clause,
    )

    return list(set(tables))


def get_table_stats(cur, tables):
    if not tables:
        return {}

    # Approximate row counts
    cur.execute(
        """
        SELECT relname, reltuples::bigint
        FROM pg_class
        WHERE relname = ANY(%s)
        """,
        (tables,),
    )

    row_counts = dict(cur.fetchall())

    # Cached pages
    cur.execute(
        """
        SELECT c.relname, count(*)
        FROM pg_buffercache b
        JOIN pg_class c
          ON b.relfilenode = c.relfilenode
        JOIN pg_database d
          ON b.reldatabase = d.oid
        WHERE d.datname = current_database()
          AND c.relname = ANY(%s)
        GROUP BY c.relname
        """,
        (tables,),
    )

    cached_pages = dict(cur.fetchall())

    # Indexes
    cur.execute(
        """
        SELECT tablename, indexname
        FROM pg_indexes
        WHERE tablename = ANY(%s)
        """,
        (tables,),
    )

    indexes = {}

    for tbl, idx in cur.fetchall():
        indexes.setdefault(tbl, []).append(idx)

    return {
        t: {
            "approx_rows": row_counts.get(t, 0),
            "cached_pages": cached_pages.get(t, 0),
            "indexes": indexes.get(t, []),
        }
        for t in tables
    }


def summarize_plan(plan_json):
    counts = {}

    def walk(n):
        counts[n["Node Type"]] = counts.get(n["Node Type"], 0) + 1

        for c in n.get("Plans", []):
            walk(c)

    walk(plan_json)

    return {
        "total_cost": plan_json["Total Cost"],
        "plan_rows": plan_json["Plan Rows"],
        "operators": counts,
    }


def get_arm_plan_summary(cur, arm_idx, query):
    set_arm(cur, arm_idx)

    cur.execute(f"EXPLAIN (FORMAT JSON) {query}")

    result = cur.fetchone()[0]

    if isinstance(result, str):
        result = json.loads(result)

    plan = result[0]["Plan"]

    return summarize_plan(plan)


def build_prompt(query, table_stats, arm_summaries):
    tables_txt = "\n".join(
        f"- {t}: ~{s['approx_rows']} rows, "
        f"{s['cached_pages']} pages currently cached, "
        f"indexes: "
        f"{', '.join(s['indexes']) if s['indexes'] else 'none'}"
        for t, s in table_stats.items()
    )

    arms_txt = ""

    for arm in range(5):
        s = arm_summaries[arm]

        ops = ", ".join(
            f"{k} x{v}"
            for k, v in s["operators"].items()
        )

        arms_txt += (
            f"\nArm {arm} "
            f"[disables: {arm_disables_text(arm)}]\n"
            f"  Estimated cost: {s['total_cost']}, "
            f"estimated rows: {s['plan_rows']}\n"
            f"  Plan shape: {ops}\n"
        )

    return f"""Query:
{query}

Referenced tables:
{tables_txt}

Candidate execution strategies:
{arms_txt}
"""


SYSTEM_PROMPT = """You are assisting a PostgreSQL query optimizer research system.

You will be shown one SQL query and five candidate execution strategies
("arms"), each defined by which join/scan operators PostgreSQL is allowed
to use.

For each arm, you are given PostgreSQL's own cost estimate, row estimate,
and a summary of the resulting plan shape, plus context about the
referenced tables.

Assign a probability to each arm reflecting your estimate of how likely
it is to be the fastest-executing option.

Respond with ONLY a JSON object in exactly this format:

{"0": p0, "1": p1, "2": p2, "3": p3, "4": p4}

The probabilities must sum to 1.0.

Do not include markdown.
Do not include explanations.
Do not include any other text.
"""


def call_llm(api_key, prompt):
    """
    Call Gemini directly through the Google Gemini HTTP API.
    Retry temporary server/rate-limit errors before giving up.
    """

    url = (
        f"https://generativelanguage.googleapis.com/"
        f"v1beta/models/{MODEL}:generateContent"
    )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    full_prompt = SYSTEM_PROMPT + "\n\n" + prompt

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": full_prompt
                    }
                ]
            }
        ]
    }

    max_retries = 5

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=120,
            )

            # Retry temporary server/rate-limit errors
            if response.status_code in [429, 500, 502, 503, 504]:

                wait_time = 10 * (attempt + 1)

                print(
                    f"  Gemini returned HTTP {response.status_code}. "
                    f"Retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/{max_retries})...",
                    flush=True,
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()

            result = response.json()

            text = (
                result["candidates"][0]
                ["content"]
                ["parts"][0]
                ["text"]
            ).strip()

            # Remove accidental Markdown code fences
            text = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                text.strip(),
                flags=re.IGNORECASE,
            ).strip()

            return json.loads(text)

        except requests.exceptions.RequestException as e:

            if attempt == max_retries - 1:
                raise

            wait_time = 10 * (attempt + 1)

            print(
                f"  Request error: {e}. "
                f"Retrying in {wait_time}s...",
                flush=True,
            )

            time.sleep(wait_time)

    raise RuntimeError(
        f"Gemini request failed after {max_retries} attempts."
    )


def main():
    # Check Gemini API key before doing any PostgreSQL work
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Run: export GEMINI_API_KEY='your-key'"
        )

    query_files = json.load(open(QUERIES_FILE))

    conn = psycopg2.connect(PG_CONN_STR)
    conn.autocommit = True

    cur = conn.cursor()

    results = []

    for qi, qf in enumerate(query_files):

        with open(qf) as f:
            query = f.read().strip().rstrip(";")

        tables = extract_tables(query)

        try:
            # Warm cache identically to ground truth generation.
            # Arm 0, one execution.
            set_arm(cur, 0)

            cur.execute(
                f"EXPLAIN (ANALYZE, FORMAT JSON) {query}"
            )

            cur.fetchone()

            # Gather database statistics.
            table_stats = get_table_stats(cur, tables)

            # Get PostgreSQL's plan under each candidate arm.
            arm_summaries = {
                arm: get_arm_plan_summary(cur, arm, query)
                for arm in range(5)
            }

            # Build teacher prompt.
            prompt = build_prompt(
                query,
                table_stats,
                arm_summaries,
            )

            # Print the complete prompt before sending it to Gemini.
            print("\n" + "=" * 80, flush=True)
            print(f"QUERY [{qi}]: {qf}", flush=True)
            print("PROMPT SENT TO GEMINI:", flush=True)
            print("=" * 80, flush=True)
            print(SYSTEM_PROMPT + "\n\n" + prompt, flush=True)
            print("=" * 80 + "\n", flush=True)

            # Query Gemini.
            llm_dist = call_llm(
                api_key,
                prompt,
            )

            results.append(
                {
                    "query_file": qf,
                    "llm_distribution": llm_dist,
                    "prompt": prompt,
                }
            )

            print(
                f"[{qi}] {qf} -> {llm_dist}",
                flush=True,
            )

            # -------------------------------------------------
            # NEW: Wait 10 seconds AFTER a successful response
            # before starting the next query.
            # -------------------------------------------------
            if qi < len(query_files) - 1:
                print(
                    f"  Waiting {WAIT_AFTER_SUCCESS}s "
                    f"before next query...",
                    flush=True,
                )

                time.sleep(WAIT_AFTER_SUCCESS)

        except Exception as e:

            print(
                f"[{qi}] FAILED {qf}: {e}",
                flush=True,
            )

            results.append(
                {
                    "query_file": qf,
                    "llm_distribution": None,
                    "error": str(e),
                }
            )

        # Save after every query so that a failed/interrupted run
        # does not lose previous results.
        with open(OUTPUT_FILE, "w") as f:
            json.dump(
                results,
                f,
                indent=2,
            )

    print(
        f"Done. Wrote {len(results)} entries to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()