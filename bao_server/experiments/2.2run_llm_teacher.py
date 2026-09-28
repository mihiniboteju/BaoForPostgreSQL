"""
Step 2: Gather statistics and query the LLM teacher.

For each validation query:
- warm the cache identically to ground truth
- snapshot buffer state
- get table row counts and indexes
- get per-arm EXPLAIN (cost/rows/plan summary)
- build the prompt
- call the local Qwen3:8b teacher through Ollama
- parse the returned distribution

Version B:
Add compact intermediate-cardinality information from each arm's
EXPLAIN plan, without sending the full plan tree to the LLM.

Output: llm_labels.json

Ollama runs on Windows.
This script runs inside WSL and connects to the Windows Ollama
server through the WSL -> Windows host IP.
"""

import json
import re
import requests
import psycopg2
import time


# ============================================================
# Configuration
# ============================================================

PG_CONN_STR = "dbname=imdb user=mini password=qwerty host=localhost"

QUERIES_FILE = "validation_queries.json"
OUTPUT_FILE = "llm_labels.json"

# ============================================================
# Ollama configuration
# ============================================================

# Windows host IP as seen from WSL.
OLLAMA_BASE_URL = "http://172.29.64.1:11434"

# Local Ollama model
MODEL = "qwen3:8b"

# Wait 10 seconds after each successful LLM response
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


# ============================================================
# PostgreSQL arm configuration
# ============================================================

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


# ============================================================
# Table extraction and statistics
# ============================================================

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


# ============================================================
# Version C:
# Compact cardinality + access paths + join structure
# ============================================================

def summarize_plan(plan_json):
    counts = {}

    all_rows = []
    join_nodes = []
    scan_nodes = []

    def walk(n):

        node_type = n.get(
            "Node Type",
            "Unknown"
        )

        plan_rows = n.get(
            "Plan Rows",
            0
        )

        if plan_rows is None:
            plan_rows = 0

        # ----------------------------------------------------
        # Operator counts
        # ----------------------------------------------------

        counts[node_type] = (
            counts.get(node_type, 0) + 1
        )

        # ----------------------------------------------------
        # Cardinality
        # ----------------------------------------------------

        all_rows.append(plan_rows)

        # ----------------------------------------------------
        # Join structure
        # ----------------------------------------------------

        if (
            "Join" in node_type
            or node_type == "Nested Loop"
        ):

            children = n.get(
                "Plans",
                []
            )

            child_desc = []

            for child in children:

                child_type = child.get(
                    "Node Type",
                    "Unknown"
                )

                relation = child.get(
                    "Alias",
                    child.get(
                        "Relation Name"
                    )
                )

                if relation:
                    child_desc.append(
                        f"{relation}:{child_type}"
                    )
                else:
                    child_desc.append(
                        child_type
                    )

            join_nodes.append(
                {
                    "type": node_type,
                    "rows": plan_rows,
                    "children": child_desc,
                }
            )

        # ----------------------------------------------------
        # Scan / access path
        # ----------------------------------------------------

        if "Scan" in node_type:

            scan_nodes.append(
                {
                    "type": node_type,
                    "relation": n.get(
                        "Relation Name"
                    ),
                    "alias": n.get(
                        "Alias"
                    ),
                    "index": n.get(
                        "Index Name"
                    ),
                    "rows": plan_rows,
                }
            )

        # ----------------------------------------------------
        # Recurse into child plans
        # ----------------------------------------------------

        for child in n.get(
            "Plans",
            []
        ):
            walk(child)

    walk(plan_json)

    # --------------------------------------------------------
    # Cardinality summaries
    # --------------------------------------------------------

    max_intermediate_rows = (
        max(all_rows)
        if all_rows
        else 0
    )

    join_rows = [
        j["rows"]
        for j in join_nodes
    ]

    max_join_output = (
        max(join_rows)
        if join_rows
        else 0
    )

    scan_rows = [
        s["rows"]
        for s in scan_nodes
    ]

    max_scan_rows = (
        max(scan_rows)
        if scan_rows
        else 0
    )

    nodes_over_10k = sum(
        1
        for r in all_rows
        if r >= 10000
    )

    nodes_over_100k = sum(
        1
        for r in all_rows
        if r >= 100000
    )

    return {
        # ----------------------------------------------------
        # Existing information
        # ----------------------------------------------------

        "total_cost": plan_json.get(
            "Total Cost",
            0
        ),

        "plan_rows": plan_json.get(
            "Plan Rows",
            0
        ),

        "operators": counts,

        # ----------------------------------------------------
        # Cardinality information
        # ----------------------------------------------------

        "max_intermediate_rows":
            max_intermediate_rows,

        "max_join_output":
            max_join_output,

        "max_scan_rows":
            max_scan_rows,

        "nodes_over_10k":
            nodes_over_10k,

        "nodes_over_100k":
            nodes_over_100k,

        # ----------------------------------------------------
        # Version C: access paths
        # ----------------------------------------------------

        "scans": scan_nodes,

        # ----------------------------------------------------
        # Version C: compact join structure
        # ----------------------------------------------------

        "joins": join_nodes,
    }
def get_arm_plan_summary(cur, arm_idx, query):
    set_arm(cur, arm_idx)

    cur.execute(f"EXPLAIN (FORMAT JSON) {query}")

    result = cur.fetchone()[0]

    if isinstance(result, str):
        result = json.loads(result)

    plan = result[0]["Plan"]

    return summarize_plan(plan)


# ============================================================
# Prompt construction
# ============================================================

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

        # ----------------------------------------------------
        # Operator counts
        # ----------------------------------------------------

        ops = ", ".join(
            f"{k} x{v}"
            for k, v in s["operators"].items()
        )

        # ----------------------------------------------------
        # Access paths
        # ----------------------------------------------------

        scans_txt = "; ".join(
            f"{x['alias'] or x['relation']}="
            f"{x['type']}"
            f"{'[' + x['index'] + ']' if x['index'] else ''}"
            f"(rows={x['rows']})"
            for x in s["scans"]
        )

        if not scans_txt:
            scans_txt = "none"

        # ----------------------------------------------------
        # Compact join structure
        # ----------------------------------------------------

        joins_txt = "; ".join(
            f"{' + '.join(j['children'])}"
            f" -> {j['type']}"
            f"(rows={j['rows']})"
            for j in s["joins"]
        )

        if not joins_txt:
            joins_txt = "none"

        # ----------------------------------------------------
        # Build arm description
        # ----------------------------------------------------

        arms_txt += (
            f"\nArm {arm} "
            f"[disables: {arm_disables_text(arm)}]\n"

            f"  Estimated cost: "
            f"{s['total_cost']}, "
            f"final rows: "
            f"{s['plan_rows']}\n"

            f"  Plan shape: "
            f"{ops}\n"

            f"  Cardinality: "
            f"max intermediate="
            f"{s['max_intermediate_rows']}, "
            f"max join output="
            f"{s['max_join_output']}, "
            f"max scan rows="
            f"{s['max_scan_rows']}, "
            f">10K nodes="
            f"{s['nodes_over_10k']}, "
            f">100K nodes="
            f"{s['nodes_over_100k']}\n"

            f"  Access: "
            f"{scans_txt}\n"

            f"  Joins: "
            f"{joins_txt}\n"
        )

    return f"""Query:
{query}

Referenced tables:
{tables_txt}

Candidate execution strategies:
{arms_txt}
"""


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """You are assisting a PostgreSQL query optimizer
research system.

You will be shown one SQL query and five candidate execution strategies
("arms").

Each arm represents a different set of PostgreSQL join and scan
operators that are allowed to be used.

For each arm, you are given:

- PostgreSQL estimated total cost
- PostgreSQL estimated final result rows
- operator counts
- compact intermediate-cardinality information
- table access paths
- compact join structure
- information about the referenced tables

Important:

PostgreSQL's estimated cost is only a heuristic.
It does not necessarily equal actual execution time.

Use all available information when estimating which arm is likely to
execute fastest.

Pay particular attention to:

1. The overall plan shape.
2. Intermediate cardinalities, not only final result rows.
3. Large join outputs.
4. Large scan outputs.
5. Access paths such as sequential scans, index scans,
   index-only scans, and bitmap scans.
6. Which indexes are being used.
7. Join algorithms such as nested loop, hash join, and merge join.
8. Whether nested loops may repeatedly process expensive inner plans.
9. Sort and materialization operators.
10. PostgreSQL estimated cost as supporting evidence, not as the
    sole decision criterion.

The "Access" information describes which relations are accessed and
which access method or index PostgreSQL selected.

The "Joins" information describes the immediate children of each join
node. It is a compact structural summary and may not reconstruct the
entire plan tree.

Your task is to estimate the probability that each arm is the
fastest-executing option.

Return a probability distribution over the five arms.

Respond with ONLY a JSON object in exactly this format:

{"0": p0, "1": p1, "2": p2, "3": p3, "4": p4}

The probabilities must sum to 1.0.

Do not include markdown.
Do not include explanations.
Do not include any other text.
"""


# ============================================================
# Ollama API call
# ============================================================

def call_llm(prompt):
    """
    Call Qwen3:8b through the Ollama API running on Windows.

    WSL connects to the Windows Ollama server using:
        http://172.29.64.1:11434
    """

    url = f"{OLLAMA_BASE_URL}/api/chat"

    full_prompt = SYSTEM_PROMPT + "\n\n" + prompt

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,

        # Disable Qwen3 thinking for this classification task.
        # This makes responses substantially faster and keeps the
        # teacher focused on returning the required JSON distribution.
        "think": False,

        # Encourage deterministic teacher labels.
        "options": {
            "temperature": 0
        },

        # Ask Ollama to return JSON.
        "format": "json",
    }

    max_retries = 5

    for attempt in range(max_retries):

        try:

            response = requests.post(
                url,
                json=data,
                timeout=180,
            )

            # Retry temporary Ollama/server errors
            if response.status_code in [429, 500, 502, 503, 504]:

                wait_time = 5 * (attempt + 1)

                print(
                    f"  Ollama returned HTTP {response.status_code}. "
                    f"Retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/{max_retries})...",
                    flush=True,
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()

            result = response.json()

            # Ollama /api/chat response format:
            #
            # {
            #   "message": {
            #       "role": "assistant",
            #       "content": "..."
            #   }
            # }

            text = result["message"]["content"].strip()

            # Remove accidental Markdown code fences
            text = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                text.strip(),
                flags=re.IGNORECASE,
            ).strip()

            llm_result = json.loads(text)

            # ------------------------------------------------
            # Basic validation
            # ------------------------------------------------

            if not isinstance(llm_result, dict):
                raise ValueError(
                    "LLM response is not a JSON object."
                )

            expected_keys = {"0", "1", "2", "3", "4"}

            if set(llm_result.keys()) != expected_keys:
                raise ValueError(
                    f"Unexpected JSON keys: {llm_result.keys()}"
                )

            probabilities = []

            for key in ["0", "1", "2", "3", "4"]:

                value = float(llm_result[key])

                if value < 0 or value > 1:
                    raise ValueError(
                        f"Invalid probability for arm {key}: {value}"
                    )

                probabilities.append(value)

            total = sum(probabilities)

            # Qwen/Ollama should normally return exactly 1.0,
            # but normalize small floating-point deviations.
            if total <= 0:
                raise ValueError(
                    "Probability sum is zero."
                )

            if abs(total - 1.0) > 0.001:

                print(
                    f"  Warning: probability sum = {total:.6f}. "
                    f"Normalizing.",
                    flush=True,
                )

                for key in ["0", "1", "2", "3", "4"]:
                    llm_result[key] = (
                        float(llm_result[key]) / total
                    )

            return llm_result

        except requests.exceptions.RequestException as e:

            if attempt == max_retries - 1:
                raise

            wait_time = 5 * (attempt + 1)

            print(
                f"  Ollama request error: {e}. "
                f"Retrying in {wait_time}s...",
                flush=True,
            )

            time.sleep(wait_time)

        except (json.JSONDecodeError, KeyError, ValueError) as e:

            # Invalid model output is worth retrying because
            # the model may occasionally produce malformed JSON.
            if attempt == max_retries - 1:
                raise

            print(
                f"  Invalid Ollama response: {e}. "
                f"Retrying...",
                flush=True,
            )

            time.sleep(2)

    raise RuntimeError(
        f"Ollama request failed after {max_retries} attempts."
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Check Ollama connection before doing PostgreSQL work.
    # --------------------------------------------------------

    print(
        f"Testing Ollama connection at "
        f"{OLLAMA_BASE_URL}...",
        flush=True,
    )

    try:

        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=10,
        )

        response.raise_for_status()

        models = response.json().get("models", [])

        model_names = [
            m.get("name")
            for m in models
        ]

        print(
            f"Ollama connection successful.",
            flush=True,
        )

        print(
            f"Available models: {model_names}",
            flush=True,
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not connect to Ollama at "
            f"{OLLAMA_BASE_URL}. "
            f"Original error: {e}"
        )

    # --------------------------------------------------------
    # Verify Qwen3 is available.
    # --------------------------------------------------------

    if MODEL not in model_names:

        raise RuntimeError(
            f"Model '{MODEL}' is not available in Ollama.\n"
            f"Available models: {model_names}\n"
            f"Run 'ollama list' on Windows to check."
        )

    print(
        f"Using Ollama model: {MODEL}",
        flush=True,
    )

    # --------------------------------------------------------
    # Load validation queries.
    # --------------------------------------------------------

    query_files = json.load(
        open(QUERIES_FILE)
    )

    conn = psycopg2.connect(PG_CONN_STR)
    conn.autocommit = True

    cur = conn.cursor()

    results = []

    for qi, qf in enumerate(query_files):

        with open(qf) as f:
            query = f.read().strip().rstrip(";")

        tables = extract_tables(query)

        try:

            # -------------------------------------------------
            # Warm cache identically to ground truth generation.
            # Arm 0, one execution.
            # -------------------------------------------------

            set_arm(cur, 0)

            cur.execute(
                f"EXPLAIN (ANALYZE, FORMAT JSON) {query}"
            )

            cur.fetchone()

            # -------------------------------------------------
            # Gather database statistics.
            # -------------------------------------------------

            table_stats = get_table_stats(
                cur,
                tables,
            )

            # -------------------------------------------------
            # Get PostgreSQL's plan under each candidate arm.
            # -------------------------------------------------

            arm_summaries = {
                arm: get_arm_plan_summary(
                    cur,
                    arm,
                    query,
                )
                for arm in range(5)
            }

            # -------------------------------------------------
            # Build teacher prompt.
            # -------------------------------------------------

            prompt = build_prompt(
                query,
                table_stats,
                arm_summaries,
            )

            # -------------------------------------------------
            # Print complete prompt before sending to Ollama.
            # -------------------------------------------------

            print("\n" + "=" * 80, flush=True)
            print(
                f"QUERY [{qi}]: {qf}",
                flush=True,
            )

            print(
                "PROMPT SENT TO OLLAMA:",
                flush=True,
            )

            print("=" * 80, flush=True)

            print(
                SYSTEM_PROMPT + "\n\n" + prompt,
                flush=True,
            )

            print(
                "=" * 80 + "\n",
                flush=True,
            )

            # -------------------------------------------------
            # Query Qwen3 through Ollama.
            # -------------------------------------------------

            llm_dist = call_llm(prompt)

            # -------------------------------------------------
            # Save result.
            # Store arm_summaries too so Version B can be
            # analyzed later without rerunning PostgreSQL.
            # -------------------------------------------------

            results.append(
                {
                    "query_file": qf,
                    "llm_distribution": llm_dist,
                    "arm_summaries": arm_summaries,
                    "prompt": prompt,
                }
            )

            print(
                f"[{qi}] {qf} -> {llm_dist}",
                flush=True,
            )

            # -------------------------------------------------
            # Wait 10 seconds after successful response.
            # -------------------------------------------------

            if qi < len(query_files) - 1:

                print(
                    f"  Waiting {WAIT_AFTER_SUCCESS}s "
                    f"before next query...",
                    flush=True,
                )

                time.sleep(
                    WAIT_AFTER_SUCCESS
                )

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

        # -----------------------------------------------------
        # Save after every query so that a failed/interrupted
        # run does not lose previous results.
        # -----------------------------------------------------

        with open(
            OUTPUT_FILE,
            "w"
        ) as f:

            json.dump(
                results,
                f,
                indent=2,
            )

    print(
        f"Done. Wrote {len(results)} entries to "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
