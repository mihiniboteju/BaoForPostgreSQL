"""
run_pipeline.py

One query, one authoritative measurement, many cheap LLM reads.

For each query:

  1. ANALYZE the tables it touches (once, up front, so the plans we record
     are the plans that actually execute)
  2. EXPLAIN the query under all 5 arms -> up to 5 hints -> unique plans
  3. extract statistics
  4. really execute every unique plan and record latency
  5. build the prompt once per --row-source variant and ask OpenRouter
  6. write one JSON record per query

Timing is the expensive part (seconds per query); the API call costs cents.
So timings are cached in their own file and every model sweep reuses them
without touching the database again.

Measurement choices that matter:

  * ``enable_bao`` is off and only the arm's own GUCs are on, so what is timed
    is the arm's plan and not Bao's policy;
  * each unique plan is timed once, then the latency is copied to every arm
    that shares it (arms 0 and 2 usually collide);
  * one untimed warm-up pass, then ``--reps`` timed passes with the starting
    plan rotated each pass, so cache warming cannot favour one plan;
  * a timeout is recorded as censored rather than dropped, which keeps the
    query usable;
  * median and min are both kept: min is the more robust statistic on a
    shared machine.

Usage:

    export OPENROUTER_API_KEY=sk-or-v1-...
    python run_pipeline.py --limit 10 --models qwen/qwen3.6-35b-a3b

Outputs:
    runs/<query>.json      full record for one query (resumable)
    timings.json           measured latencies only, reused by later sweeps
    labels_<model>.json    LLM labels, one file per model
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import psycopg2

from bao_arm_plans import (
    ARMS,
    DSN,
    KNOBS,
    collect_statistics,
    format_compact_prompt,
    get_plan,
    get_verbose_plan,
    plan_groups,
    query_aliases,
    serialize,
)
from llm_ranker import LETTERS, SYSTEM_PROMPT, build_prompt, call_openrouter

from psycopg2 import sql as sql_ident

ROW_SOURCES = ("plan", "table", "both")


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------


def set_session(cur: Any, settings: Mapping[str, str]) -> None:
    """Put the session into exactly one arm's configuration, Bao disabled."""
    try:
        cur.execute("SET enable_bao TO off")
    except psycopg2.Error:
        pass                      # Bao extension not loaded; arms still apply
    for knob in KNOBS:
        cur.execute("SET %s = %s" % (knob, "off"))
    for knob, value in settings.items():
        cur.execute("SET %s = %s" % (knob, value))


def rollback(cur: Any) -> None:
    """
    Clear a failed transaction.

    A single bad statement leaves the connection in "current transaction is
    aborted" state, and every later statement fails until a rollback. Without
    this, one missing table aborts every query that follows it.
    """
    try:
        if cur.connection is not None:
            cur.connection.rollback()
    except psycopg2.Error:
        pass


def analyze_tables(cur: Any, query: str) -> list[str]:
    """ANALYZE every table the query references, before any plan is taken."""
    present = []
    for table in sorted(set(query_aliases(query).values())):
        cur.execute("SELECT to_regclass(%s)", ["public." + table])
        if cur.fetchone()[0] is None:
            print(f"    skip ANALYZE {table}: not a relation in public", flush=True)
            continue
        try:
            cur.execute(sql_ident.SQL("ANALYZE public.{}").format(sql_ident.Identifier(table)))
            present.append(table)
        except psycopg2.Error as error:
            rollback(cur)
            print(f"    ANALYZE {table} failed: {str(error).strip()[:80]}", flush=True)
    return present


def execute_once(cur: Any, query: str, settings: Mapping[str, str],
                 timeout_ms: int) -> float | None:
    """Run the query for real; return milliseconds, or None if it timed out."""
    try:
        cur.execute("SET statement_timeout = %d" % timeout_ms)
        set_session(cur, settings)
        cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + query)
        result = cur.fetchone()[0]
        if isinstance(result, str):
            result = json.loads(result)
        node = result[0]
        elapsed = node.get("Execution Time")
        if elapsed is None:
            elapsed = node.get("Plan", {}).get("Actual Total Time")
        return float(elapsed) if elapsed is not None else None
    except psycopg2.errors.QueryCanceled:
        return None
    except psycopg2.Error as error:
        rollback(cur)
        print(f"    execution failed: {str(error).strip()[:100]}", flush=True)
        return None
    finally:
        try:
            cur.execute("RESET ALL")
        except psycopg2.Error:
            rollback(cur)


def time_plans(cur: Any, query: str, plan_settings: list[Mapping[str, str]],
               reps: int, timeout_ms: int) -> list[dict[str, Any]]:
    """Warm up once, then time each plan `reps` times with a rotating order."""
    n_plans = len(plan_settings)
    samples: list[list[float]] = [[] for _ in range(n_plans)]
    timed_out = [False] * n_plans

    for index in range(n_plans):                       # untimed warm-up
        execute_once(cur, query, plan_settings[index], timeout_ms)

    for rep in range(reps):
        for offset in range(n_plans):                 # rotated so no plan is always first
            index = (rep + offset) % n_plans
            elapsed = execute_once(cur, query, plan_settings[index], timeout_ms)
            if elapsed is None:
                timed_out[index] = True
            else:
                samples[index].append(elapsed)

    results = []
    for index in range(n_plans):
        values = samples[index]
        results.append({
            "samples_ms": values,
            "median_ms": round(statistics.median(values), 3) if values else None,
            "min_ms": round(min(values), 3) if values else None,
            "spread_ms": round(max(values) - min(values), 3) if len(values) > 1 else 0.0,
            "timed_out": timed_out[index],
        })
    return results


# ---------------------------------------------------------------------------
# prompt + LLM
# ---------------------------------------------------------------------------


def candidate_distribution(response: Mapping[str, Any], order: list[int]) -> dict[int, float]:
    """Map the per-letter probabilities back onto real candidate indices."""
    probs = response.get("letter_probs") or {}
    out: dict[int, float] = {}
    for letter_index, candidate in enumerate(order):
        letter = LETTERS[letter_index]
        if letter in probs:
            out[candidate] = float(probs[letter])
    return out


def run_model(cur_query: Mapping[str, Any], model: str, api_key: str,
              row_sources: list[str], temperature: float, max_tokens: int) -> list[dict[str, Any]]:
    """One prompt per row-source variant, all sharing the same letter order."""
    records = []
    for row_source in row_sources:
        prompt, order = build_prompt(cur_query, cur_query["query_file"], 0, row_source)
        response = call_openrouter(
            prompt, model, api_key, len(order), temperature, max_tokens,
            include_raw=True, reasoning=False,
        )
        distribution = candidate_distribution(response, order)
        if not distribution:
            # No logprobs from this provider; fall back to the argmax letter so
            # the query still yields a ranking instead of being dropped.
            letter = response.get("chosen")
            if letter and letter in LETTERS[: len(order)]:
                distribution = {order[LETTERS.index(letter)]: 1.0}
        best = max(distribution, key=distribution.get) if distribution else None
        arm_to_plan = {int(a): i for a, i in cur_query["arm_to_plan"].items()}
        records.append({
            "row_source": row_source,
            "model": model,
            "prompt": prompt,
            "letter_order": order,
            "system_prompt": SYSTEM_PROMPT,
            "response": response,
            "distribution": distribution,
            "chosen_candidate": best,
            "predicted_arms": sorted(a for a, c in arm_to_plan.items() if c == best),
        })
    return records


# ---------------------------------------------------------------------------


def build_record(query_path: Path, hints: dict[int, str], default_hint: str,
                 unique_plans: list[str], statistics_block: Mapping[str, Any],
                 timings: list[dict[str, Any]], arm_to_plan: Mapping[str, int],
                 model_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble everything known about one query."""
    by_arm: dict[int, dict[str, Any]] = {}
    for arm, candidate in arm_to_plan.items():
        timing = timings[candidate] if candidate < len(timings) else {}
        by_arm[arm] = {
            "plan_index": candidate,
            "median_ms": timing.get("median_ms"),
            "min_ms": timing.get("min_ms"),
        }

    scored = {a: t["median_ms"] for a, t in by_arm.items() if t["median_ms"] is not None}
    ordered = sorted(scored, key=lambda a: scored[a])
    best_arm = ordered[0] if ordered else None
    ratio = None
    if len(ordered) > 1 and scored[ordered[1]] > 0:
        ratio = round(scored[ordered[0]] / scored[ordered[1]], 4)

    return {
        "query_file": str(query_path),
        "arms": {str(a): h for a, h in hints.items()},
        "default": default_hint,
        "unique_arm_plans": unique_plans,
        "arm_to_plan": {str(a): c for a, c in arm_to_plan.items()},
        "statistics": statistics_block,
        "timings": timings,
        "arm_latency": by_arm,
        "best_arm": best_arm,
        "best_over_second": ratio,
        "labels": model_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-file", type=Path, default=Path("validation_queries.json"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=["qwen/qwen3.6-35b-a3b"])
    parser.add_argument("--row-sources", nargs="+", default=list(ROW_SOURCES), choices=ROW_SOURCES)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--dsn", default=DSN)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--timings-out", type=Path, default=Path("timings.json"))
    parser.add_argument("--plan-only", action="store_true",
                        help="measure timings but do not call OpenRouter")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.plan_only:
        raise SystemExit(f"set {args.api_key_env}=sk-or-v1-... or pass --plan-only")

    queries = json.loads(args.queries_file.read_text(encoding="utf-8"))[: args.limit]
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(queries)} queries | {args.reps} reps | "
          f"row sources: {', '.join(args.row_sources)} | models: {', '.join(args.models)}\n")

    timings_all: dict[str, Any] = {}
    if args.timings_out.exists():
        timings_all = json.loads(args.timings_out.read_text(encoding="utf-8"))

    started = time.time()
    with psycopg2.connect(args.dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for i, query_path in enumerate(queries, 1):
                path = Path(query_path)
                query = path.read_text(encoding="utf-8").strip().rstrip(";").rstrip()
                print(f"[{i}/{len(queries)}] {path.name}", flush=True)

                try:
                    analyze_tables(cur, query)
                    hints = {arm: serialize(get_plan(cur, query, settings))
                             for arm, settings in ARMS.items()}
                    unique_plans = sorted(set(hints.values()))
                    default_plan = get_verbose_plan(cur, query)
                    default_hint = serialize(default_plan)
                    stats = collect_statistics(cur, default_plan, query)
                except psycopg2.Error as error:
                    rollback(cur)
                    print(f"    PLANNING FAILED: {str(error).strip()[:120]}", flush=True)
                    continue

                groups = plan_groups({
                    "arms": {str(a): h for a, h in hints.items()},
                    "unique_arm_plans": unique_plans,
                })
                arm_to_plan = {arm: index
                               for index, (_, arm_list, _) in enumerate(groups)
                               for arm in arm_list}

                # Each unique plan is timed under the settings of the first arm
                # that produced it, which is what reproduces that exact plan.
                first_arm = {}
                for arm in sorted(arm_to_plan):
                    first_arm.setdefault(arm_to_plan[arm], arm)
                plan_settings = [ARMS[first_arm[c]] for c in range(len(unique_plans))]

                print(f"    {len(unique_plans)} unique plans, {len(stats['tables'])} tables "
                      f"— timing ({args.reps} reps x {len(unique_plans)} plans)...", flush=True)
                timings = time_plans(cur, query, plan_settings, args.reps, args.timeout_ms)
                for c, timing in enumerate(timings):
                    print(f"      plan {c}: median={timing['median_ms']} "
                          f"spread={timing['spread_ms']} "
                          f"{'TIMEOUT' if timing['timed_out'] else ''}", flush=True)

                labels: list[dict[str, Any]] = []
                if not args.plan_only and api_key:
                    for model in args.models:
                        labels += run_model(
                            {"query_file": str(path), "arms": {str(a): h for a, h in hints.items()},
                             "unique_arm_plans": unique_plans,
                             "statistics": stats,
                             "arm_to_plan": {str(a): c for a, c in arm_to_plan.items()}},
                            model, api_key, args.row_sources,
                            args.temperature, args.max_tokens,
                        )

                record = build_record(path, hints, default_hint, unique_plans,
                                     stats, timings, arm_to_plan, labels)
                out_path = args.runs_dir / (path.stem + ".json")
                out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

                timings_all[str(path)] = {
                    "unique_arm_plans": unique_plans,
                    "arm_to_plan": record["arm_to_plan"],
                    "timings": timings,
                    "best_arm": record["best_arm"],
                    "best_over_second": record["best_over_second"],
                }
                args.timings_out.write_text(json.dumps(timings_all, indent=2) + "\n",
                                            encoding="utf-8")
                print(f"    best arm {record['best_arm']} "
                      f"(best/2nd = {record['best_over_second']}) "
                      f"-> {out_path}  [{time.time() - started:.0f}s]", flush=True)

    print(f"\nDone in {time.time() - started:.0f}s. "
          f"Timings in {args.timings_out}, records in {args.runs_dir}/")


if __name__ == "__main__":
    main()