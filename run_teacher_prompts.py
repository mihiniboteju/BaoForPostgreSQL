"""
run_teacher_prompts.py

End-to-end teacher run on a small slice of the validation queries:

    validation_queries.json
        -> PostgreSQL  : EXPLAIN the query under all 5 arms, collect statistics
        -> prompt      : compact statistics + letter-labelled candidates
        -> OpenRouter  : one chat completion per query, read the letter logprobs
        -> output      : the exact prompt sent AND the raw response received

Everything needed to audit a label is written to disk: if a model later turns
out to have reasoned badly, the prompt that caused it is still there.

The script reuses bao_arm_plans for extraction and llm_ranker for the API
call, so there is only one implementation of each step.

Usage:

    export OPENROUTER_API_KEY=sk-or-v1-...

    # build the prompts and save them WITHOUT calling the API
    python run_teacher_prompts.py --model openai/gpt-4.1-mini --dry-run

    # real run on the first 5 validation queries
    python run_teacher_prompts.py --model openai/gpt-4.1-mini

Outputs:
    teacher_runs.json   full records (prompt, response, parsed distribution)
    teacher_runs.txt    the same, human readable
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import psycopg2

from bao_arm_plans import DSN, format_compact_prompt, plan_groups, process_file
from llm_ranker import LETTERS, SYSTEM_PROMPT, build_prompt, call_openrouter

SEPARATOR = "=" * 78


def build_entry(hints, default_hint, unique_plans, statistics) -> dict[str, Any]:
    """Assemble one entry in the bao_arm_plans.json shape."""
    entry: dict[str, Any] = {
        "arms": {str(arm): hint for arm, hint in hints.items()},
        "default": default_hint,
        "unique_arm_plans": unique_plans,
        "statistics": statistics,
    }
    entry["arm_to_plan"] = {
        str(arm): index
        for index, (_, arm_list, _) in enumerate(plan_groups(entry))
        for arm in arm_list
    }
    return entry


def render_text(records: list[dict[str, Any]]) -> str:
    """Human readable dump: the exact prompt, then the exact response."""
    chunks = []
    for record in records:
        parts = [
            SEPARATOR,
            f"QUERY      : {record['query_file']}",
            f"MODEL      : {record['model']}",
            f"CANDIDATES : {record['n_plans']} unique plans",
            f"ORDER      : letter index -> candidate index = {record['letter_order']}",
            SEPARATOR,
            "",
            "--- SYSTEM MESSAGE ---",
            record["system_prompt"],
            "",
            "--- USER MESSAGE (exactly what was sent) ---",
            record["prompt"],
            "",
            "--- RESPONSE ---",
        ]
        if record.get("dry_run"):
            parts.append("(dry run: no API call was made)")
        else:
            parsed = record["response"]
            parts.append(f"text           : {parsed.get('text')!r}")
            parts.append(f"logprobs used  : {parsed.get('supported')}")
            parts.append(f"letter probs   : {parsed.get('letter_probs')}")
            parts.append(f"residual mass  : {parsed.get('residual_mass')}")
            parts.append(f"error          : {parsed.get('error')}")
            parts.append("")
            parts.append("--- RAW RESPONSE (exactly what came back) ---")
            parts.append(json.dumps(parsed.get("raw_response"), indent=2))
        chunks.append("\n".join(parts))
    return "\n\n".join(chunks) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-file", type=Path, default=Path("validation_queries.json"))
    parser.add_argument("--limit", type=int, default=5, help="how many queries to run")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dsn", default=DSN)
    parser.add_argument("--out-json", type=Path, default=Path("teacher_runs.json"))
    parser.add_argument("--out-txt", type=Path, default=Path("teacher_runs.txt"))
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true",
                        help="build and save prompts but do not call the API")
    parser.add_argument("--thinking", action="store_true",
                        help="keep the model's reasoning mode on (needs --max-tokens > 1)")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.dry_run:
        raise SystemExit(
            f"set {args.api_key_env} first:\n"
            f"    export {args.api_key_env}=sk-or-v1-...\n"
            f"(use --dry-run to build prompts without a key)"
        )

    queries = json.loads(args.queries_file.read_text(encoding="utf-8"))[: args.limit]
    print(f"{len(queries)} queries from {args.queries_file}")
    print(f"model: {args.model}{'  [DRY RUN - no API calls]' if args.dry_run else ''}\n")

    records: list[dict[str, Any]] = []
    started = time.time()

    with psycopg2.connect(args.dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for i, query_path in enumerate(queries, 1):
                path = Path(query_path)
                print(f"[{i}/{len(queries)}] {path.name} ...", end=" ", flush=True)

                try:
                    hints, default_hint, unique_plans, statistics = process_file(cur, path)
                except Exception as error:                       # noqa: BLE001
                    print(f"EXTRACTION FAILED: {error}", flush=True)
                    continue

                entry = build_entry(hints, default_hint, unique_plans, statistics)
                prompt, order = build_prompt(entry, str(path), rotation=0)
                n_plans = len(unique_plans)

                if args.dry_run:
                    response: dict[str, Any] = {"ok": True, "dry_run": True}
                else:
                    response = call_openrouter(
                        prompt, args.model, api_key, n_plans,
                        args.temperature, args.max_tokens,
                        include_raw=True, reasoning=args.thinking,
                    )

                chosen_letter = response.get("chosen")
                chosen_candidate = None
                if chosen_letter and chosen_letter in LETTERS[:n_plans]:
                    chosen_candidate = order[LETTERS.index(chosen_letter)]
                predicted_arms = (
                    sorted(int(a) for a, idx in entry["arm_to_plan"].items()
                           if idx == chosen_candidate)
                    if chosen_candidate is not None else []
                )

                record = {
                    "query_file": str(path),
                    "model": args.model,
                    "n_plans": n_plans,
                    "letter_order": order,
                    "system_prompt": SYSTEM_PROMPT,
                    "prompt": prompt,
                    "compact_block": format_compact_prompt(entry),
                    "arms": entry["arms"],
                    "arm_to_plan": entry["arm_to_plan"],
                    "unique_arm_plans": unique_plans,
                    "chosen_letter": chosen_letter,
                    "chosen_candidate": chosen_candidate,
                    "predicted_arms": predicted_arms,
                    "response": response,
                    "dry_run": args.dry_run,
                }
                records.append(record)

                if args.dry_run:
                    print(f"{n_plans} plans, prompt {len(prompt)} chars "
                          f"({time.time() - started:.0f}s)", flush=True)
                elif response.get("ok"):
                    probs = response.get("letter_probs") or {}
                    best = max(probs, key=probs.get) if probs else "?"
                    print(f"{n_plans} plans -> letter {best} "
                          f"(p={probs.get(best, 0):.3f}, "
                          f"residual={response.get('residual_mass')})", flush=True)
                else:
                    print(f"API ERROR: {response.get('error')}", flush=True)

                # Checkpoint after every query.
                args.out_json.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
                args.out_txt.write_text(render_text(records), encoding="utf-8")

    print(f"\nWrote {len(records)} records to {args.out_json} and {args.out_txt}")
    if not args.dry_run and records:
        print(f"predicted arms: {[r['predicted_arms'] for r in records]}")
        residuals = [r["response"].get("residual_mass") for r in records
                     if r["response"].get("supported")]
        if residuals:
            print(f"mean residual mass outside A-D: {sum(residuals) / len(residuals):.4f}")


if __name__ == "__main__":
    main()