"""
llm_ranker.py

Ask a model which candidate plan is fastest, and read the answer off the
next-token distribution instead of parsing prose.

Why logprobs rather than "reply with a JSON object of probabilities":

  * one letter is the same across every provider, so a model sweep measures
    reasoning rather than JSON formatting failures;
  * softmax over the logprobs of A/B/C/D gives the ranking *and* a soft
    distribution in a single request (max_tokens=1);
  * if a provider does not expose logprobs we still get the argmax letter,
    so the ranking degrades gracefully instead of failing.

Two guards against position bias, which is large in multiple-choice tasks:

  * the candidates are presented in a seeded shuffle, because
    ``unique_arm_plans`` is sorted alphabetically and would otherwise tie
    "letter A" to one plan family;
  * ``--rotations`` runs cyclic permutations and averages the distributions
    back onto the real candidates.

Usage:

    export OPENROUTER_API_KEY=sk-or-...
    python llm_ranker.py --plans bao_arm_plans.json \
        --model openai/gpt-4.1-mini --out labels_gpt41mini.json
    python llm_ranker.py --plans bao_arm_plans.json \
        --model google/gemini-2.5-flash --rotations 3

Note: logprob support varies per provider on OpenRouter. Missing logprobs
are recorded as ``supported: false`` rather than raising.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import requests

from bao_arm_plans import format_compact_prompt, plan_groups

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
LETTERS = "ABCDEFGHIJ"

SYSTEM_PROMPT = (
    "You are an expert PostgreSQL query optimizer. You reason about join "
    "orders, access paths and selectivity using table statistics. You answer "
    "with a single letter and nothing else."
)

INSTRUCTION = (
    "Which one of the candidate plans above will execute FASTEST on this "
    "database? Consider the access paths, the join order, and how selective "
    "the filters are. Answer with exactly one letter ({letters}) and nothing "
    "else."
)


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------


def order_for_rotation(n_plans: int, seed: str, rotation: int) -> list[int]:
    """
    Permutation of candidate indices, deterministic for a given query.

    Rotation 0 is a seeded shuffle so that the alphabetical order of
    ``unique_arm_plans`` never correlates with the letter assignment; further
    rotations are cyclic shifts of it, which is what makes the position-bias
    estimate valid.
    """
    if rotation == 0:
        order = list(range(n_plans))
        random.Random(seed).shuffle(order)
        return order
    base = order_for_rotation(n_plans, seed, 0)
    shift = rotation % n_plans
    return base[shift:] + base[:shift]


def build_prompt(
    entry: Mapping[str, Any],
    seed: str,
    rotation: int,
    row_source: str = "both",
) -> tuple[str, list[int]]:
    """Return (prompt, order) where order[letter_index] = candidate index."""
    n_plans = len(plan_groups(entry))
    order = order_for_rotation(n_plans, seed, rotation)
    letters = [LETTERS[i] for i in range(n_plans)]
    # ``labels`` is indexed by CANDIDATE (format_compact_prompt walks the
    # candidates in their own order), while ``order`` maps letter -> candidate.
    # The two must be transposes of each other, otherwise the prompt shows a
    # letter that does not mean the row it is printed next to.
    labels: list[str | None] = [None] * n_plans
    for letter_index, candidate in enumerate(order):
        labels[candidate] = letters[letter_index]
    block = format_compact_prompt(
        entry,
        include_query=True,
        labels=labels,
        row_source=row_source,
    )
    instruction = INSTRUCTION.format(letters=", ".join(letters))
    return block + "\n\n" + instruction, order


# ---------------------------------------------------------------------------
# OpenRouter call
# ---------------------------------------------------------------------------


def call_openrouter(
    prompt: str,
    model: str,
    api_key: str,
    n_letters: int,
    temperature: float,
    max_tokens: int,
    retries: int = 4,
    include_raw: bool = False,
    reasoning: bool = False,
) -> dict[str, Any]:
    """
    One chat completion; returns parsed letter logprobs and raw text.

    Qwen3.x and similar hybrid models emit thinking tokens first. With
    max_tokens=1 that would make us read the logprobs of a thinking token
    instead of the answer letter, so reasoning is switched off by default.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 20,
    }
    if not reasoning:
        payload["reasoning"] = {"enabled": False}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/baolab/BaoForPostgreSQL",
        "X-Title": "Bao LLM plan ranker",
    }

    last_error = None
    for attempt in range(retries):
        try:
            response = requests.post(ENDPOINT, json=payload, headers=headers, timeout=120)
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(3 * (attempt + 1))
                continue
            response.raise_for_status()
            body = response.json()
            parsed = parse_response(body, n_letters)
            if include_raw:
                parsed["raw_response"] = body
            return parsed
        except Exception as error:                      # noqa: BLE001
            last_error = error
            time.sleep(2 * (attempt + 1))
    return {"ok": False, "error": str(last_error)}


def parse_response(body: Mapping[str, Any], n_letters: int) -> dict[str, Any]:
    """Pull the letter logprobs out of a chat completion response."""
    try:
        choice = body["choices"][0]
    except (KeyError, IndexError, TypeError):
        return {"ok": False, "error": "malformed response", "raw": str(body)[:300]}

    text = (choice.get("message") or {}).get("content") or ""
    top = ((choice.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs")
    if not top:
        # Provider does not expose logprobs: keep the letter as a hard answer.
        return {
            "ok": True,
            "supported": False,
            "text": text.strip(),
            "letter_logprobs": {},
            "chosen": next((c for c in text.strip() if c in LETTERS[:n_letters]), None),
        }

    logprobs: dict[str, float] = {}
    for item in top:
        token = (item.get("token") or "").strip()
        if len(token) == 1 and token in LETTERS[:n_letters]:
            logprobs[token] = logprobs.get(token, -math.inf)
            logprobs[token] = max(logprobs[token], float(item.get("logprob", -math.inf)))

    if not logprobs:
        return {"ok": False, "error": "no letter token in top_logprobs", "raw": str(body)[:300]}

    top_value = max(logprobs.values())
    weights = {k: math.exp(v - top_value) for k, v in logprobs.items()}
    total = sum(weights.values())
    probs = {k: v / total for k, v in weights.items()}
    return {
        "ok": True,
        "supported": True,
        "text": text.strip(),
        "letter_logprobs": logprobs,
        "letter_probs": probs,
        "residual_mass": round(1.0 - sum(probs.values()), 6),
        "chosen": max(probs, key=probs.get),
    }


# ---------------------------------------------------------------------------
# scoring one query
# ---------------------------------------------------------------------------


def score_entry(
    entry: Mapping[str, Any],
    query_key: str,
    model: str,
    api_key: str,
    rotations: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    n_plans = len(plan_groups(entry))
    n_rot = min(rotations, n_plans) if rotations > 0 else 1

    totals = [0.0] * n_plans
    per_rotation: list[dict[str, Any]] = []
    letter_picks: list[str] = []
    supported_flags: list[bool] = []

    for rotation in range(n_rot):
        prompt, order = build_prompt(entry, query_key, rotation)
        result = call_openrouter(prompt, model, api_key, n_plans, temperature, max_tokens)
        record: dict[str, Any] = {"rotation": rotation, "order": order, "result": result}
        per_rotation.append(record)

        if not result.get("ok"):
            continue
        supported_flags.append(bool(result.get("supported")))
        chosen = result.get("chosen")
        if chosen:
            letter_picks.append(chosen)
        probs = result.get("letter_probs") or {}
        for letter_index, candidate in enumerate(order):
            letter = LETTERS[letter_index]
            if letter in probs:
                totals[candidate] += probs[letter]

    total = sum(totals)
    distribution = [t / total for t in totals] if total > 0 else totals

    return {
        "query_file": query_key,
        "model": model,
        "n_plans": n_plans,
        "n_rotations": n_rot,
        "logprobs_supported": any(supported_flags) if supported_flags else False,
        "distribution": distribution,
        "ranking": [i for i, _ in sorted(enumerate(distribution), key=lambda kv: -kv[1])],
        "top_candidate": max(range(n_plans), key=lambda i: distribution[i]) if total > 0 else None,
        "letter_picks": letter_picks,
        "arm_to_plan": {
            str(arm): index
            for index, (_, arm_list, _) in enumerate(plan_groups(entry))
            for arm in arm_list
        },
        "per_rotation": per_rotation,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def report(results: list[Mapping[str, Any]], ground_truth: Mapping[str, float] | None) -> None:
    print(f"\nqueries scored : {len(results)}")
    print(f"logprobs used  : {sum(r['logprobs_supported'] for r in results)}/{len(results)}")

    picks: dict[str, int] = {}
    for result in results:
        if result["top_candidate"] is None:
            continue
        for arm, index in result["arm_to_plan"].items():
            picks[arm] = picks.get(arm, 0) + 1
    print("predicted arm distribution :", dict(sorted(picks.items())))

    letters: dict[str, int] = {}
    for result in results:
        for letter in result["letter_picks"]:
            letters[letter] = letters.get(letter, 0) + 1
    print("raw letter picks (position bias check) :", dict(sorted(letters.items())))

    if not ground_truth:
        return

    correct = 0
    for result in results:
        best = ground_truth.get(result["query_file"])
        if best is None or result["top_candidate"] is None:
            continue
        mapped = [a for a, i in result["arm_to_plan"].items() if i == result["top_candidate"]]
        if best in mapped:
            correct += 1
    n = sum(1 for r in results if ground_truth.get(r["query_file"]) is not None)
    if n:
        print(f"top-1 accuracy vs ground truth : {correct}/{n} = {correct / n:.3f}")
        print("  (trivial baseline 'always best-frequent arm' was 0.45; random is 0.20)")


def load_ground_truth(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {entry["query_file"]: entry["best_arm"] for entry in entries}


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=Path("bao_arm_plans.json"))
    parser.add_argument("--model", required=True, help="OpenRouter model id")
    parser.add_argument("--out", type=Path, default=Path("llm_labels.json"))
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--rotations", type=int, default=1,
                        help="cyclic permutations averaged per query, 1 disables")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="use 1.0; temperature 0 collapses the logprob distribution")
    parser.add_argument("--max-tokens", type=int, default=1,
                        help="1 for the cheap single-letter read")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N queries")
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--hist-points", type=int, default=12)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"set {args.api_key_env} (export {args.api_key_env}=sk-or-...)")

    entries = json.loads(args.plans.read_text(encoding="utf-8"))
    if args.limit:
        entries = dict(list(entries.items())[: args.limit])

    results = []
    for i, (query_key, entry) in enumerate(entries.items(), 1):
        result = score_entry(
            entry, query_key, args.model, api_key,
            args.rotations, args.temperature, args.max_tokens,
        )
        results.append(result)
        top = result["top_candidate"]
        top_arms = [a for a, idx in result["arm_to_plan"].items() if idx == top]
        print(f"[{i}/{len(entries)}] {query_key} -> candidate {top} (arms {top_arms}) "
              f"p={result['distribution'][top] if top is not None else 0:.3f}", flush=True)

    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(results)} results to {args.out}")
    report(results, load_ground_truth(args.ground_truth))


if __name__ == "__main__":
    main()