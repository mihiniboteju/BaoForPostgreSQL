"""Print serialized PostgreSQL plans for Bao's five optimizer arms.

This script only asks PostgreSQL for estimated plans. It never uses
``EXPLAIN ANALYZE`` and therefore does not execute the query being explained.

Examples:

    python3 bao_arm_plans.py --file job_unique/00000.sql
    python3 bao_arm_plans.py --dsn "dbname=imdb user=mini password=qwerty host=localhost"
    python3 bao_arm_plans.py --compact bao_prompts.txt --candidates 3

``--compact`` also writes one short text block per query (query, table sizes,
main values, NDV, min/max, downsampled histograms, candidate plans) that is
small enough to send to a language model. The connection string defaults to
DSN below; set BAO_PG_DSN or pass --dsn to override it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import psycopg2
from psycopg2 import sql


KNOBS = (
    "enable_hashjoin",
    "enable_mergejoin",
    "enable_nestloop",
    "enable_indexscan",
    "enable_seqscan",
    "enable_indexonlyscan",
)

ARMS_ON = {
    0: {"enable_hashjoin", "enable_indexonlyscan", "enable_indexscan",
        "enable_mergejoin", "enable_seqscan"},
    1: {"enable_hashjoin", "enable_indexonlyscan", "enable_nestloop",
        "enable_seqscan"},
    2: {"enable_hashjoin", "enable_indexonlyscan", "enable_seqscan"},
    3: {"enable_hashjoin", "enable_indexonlyscan", "enable_indexscan",
        "enable_nestloop", "enable_seqscan"},
    4: {"enable_indexonlyscan", "enable_indexscan", "enable_mergejoin",
        "enable_nestloop", "enable_seqscan"},
}

ARMS = {
    arm: {knob: ("on" if knob in enabled else "off") for knob in KNOBS}
    for arm, enabled in ARMS_ON.items()
}
DEFAULT = {knob: "on" for knob in KNOBS}

SCAN = {
    "Seq Scan": "SeqScan",
    "Index Scan": "IndexScan",
    "Index Only Scan": "IndexOnlyScan",
    "Bitmap Heap Scan": "BitmapScan",
    "Tid Scan": "TidScan",
}
JOIN = {
    "Nested Loop": "NestLoop",
    "Hash Join": "HashJoin",
    "Merge Join": "MergeJoin",
}
NUMERIC_TYPES = {
    "smallint", "integer", "bigint", "decimal", "numeric", "real",
    "double precision", "smallserial", "serial", "bigserial",
}

DSN = "dbname=imdb user=mini password=qwerty host=localhost"


def get_plan(cur: Any, sql: str, settings: Mapping[str, str]) -> dict[str, Any]:
    """Return an estimated plan after applying one arm's settings."""
    try:
        for knob, value in settings.items():
            cur.execute(f"SET {knob} = {value}")
        cur.execute("EXPLAIN (FORMAT JSON) " + sql)
        result = cur.fetchone()[0]
        if isinstance(result, str):
            result = json.loads(result)
        return result[0]["Plan"]
    finally:
        cur.execute("RESET ALL")


def get_verbose_plan(cur: Any, query: str) -> dict[str, Any]:
    cur.execute("EXPLAIN (VERBOSE, FORMAT JSON) " + query)
    result = cur.fetchone()[0]
    if isinstance(result, str):
        result = json.loads(result)
    return result[0]["Plan"]


def collect_plan_nodes(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    nodes = [plan]
    for child in plan.get("Plans", []):
        nodes.extend(collect_plan_nodes(child))
    return nodes


def extract_query_columns(
    plan: Mapping[str, Any], cur: Any
) -> dict[tuple[str, str], set[str]]:
    """Find columns mentioned by the planner for each scanned relation."""
    nodes = collect_plan_nodes(plan)
    relations: dict[tuple[str, str], set[str]] = {}
    expressions = json.dumps(nodes)

    for node in nodes:
        relation = node.get("Relation Name")
        alias = node.get("Alias")
        if not relation or not alias:
            continue
        schema = node.get("Schema", "public")
        cur.execute(
            """
            SELECT a.attname
            FROM pg_attribute AS a
            JOIN pg_class AS c ON c.oid = a.attrelid
            JOIN pg_namespace AS n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = %s
              AND a.attnum > 0 AND NOT a.attisdropped
            """,
            (schema, relation),
        )
        columns = {row[0] for row in cur.fetchall()}
        referenced = {
            column for column in columns
            if f'"{alias}"."{column}"' in expressions
            or f"{alias}.{column}" in expressions
            or f'"{relation}"."{column}"' in expressions
            or f"{relation}.{column}" in expressions
        }
        relations.setdefault((schema, relation), set()).update(referenced)
    return relations


def json_value(value: Any) -> Any:
    """Convert PostgreSQL values, including arrays and dates, to JSON values."""
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def collect_statistics(
    cur: Any, plan: Mapping[str, Any], query: str
) -> dict[str, Any]:
    relations = extract_query_columns(plan, cur)
    nodes = collect_plan_nodes(plan)
    table_stats: dict[str, Any] = {}

    for (schema, table), columns in sorted(relations.items()):
        cur.execute(
            sql.SQL("ANALYZE {}.{}").format(
                sql.Identifier(schema), sql.Identifier(table)
            )
        )
        cur.execute(
            """
            SELECT c.reltuples::float8
            FROM pg_class AS c
            JOIN pg_namespace AS n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = %s
            """,
            (schema, table),
        )
        reltuples = cur.fetchone()[0]
        plan_rows = [
            node.get("Plan Rows") for node in nodes
            if node.get("Relation Name") == table
            and node.get("Schema", "public") == schema
            and node.get("Plan Rows") is not None
        ]
        table_key = f"{schema}.{table}"
        table_stats[table_key] = {
            "Card_Tb": {
                "reltuples": reltuples,
                "default_plan_rows": plan_rows,
            },
            "columns": {},
        }

        for column in sorted(columns):
            cur.execute(
                """
                SELECT
                    s.n_distinct,
                    s.most_common_vals,
                    s.most_common_freqs,
                    s.histogram_bounds,
                    format_type(a.atttypid, a.atttypmod)
                FROM pg_stats AS s
                JOIN pg_namespace AS n ON n.nspname = s.schemaname
                JOIN pg_class AS c ON c.relnamespace = n.oid
                                  AND c.relname = s.tablename
                JOIN pg_attribute AS a ON a.attrelid = c.oid
                                       AND a.attname = s.attname
                WHERE s.schemaname = %s AND s.tablename = %s
                  AND s.attname = %s
                """,
                (schema, table, column),
            )
            row = cur.fetchone()
            if row is None:
                continue
            n_distinct, most_common_vals, frequencies, histogram, data_type = row
            column_stats: dict[str, Any] = {
                "NDV": json_value(n_distinct),
                "Main_Value": {
                    "values": json_value(most_common_vals),
                    "frequencies": json_value(frequencies),
                },
            }
            if data_type in NUMERIC_TYPES:
                cur.execute(
                    sql.SQL("SELECT min({0}), max({0}) FROM {1}.{2}").format(
                        sql.Identifier(column),
                        sql.Identifier(schema),
                        sql.Identifier(table),
                    )
                )
                minimum, maximum = cur.fetchone()
                column_stats["Min_Max"] = {
                    "min": json_value(minimum),
                    "max": json_value(maximum),
                }
                column_stats["Hist"] = json_value(histogram)
            table_stats[table_key]["columns"][column] = column_stats

    return {
        "tables": table_stats,
        "query": query,
    }


def serialize(plan: Mapping[str, Any]) -> str:
    """Serialize scans and joins into the hint format used by Bao."""
    scans: list[str] = []
    joins: list[str] = []

    def walk(node: Mapping[str, Any]) -> tuple[str, str]:
        node_type = node["Node Type"]
        if node_type in SCAN:
            alias = node.get("Alias")
            if not alias:
                raise ValueError(f"Scan node has no Alias: {node_type}")
            scans.append(f"{SCAN[node_type]}({alias})")
            return alias, alias

        children = node.get("Plans", [])
        if node_type in JOIN and len(children) == 2:
            left_hint, left_tree = walk(children[0])
            right_hint, right_tree = walk(children[1])
            joins.append(f"{JOIN[node_type]}({left_hint} {right_hint})")
            return f"{left_hint} {right_hint}", f"({left_tree} {right_tree})"
        if not children:
            raise ValueError(f"Unsupported leaf node: {node_type}")
        return walk(children[0])

    _, tree = walk(plan)
    return " ".join(scans + joins + [f"Leading({tree})"])


# ---------------------------------------------------------------------------
# compact, LLM-friendly rendering
# ---------------------------------------------------------------------------
#
# The collected statistics are far too large to send to a language model, so the
# same numbers are also rendered as a short text block:
#
#   **Query**       the SQL text
#   **Statistics**  Table Size / Main Values / NDV / Min Max / Histograms
#   **Candidates**  the distinct plans produced by the optimizer arms
#
# Only the columns the query actually mentions (filters and join keys, plus the
# primary key) are listed, most common values are shown as {value, row count},
# histograms are downsampled to a handful of bounds, and n_distinct is reported
# as an absolute count. Table Size shows the row count the planner estimates for
# the alias (row_source="plan") with the largest primary key in parentheses;
# pass row_source="table" to print the physical reltuples instead.

COLREF_RE = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")


def query_aliases(query: str) -> dict[str, str]:
    """Return {alias: table} for the FROM clause of the query."""
    match = re.search(r"\bFROM\b(.*?)\bWHERE\b", query, re.I | re.S)
    if not match:
        return {}
    aliases: dict[str, str] = {}
    for item in match.group(1).split(","):
        parts = item.split()
        if len(parts) >= 2:
            aliases[parts[-1].lower()] = parts[0].lower()
        elif parts:
            aliases[parts[0].lower()] = parts[0].lower()
    return aliases


def query_columns(query: str) -> dict[str, list[str]]:
    """
    Return {alias: [column]} for every alias.column reference in the query,
    in order of first appearance so the rendered block is reproducible.
    """
    used: dict[str, list[str]] = {}
    for alias, column in COLREF_RE.findall(query):
        alias, column = alias.lower(), column.lower()
        columns = used.setdefault(alias, [])
        if column not in columns:
            columns.append(column)
    return used


def absolute_ndv(cstat: Mapping[str, Any] | None, rows: float) -> int | None:
    """pg_stats n_distinct as an absolute count (negative means row fraction)."""
    ndv = (cstat or {}).get("NDV")
    if not ndv:
        return None
    return int(round(-ndv * rows)) if ndv < 0 else int(round(ndv))


def split_pg_array(text: str) -> list[str]:
    """Split Postgres array text such as {a,"b c",d} into its elements."""
    s = str(text).strip()
    if not (s.startswith("{") and s.endswith("}")):
        return []
    s, out, buf, quoted, i = s[1:-1], [], [], False, 0
    while i < len(s):
        ch = s[i]
        if quoted:
            if ch == "\\" and i + 1 < len(s):
                buf.append(s[i + 1])
                i += 2
                continue
            if ch == '"':
                quoted = False
            else:
                buf.append(ch)
        elif ch == '"':
            quoted = True
        elif ch == ",":
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    out.append("".join(buf))
    return out


def mcv_values(main_value: Mapping[str, Any] | None) -> list[Any]:
    """Most common values as a list, whether psycopg2 parsed them or not."""
    values = (main_value or {}).get("values")
    if values is None:
        return []
    if isinstance(values, str):
        return split_pg_array(values)
    return list(values)


def mcv_freqs(main_value: Mapping[str, Any] | None, count: int) -> list[float]:
    """Most common frequencies as a list, whether psycopg2 parsed them or not."""
    freqs = (main_value or {}).get("frequencies")
    if freqs is None:
        return []
    if isinstance(freqs, str):
        freqs = split_pg_array(freqs)
    return [float(f) for f in freqs[:count]]


def fmt_mcv_value(value: Any) -> str:
    """Render a most common value, quoting it when it is not a plain number."""
    text = str(value)
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return text
    if len(text) > 24:
        text = text[:24] + "..."
    return "'%s'" % text.replace("'", "''")


def mcv_pairs(
    cstat: Mapping[str, Any] | None, rows: float, limit: int
) -> list[tuple[Any, int]]:
    """Most common values as (value, row count) pairs, most frequent first."""
    main_value = (cstat or {}).get("Main_Value") or {}
    pairs = []
    for value, freq in zip(mcv_values(main_value), mcv_freqs(main_value, limit)):
        try:
            pairs.append((value, int(round(float(freq) * rows))))
        except (TypeError, ValueError):
            continue
    pairs.sort(key=lambda pair: -pair[1])
    return pairs[:limit]


def downsample(values: list[str], limit: int) -> list[str]:
    """Evenly spaced sample of a histogram that keeps both endpoints."""
    n = len(values)
    if n <= limit:
        return list(values)
    step = (n - 1) / float(limit - 1)
    sampled = []
    for i in range(limit):
        value = values[int(round(i * step))]
        if not sampled or sampled[-1] != value:
            sampled.append(value)
    return sampled


def plan_groups(entry: Mapping[str, Any]) -> list[tuple[str, list[int], str]]:
    """
    Group the arms by identical plan: [(label, [arms], hint)].

    Two arms can settle on the same plan (arms 0 and 2 usually do), so the
    grouping is what lets a caller map a chosen plan back to the arms it came
    from. Labels are C1, C2, ... in the order of ``unique_arm_plans``.
    """
    arms = {int(k): v for k, v in (entry.get("arms") or {}).items()}
    hints = entry.get("unique_arm_plans") or sorted(set(arms.values()))
    by_hint: dict[str, list[int]] = {}
    for arm in sorted(arms):
        by_hint.setdefault(arms[arm], []).append(arm)
    return [
        ("C%d" % i, by_hint.get(hint, []), hint)
        for i, hint in enumerate(hints, 1)
    ]


def format_compact_prompt(
    entry: Mapping[str, Any],
    candidates: int = 0,
    max_cols: int = 8,
    hist_points: int = 12,
    mcv_count: int = 3,
    row_source: str = "plan",
    include_query: bool = True,
) -> str:
    """Render one bao_arm_plans entry as a short LLM prompt block."""
    stats = entry["statistics"]
    tables = stats.get("tables", {})
    query = stats["query"]
    aliases = query_aliases(query)
    used = query_columns(query)

    size_lines: list[str] = []
    mv_lines: list[str] = []
    ndv_lines: list[str] = []
    mm_lines: list[str] = []
    hist_lines: list[str] = []

    for alias, table in aliases.items():
        table_stats = tables.get("public." + table)
        if table_stats is None:
            continue
        reltuples = table_stats["Card_Tb"]["reltuples"] or 0
        columns = table_stats.get("columns", {})

        estimates = [
            r for r in table_stats["Card_Tb"].get("default_plan_rows") or []
            if r is not None
        ]
        shown = int(round(max(estimates))) if estimates else 0
        if row_source == "table" or not shown:
            shown = int(round(reltuples))
        max_id = ((columns.get("id") or {}).get("Min_Max") or {}).get("max")
        size_lines.append(
            "%s : %d%s" % (alias, shown, " (%s)" % max_id if max_id is not None else "")
        )

        picked = [c for c in (used.get(alias) or []) if c in columns]
        if "id" in columns and "id" not in picked:
            picked.append("id")
        picked = picked[:max_cols] or sorted(columns)[:1]

        for column in picked:
            cstat = columns[column]
            pairs = mcv_pairs(cstat, reltuples, mcv_count)
            if pairs:
                mv_lines.append(
                    "%s.%s : %s"
                    % (
                        table,
                        column,
                        ", ".join(
                            "{%s, %d}" % (fmt_mcv_value(v), c) for v, c in pairs
                        ),
                    )
                )
            ndv = absolute_ndv(cstat, reltuples)
            if ndv is not None:
                ndv_lines.append("%s.%s : %d" % (table, column, ndv))
            min_max = cstat.get("Min_Max") or {}
            if min_max.get("min") is not None or min_max.get("max") is not None:
                mm_lines.append(
                    "%s.%s : {%s, %s}"
                    % (table, column, min_max.get("min"), min_max.get("max"))
                )
            histogram = cstat.get("Hist")
            if histogram:
                bounds = [
                    b for b in str(histogram).strip("{}").split(",") if b.strip()
                ]
                hist_lines.append(
                    "%s.%s : [%s]"
                    % (table, column, ", ".join(downsample(bounds, hist_points)))
                )

    out: list[str] = []
    if include_query:
        out.append("**Query**\n" + query.strip())
    out.append("**Statistics**")
    for title, lines in (
        ("Table Size", size_lines),
        ("Main Values", mv_lines),
        ("NDV", ndv_lines),
        ("Min Max", mm_lines),
        ("Histograms", hist_lines),
    ):
        if lines:
            out.append(title + ":")
            out.extend(lines)

    groups = plan_groups(entry)
    if candidates:
        groups = groups[:candidates]
    out.append("**Candidates**")
    for label, arm_list, hint in groups:
        if arm_list:
            label += " (arms %s)" % ", ".join(str(arm) for arm in arm_list)
        out.append("%s: %s" % (label, hint))
    return "\n".join(out)


def read_sql(path: Path) -> str:
    sql = path.read_text(encoding="utf-8").strip()
    return sql.rstrip(";").rstrip()


def process_file(
    cur: Any, path: Path
) -> tuple[dict[int, str], str, list[str], dict[str, Any]]:
    sql = read_sql(path)
    hints = {
        arm: serialize(get_plan(cur, sql, settings))
        for arm, settings in ARMS.items()
    }
    default_hint = serialize(get_plan(cur, sql, DEFAULT))
    unique_plans = sorted(set(hints.values()))
    default_plan = get_verbose_plan(cur, sql)
    statistics = collect_statistics(cur, default_plan, sql)
    return hints, default_hint, unique_plans, statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        action="append",
        dest="files",
        help="SQL file to explain; may be repeated (default: all job_unique/*.sql)",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("BAO_PG_DSN", DSN),
        help="psycopg2 connection string (default: BAO_PG_DSN or %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bao_arm_plans.json"),
        help="JSON output path (default: %(default)s)",
    )
    parser.add_argument(
        "--compact",
        type=Path,
        default=None,
        help="also write the compact LLM prompt of every query to this text file",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=0,
        help="candidate plans shown per --compact block, 0 for all "
             "(default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = args.files or sorted(Path(__file__).parent.joinpath("job_unique").glob("*.sql"))
    if not files:
        raise SystemExit("No SQL files found.")

    output: dict[str, dict[str, Any]] = {}
    compact_blocks: list[str] = []
    with psycopg2.connect(args.dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for path in files:
                hints, default_hint, unique_plans, statistics = process_file(cur, path)
                entry: dict[str, Any] = {
                    "arms": {str(arm): hint for arm, hint in hints.items()},
                    "default": default_hint,
                    "unique_arm_plans": unique_plans,
                    "statistics": statistics,
                }
                entry["arm_to_plan"] = {
                    str(arm): label
                    for label, arm_list, _ in plan_groups(entry)
                    for arm in arm_list
                }
                output[str(path)] = entry
                if args.compact:
                    compact_blocks.append(
                        format_compact_prompt(output[str(path)], candidates=args.candidates)
                    )
                print(f"{path}:")
                for arm in sorted(hints):
                    print(f"  arm {arm}: {hints[arm]}")
                print(f"  default: {default_hint}")
                print(f"  unique arm plans ({len(unique_plans)}):")
                for plan in unique_plans:
                    print(f"    {plan}")

    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(output)} query plan entries to {args.output}")

    if args.compact:
        separator = "\n\n" + "=" * 72 + "\n\n"
        args.compact.write_text(separator.join(compact_blocks) + "\n", encoding="utf-8")
        print(f"Wrote {len(compact_blocks)} compact prompt blocks to {args.compact}")


if __name__ == "__main__":
    main()
