"""Reproducible T32 evaluation runner for the CUSMA Trade Assistant.

Runs the repository's evaluation datasets through the full pipeline and
records routing, answer, security, timing, token, and cost results.

Example:
    python evaluation/run_evaluation.py \
        --model openai/gpt-4o-mini \
        --limit 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents import pipeline  # noqa: E402
from agents.base import OpenRouterModel  # noqa: E402
from evaluation.adversarial_suite import ATTACKS, TEST_CASES  # noqa: E402
from evaluation.metrics_calculator import (  # noqa: E402
    actual_outcome,
    calculate_cost,
    score_attack,
    score_holdout,
    score_test_case,
    summarize,
    tool_names,
)
from mcp_servers.document_records import temporary_records  # noqa: E402


DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_FILES = ("holdout", "attacks", "test_cases", "human_written")

HOLDOUT_FILE = REPO_ROOT / "data" / "test_holdout_human_reviewed.jsonl"
HUMAN_FILE = REPO_ROOT / "data" / "test_human_written.jsonl"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL dataset."""
    rows = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {err}"
                ) from err

            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object"
                )

            rows.append(row)

    return rows


def selected_files(value: str | None) -> set[str]:
    """Normalize the --files option into dataset names."""
    if not value:
        return set(DEFAULT_FILES)

    aliases = {
        "holdout": "holdout",
        "attacks": "attacks",
        "attack": "attacks",
        "test_cases": "test_cases",
        "tests": "test_cases",
        "human": "human_written",
        "human_written": "human_written",
    }

    selected = set()

    for item in value.split(","):
        key = item.strip().lower()

        if not key:
            continue

        if key not in aliases:
            raise ValueError(
                f"Unknown dataset '{key}'. "
                "Use holdout, attacks, test_cases, or human_written."
            )

        selected.add(aliases[key])

    return selected


def load_cases(files: set[str]) -> tuple[list[dict], list[str]]:
    """Load requested datasets and return cases plus skip notices."""
    cases: list[dict] = []
    notices: list[str] = []

    if "holdout" in files:
        if HOLDOUT_FILE.exists():
            for row in load_jsonl(HOLDOUT_FILE):
                cases.append({
                    "case_kind": "holdout",
                    "source_file": str(HOLDOUT_FILE.relative_to(REPO_ROOT)),
                    **row,
                })
        else:
            notices.append(f"missing optional dataset: {HOLDOUT_FILE}")

    if "attacks" in files:
        for row in ATTACKS:
            cases.append({
                "case_kind": "attack",
                "source_file": "evaluation/adversarial_suite.py",
                **row,
            })

    if "test_cases" in files:
        for row in TEST_CASES:
            cases.append({
                "case_kind": "test_case",
                "source_file": "evaluation/adversarial_suite.py",
                **row,
            })

    if "human_written" in files:
        if HUMAN_FILE.exists():
            for row in load_jsonl(HUMAN_FILE):
                cases.append({
                    "case_kind": "human_written",
                    "source_file": str(HUMAN_FILE.relative_to(REPO_ROOT)),
                    **row,
                })
        else:
            notices.append(f"missing optional dataset: {HUMAN_FILE}")

    return cases, notices



def case_input(case: dict) -> tuple[str, str | None, str, str]:
    """Return message, original text, language, and input type."""
    language = str(case.get("language") or "en")
    input_type = str(case.get("input_type") or "message")

    if case.get("case_kind") in {"attack", "test_case"}:
        original = str(case.get("input") or "")
        translated = case.get("translated_input")

        if translated:
            return str(translated), original, language, input_type

        return original, None, language, input_type

    # Holdout and human-written rows are questions, including rows whose
    # expected route is document_intake. They are not uploaded documents.
    message = str(case.get("query") or case.get("input") or "")
    return message, None, language, input_type


def score_case(case: dict, out: dict) -> dict:
    """Dispatch scoring based on the dataset row type."""
    kind = case.get("case_kind")

    if kind in {"holdout", "human_written"}:
        return score_holdout(case, out)

    if kind == "attack":
        return score_attack(case, out)

    if kind == "test_case":
        return score_test_case(case, out)

    return {}


def json_cell(value: Any) -> str:
    """Serialize nested values for later CSV storage."""
    if value is None:
        return ""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)

    return str(value)


def evaluate_case(
    case: dict,
    model: Any,
    model_name: str,
) -> dict:
    """Run and score one evaluation case without stopping the full run."""
    message, original, language, input_type = case_input(case)
    case_id = case.get("id") or case.get("frame_id") or ""

    base = {
        "case_id": case_id,
        "case_kind": case.get("case_kind"),
        "source_file": case.get("source_file"),
        "query": message,
        "original_input": original,
        "tier": case.get("tier"),
        "language": language,
        "input_type": input_type,
        "phrasing": case.get("phrasing"),
        "answer_type": case.get("answer_type"),
        "expected_route": (
            case.get("intent")
            or case.get("expected_route")
        ),
        "expected_answer": case.get("answer"),
        "expected_outcome": case.get("expected_outcome"),
        "error": "",
    }

    try:
        out = pipeline.answer(
            message,
            case.get("tier"),
            original=original,
            language=language,
            input_type=input_type,
            model=model,
        )

        usage = out.get("usage") or {}
        guard = out.get("guard") or {}
        review = out.get("review") or {}

        result = {
            **base,
            "actual_route": out.get("label"),
            "router": out.get("router"),
            "agent": out.get("agent"),
            "pipeline_decision": out.get("decision"),
            "actual_outcome": actual_outcome(out),
            "guard_decision": guard.get("decision"),
            "guard_rule_id": guard.get("rule_id"),
            "guard_category": guard.get("category"),
            "guard_flags": guard.get("flags") or [],
            "tools_used": tool_names(out),
            "trace": out.get("trace") or [],
            "reviewer_decision": review.get("decision"),
            "reviewer_rescued": review.get("rescued"),
            "reviewer_findings": review.get("findings") or [],
            "final_reply": out.get("response") or "",
            "draft_reply": out.get("draft") or "",
            "result": out.get("result"),
            "escalation": out.get("escalation"),
            "seconds": out.get("seconds"),
            "calls": int(usage.get("calls") or 0),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(
                usage.get("completion_tokens") or 0
            ),
            "tokens": (
                int(usage.get("prompt_tokens") or 0)
                + int(usage.get("completion_tokens") or 0)
            ),
            "cost_usd": calculate_cost(
                usage,
                model=model_name,
            ),
            "models": usage.get("models") or [],
        }

        result.update(score_case(case, out))
        return result

    except Exception as err:
        return {
            **base,
            "actual_route": None,
            "router": None,
            "agent": None,
            "pipeline_decision": None,
            "actual_outcome": "error",
            "guard_decision": None,
            "guard_rule_id": None,
            "guard_category": None,
            "guard_flags": [],
            "tools_used": [],
            "trace": [],
            "reviewer_decision": None,
            "reviewer_rescued": None,
            "reviewer_findings": [],
            "final_reply": "",
            "draft_reply": "",
            "result": None,
            "escalation": None,
            "seconds": None,
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tokens": 0,
            "cost_usd": None,
            "models": [],
            "route_correct": None,
            "answer_correct": None,
            "outcome_correct": None,
            "error": f"{type(err).__name__}: {err}"[:500],
        }


CSV_FIELDS = [
    "run_id",
    "case_id",
    "case_kind",
    "source_file",
    "query",
    "original_input",
    "tier",
    "language",
    "input_type",
    "phrasing",
    "answer_type",
    "expected_route",
    "actual_route",
    "route_correct",
    "expected_answer",
    "actual_verdict",
    "actual_dollar_value",
    "answer_correct",
    "expected_outcome",
    "expected_outcome_normalized",
    "actual_outcome",
    "outcome_correct",
    "attack_passed",
    "test_case_passed",
    "must_not_contain_passed",
    "must_not_match_passed",
    "matched_forbidden_strings",
    "matched_forbidden_patterns",
    "guard_decision",
    "guard_rule_id",
    "guard_category",
    "guard_flags",
    "router",
    "agent",
    "tools_used",
    "trace",
    "reviewer_decision",
    "reviewer_rescued",
    "reviewer_findings",
    "pipeline_decision",
    "final_reply",
    "draft_reply",
    "result",
    "escalation",
    "seconds",
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "tokens",
    "cost_usd",
    "models",
    "error",
]


def csv_value(value: Any) -> Any:
    """Convert nested values to JSON strings for a CSV cell."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)

    if value is None:
        return ""

    return value


def write_results_csv(
    rows: list[dict],
    path: Path,
    run_id: str,
) -> None:
    """Save one detailed CSV row for every evaluated question."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            flattened = {
                field: csv_value(row.get(field))
                for field in CSV_FIELDS
            }
            flattened["run_id"] = run_id
            writer.writerow(flattened)


def write_summary_json(
    summary: dict,
    path: Path,
    *,
    run_id: str,
    model: str,
    router: str,
    files: set[str],
    csv_path: Path,
) -> None:
    """Save the aggregate summary and run metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "router": router,
        "datasets": sorted(files),
        "results_csv": str(csv_path),
        "summary": summary,
    }

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the T32 reproducible CUSMA evaluation."
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model name (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--router",
        choices=("current", "keyword"),
        default="current",
        help="Router to evaluate (default: current)",
    )

    parser.add_argument(
        "--files",
        help=(
            "Comma-separated datasets: "
            "holdout,attacks,test_cases,human_written"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N selected cases.",
    )

    parser.add_argument(
        "--json",
        nargs="?",
        const="auto",
        help="Also save the summary as JSON, optionally to this path.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for evaluation result files.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    try:
        files = selected_files(args.files)
    except ValueError as err:
        raise SystemExit(str(err)) from err

    cases, notices = load_cases(files)

    if args.limit is not None:
        cases = cases[:args.limit]

    for notice in notices:
        print(f"NOTICE: {notice}", file=sys.stderr)

    print(f"Selected cases: {len(cases)}")
    print(f"Model: {args.model}")
    print(f"Router: {args.router}")

    previous_router = os.environ.get("CUSMA_ROUTER")

    if args.router == "keyword":
        os.environ["CUSMA_ROUTER"] = "keyword"
    else:
        os.environ.pop("CUSMA_ROUTER", None)

    pipeline._router_cache.clear()

    try:
        model = OpenRouterModel(model=args.model)
        results = []

        with temporary_records():
            for index, case in enumerate(cases, start=1):
                result = evaluate_case(
                    case,
                    model=model,
                    model_name=args.model,
                )
                results.append(result)

                status = "ERROR" if result.get("error") else "DONE"
                case_id = result.get("case_id") or index

                print(
                    f"[{index}/{len(cases)}] {status} "
                    f"{case_id} route={result.get('actual_route')} "
                    f"seconds={result.get('seconds')} "
                    f"cost={result.get('cost_usd')}"
                )

        summary = summarize(results)

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = args.output_dir / f"evaluation_{run_id}.csv"

        write_results_csv(
            results,
            csv_path,
            run_id,
        )

        json_path = None

        if args.json:
            if args.json == "auto":
                json_path = args.output_dir / f"evaluation_{run_id}.json"
            else:
                json_path = Path(args.json)

            write_summary_json(
                summary,
                json_path,
                run_id=run_id,
                model=args.model,
                router=args.router,
                files=files,
                csv_path=csv_path,
            )

        print()
        print("Evaluation summary")
        print("------------------")
        print(f"Total evaluated: {summary['total_evaluated']}")
        print(
            "Routing accuracy: "
            f"{summary['routing']['accuracy_pct']}% "
            f"({summary['routing']['correct']}/"
            f"{summary['routing']['applicable']})"
        )
        print(
            "Answer accuracy: "
            f"{summary['answers']['accuracy_pct']}% "
            f"({summary['answers']['correct']}/"
            f"{summary['answers']['applicable']})"
        )
        print(
            "Attacks stopped or answered safely: "
            f"{summary['attacks']['stopped_or_safe_pct']}% "
            f"({summary['attacks']['passed']}/"
            f"{summary['attacks']['total']})"
        )
        print(
            "Normal questions wrongly blocked: "
            f"{summary['normal_questions_wrongly_blocked']}"
        )
        print(f"Average seconds: {summary['average_seconds']}")
        print(f"Total cost USD: {summary['total_cost_usd']}")
        print(
            "Cost per question USD: "
            f"{summary['cost_per_question_usd']}"
        )
        print(
            "Seen routing / answer accuracy: "
            f"{summary['seen']['routing_accuracy_pct']}% / "
            f"{summary['seen']['answer_accuracy_pct']}%"
        )
        print(
            "Unseen routing / answer accuracy: "
            f"{summary['unseen']['routing_accuracy_pct']}% / "
            f"{summary['unseen']['answer_accuracy_pct']}%"
        )
        print(f"Errors: {summary['errors']}")
        print()
        print(f"Results CSV: {csv_path}")

        if json_path is not None:
            print(f"Summary JSON: {json_path}")

    finally:
        if previous_router is None:
            os.environ.pop("CUSMA_ROUTER", None)
        else:
            os.environ["CUSMA_ROUTER"] = previous_router

        pipeline._router_cache.clear()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
