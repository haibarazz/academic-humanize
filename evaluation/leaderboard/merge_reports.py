"""
Merge multiple Academic Humanize reports into a single leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge ah_model_eval_v1 reports into a leaderboard.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input report paths or glob patterns, for example: results/ah_eval_local_*.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ah_leaderboard.json",
        help="Output leaderboard JSON path.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum selected sample count required for a report to be included.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="overall",
        choices=["overall", "bleu", "chrfpp", "ter_inv", "style_diff", "burstiness"],
        help="Primary sort key for the merged model list.",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def expand_input_paths(raw_inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for item in raw_inputs:
        matched = sorted(glob.glob(item))
        if not matched and Path(item).exists():
            matched = [item]
        for path_str in matched:
            path = Path(path_str)
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


def load_report(path: Path) -> Tuple[Dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"File does not exist: {path}"

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        return None, f"Failed to read JSON: {exc}"

    if not isinstance(payload, dict):
        return None, "Invalid report format: expected a JSON object."
    if payload.get("report_type") != "ah_model_eval_v1":
        return None, f"Unsupported report_type: {payload.get('report_type')}"

    task_results = payload.get("task_results", {})
    if not isinstance(task_results, dict):
        return None, "Missing task_results"

    counts = payload.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}

    row = {
        "run_id": str(payload.get("run_id", "")).strip() or path.stem,
        "mode": str(payload.get("mode", "")).strip() or "unknown",
        "model_id": str(payload.get("model_id", "")).strip() or f"unknown_model@{path.stem}",
        "source_file": str(path),
        "sample_count": int(counts.get("selected_rows", task_results.get("sample_count", 0)) or 0),
        "bleu": safe_float(task_results.get("bleu", 0.0)),
        "chrfpp": safe_float(task_results.get("chrfpp", 0.0)),
        "ter": safe_float(task_results.get("ter", 0.0)),
        "style_diff": safe_float(task_results.get("style_diff_mean", 0.0)),
        "burstiness": safe_float(task_results.get("burstiness_mean", 0.0)),
    }
    return row, None


def minmax_normalize(values: List[float], lower_is_better: bool = False) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax <= vmin:
        return [1.0 for _ in values]
    norm = [(value - vmin) / (vmax - vmin) for value in values]
    if lower_is_better:
        return [1.0 - value for value in norm]
    return norm


def attach_overall_scores(rows: List[Dict[str, Any]]) -> None:
    bleu_norm = minmax_normalize([safe_float(row["bleu"]) for row in rows])
    chrfpp_norm = minmax_normalize([safe_float(row["chrfpp"]) for row in rows])
    ter_inv = minmax_normalize([safe_float(row["ter"]) for row in rows], lower_is_better=True)
    style_diff_norm = minmax_normalize([safe_float(row["style_diff"]) for row in rows])
    burstiness_norm = minmax_normalize([safe_float(row["burstiness"]) for row in rows])

    for idx, row in enumerate(rows):
        row["ter_inv"] = float(ter_inv[idx])
        score = (
            0.30 * bleu_norm[idx]
            + 0.25 * chrfpp_norm[idx]
            + 0.20 * ter_inv[idx]
            + 0.15 * style_diff_norm[idx]
            + 0.10 * burstiness_norm[idx]
        )
        row["overall_score"] = float(score)
        row["overall_score_percent"] = float(score * 100.0)


def sort_rows(rows: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    key_map = {
        "overall": "overall_score",
        "bleu": "bleu",
        "chrfpp": "chrfpp",
        "ter_inv": "ter_inv",
        "style_diff": "style_diff",
        "burstiness": "burstiness",
    }
    key = key_map.get(sort_by, "overall_score")
    return sorted(rows, key=lambda row: safe_float(row.get(key, 0.0)), reverse=True)


def build_ranked_list(rows: List[Dict[str, Any]], metric_key: str) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: safe_float(row.get(metric_key, 0.0)), reverse=True)
    return [
        {
            "rank": idx + 1,
            "model_id": row["model_id"],
            "mode": row["mode"],
            "run_id": row["run_id"],
            "value": safe_float(row.get(metric_key, 0.0)),
            "source_file": row["source_file"],
        }
        for idx, row in enumerate(ranked)
    ]


def dump_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "rank",
        "model_id",
        "mode",
        "run_id",
        "sample_count",
        "bleu",
        "chrfpp",
        "ter",
        "ter_inv",
        "style_diff",
        "burstiness",
        "overall_score",
        "overall_score_percent",
        "source_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def main() -> int:
    args = parse_args()
    input_paths = expand_input_paths(args.inputs)
    if not input_paths:
        raise ValueError("No input reports found. Check --inputs.")

    valid_rows: List[Dict[str, Any]] = []
    invalid_reports: List[Dict[str, Any]] = []
    for path in input_paths:
        row, err = load_report(path)
        if err:
            invalid_reports.append({"file": str(path), "reason": err})
            continue
        assert row is not None
        if row["sample_count"] < args.min_samples:
            invalid_reports.append(
                {
                    "file": str(path),
                    "reason": f"Not enough samples: sample_count={row['sample_count']}",
                }
            )
            continue
        valid_rows.append(row)

    if not valid_rows:
        raise ValueError("No valid Academic Humanize reports available for merge.")

    attach_overall_scores(valid_rows)
    sorted_rows = sort_rows(valid_rows, args.sort_by)
    for idx, row in enumerate(sorted_rows):
        row["rank"] = idx + 1

    output_payload = {
        "report_type": "ah_leaderboard_v1",
        "generated_at": datetime.now().isoformat(),
        "settings": {
            "inputs": [str(path) for path in input_paths],
            "min_samples": args.min_samples,
            "sort_by": args.sort_by,
        },
        "counts": {
            "input_reports": len(input_paths),
            "valid_reports": len(sorted_rows),
            "invalid_reports": len(invalid_reports),
        },
        "invalid_reports": invalid_reports,
        "models": sorted_rows,
        "leaderboards": {
            "overall": build_ranked_list(sorted_rows, "overall_score"),
            "bleu": build_ranked_list(sorted_rows, "bleu"),
            "chrfpp": build_ranked_list(sorted_rows, "chrfpp"),
            "ter_inv": build_ranked_list(sorted_rows, "ter_inv"),
            "style_diff": build_ranked_list(sorted_rows, "style_diff"),
            "burstiness": build_ranked_list(sorted_rows, "burstiness"),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    if args.csv_output:
        dump_csv(Path(args.csv_output), sorted_rows)

    print("=" * 72)
    print("Academic Humanize leaderboard merge finished")
    print("=" * 72)
    print(f"input_reports: {len(input_paths)}")
    print(f"valid_reports: {len(sorted_rows)}")
    print(f"invalid_reports: {len(invalid_reports)}")
    print(f"json_output: {output_path}")
    if args.csv_output:
        print(f"csv_output: {args.csv_output}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
