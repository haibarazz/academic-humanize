"""
Merge multiple Academic Humanize judge reports into a single judge leaderboard.

The current canonical report type is `ah_judge_report_v2`. This merger supports:
- older v1 reports
- old v2 reports with 4 dimensions
- current v2 reports with 6 dimensions and total_normalized
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NEW_V2_DIMENSIONS = [
    "d1_lexical_markers",
    "d2_structural_patterns",
    "d3_naturalness",
    "d4_semantic_faithfulness",
    "d5_terminology_accuracy",
    "d6_edit_value",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge ah_judge_report_v1/v2 files into a leaderboard.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input judge report paths or glob patterns.")
    parser.add_argument("--output", type=str, default="results/judge/ah_judge_leaderboard.json", help="Output leaderboard JSON path.")
    parser.add_argument("--csv-output", type=str, default=None, help="Optional CSV output path.")
    parser.add_argument("--min-samples", type=int, default=1, help="Minimum judged_rows required for a report to be included.")
    parser.add_argument(
        "--sort-by",
        type=str,
        default="total_normalized",
        choices=[
            "total_normalized",
            "overall",
            "naturalness",
            "semantic_fidelity",
            "terminology_accuracy",
            "edit_value",
            "lexical_markers",
            "structural_patterns",
            "ai_tone_inv",
        ],
        help="Primary sort key for the merged judge leaderboard.",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
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


def summary_mean(summary: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = summary.get(key, {})
    if isinstance(value, dict):
        return safe_float(value.get("mean", default), default)
    return safe_float(value, default)


def normalize_old_judge_human_like(ai_tone_mean: float, naturalness_mean: float) -> float:
    naturalness_norm = max(0.0, min(1.0, (naturalness_mean - 1.0) / 4.0))
    ai_tone_inv = max(0.0, min(1.0, (5.0 - ai_tone_mean) / 4.0))
    return float(0.5 * naturalness_norm + 0.5 * ai_tone_inv)


def load_report(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, f"File does not exist: {path}"

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        return None, f"Failed to read JSON: {exc}"

    if not isinstance(payload, dict):
        return None, "Invalid report format: expected a JSON object."
    report_type = payload.get("report_type")
    if report_type not in {"ah_judge_report_v1", "ah_judge_report_v2"}:
        return None, f"Unsupported report_type: {payload.get('report_type')}"

    summary = payload.get("summary", {})
    counts = payload.get("counts", {})
    if not isinstance(summary, dict):
        return None, "Missing summary"
    if not isinstance(counts, dict):
        counts = {}

    schema = "v1"
    ai_tone_score_mean = 0.0
    d1_lexical_markers_mean = 0.0
    d2_structural_patterns_mean = 0.0
    naturalness_mean = 0.0
    semantic_fidelity_mean = 0.0
    terminology_accuracy_mean = 0.0
    edit_value_mean = 0.0
    total_score_mean = 0.0
    total_normalized_mean = 0.0

    if report_type == "ah_judge_report_v1":
        ai_tone_score_mean = summary_mean(summary, "ai_tone_score")
        naturalness_mean = summary_mean(summary, "naturalness_score")
        total_normalized_mean = normalize_old_judge_human_like(ai_tone_score_mean, naturalness_mean)
        total_score_mean = total_normalized_mean
    elif "total_normalized" in summary and "d1_lexical_markers" in summary:
        schema = "v2_6dim_8pt"
        d1_lexical_markers_mean = summary_mean(summary, "d1_lexical_markers")
        d2_structural_patterns_mean = summary_mean(summary, "d2_structural_patterns")
        naturalness_mean = summary_mean(summary, "d3_naturalness")
        semantic_fidelity_mean = summary_mean(summary, "d4_semantic_faithfulness")
        terminology_accuracy_mean = summary_mean(summary, "d5_terminology_accuracy")
        edit_value_mean = summary_mean(summary, "d6_edit_value")
        total_score_mean = summary_mean(summary, "total")
        total_normalized_mean = summary_mean(summary, "total_normalized")
        ai_tone_score_mean = 1.0 - ((d1_lexical_markers_mean + d2_structural_patterns_mean) / 2.0)
    else:
        schema = "v2_4dim_5pt"
        overall_mean = summary_mean(summary, "overall")
        naturalness_mean = summary_mean(summary, "naturalness")
        semantic_fidelity_mean = summary_mean(summary, "semantic_fidelity")
        terminology_accuracy_mean = summary_mean(summary, "terminology_accuracy")
        edit_value_mean = summary_mean(summary, "edit_value")
        total_score_mean = overall_mean
        total_normalized_mean = max(0.0, min(1.0, (overall_mean - 1.0) / 4.0))

    row = {
        "run_id": str(payload.get("run_id", "")).strip() or path.stem,
        "report_type": str(report_type),
        "judge_schema": schema,
        "source_run_id": str(payload.get("source_run_id", "")).strip(),
        "model_id": str(payload.get("source_model_id", "")).strip() or f"unknown_model@{path.stem}",
        "judge_model": str(payload.get("judge_model", "")).strip() or "unknown",
        "sample_source": str(payload.get("sample_source", "")).strip() or "unknown",
        "prompt_version": str(payload.get("prompt_version", "")).strip(),
        "source_file": str(path),
        "judged_rows": int(counts.get("judged_rows", 0) or 0),
        "parsed_rows": int(counts.get("parsed_rows", 0) or 0),
        "failed_parse_rows": int(counts.get("failed_parse_rows", 0) or 0),
        "ai_tone_score_mean": ai_tone_score_mean,
        "ai_tone_inv": max(0.0, min(1.0, 1.0 - ai_tone_score_mean)),
        "d1_lexical_markers_mean": d1_lexical_markers_mean,
        "d2_structural_patterns_mean": d2_structural_patterns_mean,
        "naturalness_score_mean": naturalness_mean,
        "semantic_fidelity_score_mean": semantic_fidelity_mean,
        "terminology_accuracy_score_mean": terminology_accuracy_mean,
        "edit_value_score_mean": edit_value_mean,
        "total_score_mean": total_score_mean,
        "total_normalized_mean": total_normalized_mean,
        "overall_score_mean": total_normalized_mean,
        "overall_score": total_normalized_mean,
        "overall_score_percent": total_normalized_mean * 100.0,
    }
    return row, None


def sort_rows(rows: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    key_map = {
        "total_normalized": "total_normalized_mean",
        "overall": "overall_score",
        "naturalness": "naturalness_score_mean",
        "semantic_fidelity": "semantic_fidelity_score_mean",
        "terminology_accuracy": "terminology_accuracy_score_mean",
        "edit_value": "edit_value_score_mean",
        "lexical_markers": "d1_lexical_markers_mean",
        "structural_patterns": "d2_structural_patterns_mean",
        "ai_tone_inv": "ai_tone_inv",
    }
    key = key_map.get(sort_by, "total_normalized_mean")
    return sorted(rows, key=lambda row: safe_float(row.get(key, 0.0)), reverse=True)


def build_ranked_list(rows: List[Dict[str, Any]], metric_key: str) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: safe_float(row.get(metric_key, 0.0)), reverse=True)
    return [
        {
            "rank": idx + 1,
            "model_id": row["model_id"],
            "judge_model": row["judge_model"],
            "sample_source": row["sample_source"],
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
        "report_type",
        "judge_schema",
        "judge_model",
        "sample_source",
        "prompt_version",
        "run_id",
        "source_run_id",
        "judged_rows",
        "parsed_rows",
        "failed_parse_rows",
        "d1_lexical_markers_mean",
        "d2_structural_patterns_mean",
        "naturalness_score_mean",
        "semantic_fidelity_score_mean",
        "terminology_accuracy_score_mean",
        "edit_value_score_mean",
        "total_score_mean",
        "total_normalized_mean",
        "ai_tone_score_mean",
        "ai_tone_inv",
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
        raise ValueError("No input judge reports found. Check --inputs.")

    valid_rows: List[Dict[str, Any]] = []
    invalid_reports: List[Dict[str, Any]] = []
    for path in input_paths:
        row, err = load_report(path)
        if err:
            invalid_reports.append({"file": str(path), "reason": err})
            continue
        assert row is not None
        if row["judged_rows"] < args.min_samples:
            invalid_reports.append({"file": str(path), "reason": f"Not enough judged rows: judged_rows={row['judged_rows']}"})
            continue
        valid_rows.append(row)

    if not valid_rows:
        raise ValueError("No valid Academic Humanize judge reports available for merge.")

    sorted_rows = sort_rows(valid_rows, args.sort_by)
    for idx, row in enumerate(sorted_rows):
        row["rank"] = idx + 1

    output_payload = {
        "report_type": "ah_judge_leaderboard_v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sort_by": args.sort_by,
        "counts": {
            "input_files": len(input_paths),
            "valid_reports": len(valid_rows),
            "invalid_reports": len(invalid_reports),
        },
        "rankings": {
            "total_normalized": build_ranked_list(sorted_rows, "total_normalized_mean"),
            "naturalness": build_ranked_list(sorted_rows, "naturalness_score_mean"),
            "semantic_fidelity": build_ranked_list(sorted_rows, "semantic_fidelity_score_mean"),
            "terminology_accuracy": build_ranked_list(sorted_rows, "terminology_accuracy_score_mean"),
            "edit_value": build_ranked_list(sorted_rows, "edit_value_score_mean"),
            "lexical_markers": build_ranked_list(sorted_rows, "d1_lexical_markers_mean"),
            "structural_patterns": build_ranked_list(sorted_rows, "d2_structural_patterns_mean"),
        },
        "models": sorted_rows,
        "invalid_reports": invalid_reports,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    if args.csv_output:
        dump_csv(Path(args.csv_output), sorted_rows)

    print("=" * 72)
    print("Academic Humanize judge leaderboard complete")
    print("=" * 72)
    print(f"inputs: {len(input_paths)} | valid: {len(valid_rows)} | invalid: {len(invalid_reports)}")
    print(f"output: {output_path}")
    if args.csv_output:
        print(f"csv_output: {args.csv_output}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
