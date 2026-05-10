"""
Compute Academic Humanize metrics from an existing prediction report.

This script does not call any model or API. It rebuilds row-level metrics,
task summary metrics, badcases, and counts from the cached `rows` field.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import (
    AH_TASK,
    build_badcases,
    build_eval_row,
    count_non_ah,
    summarize_task_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Academic Humanize metrics from a cached prediction report.")
    parser.add_argument("--report-file", type=str, required=True, help="Path to a ah_model_eval_v1 report.")
    parser.add_argument("--output", type=str, default=None, help="Optional output path. Default: overwrite report-file.")
    parser.add_argument("--badcase-limit", type=int, default=None, help="Optional override for badcase count.")
    return parser.parse_args()


def load_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Report file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Invalid report format: expected a JSON object.")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("The report is missing `rows`. Re-run the Academic Humanize evaluator first.")
    return payload


def save_report_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def rebuild_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rebuilt: List[Dict[str, Any]] = []
    total = len(raw_rows)
    for idx, row in enumerate(tqdm(raw_rows, desc="Recomputing rows", total=total), start=1):
        if not isinstance(row, dict):
            continue
        normalized = {
            "sample_id": str(row.get("sample_id", f"row_{idx}")),
            "paper_id": str(row.get("paper_id", "unknown")),
            "task_type": AH_TASK,
            "instruction": str(row.get("instruction", "")).strip(),
            "input": str(row.get("input", "")).strip(),
            "output": str(row.get("reference", row.get("output", ""))).strip(),
        }
        prediction = str(row.get("prediction", "")).strip()
        rebuilt.append(build_eval_row(normalized, prediction, idx))
    return rebuilt


def resolve_badcase_limit(payload: Dict[str, Any], override: int | None) -> int:
    if override is not None:
        return max(int(override), 0)
    old_badcases = payload.get("badcases")
    if isinstance(old_badcases, list) and old_badcases:
        return len(old_badcases)
    return 20


def main() -> int:
    args = parse_args()
    report_path = Path(args.report_file)
    payload = load_report(report_path)

    rebuilt_rows = rebuild_rows(payload.get("rows", []))
    skipped_samples = payload.get("skipped_samples", [])
    if not isinstance(skipped_samples, list):
        skipped_samples = []

    print("Computing corpus metrics (BLEU / chrF++ / TER / BERTScore)...")
    task_results = summarize_task_results(rebuilt_rows)
    badcase_limit = resolve_badcase_limit(payload, args.badcase_limit)

    old_counts = payload.get("counts", {})
    if not isinstance(old_counts, dict):
        old_counts = {}

    selected_rows = len(rebuilt_rows)
    skipped_rows = len(skipped_samples)
    total_rows = int(old_counts.get("total_rows", selected_rows + skipped_rows) or 0)
    total_rows = max(total_rows, selected_rows + skipped_rows)

    payload["task"] = AH_TASK
    payload["task_results"] = task_results
    payload["rows"] = rebuilt_rows
    print("Building badcases...")
    payload["badcases"] = build_badcases(rebuilt_rows, badcase_limit)
    payload["report_type"] = "ah_model_eval_v1"
    payload["counts"] = {
        "total_rows": total_rows,
        "selected_rows": selected_rows,
        "skipped_rows": skipped_rows,
        "non_ah_rows": count_non_ah(skipped_samples),
    }

    print("Saving report...")
    output_path = Path(args.output) if args.output else report_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_report_atomic(output_path, payload)

    print("=" * 72)
    print("Academic Humanize report recompute finished")
    print("=" * 72)
    print(f"source_report: {report_path}")
    print(f"output_report: {output_path}")
    print(f"selected_rows: {selected_rows} | skipped_rows: {skipped_rows}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
