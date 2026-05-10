"""
Batch-compute Academic Humanize metrics for prediction reports.

This script does not call any model or API. It reads prediction JSON reports
from one folder and writes scored reports to another folder.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import AH_TASK, build_badcases, count_non_ah, summarize_task_results
from evaluation.metrics.recompute_from_report import (
    load_report,
    rebuild_rows,
    resolve_badcase_limit,
    save_report_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-compute AH metrics for prediction reports.")
    parser.add_argument("--input-dir", type=str, default="results/predictions", help="Prediction report folder.")
    parser.add_argument("--pattern", type=str, default="*.json", help="Glob pattern under input-dir.")
    parser.add_argument("--report-files", nargs="*", default=[], help="Extra explicit report files.")
    parser.add_argument("--output-dir", type=str, default="results/scored", help="Scored report output folder.")
    parser.add_argument("--suffix", type=str, default="_scored", help="Output filename suffix before .json.")
    parser.add_argument("--badcase-limit", type=int, default=None, help="Optional override for badcase count.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing scored reports.")
    return parser.parse_args()


def collect_report_paths(input_dir: Path, pattern: str, report_files: List[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for path in sorted(input_dir.glob(pattern)):
        if path.is_file():
            key = str(path.resolve())
            if key not in seen:
                paths.append(path)
                seen.add(key)

    for item in report_files:
        path = Path(item)
        if path.is_file():
            key = str(path.resolve())
            if key not in seen:
                paths.append(path)
                seen.add(key)
    return paths


def output_path_for(input_path: Path, output_dir: Path, suffix: str) -> Path:
    return output_dir / f"{input_path.stem}{suffix}.json"


def score_one(report_path: Path, output_path: Path, badcase_limit_override: int | None) -> Dict[str, Any]:
    payload = load_report(report_path)
    rebuilt_rows = rebuild_rows(payload.get("rows", []))
    skipped_samples = payload.get("skipped_samples", [])
    if not isinstance(skipped_samples, list):
        skipped_samples = []

    task_results = summarize_task_results(rebuilt_rows)
    badcase_limit = resolve_badcase_limit(payload, badcase_limit_override)

    old_counts = payload.get("counts", {})
    if not isinstance(old_counts, dict):
        old_counts = {}

    selected_rows = len(rebuilt_rows)
    skipped_rows = len(skipped_samples)
    total_rows = int(old_counts.get("total_rows", selected_rows + skipped_rows) or 0)
    total_rows = max(total_rows, selected_rows + skipped_rows)

    payload["report_type"] = "ah_model_eval_v1"
    payload["task"] = AH_TASK
    payload["status"] = "scored"
    payload["metrics_computed_at"] = datetime.now().isoformat(timespec="seconds")
    payload["task_results"] = task_results
    payload["rows"] = rebuilt_rows
    payload["badcases"] = build_badcases(rebuilt_rows, badcase_limit)
    payload["counts"] = {
        "total_rows": total_rows,
        "selected_rows": selected_rows,
        "skipped_rows": skipped_rows,
        "non_ah_rows": count_non_ah(skipped_samples),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_report_atomic(output_path, payload)
    return {
        "input": str(report_path),
        "output": str(output_path),
        "model_id": str(payload.get("model_id", "")),
        "sample_count": selected_rows,
        "bleu": float(task_results.get("bleu", 0.0)),
        "chrfpp": float(task_results.get("chrfpp", 0.0)),
        "ter": float(task_results.get("ter", 0.0)),
        "bertscore_f1": float(task_results.get("bertscore_f1", 0.0)),
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_paths = collect_report_paths(input_dir, args.pattern, args.report_files)

    if not report_paths:
        raise FileNotFoundError(f"No prediction reports found in {input_dir} with pattern {args.pattern}")

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for report_path in tqdm(report_paths, desc="Scoring reports", unit="report"):
        out_path = output_path_for(report_path, output_dir, args.suffix)
        if out_path.exists() and not args.overwrite:
            results.append({"input": str(report_path), "output": str(out_path), "status": "skipped_exists"})
            continue
        try:
            item = score_one(report_path, out_path, args.badcase_limit)
            item["status"] = "scored"
            results.append(item)
        except Exception as exc:
            failures.append({"input": str(report_path), "error": str(exc)})

    summary_path = output_dir / "batch_metrics_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_report_atomic(
        summary_path,
        {
            "report_type": "ah_metrics_batch_summary_v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "pattern": args.pattern,
            "counts": {
                "found": len(report_paths),
                "scored_or_skipped": len(results),
                "failed": len(failures),
            },
            "results": results,
            "failures": failures,
        },
    )

    print("=" * 72)
    print("AH batch metrics complete")
    print("=" * 72)
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"summary: {summary_path}")
    print(f"found: {len(report_paths)} | failed: {len(failures)}")
    print("=" * 72)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
