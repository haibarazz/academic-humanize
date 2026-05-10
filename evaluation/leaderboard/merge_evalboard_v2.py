"""
Merge Academic Humanize model reports, judge reports, and detector reports into evalboard_v2.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import AH_TASK, build_eval_row, summarize_task_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Academic Humanize reports into evalboard_v2.")
    parser.add_argument("--model-reports", nargs="+", required=True, help="Input ah_model_eval_v1 paths or glob patterns.")
    parser.add_argument("--champion-report", type=str, required=True, help="Champion ah_model_eval_v1 report.")
    parser.add_argument("--judge-reports", nargs="*", default=[], help="Optional ah_judge_report_v1 paths or globs.")
    parser.add_argument("--detector-reports", nargs="*", default=[], help="Optional ah_detector_report_v1 paths or globs.")
    parser.add_argument("--output", type=str, default="results/ah_evalboard_v2.json", help="Output JSON path.")
    parser.add_argument("--csv-output", type=str, default=None, help="Optional CSV output path.")
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
            try:
                key = str(path.resolve())
            except Exception:
                key = str(path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


def resolve_key(path_str: str) -> str:
    try:
        return str(Path(path_str).resolve())
    except Exception:
        return str(Path(path_str))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return payload


def ensure_model_report_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    task_results = payload.get("task_results", {})
    rows = payload.get("rows", [])
    if not isinstance(task_results, dict):
        task_results = {}
    if not isinstance(rows, list):
        rows = []

    required_keys = {
        "bleu",
        "chrfpp",
        "ter",
        "bertscore_f1",
        "style_diff_mean",
        "burstiness_mean",
        "format_violation_rate",
    }
    if required_keys.issubset(task_results.keys()):
        return task_results
    if not rows:
        return task_results

    rebuilt_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
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
        rebuilt_rows.append(build_eval_row(normalized, prediction, idx))
    return summarize_task_results(rebuilt_rows)


def load_model_report(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    if payload.get("report_type") != "ah_model_eval_v1":
        raise ValueError(f"Unsupported model report_type: {payload.get('report_type')}")

    task_results = ensure_model_report_metrics(payload)
    counts = payload.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}

    return {
        "run_id": str(payload.get("run_id", "")).strip() or path.stem,
        "model_id": str(payload.get("model_id", "")).strip() or path.stem,
        "mode": str(payload.get("mode", "")).strip() or "unknown",
        "source_file": str(path),
        "source_key": resolve_key(str(path)),
        "sample_count": int(counts.get("selected_rows", task_results.get("sample_count", 0)) or 0),
        "bleu": safe_float(task_results.get("bleu", 0.0)),
        "chrfpp": safe_float(task_results.get("chrfpp", 0.0)),
        "ter": safe_float(task_results.get("ter", 0.0)),
        "bertscore_f1": safe_float(task_results.get("bertscore_f1", 0.0)),
        "style_diff": safe_float(task_results.get("style_diff_mean", 0.0)),
        "burstiness": safe_float(task_results.get("burstiness_mean", 0.0)),
        "format_violation_rate": safe_float(task_results.get("format_violation_rate", 0.0)),
        "contains_cjk_rate": safe_float(task_results.get("contains_cjk_rate", 0.0)),
        "meta_prefix_rate": safe_float(task_results.get("meta_prefix_rate", 0.0)),
        "meta_suffix_rate": safe_float(task_results.get("meta_suffix_rate", 0.0)),
        "explanation_block_rate": safe_float(task_results.get("explanation_block_rate", 0.0)),
        "bullet_or_list_tail_rate": safe_float(task_results.get("bullet_or_list_tail_rate", 0.0)),
    }


def load_judge_report(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    report_type = payload.get("report_type")
    if report_type not in {"ah_judge_report_v1", "ah_judge_report_v2"}:
        raise ValueError(f"Unsupported judge report_type: {payload.get('report_type')}")
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    if report_type == "ah_judge_report_v1":
        ai_tone = summary.get("ai_tone_score", {})
        naturalness = summary.get("naturalness_score", {})
        if not isinstance(ai_tone, dict):
            ai_tone = {}
        if not isinstance(naturalness, dict):
            naturalness = {}
        ai_tone_mean = safe_float(ai_tone.get("mean", 0.0))
        naturalness_mean = safe_float(naturalness.get("mean", 0.0))
        judge_human_like = normalize_judge_human_like(ai_tone_mean, naturalness_mean)
        semantic_fidelity_mean = 0.0
        terminology_accuracy_mean = 0.0
        edit_value_mean = 0.0
        overall_mean = 0.5 * naturalness_mean + 0.5 * (6.0 - ai_tone_mean)
    else:
        if "total_normalized" in summary and "d1_lexical_markers" in summary:
            def dim_mean(key: str) -> float:
                item = summary.get(key, {})
                return safe_float(item.get("mean", 0.0)) if isinstance(item, dict) else 0.0

            d1_mean = dim_mean("d1_lexical_markers")
            d2_mean = dim_mean("d2_structural_patterns")
            ai_tone_mean = 1.0 - ((d1_mean + d2_mean) / 2.0)
            naturalness_mean = dim_mean("d3_naturalness")
            semantic_fidelity_mean = dim_mean("d4_semantic_faithfulness")
            terminology_accuracy_mean = dim_mean("d5_terminology_accuracy")
            edit_value_mean = dim_mean("d6_edit_value")
            total = summary.get("total", {})
            total_normalized = summary.get("total_normalized", {})
            overall_mean = safe_float(total.get("mean", 0.0)) if isinstance(total, dict) else 0.0
            judge_human_like = (
                safe_float(total_normalized.get("mean", 0.0))
                if isinstance(total_normalized, dict)
                else 0.0
            )
        else:
            overall = summary.get("overall", {})
            naturalness = summary.get("naturalness", {})
            semantic_fidelity = summary.get("semantic_fidelity", {})
            terminology_accuracy = summary.get("terminology_accuracy", {})
            edit_value = summary.get("edit_value", {})
            if not isinstance(overall, dict):
                overall = {}
            if not isinstance(naturalness, dict):
                naturalness = {}
            if not isinstance(semantic_fidelity, dict):
                semantic_fidelity = {}
            if not isinstance(terminology_accuracy, dict):
                terminology_accuracy = {}
            if not isinstance(edit_value, dict):
                edit_value = {}
            ai_tone_mean = 0.0
            naturalness_mean = safe_float(naturalness.get("mean", 0.0))
            semantic_fidelity_mean = safe_float(semantic_fidelity.get("mean", 0.0))
            terminology_accuracy_mean = safe_float(terminology_accuracy.get("mean", 0.0))
            edit_value_mean = safe_float(edit_value.get("mean", 0.0))
            overall_mean = safe_float(overall.get("mean", 0.0))
            judge_human_like = max(0.0, min(1.0, (overall_mean - 1.0) / 4.0))

    return {
        "source_report": str(payload.get("source_report", "")).strip(),
        "source_key": resolve_key(str(payload.get("source_report", "")).strip()) if payload.get("source_report") else "",
        "source_run_id": str(payload.get("source_run_id", "")).strip(),
        "source_model_id": str(payload.get("source_model_id", "")).strip(),
        "report_type": str(report_type),
        "judge_model": str(payload.get("judge_model", "")).strip(),
        "judged_rows": int(payload.get("counts", {}).get("judged_rows", 0) or 0) if isinstance(payload.get("counts"), dict) else 0,
        "ai_tone_score_mean": ai_tone_mean,
        "naturalness_score_mean": naturalness_mean,
        "semantic_fidelity_score_mean": semantic_fidelity_mean,
        "terminology_accuracy_score_mean": terminology_accuracy_mean,
        "edit_value_score_mean": edit_value_mean,
        "overall_score_mean": overall_mean,
        "judge_human_like": judge_human_like,
    }


def load_detector_report(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    if payload.get("report_type") != "ah_detector_report_v1":
        raise ValueError(f"Unsupported detector report_type: {payload.get('report_type')}")
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    human = summary.get("human_likeness", {})
    machine = summary.get("machine_generated_prob", {})
    if not isinstance(human, dict):
        human = {}
    if not isinstance(machine, dict):
        machine = {}

    counts = payload.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}

    return {
        "source_report": str(payload.get("source_report", "")).strip(),
        "source_key": resolve_key(str(payload.get("source_report", "")).strip()) if payload.get("source_report") else "",
        "source_run_id": str(payload.get("source_run_id", "")).strip(),
        "source_model_id": str(payload.get("source_model_id", "")).strip(),
        "backend": str(payload.get("backend", "")).strip(),
        "detected_rows": int(counts.get("detected_rows", 0) or 0),
        "failed_rows": int(counts.get("failed_rows", 0) or 0),
        "detector_coverage_rate": safe_float(summary.get("detector_coverage_rate", 0.0)),
        "human_likeness_mean": safe_float(human.get("mean", 0.0)),
        "machine_generated_prob_mean": safe_float(machine.get("mean", 0.0)),
    }


def build_lookup(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for item in items:
        for key in [item.get("source_key", ""), item.get("source_run_id", ""), item.get("source_model_id", "")]:
            key_text = str(key or "").strip()
            if key_text and key_text not in lookup:
                lookup[key_text] = item
    return lookup


def match_sidecar(model_row: Dict[str, Any], lookup: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for key in [model_row.get("source_key", ""), model_row.get("run_id", ""), model_row.get("model_id", "")]:
        key_text = str(key or "").strip()
        if key_text and key_text in lookup:
            return lookup[key_text]
    return None


def normalize_judge_human_like(ai_tone_mean: float, naturalness_mean: float) -> float:
    naturalness_norm = max(0.0, min(1.0, (naturalness_mean - 1.0) / 4.0))
    ai_tone_inv = max(0.0, min(1.0, (5.0 - ai_tone_mean) / 4.0))
    return float(0.5 * naturalness_norm + 0.5 * ai_tone_inv)


def build_guardrail(champion: Dict[str, Any]) -> Dict[str, float]:
    return {
        "bleu_min": champion["bleu"] * 0.90,
        "chrfpp_min": champion["chrfpp"] * 0.93,
        "ter_max": champion["ter"] * 1.08,
        "bertscore_f1_min": champion["bertscore_f1"] - 0.01,
    }


def semantic_gate_result(model: Dict[str, Any], guardrail: Dict[str, float]) -> Dict[str, Any]:
    checks = {
        "bleu": safe_float(model["bleu"]) >= guardrail["bleu_min"],
        "chrfpp": safe_float(model["chrfpp"]) >= guardrail["chrfpp_min"],
        "ter": safe_float(model["ter"]) <= guardrail["ter_max"],
        "bertscore_f1": safe_float(model["bertscore_f1"]) >= guardrail["bertscore_f1_min"],
    }
    failed = [metric for metric, passed in checks.items() if not passed]
    return {
        "semantic_gate_passed": len(failed) == 0,
        "semantic_gate_failed": len(failed) > 0,
        "semantic_gate_failed_metrics": failed,
    }


def compute_ai_humanization_score(detector_human_like: float | None, judge_human_like: float | None) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    if detector_human_like is not None:
        weighted_sum += 0.60 * detector_human_like
        total_weight += 0.60
    if judge_human_like is not None:
        weighted_sum += 0.40 * judge_human_like
        total_weight += 0.40
    if total_weight <= 0:
        return 0.0
    return float(weighted_sum / total_weight)


def dump_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "rank",
        "semantic_tier",
        "model_id",
        "mode",
        "run_id",
        "sample_count",
        "semantic_gate_passed",
        "ai_humanization_score",
        "fastdetect_human_like",
        "judge_human_like",
        "bleu",
        "chrfpp",
        "ter",
        "bertscore_f1",
        "style_diff",
        "burstiness",
        "format_violation_rate",
        "source_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def build_ranked_list(rows: List[Dict[str, Any]], metric_key: str) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: safe_float(row.get(metric_key, 0.0)), reverse=True)
    return [
        {
            "rank": idx + 1,
            "model_id": row["model_id"],
            "mode": row["mode"],
            "run_id": row["run_id"],
            "value": safe_float(row.get(metric_key, 0.0)),
            "semantic_gate_passed": bool(row.get("semantic_gate_passed", False)),
            "source_file": row["source_file"],
        }
        for idx, row in enumerate(ranked)
    ]


def main() -> int:
    args = parse_args()
    model_paths = expand_input_paths(args.model_reports)
    judge_paths = expand_input_paths(args.judge_reports)
    detector_paths = expand_input_paths(args.detector_reports)
    if not model_paths:
        raise ValueError("No model reports found. Check --model-reports.")

    champion = load_model_report(Path(args.champion_report))
    guardrail = build_guardrail(champion)

    judge_lookup = build_lookup([load_judge_report(path) for path in judge_paths]) if judge_paths else {}
    detector_lookup = build_lookup([load_detector_report(path) for path in detector_paths]) if detector_paths else {}

    models: List[Dict[str, Any]] = []
    for path in model_paths:
        model_row = load_model_report(path)
        gate = semantic_gate_result(model_row, guardrail)

        judge_row = match_sidecar(model_row, judge_lookup)
        detector_row = match_sidecar(model_row, detector_lookup)

        judge_human_like = None
        if judge_row:
            judge_human_like = safe_float(judge_row.get("judge_human_like", 0.0))

        detector_human_like = None
        if detector_row:
            detector_human_like = safe_float(detector_row.get("human_likeness_mean", 0.0))

        model_row.update(gate)
        model_row["fastdetect_human_like"] = detector_human_like
        model_row["judge_human_like"] = judge_human_like
        model_row["judge_overall_score"] = safe_float(judge_row.get("overall_score_mean", 0.0)) if judge_row else 0.0
        model_row["judge_naturalness"] = safe_float(judge_row.get("naturalness_score_mean", 0.0)) if judge_row else 0.0
        model_row["judge_semantic_fidelity"] = safe_float(judge_row.get("semantic_fidelity_score_mean", 0.0)) if judge_row else 0.0
        model_row["judge_terminology_accuracy"] = safe_float(judge_row.get("terminology_accuracy_score_mean", 0.0)) if judge_row else 0.0
        model_row["judge_edit_value"] = safe_float(judge_row.get("edit_value_score_mean", 0.0)) if judge_row else 0.0
        model_row["detector_backend"] = detector_row.get("backend", "") if detector_row else ""
        model_row["judge_model"] = judge_row.get("judge_model", "") if judge_row else ""
        model_row["detector_coverage_rate"] = safe_float(detector_row.get("detector_coverage_rate", 0.0)) if detector_row else 0.0
        model_row["ai_humanization_score"] = compute_ai_humanization_score(detector_human_like, judge_human_like)
        model_row["semantic_tier"] = "tier_1" if model_row["semantic_gate_passed"] else "tier_2"
        models.append(model_row)

    ranked_models = sorted(
        models,
        key=lambda row: (
            0 if row.get("semantic_gate_passed") else 1,
            -safe_float(row.get("ai_humanization_score", 0.0)),
            -safe_float(row.get("fastdetect_human_like", -1.0)),
            -safe_float(row.get("judge_human_like", -1.0)),
        ),
    )
    for idx, row in enumerate(ranked_models, start=1):
        row["rank"] = idx

    payload = {
        "report_type": "ah_evalboard_v2",
        "generated_at": datetime.now().isoformat(),
        "settings": {
            "model_reports": [str(path) for path in model_paths],
            "judge_reports": [str(path) for path in judge_paths],
            "detector_reports": [str(path) for path in detector_paths],
            "champion_report": str(args.champion_report),
            "ai_humanization_formula": "0.60 * fastdetect_human_like + 0.40 * judge_human_like",
        },
        "champion_guardrail": {
            "champion_model_id": champion["model_id"],
            "champion_run_id": champion["run_id"],
            "thresholds": guardrail,
        },
        "counts": {
            "model_reports": len(model_paths),
            "judge_reports": len(judge_paths),
            "detector_reports": len(detector_paths),
            "merged_models": len(ranked_models),
        },
        "models": ranked_models,
        "leaderboards": {
            "ai_humanization": build_ranked_list(ranked_models, "ai_humanization_score"),
            "fastdetect_human_like": build_ranked_list(ranked_models, "fastdetect_human_like"),
            "judge_human_like": build_ranked_list(ranked_models, "judge_human_like"),
            "bertscore_f1": build_ranked_list(ranked_models, "bertscore_f1"),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.csv_output:
        dump_csv(Path(args.csv_output), ranked_models)

    print("=" * 72)
    print("Academic Humanize evalboard v2 merge finished")
    print("=" * 72)
    print(f"model_reports: {len(model_paths)}")
    print(f"merged_models: {len(ranked_models)}")
    print(f"output: {output_path}")
    if args.csv_output:
        print(f"csv_output: {args.csv_output}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
