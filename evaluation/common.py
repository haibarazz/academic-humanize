"""
Academic Humanize 评测公共函数。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

AH_TASK = "ah"


def _load_metrics():
    """
    Lazy-load metric dependencies.

    Prediction-only scripts should not require sacrebleu / bert-score. Metrics
    are imported only when a scoring function is actually called.
    """
    from evaluation.metrics import metrics

    return metrics


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"验证集文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("验证集必须为 JSON 数组")
    return [item for item in payload if isinstance(item, dict)]


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def rate_from_bools(values: List[bool]) -> float:
    if not values:
        return 0.0
    return float(sum(1 for value in values if value) / len(values))


def _skip_row(idx: int, row: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "index": idx,
        "sample_id": str(row.get("sample_id", f"row_{idx}")),
        "reason": reason,
        "task_type": str(row.get("task_type", "")).strip(),
    }


def select_ah_samples(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        task_type = str(row.get("task_type", AH_TASK)).strip() or AH_TASK
        if task_type != AH_TASK:
            skipped.append(_skip_row(idx, row, "non_ah"))
            continue

        instruction = str(row.get("instruction", "")).strip()
        input_text = str(row.get("input", "")).strip()
        output_text = str(row.get("output", "")).strip()
        if not instruction or not output_text:
            skipped.append(_skip_row(idx, row, "missing_instruction_or_output"))
            continue
        if not input_text:
            skipped.append(_skip_row(idx, row, "missing_input"))
            continue
        selected.append(row)
    return selected, skipped


def build_ah_prompt(instruction: str, input_text: str) -> str:
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def build_default_output_path(prefix: str, model_id: str) -> Path:
    safe_model = model_id.replace("/", "_").replace(" ", "_")
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{prefix}_{safe_model}_{stamp}.json"


def resolve_local_model_id(model_path: str, adapter_path: str | None = None) -> str:
    if not adapter_path:
        return model_path
    adapter_name = Path(adapter_path).name or adapter_path
    return f"{model_path}+{adapter_name}"


def build_eval_row(row: Dict[str, Any], prediction: str, default_index: int) -> Dict[str, Any]:
    metrics = _load_metrics()
    input_text = str(row.get("input", "")).strip()
    reference = str(row.get("output", row.get("reference", ""))).strip()
    prediction_text = (prediction or "").strip()
    format_diag = metrics.analyze_format_violations(prediction_text)

    one: Dict[str, Any] = {
        "sample_id": str(row.get("sample_id", f"row_{default_index}")),
        "paper_id": str(row.get("paper_id", "unknown")),
        "task_type": AH_TASK,
        "instruction": str(row.get("instruction", "")).strip(),
        "input": input_text,
        "reference": reference,
        "prediction": prediction_text,
        "sentence_bleu": float(metrics.compute_sentence_bleu(prediction_text, reference)),
        "style_diff": float(metrics.compute_style_diff(input_text, prediction_text)),
        "burstiness": float(metrics.compute_burstiness(prediction_text)),
    }
    one.update({key: bool(value) if isinstance(value, bool) else value for key, value in format_diag.items()})
    return one


def build_prediction_row(row: Dict[str, Any], prediction: str, default_index: int) -> Dict[str, Any]:
    """Build a dependency-light prediction row without computing metrics."""
    return {
        "sample_id": str(row.get("sample_id", f"row_{default_index}")),
        "paper_id": str(row.get("paper_id", "unknown")),
        "task_type": AH_TASK,
        "instruction": str(row.get("instruction", "")).strip(),
        "input": str(row.get("input", "")).strip(),
        "reference": str(row.get("output", row.get("reference", ""))).strip(),
        "prediction": (prediction or "").strip(),
    }


def summarize_task_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "bleu": 0.0,
            "chrfpp": 0.0,
            "ter": 0.0,
            "bertscore_f1": 0.0,
            "style_diff_mean": 0.0,
            "burstiness_mean": 0.0,
            "format_violation_rate": 0.0,
            "contains_cjk_rate": 0.0,
            "meta_prefix_rate": 0.0,
            "meta_suffix_rate": 0.0,
            "explanation_block_rate": 0.0,
            "bullet_or_list_tail_rate": 0.0,
        }

    hypotheses = [item["prediction"] for item in rows]
    references = [item["reference"] for item in rows]
    metrics = _load_metrics()
    return {
        "sample_count": len(rows),
        "bleu": float(metrics.compute_bleu(hypotheses, references)),
        "chrfpp": float(metrics.compute_chrfpp(hypotheses, references)),
        "ter": float(metrics.compute_ter(hypotheses, references)),
        "bertscore_f1": float(metrics.compute_bertscore_f1(hypotheses, references)),
        "style_diff_mean": mean([float(item.get("style_diff", 0.0)) for item in rows]),
        "burstiness_mean": mean([float(item.get("burstiness", 0.0)) for item in rows]),
        "format_violation_rate": rate_from_bools([bool(item.get("format_violation", False)) for item in rows]),
        "contains_cjk_rate": rate_from_bools([bool(item.get("contains_cjk", False)) for item in rows]),
        "meta_prefix_rate": rate_from_bools([bool(item.get("meta_prefix", False)) for item in rows]),
        "meta_suffix_rate": rate_from_bools([bool(item.get("meta_suffix", False)) for item in rows]),
        "explanation_block_rate": rate_from_bools([bool(item.get("explanation_block", False)) for item in rows]),
        "bullet_or_list_tail_rate": rate_from_bools([bool(item.get("bullet_or_list_tail", False)) for item in rows]),
    }


def build_badcases(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=lambda item: float(item.get("sentence_bleu", 0.0)))
    items = ranked[: max(int(limit), 0)]
    return [
        {
            "sample_id": item["sample_id"],
            "paper_id": item["paper_id"],
            "sentence_bleu": float(item.get("sentence_bleu", 0.0)),
            "input": item["input"],
            "prediction": item["prediction"],
            "reference": item["reference"],
            "format_violation": bool(item.get("format_violation", False)),
        }
        for item in items
    ]


def count_non_ah(skipped_rows: List[Dict[str, Any]]) -> int:
    return sum(1 for item in skipped_rows if str(item.get("reason", "")).strip() == "non_ah")
