"""
Academic Humanize API 模型评测脚本。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import (
    AH_TASK,
    build_badcases,
    build_default_output_path,
    build_eval_row,
    build_ah_prompt,
    build_prediction_row,
    count_non_ah,
    load_rows,
    select_ah_samples,
    summarize_task_results,
)
from scripts.utils.api_client import create_client

_THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Academic Humanize API 模型评测")
    parser.add_argument("--val-file", type=str, default="cloud_data/ah_v2/val/final_val_v2.json", help="验证集文件路径（JSON）")
    parser.add_argument("--api-model", type=str, required=True, help="API 模型名称")
    parser.add_argument("--output", type=str, default=None, help="输出报告路径（默认 results/predictions/ah_eval_api_*.json）")
    parser.add_argument("--max-samples", type=int, default=None, help="最多评测样本数（调试用）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--badcase-limit", type=int, default=20, help="badcase 条数")
    parser.add_argument("--max-tokens", type=int, default=1600, help="API 最大输出 token 数")
    parser.add_argument("--connection-test", action="store_true", help="只测试 API 连通性，不写评估报告")
    parser.add_argument("--compute-metrics", action="store_true", help="生成后立即计算指标；默认只保存预测结果")
    parser.add_argument("--max-concurrency", type=int, default=1, help="并发 API 调用数；默认 1")
    parser.add_argument("--resume", action="store_true", help="从已有 output 断点续跑，跳过已完成 sample_id")
    parser.add_argument("--save-every", type=int, default=10, help="每 N 条保存一次中间预测；0 表示只在结束时保存")
    return parser.parse_args()


def build_prediction_output_path(model_id: str) -> Path:
    base_path = build_default_output_path("ah_eval_api", model_id)
    out_dir = Path("results/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / base_path.name


def configure_client(client, max_tokens: int):
    client.max_tokens = int(max_tokens)
    return client


def get_thread_client(args: argparse.Namespace):
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = configure_client(create_client(model=args.api_model), args.max_tokens)
        _THREAD_LOCAL.client = client
    return client


def predict_one_row(
    *,
    idx: int,
    row: Dict[str, Any],
    args: argparse.Namespace,
    client=None,
) -> Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]:
    active_client = client or get_thread_client(args)
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    prompt = build_ah_prompt(instruction, input_text)
    system_prompt = "你是学术英语润色助手，请降低 AI 味并保持原意准确。"
    prediction = (active_client.call(system_prompt, prompt) or "").strip()
    metric_row = build_eval_row(row, prediction, idx) if args.compute_metrics else None
    return idx, build_prediction_row(row, prediction, idx), metric_row


def completed_rows_in_order(rows: List[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row is not None]


def is_completed_prediction(row: Dict[str, Any]) -> bool:
    prediction = str(row.get("prediction", "")).strip()
    return bool(prediction) and not prediction.startswith("ERROR:")


def load_resume_rows(output_path: Path, api_model: str) -> Dict[str, Dict[str, Any]]:
    if not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"无法断点续跑：已有 output 不是 JSON object: {output_path}")
    if payload.get("report_type") != "ah_model_eval_v1":
        raise ValueError(f"无法断点续跑：已有 output report_type 不匹配: {payload.get('report_type')}")
    existing_model = str(payload.get("model_id", "")).strip()
    if existing_model and existing_model != api_model:
        raise ValueError(f"无法断点续跑：已有 output model_id={existing_model}, 当前 api_model={api_model}")

    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        return {}

    completed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not is_completed_prediction(row):
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            completed[sample_id] = row
    return completed


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    val_file = Path(args.val_file)
    all_rows = load_rows(val_file)
    selected_rows, skipped_rows = select_ah_samples(all_rows)
    if args.max_samples and args.max_samples > 0 and len(selected_rows) > args.max_samples:
        selected_rows = random.sample(selected_rows, k=args.max_samples)

    output_path = Path(args.output) if args.output else build_prediction_output_path(args.api_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and not args.output:
        print("⚠️ --resume 通常需要配合 --output 指向已有结果文件；未指定 --output 时默认会创建新的时间戳文件。")

    print("=" * 72)
    print("Academic Humanize API 模型评测")
    print("=" * 72)
    print(f"val_file: {val_file}")
    print(f"api_model: {args.api_model}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"concurrency: {max(1, int(args.max_concurrency))}")
    print(f"resume: {args.resume}")
    print(f"selected: {len(selected_rows)} | skipped: {len(skipped_rows)}")
    print("=" * 72)

    client = configure_client(create_client(model=args.api_model), args.max_tokens)
    if args.connection_test:
        if not selected_rows:
            raise ValueError("没有可用于测试的 AH 验证样本")
        row = selected_rows[0]
        prompt = build_ah_prompt(
            str(row.get("instruction", "")).strip(),
            str(row.get("input", "")).strip(),
        )
        system_prompt = "你是学术英语润色助手，请降低 AI 味并保持原意准确。"
        prediction = (client.call(system_prompt, prompt) or "").strip()
        print("\n" + "=" * 72)
        print("API 连通性测试完成")
        print("=" * 72)
        print(f"api_model: {args.api_model}")
        print(f"max_tokens: {client.max_tokens}")
        print(f"response_chars: {len(prediction)}")
        print(f"response_preview: {prediction[:300]}")
        print("=" * 72)
        if not prediction:
            raise RuntimeError("API 连通成功但返回内容为空")
        return 0

    run_id = datetime.now().strftime("ah_api_%Y%m%d_%H%M%S")

    def build_report(rows: List[Dict[str, Any]], task_results=None, badcases=None, status: str = "running") -> Dict[str, Any]:
        return {
            "report_type": "ah_model_eval_v1",
            "run_id": run_id,
            "mode": "api",
            "task": AH_TASK,
            "model_id": args.api_model,
            "status": status,
            "settings": {
                "val_file": str(val_file),
                "api_model": args.api_model,
                "max_samples": args.max_samples,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "compute_metrics": args.compute_metrics,
                "max_concurrency": args.max_concurrency,
                "resume": args.resume,
                "save_every": args.save_every,
            },
            "counts": {
                "total_rows": len(all_rows),
                "selected_rows": len(rows),
                "skipped_rows": len(skipped_rows),
                "non_ah_rows": count_non_ah(skipped_rows),
            },
            "task_results": task_results,
            "rows": rows,
            "badcases": badcases or [],
            "skipped_samples": skipped_rows,
        }

    def save_report(payload: Dict[str, Any]) -> None:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    max_concurrency = max(1, int(args.max_concurrency))
    indexed_rows = list(enumerate(selected_rows, start=1))
    prediction_slots: List[Optional[Dict[str, Any]]] = [None] * len(indexed_rows)
    metric_slots: List[Optional[Dict[str, Any]]] = [None] * len(indexed_rows)
    pending_rows: List[Tuple[int, Dict[str, Any]]] = indexed_rows

    if args.resume:
        completed_by_sample_id = load_resume_rows(output_path, args.api_model)
        pending_rows = []
        for idx, row in indexed_rows:
            sample_id = str(row.get("sample_id", f"row_{idx}")).strip()
            completed_row = completed_by_sample_id.get(sample_id)
            if completed_row is None:
                pending_rows.append((idx, row))
                continue
            prediction_slots[idx - 1] = completed_row
            if args.compute_metrics:
                metric_slots[idx - 1] = build_eval_row(row, str(completed_row.get("prediction", "")), idx)
        reused_count = len(indexed_rows) - len(pending_rows)
        print(f"resume: reused={reused_count} | pending={len(pending_rows)}")

    if max_concurrency == 1:
        iterator = tqdm(pending_rows, desc="Evaluating API", unit="sample")
        for done_count, (idx, row) in enumerate(iterator, start=1):
            _, prediction_row, metric_row = predict_one_row(idx=idx, row=row, args=args, client=client)
            prediction_slots[idx - 1] = prediction_row
            if args.compute_metrics:
                metric_slots[idx - 1] = metric_row
            if args.save_every > 0 and done_count % args.save_every == 0:
                partial_rows = completed_rows_in_order(metric_slots if args.compute_metrics else prediction_slots)
                save_report(build_report(partial_rows, status="running"))
    else:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(predict_one_row, idx=idx, row=row, args=args): (idx, row)
                for idx, row in pending_rows
            }
            iterator = tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating API x{max_concurrency}", unit="sample")
            for done_count, future in enumerate(iterator, start=1):
                idx, row = futures[future]
                try:
                    _, prediction_row, metric_row = future.result()
                except Exception as exc:
                    prediction = f"ERROR: {exc}"
                    prediction_row = build_prediction_row(row, prediction, idx)
                    metric_row = build_eval_row(row, prediction, idx) if args.compute_metrics else None
                prediction_slots[idx - 1] = prediction_row
                if args.compute_metrics:
                    metric_slots[idx - 1] = metric_row
                if args.save_every > 0 and done_count % args.save_every == 0:
                    partial_rows = completed_rows_in_order(metric_slots if args.compute_metrics else prediction_slots)
                    save_report(build_report(partial_rows, status="running"))

    prediction_rows = completed_rows_in_order(prediction_slots)
    metric_rows = completed_rows_in_order(metric_slots) if args.compute_metrics else []

    save_report(build_report(metric_rows if args.compute_metrics else prediction_rows, status="predictions_complete"))
    task_results = summarize_task_results(metric_rows) if args.compute_metrics else None
    output_rows = metric_rows if args.compute_metrics else prediction_rows
    report = build_report(
        output_rows,
        task_results=task_results,
        badcases=build_badcases(metric_rows, args.badcase_limit) if args.compute_metrics else [],
        status="complete",
    )
    save_report(report)

    print("\n" + "=" * 72)
    print("API 预测完成")
    print("=" * 72)
    if args.compute_metrics and task_results:
        print(
            f"Academic Humanize: n={task_results['sample_count']} | BLEU={task_results['bleu']:.4f} | "
            f"chrF++={task_results['chrfpp']:.4f} | TER={task_results['ter']:.4f} | "
            f"BERTScore={task_results['bertscore_f1']:.4f} | "
            f"style_diff={task_results['style_diff_mean']:.4f} | burstiness={task_results['burstiness_mean']:.4f} | "
            f"format_violation={task_results['format_violation_rate']:.4f}"
        )
    else:
        print("metrics: skipped (run evaluation/metrics/compute_metrics.py later)")
    print(f"报告文件: {output_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
