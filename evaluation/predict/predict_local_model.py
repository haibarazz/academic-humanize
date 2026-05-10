"""
Academic Humanize 本地模型评测脚本（Transformers + 可选 LoRA）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    resolve_local_model_id,
    select_ah_samples,
    summarize_task_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Academic Humanize 本地模型评测（Transformers）")
    parser.add_argument("--val-file", type=str, default="cloud_data/ah_v2/val/final_val_v2.json", help="验证集文件路径（JSON）")
    parser.add_argument("--model-path", type=str, required=True, help="基础模型名称或本地路径")
    parser.add_argument("--adapter-path", type=str, default=None, help="可选 LoRA adapter 路径")
    parser.add_argument("--output", type=str, default=None, help="输出报告路径（默认 results/predictions/ah_eval_local_*.json）")
    parser.add_argument("--max-samples", type=int, default=None, help="最多评测样本数（调试用）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--badcase-limit", type=int, default=20, help="badcase 条数")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="生成最大 token")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    parser.add_argument("--top-p", type=float, default=1.0, help="top_p")
    parser.add_argument("--do-sample", action="store_true", help="是否开启采样（默认关闭）")
    parser.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=True, help="开启 4bit 量化加载（默认）")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false", help="关闭 4bit 量化加载")
    parser.add_argument("--compute-metrics", action="store_true", help="生成后立即计算指标；默认只保存预测结果")
    return parser.parse_args()


def build_prediction_output_path(model_id: str) -> Path:
    base_path = build_default_output_path("ah_eval_local", model_id)
    out_dir = Path("results/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / base_path.name


def _looks_like_local_path(value: str) -> bool:
    return value.startswith((".", "/", "~")) or value.startswith("models/") or value.startswith("checkpoints/")


def validate_local_model_path(model_path: str) -> None:
    if not _looks_like_local_path(model_path):
        return
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"本地模型路径不存在: {path}")
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"本地模型路径不完整，缺少 config.json: {path}")


def load_local_model(model_path: str, adapter_path: Optional[str], load_in_4bit: bool):
    validate_local_model_path(model_path)
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_prediction(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    messages = [
        {"role": "system", "content": "你是学术英语润色助手，请降低 AI 味并保持原意准确。"},
        {"role": "user", "content": build_ah_prompt(instruction, input_text)},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = {
        "input_ids": input_ids,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    gen_ids = model.generate(**gen_kwargs)
    output_ids = gen_ids[0, input_ids.shape[-1] :]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    val_file = Path(args.val_file)
    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    if adapter_path and not adapter_path.exists():
        raise FileNotFoundError(f"adapter 路径不存在: {adapter_path}")

    all_rows = load_rows(val_file)
    selected_rows, skipped_rows = select_ah_samples(all_rows)
    if args.max_samples and args.max_samples > 0 and len(selected_rows) > args.max_samples:
        selected_rows = random.sample(selected_rows, k=args.max_samples)

    model_id = resolve_local_model_id(args.model_path, str(adapter_path) if adapter_path else None)
    output_path = Path(args.output) if args.output else build_prediction_output_path(model_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Academic Humanize 本地模型评测")
    print("=" * 72)
    print(f"val_file: {val_file}")
    print(f"model_path: {args.model_path}")
    print(f"adapter_path: {adapter_path if adapter_path else 'None'}")
    print(f"selected: {len(selected_rows)} | skipped: {len(skipped_rows)}")
    print("=" * 72)

    model, tokenizer = load_local_model(
        model_path=args.model_path,
        adapter_path=str(adapter_path) if adapter_path else None,
        load_in_4bit=args.load_in_4bit,
    )

    prediction_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(tqdm(selected_rows, desc="Generating", unit="sample"), start=1):
        prediction = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            instruction=str(row.get("instruction", "")).strip(),
            input_text=str(row.get("input", "")).strip(),
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        prediction_rows.append(build_prediction_row(row, prediction, i))
        if args.compute_metrics:
            metric_rows.append(build_eval_row(row, prediction, i))
        if i % 10 == 0 or i == len(selected_rows):
            print(f"进度: {i}/{len(selected_rows)}")

    task_results = summarize_task_results(metric_rows) if args.compute_metrics else None
    output_rows = metric_rows if args.compute_metrics else prediction_rows
    report: Dict[str, Any] = {
        "report_type": "ah_model_eval_v1",
        "run_id": datetime.now().strftime("ah_local_%Y%m%d_%H%M%S"),
        "mode": "local",
        "task": AH_TASK,
        "model_id": model_id,
        "settings": {
            "val_file": str(val_file),
            "model_path": args.model_path,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "load_in_4bit": args.load_in_4bit,
            "compute_metrics": args.compute_metrics,
        },
        "counts": {
            "total_rows": len(all_rows),
            "selected_rows": len(output_rows),
            "skipped_rows": len(skipped_rows),
            "non_ah_rows": count_non_ah(skipped_rows),
        },
        "task_results": task_results,
        "rows": output_rows,
        "badcases": build_badcases(metric_rows, args.badcase_limit) if args.compute_metrics else [],
        "skipped_samples": skipped_rows,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("本地模型预测完成")
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
