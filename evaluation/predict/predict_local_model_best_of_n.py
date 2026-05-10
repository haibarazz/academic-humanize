"""
Generate Best-of-N candidates with a local Transformers model and optional LoRA.

This script is prediction-only. It produces one flat row per candidate so the
existing LLM-as-Judge script can score candidates with minimal changes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

AH_TASK = "ah"
SYSTEM_PROMPT = "你是学术英语润色助手，请降低 AI 味并保持原意准确。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Academic Humanize Best-of-N 本地模型候选生成")
    parser.add_argument("--val-file", type=str, default="cloud_data/ah_v2/train/final_train_v2.json", help="输入数据文件路径（JSON）")
    parser.add_argument("--model-path", type=str, required=True, help="基础模型名称或本地路径")
    parser.add_argument("--adapter-path", type=str, default=None, help="可选 LoRA adapter 路径")
    parser.add_argument("--output", type=str, required=True, help="输出候选报告路径")
    parser.add_argument("--max-samples", type=int, default=None, help="最多抽取多少条 source 样本；调试用")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-candidates", type=int, default=4, help="每条 input 生成几个候选")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="每个候选最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度，建议 0.7-0.9")
    parser.add_argument("--top-p", type=float, default=0.9, help="top_p 采样参数")
    parser.add_argument("--top-k", type=int, default=50, help="top_k 采样参数；<=0 表示不传入 top_k")
    parser.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=True, help="开启 4bit 量化加载（默认）")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false", help="关闭 4bit 量化加载")
    parser.add_argument("--resume", action="store_true", help="从已有 output 断点续跑，跳过已完成 candidate_id")
    parser.add_argument("--save-every", type=int, default=20, help="每 N 个候选保存一次中间结果；0 表示只在结束时保存")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("输入数据必须是 JSON 数组")
    return [row for row in payload if isinstance(row, dict)]


def select_ah_samples(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        task_type = str(row.get("task_type", AH_TASK)).strip() or AH_TASK
        if task_type != AH_TASK:
            skipped.append({"index": idx, "sample_id": str(row.get("sample_id", f"row_{idx}")), "reason": "non_ah"})
            continue
        instruction = str(row.get("instruction", "")).strip()
        input_text = str(row.get("input", "")).strip()
        output_text = str(row.get("output", row.get("reference", ""))).strip()
        if not instruction or not input_text or not output_text:
            skipped.append({"index": idx, "sample_id": str(row.get("sample_id", f"row_{idx}")), "reason": "missing_required_fields"})
            continue
        selected.append(row)
    return selected, skipped


def build_ah_prompt(instruction: str, input_text: str) -> str:
    return f"{instruction}\n\n{input_text}" if input_text else instruction


def resolve_model_id(model_path: str, adapter_path: Optional[str]) -> str:
    if not adapter_path:
        return model_path
    adapter_name = Path(adapter_path).name or adapter_path
    return f"{model_path}+{adapter_name}"


def looks_like_local_path(value: str) -> bool:
    return value.startswith((".", "/", "~")) or value.startswith("models/") or value.startswith("checkpoints/")


def validate_local_model_path(model_path: str) -> None:
    if not looks_like_local_path(model_path):
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


def model_input_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def set_generation_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_candidate(
    *,
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> str:
    set_generation_seed(seed)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_ah_prompt(instruction, input_text)},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model_input_device(model))

    gen_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "do_sample": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if top_k and top_k > 0:
        gen_kwargs["top_k"] = int(top_k)

    gen_ids = model.generate(**gen_kwargs)
    output_ids = gen_ids[0, input_ids.shape[-1] :]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def build_candidate_id(source_sample_id: str, candidate_rank: int) -> str:
    return f"{source_sample_id}__bon_{candidate_rank:02d}"


def build_candidate_row(
    *,
    row: Dict[str, Any],
    source_index: int,
    candidate_rank: int,
    num_candidates: int,
    prediction: str,
    generation_seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    source_sample_id = str(row.get("sample_id", f"row_{source_index}"))
    candidate_id = build_candidate_id(source_sample_id, candidate_rank)
    return {
        "sample_id": candidate_id,
        "candidate_id": candidate_id,
        "original_sample_id": source_sample_id,
        "source_index": source_index,
        "candidate_rank": candidate_rank,
        "num_candidates": num_candidates,
        "paper_id": str(row.get("paper_id", "unknown")),
        "task_type": AH_TASK,
        "instruction": str(row.get("instruction", "")).strip(),
        "input": str(row.get("input", "")).strip(),
        "reference": str(row.get("output", row.get("reference", ""))).strip(),
        "prediction": (prediction or "").strip(),
        "generation": {
            "seed": generation_seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": True,
        },
    }


def is_completed_candidate(row: Dict[str, Any]) -> bool:
    prediction = str(row.get("prediction", "")).strip()
    return bool(prediction) and not prediction.startswith("ERROR:")


def load_resume_candidates(output_path: Path, model_id: str) -> Dict[str, Dict[str, Any]]:
    if not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"无法断点续跑：已有 output 不是 JSON object: {output_path}")
    if payload.get("report_type") != "ah_model_eval_v1":
        raise ValueError(f"无法断点续跑：已有 output report_type 不匹配: {payload.get('report_type')}")
    existing_model = str(payload.get("model_id", "")).strip()
    if existing_model and existing_model != model_id:
        raise ValueError(f"无法断点续跑：已有 output model_id={existing_model}, 当前 model_id={model_id}")

    completed: Dict[str, Dict[str, Any]] = {}
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        return completed
    for row in rows:
        if not isinstance(row, dict) or not is_completed_candidate(row):
            continue
        candidate_id = str(row.get("candidate_id", row.get("sample_id", ""))).strip()
        if candidate_id:
            completed[candidate_id] = row
    return completed


def count_non_ah(skipped_rows: List[Dict[str, Any]]) -> int:
    return sum(1 for row in skipped_rows if str(row.get("reason", "")).strip() == "non_ah")


def build_report(
    *,
    run_id: str,
    status: str,
    rows: List[Dict[str, Any]],
    all_rows_count: int,
    source_count: int,
    skipped_rows: List[Dict[str, Any]],
    model_id: str,
    adapter_path: Optional[Path],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "report_type": "ah_model_eval_v1",
        "run_id": run_id,
        "mode": "local_best_of_n",
        "task": AH_TASK,
        "model_id": model_id,
        "status": status,
        "settings": {
            "val_file": args.val_file,
            "model_path": args.model_path,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "num_candidates": args.num_candidates,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": True,
            "load_in_4bit": args.load_in_4bit,
            "resume": args.resume,
            "save_every": args.save_every,
            "compute_metrics": False,
        },
        "counts": {
            "total_rows": all_rows_count,
            "source_rows": source_count,
            "selected_rows": len(rows),
            "candidate_rows": len(rows),
            "expected_candidate_rows": source_count * int(args.num_candidates),
            "skipped_rows": len(skipped_rows),
            "non_ah_rows": count_non_ah(skipped_rows),
        },
        "task_results": None,
        "rows": rows,
        "badcases": [],
        "skipped_samples": skipped_rows,
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def main() -> int:
    args = parse_args()
    if args.num_candidates <= 0:
        raise ValueError("--num-candidates 必须大于 0")
    if args.temperature <= 0:
        raise ValueError("Best-of-N 需要采样多样性，--temperature 建议设置为 0.7-0.9")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_file = Path(args.val_file)
    output_path = Path(args.output)
    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    if adapter_path and not adapter_path.exists():
        raise FileNotFoundError(f"adapter 路径不存在: {adapter_path}")

    all_rows = load_rows(data_file)
    selected_rows, skipped_rows = select_ah_samples(all_rows)
    if args.max_samples and args.max_samples > 0 and len(selected_rows) > args.max_samples:
        selected_rows = random.sample(selected_rows, k=args.max_samples)

    model_id = resolve_model_id(args.model_path, str(adapter_path) if adapter_path else None)
    run_id = datetime.now().strftime("ah_bestofn_%Y%m%d_%H%M%S")

    print("=" * 72)
    print("Academic Humanize Best-of-N 本地候选生成")
    print("=" * 72)
    print(f"val_file:        {data_file}")
    print(f"model_path:      {args.model_path}")
    print(f"adapter_path:    {adapter_path if adapter_path else 'None'}")
    print(f"source_samples:  {len(selected_rows)} | skipped: {len(skipped_rows)}")
    print(f"num_candidates:  {args.num_candidates}")
    print(f"temperature:     {args.temperature}")
    print(f"top_p/top_k:     {args.top_p} / {args.top_k}")
    print(f"max_new_tokens:  {args.max_new_tokens}")
    print(f"resume:          {args.resume}")
    print(f"output:          {output_path}")
    print("=" * 72)

    indexed_jobs: List[Tuple[int, Dict[str, Any], int]] = []
    for source_index, row in enumerate(selected_rows, start=1):
        for candidate_rank in range(int(args.num_candidates)):
            indexed_jobs.append((source_index, row, candidate_rank))

    candidate_slots: List[Optional[Dict[str, Any]]] = [None] * len(indexed_jobs)
    pending_jobs: List[Tuple[int, Dict[str, Any], int, int]] = []
    completed_by_candidate_id: Dict[str, Dict[str, Any]] = {}
    if args.resume:
        completed_by_candidate_id = load_resume_candidates(output_path, model_id)

    for slot_index, (source_index, row, candidate_rank) in enumerate(indexed_jobs):
        source_sample_id = str(row.get("sample_id", f"row_{source_index}"))
        candidate_id = build_candidate_id(source_sample_id, candidate_rank)
        completed_row = completed_by_candidate_id.get(candidate_id)
        if completed_row is not None:
            candidate_slots[slot_index] = completed_row
            continue
        pending_jobs.append((slot_index, row, source_index, candidate_rank))

    if args.resume:
        print(f"resume: reused={len(indexed_jobs) - len(pending_jobs)} | pending={len(pending_jobs)}")

    model, tokenizer = load_local_model(
        model_path=args.model_path,
        adapter_path=str(adapter_path) if adapter_path else None,
        load_in_4bit=args.load_in_4bit,
    )

    def completed_rows() -> List[Dict[str, Any]]:
        return [row for row in candidate_slots if row is not None]

    iterator = tqdm(pending_jobs, desc="Generating Best-of-N", unit="candidate")
    for done_count, (slot_index, row, source_index, candidate_rank) in enumerate(iterator, start=1):
        generation_seed = int(args.seed) + source_index * 1000 + candidate_rank
        try:
            prediction = generate_candidate(
                model=model,
                tokenizer=tokenizer,
                instruction=str(row.get("instruction", "")).strip(),
                input_text=str(row.get("input", "")).strip(),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=generation_seed,
            )
        except Exception as exc:
            prediction = f"ERROR: {exc}"

        candidate_slots[slot_index] = build_candidate_row(
            row=row,
            source_index=source_index,
            candidate_rank=candidate_rank,
            num_candidates=int(args.num_candidates),
            prediction=prediction,
            generation_seed=generation_seed,
            args=args,
        )

        if args.save_every > 0 and done_count % args.save_every == 0:
            save_json(
                output_path,
                build_report(
                    run_id=run_id,
                    status="running",
                    rows=completed_rows(),
                    all_rows_count=len(all_rows),
                    source_count=len(selected_rows),
                    skipped_rows=skipped_rows,
                    model_id=model_id,
                    adapter_path=adapter_path,
                    args=args,
                ),
            )

    report = build_report(
        run_id=run_id,
        status="complete",
        rows=completed_rows(),
        all_rows_count=len(all_rows),
        source_count=len(selected_rows),
        skipped_rows=skipped_rows,
        model_id=model_id,
        adapter_path=adapter_path,
        args=args,
    )
    save_json(output_path, report)

    print("\n" + "=" * 72)
    print("Best-of-N 候选生成完成")
    print("=" * 72)
    print(f"source_rows:      {report['counts']['source_rows']}")
    print(f"candidate_rows:   {report['counts']['candidate_rows']} / {report['counts']['expected_candidate_rows']}")
    print("metrics: skipped")
    print(f"output:           {output_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
