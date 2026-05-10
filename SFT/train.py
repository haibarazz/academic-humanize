"""
Qwen2.5 QLoRA 微调训练脚本（YAML 配置化 + CLI 覆盖）。

功能目标：
1) 支持 --config YAML 统一管理训练参数
2) 兼容旧 CLI 参数，且优先级为 CLI > YAML > 默认值
3) 支持 train_file / val_file
4) 训练目录自动保存解析后的最终配置
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
try:
    # 作为包导入（如 from SFT.train import ...）
    from .train_utils import build_early_stopping_components
except ImportError:
    # 作为脚本运行（python SFT/train.py）
    from train_utils import build_early_stopping_components

# 设置模型下载镜像（使用 HF-Mirror 镜像源）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "load_in_4bit": True,
    },
    "data": {
        "train_file": "cloud_data/ah_v2/train/final_train_v2.json",
        "val_file": "cloud_data/ah_v2/val/final_val_v2.json",
        "max_seq_len": 2048,
    },
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
    },
    "train": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "linear",
        "save_steps": 100,
        "logging_steps": 5,
        "logging_first_step": True,
        "save_total_limit": 2,
        "evaluation_strategy": "steps",
        "eval_steps": 100,
        "early_stopping_enabled": True,
        "early_stopping_patience": 2,
        "early_stopping_threshold": 0.0,
    },
    "output": {
        "output_dir": "./checkpoints",
    },
    "runtime": {
        "seed": 42,
        "max_new_tokens": 512,
    },
}


@dataclass
class TrainConfig:
    model_name_or_path: str
    load_in_4bit: bool
    train_file: str
    val_file: Optional[str]
    max_seq_len: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    lr_scheduler_type: str
    save_steps: int
    logging_steps: int
    logging_first_step: bool
    save_total_limit: int
    evaluation_strategy: str
    eval_steps: int
    early_stopping_enabled: bool
    early_stopping_patience: int
    early_stopping_threshold: float
    output_dir: str
    seed: int
    max_new_tokens: int


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def normalize_yaml_shape(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(raw_cfg)
    # 兼容旧写法：顶层 data_path
    data_path = cfg.pop("data_path", None)
    if data_path:
        cfg.setdefault("data", {})
        cfg["data"]["train_file"] = data_path
    return cfg


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with p.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML 配置必须是对象结构")
    return normalize_yaml_shape(payload)


def apply_cli_overrides(merged_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = deep_merge({}, merged_cfg)

    def set_if_not_none(path: List[str], value: Any):
        if value is None:
            return
        root = cfg
        for key in path[:-1]:
            root = root.setdefault(key, {})
        root[path[-1]] = value

    # 数据
    set_if_not_none(["data", "train_file"], args.data_path)
    set_if_not_none(["data", "train_file"], args.train_file)
    set_if_not_none(["data", "val_file"], args.val_file)
    set_if_not_none(["data", "max_seq_len"], args.max_seq_len)

    # 模型
    set_if_not_none(["model", "name_or_path"], args.model)
    set_if_not_none(["model", "load_in_4bit"], args.load_in_4bit)

    # LoRA
    set_if_not_none(["lora", "r"], args.lora_r)
    set_if_not_none(["lora", "alpha"], args.lora_alpha)
    set_if_not_none(["lora", "dropout"], args.lora_dropout)

    # 训练
    set_if_not_none(["train", "num_train_epochs"], args.epochs)
    set_if_not_none(["train", "per_device_train_batch_size"], args.batch_size)
    set_if_not_none(["train", "per_device_eval_batch_size"], args.eval_batch_size)
    set_if_not_none(["train", "gradient_accumulation_steps"], args.grad_accum)
    set_if_not_none(["train", "learning_rate"], args.lr)
    set_if_not_none(["train", "warmup_ratio"], args.warmup_ratio)
    set_if_not_none(["train", "lr_scheduler_type"], args.lr_scheduler_type)
    set_if_not_none(["train", "save_steps"], args.save_steps)
    set_if_not_none(["train", "logging_steps"], args.logging_steps)
    set_if_not_none(["train", "save_total_limit"], args.save_total_limit)
    set_if_not_none(["train", "eval_steps"], args.eval_steps)
    set_if_not_none(["train", "early_stopping_enabled"], args.early_stop)
    set_if_not_none(["train", "early_stopping_patience"], args.early_stop_patience)
    set_if_not_none(["train", "early_stopping_threshold"], args.early_stop_threshold)

    # 输出
    set_if_not_none(["output", "output_dir"], args.output_dir)

    # 运行时
    set_if_not_none(["runtime", "seed"], args.seed)
    set_if_not_none(["runtime", "max_new_tokens"], args.max_new_tokens)
    return cfg


def build_train_config(cfg: Dict[str, Any]) -> TrainConfig:
    model = cfg.get("model", {})
    data = cfg.get("data", {})
    lora = cfg.get("lora", {})
    train = cfg.get("train", {})
    output = cfg.get("output", {})
    runtime = cfg.get("runtime", {})

    val_file = data.get("val_file")
    if isinstance(val_file, str) and not val_file.strip():
        val_file = None

    return TrainConfig(
        model_name_or_path=str(model.get("name_or_path")),
        load_in_4bit=bool(model.get("load_in_4bit", True)),
        train_file=str(data.get("train_file")),
        val_file=val_file if val_file is None else str(val_file),
        max_seq_len=int(data.get("max_seq_len", 2048)),
        lora_r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        num_train_epochs=int(train.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(train.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train.get("learning_rate", 1e-4)),
        warmup_ratio=float(train.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(train.get("lr_scheduler_type", "linear")),
        save_steps=int(train.get("save_steps", 100)),
        logging_steps=int(train.get("logging_steps", 5)),
        logging_first_step=bool(train.get("logging_first_step", True)),
        save_total_limit=int(train.get("save_total_limit", 2)),
        evaluation_strategy=str(train.get("evaluation_strategy", "steps")),
        eval_steps=int(train.get("eval_steps", 100)),
        early_stopping_enabled=bool(train.get("early_stopping_enabled", True)),
        early_stopping_patience=int(train.get("early_stopping_patience", 2)),
        early_stopping_threshold=float(train.get("early_stopping_threshold", 0.0)),
        output_dir=str(output.get("output_dir", "./checkpoints")),
        seed=int(runtime.get("seed", 42)),
        max_new_tokens=int(runtime.get("max_new_tokens", 512)),
    )


def load_training_data(data_path: str) -> Dataset:
    """
    加载训练/验证数据，支持 JSON 与 JSONL。
    """
    if data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_path, split="train")
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{data_path} 必须是 JSON 数组")
        dataset = Dataset.from_list(data)

    print(f"📊 加载数据: {data_path} -> {len(dataset)} 条")
    return dataset


def format_sample_for_qwen(record, tokenizer, max_seq_len):
    instruction = (record.get("instruction") or "").strip()
    input_text = (record.get("input") or "").strip()
    output_text = (record.get("output") or "").strip()

    if not instruction or not output_text:
        return {"input_ids": [], "labels": []}

    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction

    msgs_no_assist = [
        {"role": "system", "content": "你是一个专业的学术助手，请按照用户要求完成任务。"},
        {"role": "user", "content": user_content},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        msgs_no_assist,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )

    msgs_full = msgs_no_assist + [{"role": "assistant", "content": output_text}]
    full_ids = tokenizer.apply_chat_template(
        msgs_full,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    )

    # 至少给 answer 留一点空间
    min_answer_space = 64
    if len(prompt_ids) > max_seq_len - min_answer_space:
        return {"input_ids": [], "labels": []}

    full_ids = full_ids[:max_seq_len]
    cut = len(prompt_ids)

    if cut >= len(full_ids):
        return {"input_ids": [], "labels": []}

    labels = [-100] * cut + full_ids[cut:]
    return {"input_ids": full_ids, "labels": labels}



def process_dataset(dataset, tokenizer, max_seq_len):
    proc_dataset = dataset.map(
        lambda x: format_sample_for_qwen(x, tokenizer, max_seq_len),
        remove_columns=dataset.column_names,
    )

    def valid_sample(x):
        return (
            len(x["input_ids"]) > 0
            and len(x["labels"]) > 0
            and any(l != -100 for l in x["labels"])
        )

    proc_dataset = proc_dataset.filter(valid_sample)
    return proc_dataset


class QwenSftCollator:
    def __init__(self, pad_id: int, max_length: int = 2048, ignore_id: int = -100):
        self.pad_id = pad_id
        self.max_length = max_length
        self.ignore_id = ignore_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = min(max_len, self.max_length)

        input_ids = []
        labels = []
        attention_mask = []

        for f in features:
            ids = f["input_ids"][:max_len]
            lbs = f["labels"][:max_len]

            real_len = len(ids)
            pad = max_len - real_len

            if pad > 0:
                ids = ids + [self.pad_id] * pad
                lbs = lbs + [self.ignore_id] * pad

            input_ids.append(torch.tensor(ids, dtype=torch.long))
            labels.append(torch.tensor(lbs, dtype=torch.long))
            attention_mask.append(
                torch.tensor([1] * real_len + [0] * pad, dtype=torch.long)
            )

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }

def load_model_and_tokenizer(model_name_or_path: str, load_in_4bit: bool):
    """
    加载基础模型与 tokenizer。
    """
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
        model_name_or_path,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def setup_lora(model, lora_r: int, lora_alpha: int, lora_dropout: float):
    """
    构建 LoRA 适配器。
    """
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["qwen2"],
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.enable_input_require_grads()
    peft_model.config.use_cache = False
    peft_model.print_trainable_parameters()
    return peft_model


def save_resolved_config(run_dir: str, full_config_dict: Dict[str, Any], final_config: TrainConfig) -> None:
    """
    保存解析后的配置，确保实验可复现。
    """
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)

    yaml_path = p / "resolved_config.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(full_config_dict, f, allow_unicode=True, sort_keys=False)

    json_path = p / "resolved_config.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(final_config), f, ensure_ascii=False, indent=2)


def train(config: TrainConfig, full_config_dict: Dict[str, Any]) -> str:
    """
    执行训练主流程，返回 run_dir。
    """
    torch.manual_seed(config.seed)

    now_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.output_dir, f"qwen25_academic_lora_{now_tag}")
    os.makedirs(run_dir, exist_ok=True)
    save_resolved_config(run_dir, full_config_dict, config)

    print("=" * 68)
    print("Qwen2.5 学术任务 QLoRA 训练")
    print("=" * 68)
    print(f"train_file: {config.train_file}")
    print(f"val_file: {config.val_file}")
    print(f"output_dir: {run_dir}")
    print(f"epochs: {config.num_train_epochs}")
    print(f"batch_size: {config.per_device_train_batch_size}")
    print(f"lr: {config.learning_rate}")
    print("=" * 68)

    raw_train = load_training_data(config.train_file)
    raw_val = load_training_data(config.val_file) if config.val_file else None

    model, tokenizer = load_model_and_tokenizer(config.model_name_or_path, config.load_in_4bit)
    proc_train = process_dataset(raw_train, tokenizer, config.max_seq_len)
    proc_val = process_dataset(raw_val, tokenizer, config.max_seq_len) if raw_val is not None else None

    peft_model = setup_lora(model, config.lora_r, config.lora_alpha, config.lora_dropout)
    collator = QwenSftCollator(pad_id=tokenizer.pad_token_id, max_length=config.max_seq_len)

    has_val = proc_val is not None and len(proc_val) > 0
    callbacks, training_extras, resolved_save_steps, early_stop_note = build_early_stopping_components(
        has_val=has_val,
        enabled=config.early_stopping_enabled,
        evaluation_strategy=config.evaluation_strategy,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        patience=config.early_stopping_patience,
        threshold=config.early_stopping_threshold,
    )
    if early_stop_note:
        print(early_stop_note)

    training_args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        logging_first_step=config.logging_first_step,
        save_steps=resolved_save_steps,
        save_total_limit=config.save_total_limit,
        optim="adamw_torch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        report_to=[],
        seed=config.seed,
        evaluation_strategy=config.evaluation_strategy if has_val else "no",
        eval_steps=config.eval_steps if has_val else None,
        **training_extras,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=proc_train,
        eval_dataset=proc_val if has_val else None,
        data_collator=collator,
        callbacks=callbacks,
    )

    print("\n开始训练...")
    train_output = trainer.train()

    print("\n训练完成")
    print(f"最终训练损失: {train_output.training_loss:.4f}")

    if has_val:
        eval_result = trainer.evaluate()
        eval_path = Path(run_dir) / "eval_metrics.json"
        with eval_path.open("w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)
        print(f"已保存验证集指标: {eval_path}")

    peft_model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    print(f"模型已保存: {run_dir}")
    return run_dir


@torch.no_grad()
def test_inference(model, tokenizer, questions: List[str], max_new_tokens: int):
    """
    训练后做少量问答自检。
    """
    model.eval()
    print("\n" + "=" * 68)
    print("训练后推理测试")
    print("=" * 68)
    for q in questions:
        msgs = [
            {"role": "system", "content": "You are a professional academic writing assistant for natural scholarly rewriting."},
            {"role": "user", "content": q},
        ]
        input_ids = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        gen_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        out_ids = gen_ids[0, input_ids.shape[-1] :]
        answer = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        print(f"Q: {q}")
        print(f"A: {answer}")
        print("-" * 68)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2.5 QLoRA 微调训练（配置化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
示例：
  python SFT/train.py --config configs/ah_sft_v2.yaml
  python SFT/train.py --config configs/ah_sft_v2.yaml --epochs 1 --lr 1e-4
  python SFT/train.py --train-file cloud_data/ah_v2/train/final_train_v2.json --val-file cloud_data/ah_v2/val/final_val_v2.json
        """,
    )

    # 新增：配置文件入口
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")

    # 兼容旧参数 + 新增 train/val
    parser.add_argument("--data-path", type=str, default=None, help="兼容参数：等价于 --train-file")
    parser.add_argument("--train-file", type=str, default=None, help="训练数据路径")
    parser.add_argument("--val-file", type=str, default=None, help="验证数据路径")
    parser.add_argument("--max-seq-len", type=int, default=None, help="最大序列长度")

    parser.add_argument("--model", type=str, default=None, help="模型名称或路径")
    parser.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=None, help="启用 4bit 量化加载")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false", help="关闭 4bit 量化加载")

    parser.add_argument("--lora-r", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=None, help="LoRA dropout")

    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="每设备训练 batch")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="每设备验证 batch")
    parser.add_argument("--grad-accum", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--warmup-ratio", type=float, default=None, help="预热比例")
    parser.add_argument("--lr-scheduler-type", type=str, default=None, help="学习率调度器")
    parser.add_argument("--save-steps", type=int, default=None, help="保存步数间隔")
    parser.add_argument("--logging-steps", type=int, default=None, help="日志步数间隔")
    parser.add_argument("--save-total-limit", type=int, default=None, help="最多保留 checkpoint 数")
    parser.add_argument("--eval-steps", type=int, default=None, help="验证步数间隔")
    parser.add_argument("--early-stop", dest="early_stop", action="store_true", default=None, help="启用早停（需有 val_file）")
    parser.add_argument("--no-early-stop", dest="early_stop", action="store_false", help="关闭早停")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="早停耐心轮次")
    parser.add_argument("--early-stop-threshold", type=float, default=None, help="早停阈值")

    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="测试推理最大 token")
    parser.add_argument("--test", action="store_true", help="训练后进行推理测试")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        yaml_cfg = load_yaml_config(args.config)
        merged = deep_merge(DEFAULT_CONFIG, yaml_cfg)
        merged = apply_cli_overrides(merged, args)
        final_cfg = build_train_config(merged)

        run_dir = train(final_cfg, merged)

        if args.test:
            base_model = final_cfg.model_name_or_path
            model, tokenizer = load_model_and_tokenizer(base_model, final_cfg.load_in_4bit)
            model = PeftModel.from_pretrained(model, run_dir)

            questions = [
                "Rewrite this academic sentence to sound more natural: This paper endeavors to explore a novel graph neural network method.",
                "Rewrite this academic sentence while preserving meaning: This study underscores the pivotal role of adaptive feedback.",
                "Rewrite this academic sentence in a concise scholarly style: The proposed method achieves state-of-the-art performance.",
            ]
            test_inference(model, tokenizer, questions, max_new_tokens=final_cfg.max_new_tokens)

        print(f"\n全部完成，模型目录：{run_dir}")
        return 0
    except Exception as exc:
        print(f"\n训练失败: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
