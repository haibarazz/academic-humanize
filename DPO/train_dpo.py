"""
Academic Humanize DPO 训练脚本。

目标：
1. 从当前最佳 Academic Humanize SFT LoRA 继续做 DPO
2. 输入数据固定为 prompt/chosen/rejected 的 pair 数据集
3. 与现有 SFT 链和 Academic Humanize 评测链分开维护
"""

from __future__ import annotations

import argparse
import datetime
import inspect
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# AutoDL 环境里 OMP_NUM_THREADS 偶尔是非法值；DPO 又很贴显存。
# 这些环境变量必须在 torch 初始化 CUDA 前设置。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

from SFT.train_utils import build_early_stopping_components

try:
    from trl import DPOConfig, DPOTrainer
    TRL_IMPORT_ERROR = None
except ImportError as exc:
    DPOConfig = None
    DPOTrainer = None
    TRL_IMPORT_ERROR = exc


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "sft_adapter_path": "./checkpoints/qwen25_academic_lora_best",
        "load_in_4bit": True,
    },
    "data": {
        "train_pair_file": "data/generated/ah_dpo_pairs_train.jsonl",
        "val_pair_file": "data/generated/ah_dpo_pairs_val.jsonl",
        "max_seq_len": 2048,
        "max_prompt_length": 1536,
    },
    "prompt": {
        "system_prompt": "You are a professional academic writing assistant. Rewrite the text into natural academic English while preserving meaning.",
        "require_chat_template": True,
    },
    "train": {
        "beta": 0.1,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-6,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
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
        "output_dir": "./checkpoints_dpo",
    },
    "runtime": {
        "seed": 42,
    },
}


@dataclass
class DpoTrainConfig:
    base_model: str
    sft_adapter_path: str
    load_in_4bit: bool
    train_pair_file: str
    val_pair_file: Optional[str]
    max_seq_len: int
    max_prompt_length: int
    system_prompt: str
    require_chat_template: bool
    beta: float
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


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML 配置必须是字典结构")
    return payload


def build_config_dict(config_path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = deep_merge(DEFAULT_CONFIG, load_yaml_config(config_path))
    if overrides:
        merged = deep_merge(merged, overrides)
    return merged


def build_train_config(cfg: Dict[str, Any]) -> DpoTrainConfig:
    model = cfg.get("model", {})
    data = cfg.get("data", {})
    prompt = cfg.get("prompt", {})
    train = cfg.get("train", {})
    output = cfg.get("output", {})
    runtime = cfg.get("runtime", {})

    val_pair_file = data.get("val_pair_file")
    if isinstance(val_pair_file, str) and not val_pair_file.strip():
        val_pair_file = None

    max_seq_len = int(data.get("max_seq_len", 2048))
    max_prompt_length = int(data.get("max_prompt_length", min(1536, max_seq_len - 256)))
    max_prompt_length = max(64, min(max_prompt_length, max_seq_len - 64))

    return DpoTrainConfig(
        base_model=str(model.get("base_model")),
        sft_adapter_path=str(model.get("sft_adapter_path")),
        load_in_4bit=bool(model.get("load_in_4bit", True)),
        train_pair_file=str(data.get("train_pair_file")),
        val_pair_file=val_pair_file if val_pair_file is None else str(val_pair_file),
        max_seq_len=max_seq_len,
        max_prompt_length=max_prompt_length,
        system_prompt=str(prompt.get("system_prompt", DEFAULT_CONFIG["prompt"]["system_prompt"])),
        require_chat_template=bool(prompt.get("require_chat_template", True)),
        beta=float(train.get("beta", 0.1)),
        num_train_epochs=int(train.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(train.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train.get("learning_rate", 5e-6)),
        warmup_ratio=float(train.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(train.get("lr_scheduler_type", "cosine")),
        save_steps=int(train.get("save_steps", 100)),
        logging_steps=int(train.get("logging_steps", 5)),
        logging_first_step=bool(train.get("logging_first_step", True)),
        save_total_limit=int(train.get("save_total_limit", 2)),
        evaluation_strategy=str(train.get("evaluation_strategy", "steps")),
        eval_steps=int(train.get("eval_steps", 100)),
        early_stopping_enabled=bool(train.get("early_stopping_enabled", True)),
        early_stopping_patience=int(train.get("early_stopping_patience", 2)),
        early_stopping_threshold=float(train.get("early_stopping_threshold", 0.0)),
        output_dir=str(output.get("output_dir", "./checkpoints_dpo")),
        seed=int(runtime.get("seed", 42)),
    )


def load_pair_data(data_path: str) -> Dataset:
    if data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_path, split="train")
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{data_path} 必须是 JSON 数组")
        dataset = Dataset.from_list(data)

    required = {"prompt", "chosen", "rejected"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{data_path} 缺少必要字段: {sorted(missing)}")
    return dataset


def render_messages_fallback(messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    chunks: List[str] = []
    for message in messages:
        role = (message.get("role") or "").strip().upper()
        content = (message.get("content") or "").strip()
        if role and content:
            chunks.append(f"{role}:\n{content}")
    if add_generation_prompt:
        chunks.append("ASSISTANT:\n")
    return "\n\n".join(chunks)


def render_prompt_text(tokenizer, system_prompt: str, user_prompt: str, require_chat_template: bool) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if has_chat_template:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    if require_chat_template:
        raise ValueError(
            "当前 tokenizer 不包含 chat_template。"
            "请使用 chat/instruct 模型，或把 prompt.require_chat_template 设为 false。"
        )
    return render_messages_fallback(messages, add_generation_prompt=True)


def prepare_pair_dataset(dataset: Dataset, tokenizer, system_prompt: str, require_chat_template: bool) -> Dataset:
    rows: List[Dict[str, str]] = []
    for record in dataset:
        user_prompt = str(record.get("prompt", "") or "").strip()
        chosen = str(record.get("chosen", "") or "").strip()
        rejected = str(record.get("rejected", "") or "").strip()
        if not user_prompt or not chosen or not rejected:
            continue
        prompt_text = render_prompt_text(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            require_chat_template=require_chat_template,
        )
        rows.append(
            {
                "prompt": prompt_text,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return Dataset.from_list(rows)


def load_model_and_tokenizer(
    model_name_or_path: str,
    load_in_4bit: bool,
    require_chat_template: bool,
    prepare_for_training: bool,
):
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
    if prepare_for_training:
        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if require_chat_template and not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "当前 tokenizer 不包含 chat_template。"
            "请使用 chat/instruct 模型，或把 prompt.require_chat_template 设为 false。"
        )
    return model, tokenizer


def load_peft_adapter(model, adapter_path: str, is_trainable: bool):
    try:
        peft_model = PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)
    except TypeError:
        peft_model = PeftModel.from_pretrained(model, adapter_path)
        for name, param in peft_model.named_parameters():
            is_adapter_param = "lora_" in name or "modules_to_save" in name
            param.requires_grad = bool(is_trainable and is_adapter_param)

    if is_trainable:
        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()
        if hasattr(peft_model, "print_trainable_parameters"):
            peft_model.print_trainable_parameters()
    else:
        for param in peft_model.parameters():
            param.requires_grad = False
        peft_model.eval()

    peft_model.config.use_cache = False
    return peft_model


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", (text or "").strip())
    slug = slug.strip("._-")
    return slug or "model"


def save_resolved_config(run_dir: str, full_config_dict: Dict[str, Any], final_config: DpoTrainConfig) -> None:
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = out_dir / "resolved_config.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(full_config_dict, f, allow_unicode=True, sort_keys=False)

    json_path = out_dir / "resolved_config.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(final_config), f, ensure_ascii=False, indent=2)


def _filter_supported_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(callable_obj)
    return {key: value for key, value in kwargs.items() if key in signature.parameters and value is not None}


def build_training_args(
    config: DpoTrainConfig,
    run_dir: str,
    has_val: bool,
    training_extras: Dict[str, Any],
    resolved_save_steps: int,
):
    common_kwargs: Dict[str, Any] = {
        "output_dir": run_dir,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "logging_first_step": config.logging_first_step,
        "save_steps": resolved_save_steps,
        "save_total_limit": config.save_total_limit,
        "evaluation_strategy": config.evaluation_strategy if has_val else "no",
        "eval_steps": config.eval_steps if has_val else None,
        "optim": "adamw_torch",
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "report_to": [],
        "seed": config.seed,
        "remove_unused_columns": False,
    }
    common_kwargs.update(training_extras)

    dpo_specific = {
        "beta": config.beta,
        "max_length": config.max_seq_len,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": max(64, config.max_seq_len - config.max_prompt_length),
    }

    if DPOConfig is not None:
        config_kwargs = _filter_supported_kwargs(DPOConfig.__init__, {**common_kwargs, **dpo_specific})
        return DPOConfig(**config_kwargs), dpo_specific

    config_kwargs = _filter_supported_kwargs(TrainingArguments.__init__, common_kwargs)
    return TrainingArguments(**config_kwargs), dpo_specific


def build_trainer(
    model,
    ref_model,
    tokenizer,
    training_args,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    callbacks: List[Any],
    dpo_specific: Dict[str, Any],
):
    if DPOTrainer is None:
        raise ImportError(
            "当前环境缺少 trl，无法运行 DPO 训练。请先安装 requirements.txt 中新增的 trl 依赖。"
        ) from TRL_IMPORT_ERROR

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": ref_model,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    signature = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    if "callbacks" in signature.parameters:
        trainer_kwargs["callbacks"] = callbacks

    for key, value in dpo_specific.items():
        if key in signature.parameters and key not in trainer_kwargs:
            trainer_kwargs[key] = value

    return DPOTrainer(**trainer_kwargs)


def train(config: DpoTrainConfig, full_config_dict: Dict[str, Any]) -> str:
    if DPOTrainer is None:
        raise ImportError(
            "当前环境缺少 trl，无法运行 DPO 训练。请先安装 requirements.txt 中新增的 trl 依赖。"
        ) from TRL_IMPORT_ERROR

    torch.manual_seed(config.seed)

    adapter_slug = _slugify(Path(config.sft_adapter_path).name or config.sft_adapter_path)
    now_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.output_dir, f"ah_dpo_{adapter_slug}_{now_tag}")
    os.makedirs(run_dir, exist_ok=True)
    save_resolved_config(run_dir, full_config_dict, config)

    print("=" * 68)
    print("Academic Humanize DPO 训练")
    print("=" * 68)
    print(f"base_model: {config.base_model}")
    print(f"sft_adapter_path: {config.sft_adapter_path}")
    print(f"train_pair_file: {config.train_pair_file}")
    print(f"val_pair_file: {config.val_pair_file}")
    print(f"output_dir: {run_dir}")
    print(f"epochs: {config.num_train_epochs}")
    print(f"lr: {config.learning_rate}")
    print(f"beta: {config.beta}")
    print("=" * 68)

    raw_train = load_pair_data(config.train_pair_file)
    raw_val = load_pair_data(config.val_pair_file) if config.val_pair_file else None

    policy_base_model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=config.base_model,
        load_in_4bit=config.load_in_4bit,
        require_chat_template=config.require_chat_template,
        prepare_for_training=True,
    )
    ref_base_model, _ = load_model_and_tokenizer(
        model_name_or_path=config.base_model,
        load_in_4bit=config.load_in_4bit,
        require_chat_template=config.require_chat_template,
        prepare_for_training=False,
    )

    train_dataset = prepare_pair_dataset(raw_train, tokenizer, config.system_prompt, config.require_chat_template)
    val_dataset = prepare_pair_dataset(raw_val, tokenizer, config.system_prompt, config.require_chat_template) if raw_val is not None else None

    if len(train_dataset) == 0:
        raise ValueError("训练 pair 数据为空，无法开始 DPO")

    policy_model = load_peft_adapter(policy_base_model, config.sft_adapter_path, is_trainable=True)
    ref_model = load_peft_adapter(ref_base_model, config.sft_adapter_path, is_trainable=False)

    has_val = val_dataset is not None and len(val_dataset) > 0
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

    training_args, dpo_specific = build_training_args(
        config=config,
        run_dir=run_dir,
        has_val=has_val,
        training_extras=training_extras,
        resolved_save_steps=resolved_save_steps,
    )

    trainer = build_trainer(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if has_val else None,
        callbacks=callbacks,
        dpo_specific=dpo_specific,
    )

    train_output = trainer.train()
    train_metrics = getattr(train_output, "metrics", {}) or {}
    train_metrics_path = Path(run_dir) / "train_metrics.json"
    with train_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(train_metrics, f, ensure_ascii=False, indent=2)

    if has_val:
        eval_result = trainer.evaluate()
        eval_path = Path(run_dir) / "eval_metrics.json"
        with eval_path.open("w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)

    policy_model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)

    print(f"训练集有效 pair: {len(train_dataset)}")
    if has_val:
        print(f"验证集有效 pair: {len(val_dataset)}")
    print(f"模型已保存到: {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Academic Humanize DPO 训练脚本")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件")
    parser.add_argument("--base-model", type=str, default=None, help="覆盖 model.base_model")
    parser.add_argument("--sft-adapter-path", type=str, default=None, help="覆盖 model.sft_adapter_path")
    parser.add_argument("--train-pair-file", type=str, default=None, help="覆盖 data.train_pair_file")
    parser.add_argument("--val-pair-file", type=str, default=None, help="覆盖 data.val_pair_file")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 train.num_train_epochs")
    parser.add_argument("--lr", type=float, default=None, help="覆盖 train.learning_rate")
    parser.add_argument("--beta", type=float, default=None, help="覆盖 train.beta")
    parser.add_argument("--output-dir", type=str, default=None, help="覆盖 output.output_dir")
    return parser.parse_args()


def apply_cli_overrides(config_dict: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    def set_path(path: List[str], value: Any) -> None:
        if value is None:
            return
        cursor = overrides
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = value

    set_path(["model", "base_model"], args.base_model)
    set_path(["model", "sft_adapter_path"], args.sft_adapter_path)
    set_path(["data", "train_pair_file"], args.train_pair_file)
    set_path(["data", "val_pair_file"], args.val_pair_file)
    set_path(["train", "num_train_epochs"], args.epochs)
    set_path(["train", "learning_rate"], args.lr)
    set_path(["train", "beta"], args.beta)
    set_path(["output", "output_dir"], args.output_dir)
    return deep_merge(config_dict, overrides)


def main() -> int:
    args = parse_args()
    merged = build_config_dict(args.config)
    merged = apply_cli_overrides(merged, args)
    final_cfg = build_train_config(merged)
    train(final_cfg, merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
