<p align="center">
  <img src="assets/banner.svg" alt="Academic Humanize banner" width="100%">
</p>

<p align="center">
  <a href="README.zh-CN.md">中文</a> · <a href="#results">Results</a> · <a href="#quick-start">Quick Start</a> · <a href="#reproduce-the-pipeline">Reproduce</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-12343B">
  <img alt="Model" src="https://img.shields.io/badge/Base-Qwen2.5--7B--Instruct-2D4F44">
  <img alt="Training" src="https://img.shields.io/badge/Post--training-SFT%20%2B%20DPO-F2C14E">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-6BCB77">
</p>

# Academic Humanize

Academic Humanize is a post-training and evaluation pipeline for reducing
AI-like academic prose while preserving meaning, terminology, numbers,
citations, and logical relationships.

The core task is simple:

```text
Input    : an over-polished, generic, AI-like academic paragraph
Output   : a more natural scholarly rewrite
Constraint: preserve semantic fidelity and domain terminology
```

<p align="center">
  <img src="assets/pipeline.svg" alt="Academic Humanize pipeline" width="100%">
</p>

## Why this project

Academic rewriting models often improve fluency by making the text more generic,
more formulaic, or less faithful to the original claim. This project focuses on
the trade-off between two objectives:

- semantic fidelity: the rewrite should not change facts, numbers, terminology, or logic;
- human-like academic style: the rewrite should avoid common AI lexical markers and sentence templates.

## What is included

- `SFT/train.py`: QLoRA SFT training for Academic Humanize pairs.
- `DPO/train_dpo.py`: DPO training from an SFT or previous DPO LoRA adapter.
- `evaluation/predict/predict_local_model.py`: local model / LoRA prediction.
- `evaluation/predict/predict_api.py`: API baseline prediction with resume and concurrency.
- `evaluation/metrics/compute_metrics.py`: BLEU, chrF++, TER, BERTScore, and format diagnostics.
- `evaluation/judge/llm_judge.py`: six-dimension LLM-as-Judge evaluation.
- `scripts/dpo/build_dpo_pairs_from_predictions.py`: SPIN-style DPO pair construction.
- `data/examples/`: tiny schema-compatible examples for smoke tests.

The full paper corpus, generated training data, predictions, judge outputs,
checkpoints, and model weights are intentionally not included.

## Method

### SFT

The SFT dataset uses pairs of AI-like academic drafts and human or high-quality
reference rewrites:

```text
instruction + input -> output
```

### DPO-v1: SPIN-style preference training

DPO-v1 uses the current SFT model to generate a response for each training
input. The preference pair is:

```text
prompt   = instruction + input
chosen   = human / high-quality reference
rejected = SFT model prediction
```

### DPO-v2: iterative DPO

DPO-v2 repeats the same idea from the DPO-v1 model with more conservative
hyperparameters:

```text
prompt   = instruction + input
chosen   = human / high-quality reference
rejected = DPO-v1 model prediction
```

## Evaluation

Automatic metrics measure closeness to the held-out reference. LLM-as-Judge
measures subjective quality with a fixed six-dimension rubric.

Judge dimensions:

| Dimension | Range | Meaning |
|---|---:|---|
| lexical markers | 0-1 | Avoids AI-style words and template phrases |
| structural patterns | 0-1 | Avoids formulaic AI sentence structures |
| naturalness | 0-2 | Reads like natural scholarly English |
| semantic faithfulness | 0-2 | Preserves meaning, data, and logic |
| terminology accuracy | 0-1 | Preserves and uses domain terms correctly |
| edit value | 0-1 | Improves the input rather than making trivial edits |

## Results

Held-out validation set: 346 Academic Humanize paragraphs. Judge model:
`deepseek-v4-flash` with `evaluation/judge/prompts_fast.md`.

<p align="center">
  <img src="assets/results.svg" alt="SFT DPO result trade-off" width="100%">
</p>

### Automatic Metrics

| Model | BERTScore-F1 | chrF++ | BLEU | TER | Format Violation |
|---|---:|---:|---:|---:|---:|
| SFT LoRA | 0.9738 | 84.72 | 72.01 | 24.93 | 0.023 |
| DPO-v1 | 0.9664 | 78.26 | 63.95 | 31.73 | 0.023 |
| DPO-v2 | 0.9709 | 81.89 | 68.95 | 27.73 | 0.023 |
| GPT-4o-mini | 0.9426 | 65.84 | 34.92 | 64.35 | 0.020 |
| Qwen2.5-7B-Instruct API | 0.8438 | 36.37 | 6.96 | 441.07 | 0.029 |
| Kimi-K2-Instruct | 0.8870 | 38.87 | 12.91 | 91.75 | 0.026 |
| DeepSeek-v4-flash | 0.9400 | 65.72 | 37.94 | 61.63 | 0.055 |
| Gemini 3.1 Flash Lite | 0.8294 | 52.28 | 13.07 | 172.10 | 1.000 |

### LLM-as-Judge

| Model | Judge Norm | Total | Lexical | Structure | Naturalness | Semantic | Terminology | Edit Value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT LoRA | 0.9003 | 7.202 | 0.908 | 0.905 | 1.725 | 1.731 | 0.994 | 0.939 |
| DPO-v1 | 0.9241 | 7.393 | 0.994 | 0.986 | 1.827 | 1.633 | 0.988 | 0.965 |
| DPO-v2 | 0.9223 | 7.379 | 0.977 | 0.968 | 1.795 | 1.691 | 0.991 | 0.957 |
| GPT-4o-mini | 0.7056 | 5.645 | 0.610 | 0.488 | 1.301 | 1.627 | 0.962 | 0.656 |
| Qwen2.5-7B-Instruct API | 0.2738 | 2.191 | 0.214 | 0.220 | 0.494 | 0.659 | 0.396 | 0.208 |
| Kimi-K2-Instruct | 0.9722 | 7.777 | 0.997 | 0.991 | 1.945 | 1.870 | 0.983 | 0.991 |
| DeepSeek-v4-flash | 0.7764 | 6.211 | 0.642 | 0.627 | 1.491 | 1.682 | 0.991 | 0.777 |
| Gemini 3.1 Flash Lite | 0.8233 | 6.587 | 0.801 | 0.786 | 1.616 | 1.572 | 0.931 | 0.882 |

### Main finding

SFT preserves the reference most strongly on automatic metrics. DPO-v1 improves
judge preference but introduces semantic drift. DPO-v2 recovers much of the
semantic fidelity while retaining most of the preference gain, making it the
best local trade-off among the trained adapters.

## Quick Start

For local API prediction and judge evaluation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For AutoDL / CUDA training, start from a PyTorch CUDA image and install:

```bash
pip install -r requirements_autodl.txt
```

## Data

The repository includes toy examples only:

```text
data/examples/sample_train.json
data/examples/sample_val.json
data/examples/sample_dpo_pairs.jsonl
```

For a toy smoke test:

```bash
mkdir -p cloud_data/ah_v2/train cloud_data/ah_v2/val
cp data/examples/sample_train.json cloud_data/ah_v2/train/final_train_v2.json
cp data/examples/sample_val.json cloud_data/ah_v2/val/final_val_v2.json
```

## Reproduce the Pipeline

### SFT

```bash
python SFT/train.py --config configs/ah_sft_v2.yaml
```

### Local prediction

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python evaluation/predict/predict_local_model.py \
  --val-file cloud_data/ah_v2/val/final_val_v2.json \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --adapter-path checkpoints/ah_sft_v2/YOUR_SFT_ADAPTER \
  --max-new-tokens 1024 \
  --output results/predictions/ah_sft_val_pred.json
```

### API baseline prediction

```bash
python evaluation/predict/predict_api.py \
  --val-file cloud_data/ah_v2/val/final_val_v2.json \
  --api-model openai/gpt-4o-mini \
  --max-tokens 1600 \
  --max-concurrency 4 \
  --output results/predictions/ah_api_gpt4o_mini_pred.json \
  --resume \
  --save-every 20
```

### Metrics

```bash
python evaluation/metrics/compute_metrics.py \
  --report-file results/predictions/ah_sft_val_pred.json \
  --output results/scored/ah_sft_val_scored.json
```

### LLM-as-Judge

```bash
python evaluation/judge/llm_judge.py \
  --report-file results/predictions/ah_sft_val_pred.json \
  --api-model deepseek-v4-flash \
  --prompt-file evaluation/judge/prompts_fast.md \
  --max-samples 0 \
  --max-concurrency 4 \
  --max-tokens 1200 \
  --output results/judge/ah_judge_sft_deepseek_v4_flash.json \
  --resume \
  --save-every 20
```

If a few rows fail to parse, rerun the same command with the same `--output` and
`--resume`. The script reuses parsed rows and retries failed rows only.

### Build DPO pairs

Generate train-split predictions first:

```bash
python evaluation/predict/predict_local_model.py \
  --val-file cloud_data/ah_v2/train/final_train_v2.json \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --adapter-path checkpoints/ah_sft_v2/YOUR_SFT_ADAPTER \
  --max-new-tokens 1024 \
  --output results/predictions/ah_sft_train_pred_for_dpo.json
```

Then build SPIN-style pairs:

```bash
python scripts/dpo/build_dpo_pairs_from_predictions.py \
  --train-file cloud_data/ah_v2/train/final_train_v2.json \
  --prediction-report results/predictions/ah_sft_train_pred_for_dpo.json \
  --output-all cloud_data/ah_v2/dpo/ah_dpo_pairs_all.jsonl \
  --output-train cloud_data/ah_v2/dpo/train/ah_dpo_pairs_train.jsonl \
  --output-val cloud_data/ah_v2/dpo/val/ah_dpo_pairs_val.jsonl \
  --report-file cloud_data/ah_v2/dpo/ah_dpo_pairs_report.json
```

### DPO

Set `model.sft_adapter_path` in `configs/ah_dpo.yaml`, then run:

```bash
python DPO/train_dpo.py --config configs/ah_dpo.yaml
```

For iterative DPO, generate DPO-v1 train predictions, build
`cloud_data/ah_v2/dpo_iter2/`, set `configs/ah_dpo_iter2.yaml`, and run:

```bash
python DPO/train_dpo.py --config configs/ah_dpo_iter2.yaml
```

## Repository Hygiene

The `.gitignore` excludes private corpus files, full generated datasets,
checkpoints, model weights, result JSON files, local `.env` files, notebooks,
and scratch artifacts. Only toy examples and source code should be committed.

## License

MIT.
