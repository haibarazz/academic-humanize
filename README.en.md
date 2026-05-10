<p align="center">
  <img src="assets/hero_generated.png" alt="Academic Humanize banner" width="100%">
</p>

<p align="center">
  <a href="README.md">中文</a> · <a href="#technical-pipeline">Technical Pipeline</a> · <a href="#results">Results</a> · <a href="#quick-start">Quick Start</a> · <a href="#reproduce-the-pipeline">Reproduce</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-12343B">
  <img alt="Model" src="https://img.shields.io/badge/Base-Qwen2.5--7B--Instruct-2D4F44">
  <img alt="Training" src="https://img.shields.io/badge/Training-QLoRA%20SFT-F2C14E">
  <img alt="Alignment" src="https://img.shields.io/badge/Alignment-SPIN--style%20DPO-2D4F44">
  <img alt="Evaluation" src="https://img.shields.io/badge/Eval-Metrics%20%2B%20LLM--Judge-12343B">
</p>

# Academic Humanize

Academic Humanize is a post-training project for academic English rewriting. Its goal is to turn "AI-style reduction" into a trainable, optimizable, and measurable engineering pipeline.

The project covers data construction, QLoRA SFT, SPIN-style DPO, iterative DPO, automatic metrics, and LLM-as-Judge evaluation.

```text
Input      : an over-polished, formulaic, AI-like academic paragraph
Output     : a more natural scholarly rewrite
Hard rules : preserve meaning, numbers, citations, terminology, conclusions, and logic
```

<p align="center">
  <img src="assets/training_pipeline_generated.png" alt="Academic Humanize training pipeline" width="100%">
</p>

## Project Highlights

- **Local 7B post-training loop**: QLoRA SFT, SPIN-style DPO, and iterative DPO on Qwen2.5-7B-Instruct, with API baselines for comparison.
- **Sharper task definition**: academic AI-style reduction is decomposed into AI-like draft construction, semantic fidelity, terminology preservation, naturalness, and edit value.
- **Low-cost preference data**: current model outputs become rejected responses, while human references become chosen responses.
- **Two-layer evaluation**: BERTScore-F1, chrF++, BLEU, TER, format diagnostics, and a six-dimensional LLM-as-Judge rubric.
- **Comparable baselines**: SFT, DPO-v1, DPO-v2, and multiple API models are evaluated under the same pipeline.

## My Contributions

- I designed the Academic Humanize V2 data format: `instruction + AI-like input -> human reference output`.
- I implemented QLoRA SFT training, LoRA prediction, API baseline prediction, resume support, and concurrent API calls.
- I built the SPIN-style DPO pair construction flow and completed the DPO-v1 and iterative DPO-v2 training loop.
- I built the automatic metrics and LLM-as-Judge evaluation stack with fixed prompts and six scoring dimensions.
- I packaged the open-source version with toy examples, configs, training scripts, evaluation scripts, and README visuals.

## Why this project

General-purpose LLMs can make academic text more fluent, but they often introduce three problems:

- More AI-like wording: frequent use of words such as `pivotal`, `underscore`, and formulaic paired clauses.
- Unstable semantics: claims, numeric details, terminology, citations, or logical strength may change during polishing.
- Weak evaluation signals: BLEU, chrF, and BERTScore measure reference similarity, but they do not directly answer whether the rewrite reads like human scholarly English.

This project focuses on a narrower and harder problem: reducing AI writing traces while keeping academic meaning safe.

## What you get

- A reproducible post-training pipeline for academic text humanization.
- A lightweight RLHF / preference optimization example from SFT to DPO and iterative DPO.
- A reusable evaluation stack: automatic metrics for semantic fidelity and LLM-as-Judge for naturalness, terminology, and edit value.
- API baseline comparisons for understanding the gap between local 7B LoRA adapters and proprietary models.
- A transferable data construction recipe: if you have AI-like inputs and human references, the pipeline can be adapted to other academic-writing scenarios.

<p align="center">
  <img src="assets/visual_overview_generated.png" alt="Academic Humanize visual overview" width="100%">
</p>

## Technical Pipeline

The core contribution is to decompose "AI-style reduction" into four operational modules.

### 1. AH V2 data construction

Training pairs follow an "AI-like draft -> human reference" structure. Each sample has:

```text
input  = an AI-like academic draft
output = a human or high-quality reference academic rewrite
```

The model therefore learns to move from formulaic, over-polished, AI-like prose back to more natural scholarly English.

### 2. QLoRA SFT

The SFT stage uses Qwen2.5-7B-Instruct as the base model and trains a low-cost LoRA adapter with QLoRA.

```text
instruction + input -> output
```

This stage teaches the model the task format, terminology preservation, citation preservation, and the basic humanization style.

### 3. SPIN-style DPO

DPO-v1 does not require extra human preference labels. It asks the current SFT model to generate rejected responses:

```text
prompt   = instruction + input
chosen   = human / high-quality reference
rejected = SFT model prediction
```

The intuition is simple: if the human reference is better than the current model output, the model can learn the preference gap between them. This creates on-policy negative samples at low cost.

### 4. Iterative DPO

DPO-v2 repeats the same recipe using DPO-v1 predictions as rejected responses, with a more conservative learning rate and beta:

```text
prompt   = instruction + input
chosen   = human / high-quality reference
rejected = DPO-v1 model prediction
```

The rejected response is now closer to the model's current capability boundary, giving a finer preference signal. In the current experiments, DPO-v2 recovers more semantic fidelity while retaining most of the judge preference gain.

## Example

The following example is from the toy data in [data/examples/sample_train.json](data/examples/sample_train.json).

**AI-like input**

```text
This study endeavors to explore the multifaceted role of adaptive feedback mechanisms in online learning environments. The results underscore the pivotal importance of personalized intervention for improving student engagement.
```

**Humanized reference**

```text
This study examines how adaptive feedback mechanisms support online learning. The results show that personalized intervention can improve student engagement.
```

The rewrite removes common AI-style markers such as `endeavors`, `multifaceted`, `underscore`, and `pivotal`, while preserving key information such as `adaptive feedback mechanisms`, `online learning environments`, and `student engagement`.

## Evaluation

The project uses two layers of evaluation to cover semantic fidelity and subjective writing quality.

### Automatic semantic metrics

| Metric | Purpose | Role |
|---|---|---|
| BERTScore-F1 | Semantic similarity between prediction and reference | Primary |
| chrF++ | Character-level preservation of terms and spelling | Auxiliary |
| BLEU | Traditional n-gram overlap | Reference |
| TER | Edit-distance style metric | Reference |
| Format Violation | Empty output, malformed output, and obvious failures | Quality control |

### LLM-as-Judge

The judge uses a fixed prompt and six fixed dimensions, producing a total score from 0 to 8.

| Dimension | Range | Meaning |
|---|---:|---|
| lexical markers | 0-1 | Avoids AI-style words and template phrases |
| structural patterns | 0-1 | Avoids formulaic AI sentence structures |
| naturalness | 0-2 | Reads like natural scholarly English |
| semantic faithfulness | 0-2 | Preserves meaning, data, and logic |
| terminology accuracy | 0-1 | Preserves and uses domain terms correctly |
| edit value | 0-1 | Measures whether the rewrite provides a meaningful improvement |

## Prompt Assets

Prompts are maintained as versioned experimental assets:

- `evaluation/judge/prompts.md`: full judge prompt with the AI-writing word bank, structural patterns, and scoring rubric.
- `evaluation/judge/prompts_fast.md`: compact judge prompt used for full-scale evaluation.
- `scripts/dpo/prompt.md`: optional controlled rejected-candidate prompt for generating AI-like but semantically close DPO negatives.

## Results

Held-out validation set: 346 Academic Humanize paragraphs. Judge model: `deepseek-v4-flash` with `evaluation/judge/prompts_fast.md`.

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

### Main findings

- SFT LoRA is closest to the reference on automatic semantic metrics, making it the most conservative model.
- DPO-v1 significantly improves LLM-as-Judge preference scores, but sacrifices part of reference similarity.
- DPO-v2 recovers much of the semantic metric performance while preserving the DPO preference gain. It is the best local trade-off among the trained 7B adapters.
- Kimi-K2-Instruct scores very high on judge evaluation, but it is an API baseline. The main focus of this project is a reproducible, trainable, and iterative local post-training workflow.

## Who is this for

- Developers who want a practical example of SFT, DPO, and SPIN-style self-play alignment.
- Researchers who want a small, reproducible LLM post-training project.
- Builders working on academic writing, paper polishing, or AI text humanization.
- Anyone interested in combining traditional NLP metrics with LLM-as-Judge evaluation.

## Repository Structure

```text
academic-humanize/
├── SFT/                         # QLoRA SFT training
├── DPO/                         # DPO training from SFT or DPO adapter
├── configs/                     # SFT, DPO, eval configs
├── evaluation/
│   ├── predict/                 # local/API prediction
│   ├── metrics/                 # BLEU, chrF++, TER, BERTScore
│   ├── judge/                   # LLM-as-Judge
│   ├── leaderboard/             # report merging
│   └── detector/                # optional detector sidecar
├── scripts/dpo/                 # DPO pair construction tools
├── data/examples/               # toy examples only
└── assets/                      # README figures
```

The full paper corpus, generated training data, prediction files, judge outputs, checkpoints, and model weights are not included. The repository contains toy examples and reproducible code only.

## Limitations

- The full paper corpus, training data, and checkpoints are not released; this repository provides toy examples and reproducible scripts.
- LLM-as-Judge is not a replacement for human evaluation, so automatic semantic metrics are kept for cross-checking.
- The current validation set contains 346 Academic Humanize samples; new domains or writing styles should be re-evaluated.
- DPO improves preference scores but can introduce semantic drift. DPO-v2 mitigates this risk but does not eliminate it.
- API baselines can change with provider routing, model versions, and serving behavior, so they should be treated as reference points.

## Quick Start

For local API prediction and judge evaluation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For AutoDL / CUDA training:

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

If a few rows fail to parse, rerun the same command with the same `--output` and `--resume`. The script reuses parsed rows and retries failed rows only.

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

For iterative DPO, generate DPO-v1 train predictions, build `cloud_data/ah_v2/dpo_iter2/`, set `configs/ah_dpo_iter2.yaml`, and run:

```bash
python DPO/train_dpo.py --config configs/ah_dpo_iter2.yaml
```

## Contributing and Contact

Issues, suggestions, and reproduction reports are welcome. If you run this pipeline on your own academic-writing data, feel free to open an issue and share your setup and evaluation results.

Contact: 2812156857@qq.com
