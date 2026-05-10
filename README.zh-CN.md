<p align="center">
  <img src="assets/banner.svg" alt="Academic Humanize banner" width="100%">
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="#实验结果">实验结果</a> · <a href="#快速开始">快速开始</a> · <a href="#复现流程">复现流程</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-12343B">
  <img alt="Model" src="https://img.shields.io/badge/Base-Qwen2.5--7B--Instruct-2D4F44">
  <img alt="Training" src="https://img.shields.io/badge/Post--training-SFT%20%2B%20DPO-F2C14E">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-6BCB77">
</p>

# Academic Humanize

Academic Humanize 是一个面向学术英文润色的后训练与评测项目，目标是降低
AI 味、模板感和过度润色痕迹，同时尽量保持语义、术语、数字、引用和逻辑关系不变。

核心任务：

```text
输入：带有 AI 味、模板化、过度正式的学术段落
输出：更自然、更像真人学者写作的学术英文
约束：不改变原意，不丢术语、数字、引用和结论
```

<p align="center">
  <img src="assets/pipeline.svg" alt="Academic Humanize pipeline" width="100%">
</p>

## 项目动机

普通润色模型经常会把文本改得更流畅，但也更空泛、更模板化，甚至改变原文含义。
这个项目关注两个目标之间的平衡：

- 语义保真：不能改错事实、数字、术语和逻辑。
- 去 AI 味：减少常见 AI 词汇、套话和公式化句式。

## 仓库包含什么

- `SFT/train.py`：Academic Humanize 的 QLoRA SFT 训练脚本。
- `DPO/train_dpo.py`：从 SFT 或上一轮 DPO LoRA 继续做 DPO。
- `evaluation/predict/predict_local_model.py`：本地模型 / LoRA 推理。
- `evaluation/predict/predict_api.py`：API baseline 推理，支持并发和断点续跑。
- `evaluation/metrics/compute_metrics.py`：BLEU、chrF++、TER、BERTScore 和格式诊断。
- `evaluation/judge/llm_judge.py`：六维 LLM-as-Judge 评测。
- `scripts/dpo/build_dpo_pairs_from_predictions.py`：SPIN 风格 DPO pair 构造。
- `data/examples/`：用于 smoke test 的极小 toy 数据。

真实论文语料、完整训练数据、预测结果、judge 结果、checkpoint 和模型权重不随仓库发布。

## 方法

### SFT

SFT 数据由 AI-like draft 和 human/high-quality reference rewrite 组成：

```text
instruction + input -> output
```

### DPO-v1：SPIN 风格偏好训练

DPO-v1 用当前 SFT 模型对训练集 input 生成 response，然后构造偏好对：

```text
prompt   = instruction + input
chosen   = human / high-quality reference
rejected = SFT model prediction
```

### DPO-v2：迭代 DPO

DPO-v2 从 DPO-v1 出发，用更保守的超参数再做一轮：

```text
prompt   = instruction + input
chosen   = human / high-quality reference
rejected = DPO-v1 model prediction
```

## 评测框架

自动指标衡量 prediction 和 reference 的接近程度；LLM-as-Judge 衡量主观改写质量。

Judge 六个维度如下：

| 维度 | 分值 | 含义 |
|---|---:|---|
| lexical markers | 0-1 | 是否避免 AI 高频词和模板短语 |
| structural patterns | 0-1 | 是否避免公式化 AI 句式 |
| naturalness | 0-2 | 是否像自然的学术英文 |
| semantic faithfulness | 0-2 | 是否保留原意、数据和逻辑 |
| terminology accuracy | 0-1 | 术语是否保留且使用准确 |
| edit value | 0-1 | 是否相比输入有实质改进 |

## 实验结果

验证集包含 346 条 Academic Humanize 段落。Judge 模型为 `deepseek-v4-flash`，
prompt 使用 `evaluation/judge/prompts_fast.md`。

<p align="center">
  <img src="assets/results.svg" alt="SFT DPO result trade-off" width="100%">
</p>

### 自动指标

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

### 主要结论

SFT 在自动语义指标上最接近 reference。DPO-v1 明显提高 judge 偏好分数，
但带来一定语义漂移。DPO-v2 恢复了大部分语义指标，同时保留了大部分偏好收益，
因此是本项目当前本地训练模型里最好的折中版本。

## 快速开始

本地 API 推理和 judge 评测：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

AutoDL / CUDA 训练环境：

```bash
pip install -r requirements_autodl.txt
```

## 数据

仓库只包含 toy examples：

```text
data/examples/sample_train.json
data/examples/sample_val.json
data/examples/sample_dpo_pairs.jsonl
```

toy smoke test：

```bash
mkdir -p cloud_data/ah_v2/train cloud_data/ah_v2/val
cp data/examples/sample_train.json cloud_data/ah_v2/train/final_train_v2.json
cp data/examples/sample_val.json cloud_data/ah_v2/val/final_val_v2.json
```

## 复现流程

### SFT

```bash
python SFT/train.py --config configs/ah_sft_v2.yaml
```

### 本地模型推理

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python evaluation/predict/predict_local_model.py \
  --val-file cloud_data/ah_v2/val/final_val_v2.json \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --adapter-path checkpoints/ah_sft_v2/YOUR_SFT_ADAPTER \
  --max-new-tokens 1024 \
  --output results/predictions/ah_sft_val_pred.json
```

### API baseline 推理

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

### 计算 metrics

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

如果少量行解析失败，使用同一个 `--output` 和 `--resume` 重新运行即可；脚本会复用已解析行，只补失败行。

### 构造 DPO pair

先在 train split 上生成预测：

```bash
python evaluation/predict/predict_local_model.py \
  --val-file cloud_data/ah_v2/train/final_train_v2.json \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --adapter-path checkpoints/ah_sft_v2/YOUR_SFT_ADAPTER \
  --max-new-tokens 1024 \
  --output results/predictions/ah_sft_train_pred_for_dpo.json
```

然后构造 SPIN-style pairs：

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

设置 `configs/ah_dpo.yaml` 里的 `model.sft_adapter_path`，然后运行：

```bash
python DPO/train_dpo.py --config configs/ah_dpo.yaml
```

迭代 DPO 则先生成 DPO-v1 train predictions，构造 `cloud_data/ah_v2/dpo_iter2/`，
设置 `configs/ah_dpo_iter2.yaml`，然后运行：

```bash
python DPO/train_dpo.py --config configs/ah_dpo_iter2.yaml
```

## 仓库卫生

`.gitignore` 会排除真实语料、完整生成数据、checkpoint、模型权重、结果 JSON、
本地 `.env`、notebook 和临时文件。仓库只应该提交 toy examples 和源码。

## License

MIT.
