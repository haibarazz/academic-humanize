<p align="center">
  <img src="assets/hero_generated.png" alt="Academic Humanize banner" width="100%">
</p>

<p align="center">
  <a href="README.en.md">English</a> · <a href="https://huggingface.co/XiaoXu123123/academic-humanize-qwen25-7b-dpo-v2-lora">Hugging Face Adapter</a> · <a href="#核心技术路线">核心技术路线</a> · <a href="#实验结果">实验结果</a> · <a href="#快速开始">快速开始</a> · <a href="#复现流程">复现流程</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-12343B">
  <img alt="Model" src="https://img.shields.io/badge/Base-Qwen2.5--7B--Instruct-2D4F44">
  <img alt="Training" src="https://img.shields.io/badge/Training-QLoRA%20SFT-F2C14E">
  <img alt="Alignment" src="https://img.shields.io/badge/Alignment-SPIN--style%20DPO-2D4F44">
  <img alt="Evaluation" src="https://img.shields.io/badge/Eval-Metrics%20%2B%20LLM--Judge-12343B">
</p>

# Academic Humanize

Academic Humanize 是一个面向学术英文改写的后训练项目，核心目标是把“去 AI 味”这件事做成一套可训练、可优化、可评测的工程闭环。

项目覆盖完整流程：从论文 PDF/Markdown 语料准备、段落抽取和质量筛选，到 AI-like draft 构造、QLoRA SFT、SPIN-style DPO 和迭代 DPO，最后用自动语义指标与 LLM-as-Judge 同时评估语义保真和自然度。

```text
输入：带有 AI 味、模板感、过度正式表达的学术段落
输出：更自然、更像真人学者写作的学术英文
硬约束：不改变原意，不丢数字、引用、术语、结论和逻辑关系
```

<p align="center">
  <img src="assets/training_pipeline_generated.png" alt="Academic Humanize training pipeline" width="100%">
</p>

## 项目亮点

- **本地 7B 后训练闭环**：基于 Qwen2.5-7B-Instruct 构建 QLoRA SFT、SPIN-style DPO 和迭代 DPO，并使用 API baseline 做对照。
- **任务定义更细**：把“学术去 AI 味”拆成 AI-like draft 构造、语义保真、术语保留、自然度提升和编辑价值评估。
- **偏好数据低成本构造**：用当前模型输出作为 rejected response，用 human reference 作为 chosen response，避免额外人工偏好标注。
- **双层评测体系**：同时使用 BERTScore-F1、chrF++、BLEU、TER、格式诊断和六维 LLM-as-Judge。
- **有可比较 baseline**：对比 SFT、DPO-v1、DPO-v2 和多个 API 模型，验证本地 adapter 的优势和边界。
- **模型参数已公开**：DPO-v2 LoRA adapter 已上传到 [Hugging Face](https://huggingface.co/XiaoXu123123/academic-humanize-qwen25-7b-dpo-v2-lora)，可直接配合 `Qwen/Qwen2.5-7B-Instruct` 加载使用。

## 我做了什么

- 我设计了段落级 Academic Humanize 数据格式：`instruction + AI-like input -> human reference output`。
- 我实现了 QLoRA SFT 训练、LoRA 推理、API baseline 推理、断点续跑和并发调用。
- 我设计了 SPIN-style DPO pair 构造流程，并完成 DPO-v1 与迭代 DPO-v2 训练闭环。
- 我建立了自动 metrics 与 LLM-as-Judge 双层评测框架，固定 judge prompt 和六维评分标准。
- 我整理了开源版本，包括 toy examples、配置文件、训练脚本、评测脚本和 README 可视化说明。

## 为什么做这个项目

很多通用大模型可以把英文改得更流畅，但它们经常带来三个问题：

- 表达更像 AI：使用 `pivotal`、`underscore`、套话式并列结构等高频模板。
- 语义不稳定：为了“更高级”而改动事实、强弱关系、数字、引用或术语。
- 难以量化：传统 BLEU/chrF/BERTScore 只能看和 reference 的接近程度，不能直接判断“像不像真人学者写的”。

本项目关注更窄、更难的 academic humanization 任务：在保持学术语义安全的前提下，减少 AI 写作痕迹。

## 你可以从这个项目获得什么

- 一套可复现的学术文本 humanization 后训练流程。
- 一个从 SFT 到 DPO 再到迭代 DPO 的轻量 RLHF / preference optimization 示例。
- 一个可直接复用的评测框架：自动指标负责语义保真，LLM-as-Judge 负责自然度、术语、编辑价值等主观质量。
- 一组 API baseline 对比结果，用于判断本地 7B LoRA 和闭源模型的差距。
- 可迁移的数据构造思路：只要你有“AI-like input”和“human reference”，就可以迁移到其他学术写作场景。

<p align="center">
  <img src="assets/visual_overview_generated.png" alt="Academic Humanize visual overview" width="100%">
</p>

## 核心技术路线

本项目的关键技术含量在于把“去 AI 味”拆成五个可操作环节：语料准备、数据构造、SFT、DPO 和评测。

### 1. 论文语料准备：从 PDF 到段落库

数据构造从真实学术论文开始。语料主要来自管理学与信息系统（IS）领域多个学术期刊的论文 PDF。我先将 PDF 转成可处理的 Markdown/JSON 文本，随后按论文结构抽取段落级语料。

处理流程大致如下：

```text
paper PDF
-> Markdown / structured JSON
-> section split
-> paragraph extraction
-> quality filtering
-> paragraph inventory
```

这一阶段主要做四件事：

- 按章节标题切分论文，优先保留 abstract、introduction、related work、results、discussion 和 conclusion 等更适合学术表达学习的部分。
- 过滤 references、appendix、acknowledgement、实验配置、指标列表、表格密集段落、公式密集段落和明显元数据。
- 清理 OCR 噪声、乱码、版权行、引用说明、残缺句子、过短或过长段落。
- 为每个段落保留 `paper_id`、`section_title`、`paragraph_id`、词数、句子数和质量信号，方便后续做 paper-level split 和泄漏检查。

我在开源仓库里只提供 toy examples 和处理脚本，没有发布原始论文 PDF、抽取后的真实段落和完整训练数据。你可以沿用这套流程，并按自己的研究领域选择对应的论文语料库。

### 2. Academic Humanize 数据构造

数据构造是这个项目最核心的部分。我先从真实论文中抽取高质量 human paragraph，把它作为稳定的质量锚点，再反向生成一个语义一致但带有 AI 味的 academic draft。这样每条样本都有明确的学习方向：从模板化、过度润色、机械连接的表达，回到自然、准确、领域术语稳定的学术英文。

最终训练样本采用下面的结构：

```text
instruction = 保持原意、数字、引用和术语，降低 AI 味
input       = 带有 AI 味的学术 draft
output      = human reference 学术段落
```

这里的 `output` 来自真实论文段落或高质量 reference，负责提供语义、术语和写作风格的上限；`input` 是受控生成的 AI-like draft，负责提供模型需要修正的问题。这个设计让 SFT 可以学习具体的编辑映射，也让后续 DPO 可以围绕同一个 prompt 比较不同 response 的质量差异。

AI-like draft 采用受控生成。我在 prompt 中显式控制它保留原文含义，同时加入适度的 AI 写作痕迹，例如：

- 高频 AI lexical markers：`underscore`、`pivotal`、`intricate`、`leverage`、`delve into`、`taken together` 等。
- 模板化结构：双重并列句式、泛化结尾句、机械过渡词、过度名词化表达。
- 过度正式的 academic polish：表达流畅但略显泛化，读起来像普通 LLM 生成的学术改写。
- 语义保护约束：保留数字、年份、引用、缩写、专有名词、技术术语和因果逻辑。

候选 draft 生成后还会经过质量筛选，避免把语义错误样本放进训练集。筛选重点包括：

- 数字、引用、缩写和术语是否完整保留。
- draft 与 reference 的长度比例是否合理。
- draft 是否保留原文核心结论和逻辑关系。
- draft 是否真的带有可学习的 AI-like 痕迹。
- reference 与 draft 之间是否存在足够的 humanization gap。

最终 train/val 使用 paper-level split，避免同一篇论文的段落同时进入训练集和验证集。这样评测时更接近真实迁移场景：模型需要处理训练集之外的论文表达，降低相邻段落记忆带来的评估偏差。

### 3. QLoRA SFT

SFT 阶段使用 Qwen2.5-7B-Instruct 作为基座，通过 QLoRA 训练低成本 LoRA adapter。

```text
instruction + input -> output
```

这一阶段主要让模型掌握任务格式、保留术语和引用，并学习基础的 humanization 风格。

### 4. SPIN-style DPO

DPO-v1 无需额外人工偏好标注，直接用当前 SFT 模型生成 rejected response：

```text
prompt   = instruction + input
chosen   = human / high-quality reference
rejected = SFT model prediction
```

逻辑是：如果人写 reference 比当前模型输出更好，就让模型继续学习两者之间的偏好差异。这相当于用模型自己的输出构造 on-policy 负样本，成本低，适合小项目快速迭代。

### 5. 迭代 DPO

DPO-v2 继续使用 DPO-v1 的输出作为 rejected，并使用更保守的学习率和 beta：

```text
prompt   = instruction + input
chosen   = human / high-quality reference
rejected = DPO-v1 model prediction
```

这样 rejected 会更接近当前模型能力边界，学习信号比第一轮更细。实验结果显示，DPO-v2 在保留 judge 偏好收益的同时，恢复了更多语义指标。

## 示例

下面的例子来自仓库中的 toy data：[data/examples/sample_train.json](data/examples/sample_train.json)。

**AI-like input**

```text
This study endeavors to explore the multifaceted role of adaptive feedback mechanisms in online learning environments. The results underscore the pivotal importance of personalized intervention for improving student engagement.
```

**Humanized reference**

```text
This study examines how adaptive feedback mechanisms support online learning. The results show that personalized intervention can improve student engagement.
```

这个例子体现了项目的核心目标：去掉 `endeavors`、`multifaceted`、`underscore`、`pivotal` 这类常见 AI 写作痕迹，同时保留 `adaptive feedback mechanisms`、`online learning environments` 和 `student engagement` 等关键信息。

## 评测框架

本项目把评测拆成两层，分别覆盖语义保真和主观写作质量。

### 自动语义指标

| 指标 | 作用 | 定位 |
|---|---|---|
| BERTScore-F1 | 衡量 prediction 和 reference 的语义接近程度 | 主指标 |
| chrF++ | 对术语、拼写和字符级保留敏感 | 辅助指标 |
| BLEU | 传统 n-gram overlap | 参考 |
| TER | 编辑距离类指标 | 参考 |
| Format Violation | 检查空输出、异常格式和明显失败输出 | 质量控制 |

### LLM-as-Judge

Judge 使用固定 prompt 和固定六维 rubric，输出 0 到 8 的总分。

| 维度 | 分值 | 含义 |
|---|---:|---|
| lexical markers | 0-1 | 是否避免 AI 高频词和模板短语 |
| structural patterns | 0-1 | 是否避免公式化 AI 句式 |
| naturalness | 0-2 | 是否像自然的学术英文 |
| semantic faithfulness | 0-2 | 是否保留原意、数据和逻辑 |
| terminology accuracy | 0-1 | 术语是否保留且使用准确 |
| edit value | 0-1 | 是否相比输入有实质改进 |

## Prompt 资产

Prompt 在这个项目里作为可版本管理的实验资产维护：

- `evaluation/judge/prompts.md`：完整版 judge prompt，包含 AI 写作词表、结构模式和评分标准。
- `evaluation/judge/prompts_fast.md`：正式批量评测使用的轻量 judge prompt。
- `scripts/dpo/prompt.md`：可选的 controlled rejected candidate 生成 prompt，用于构造带 AI 味但语义接近的 DPO 负样本。

## 实验结果

验证集包含 346 条 Academic Humanize 段落。Judge 模型为 `deepseek-v4-flash`，prompt 使用 `evaluation/judge/prompts_fast.md`。

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

- SFT LoRA 在自动语义指标上最接近 reference，说明它最稳地保留了原始语义。
- DPO-v1 明显提高 LLM-as-Judge 偏好分数，但会牺牲一部分 reference 接近度。
- DPO-v2 恢复了大部分语义指标，同时保留了 DPO 的偏好收益，是当前本地 7B adapter 里的最佳折中版本。
- Kimi-K2-Instruct 的 judge 分数很高，但它是 API baseline；本项目的重点是复现一套可训练、可迭代的本地后训练流程。

## 适合谁

- 想了解 SFT、DPO、SPIN-style self-play alignment 的实践流程。
- 想复现一个小成本、可解释的 LLM 后训练项目。
- 想做学术写作、论文润色、AI text humanization 方向的实验。
- 想把 LLM-as-Judge 和传统 NLP 指标结合起来做评测。

## 项目结构

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

我在仓库里只保留 toy examples 和可复现代码；真实论文语料、完整训练数据、预测结果、judge 结果、checkpoint 和大文件权重不会放进 GitHub。DPO-v2 LoRA adapter 已单独上传到 Hugging Face。

## 局限性

- 我没有发布原始论文 PDF、完整训练数据和训练 checkpoint；仓库中只保留 toy examples 和可复现脚本，方便其他人替换成自己领域的语料库。
- 我使用 LLM-as-Judge 做结构化评测，但它无法替代人工评审，所以我同时保留 BERTScore-F1、chrF++、BLEU、TER 等自动语义指标做交叉验证。
- 我目前的验证集规模是 346 条 Academic Humanize 样本；如果迁移到其他学科、期刊类型或写作风格，需要重新评估。
- 我观察到 DPO 可以提升偏好分数，也可能带来语义漂移；DPO-v2 主要用于缓解这个风险，最终仍需要结合语义指标和样例检查一起判断。
- 我使用 API baseline 做横向对比，但这类结果会受到供应商、模型版本和路由策略影响，更适合作为阶段性参考。

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

迭代 DPO 则先生成 DPO-v1 train predictions，构造 `cloud_data/ah_v2/dpo_iter2/`，设置 `configs/ah_dpo_iter2.yaml`，然后运行：

```bash
python DPO/train_dpo.py --config configs/ah_dpo_iter2.yaml
```

## 参与和反馈

欢迎通过 GitHub Issues 提交问题、建议或复现实验结果。如果你基于自己的数据集运行了这套流程，也欢迎分享你的设置和评测结果。

联系邮箱：2812156857@qq.com
