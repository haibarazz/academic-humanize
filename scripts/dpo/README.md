# DPO Data Construction

This folder contains preference-data builders for Academic Humanize.
Training code lives in `DPO/train_dpo.py`.

## Main path: SPIN-style DPO

The recommended low-cost path is to build preference pairs from model
predictions:

```text
prompt   = instruction + input
chosen   = human or high-quality reference
rejected = current model prediction for the same input
```

Generate train-split predictions first:

```bash
python evaluation/predict/predict_local_model.py \
  --val-file cloud_data/ah_v2/train/final_train_v2.json \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --adapter-path checkpoints/ah_sft_v2/YOUR_SFT_ADAPTER \
  --max-new-tokens 1024 \
  --output results/predictions/ah_sft_train_pred_for_dpo.json
```

Build DPO pairs:

```bash
python scripts/dpo/build_dpo_pairs_from_predictions.py \
  --train-file cloud_data/ah_v2/train/final_train_v2.json \
  --prediction-report results/predictions/ah_sft_train_pred_for_dpo.json \
  --output-all cloud_data/ah_v2/dpo/ah_dpo_pairs_all.jsonl \
  --output-train cloud_data/ah_v2/dpo/train/ah_dpo_pairs_train.jsonl \
  --output-val cloud_data/ah_v2/dpo/val/ah_dpo_pairs_val.jsonl \
  --report-file cloud_data/ah_v2/dpo/ah_dpo_pairs_report.json
```

Do not build DPO pairs from the held-out validation split.

## Optional path: generated rejected candidates

`build_dpo_rejected_candidates.py` and `prompt.md` can generate controlled
AI-like rejected candidates with an API model. This is useful when you want
explicit lexical/structural negative examples rather than on-policy model
responses.

```bash
API_MAX_CONCURRENCY=4 python scripts/dpo/build_dpo_rejected_candidates.py \
  --input-file cloud_data/ah_v2/train/final_train_v2.json \
  --api-model deepseek-v4-flash \
  --num-samples 100 \
  --num-candidates 1 \
  --output-file data/generated/dpo_rejected_candidates_raw.jsonl \
  --report-file data/generated/dpo_rejected_candidates_report.json
```

Then convert generated candidates into DPO pairs:

```bash
python scripts/dpo/build_dpo_pairs.py \
  --sft-train-file cloud_data/ah_v2/train/final_train_v2.json \
  --candidate-file data/generated/dpo_rejected_candidates_raw.jsonl \
  --output-all cloud_data/ah_v2/dpo/ah_dpo_pairs_all.jsonl \
  --output-train cloud_data/ah_v2/dpo/train/ah_dpo_pairs_train.jsonl \
  --output-val cloud_data/ah_v2/dpo/val/ah_dpo_pairs_val.jsonl \
  --report-file cloud_data/ah_v2/dpo/ah_dpo_pairs_report.json
```
