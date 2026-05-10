# Evaluation

Evaluation is split into four stages.

## 1. Prediction

Prediction scripts only generate model outputs. They do not compute BLEU/BERTScore by default.

```text
evaluation/predict/predict_api.py
evaluation/predict/predict_local_model.py
```

Outputs go to:

```text
results/predictions/
```

## 2. Metrics

Metrics scripts read prediction reports and write scored reports.

```text
evaluation/metrics/compute_metrics.py
evaluation/metrics/compute_metrics_batch.py
```

Outputs go to:

```text
results/scored/
```

## 3. LLM-as-Judge

Judge scripts read prediction reports and score subjective quality dimensions.

```text
evaluation/judge/llm_judge.py
evaluation/judge/merge_judge_reports.py
evaluation/judge/prompts.md
```

Current v2 judge dimensions:

- `d1_lexical_markers` (0/1)
- `d2_structural_patterns` (0/1)
- `d3_naturalness` (0/1/2)
- `d4_semantic_faithfulness` (0/1/2)
- `d5_terminology_accuracy` (0/1)
- `d6_edit_value` (0/1)

The judge report stores `total` on a 0-8 scale and `total_normalized` on a 0-1 scale.

## 4. Leaderboard

Leaderboard scripts merge metrics and judge results.

```text
evaluation/leaderboard/merge_evalboard_v2.py
```

## Data Flow

```text
prediction json -> scored json -> judge json -> leaderboard
```
