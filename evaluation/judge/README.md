# Academic Humanize Judge

This folder contains the LLM-as-Judge evaluation layer.

## Current layout

```text
evaluation/judge/
├── llm_judge.py              # schema-driven generic runner
├── prompts/
│   ├── judge_6d.md
│   ├── judge_6d_fast.md
│   └── binary24.md
├── schemas/
│   ├── judge_6d.yaml
│   └── binary24.yaml
└── README.md
```

Old prompt files such as `prompts.md` and `prompts_fast.md` are kept
temporarily for compatibility with earlier commands. New runs should use
`llm_judge.py` plus a schema.

## Schemas

A schema defines:

- report type
- prompt file
- prompt version
- dimensions and score ranges
- block grouping
- hard-fail rules
- default max tokens

This lets the same runner support both the 6D judge and the binary-24 judge.

## Run the 6D judge

```bash
python evaluation/judge/llm_judge.py \
  --schema evaluation/judge/schemas/judge_6d.yaml \
  --report-file results/predictions/ah_v2_api_kimi2_pred.json \
  --api-model deepseek-v4-flash \
  --max-samples 100 \
  --max-concurrency 4 \
  --resume \
  --output results/judge/ah_judge_kimi2_deepseek_v4_6d_100.json
```

## Run binary-24 judge

```bash
python evaluation/judge/llm_judge.py \
  --schema evaluation/judge/schemas/binary24.yaml \
  --report-file results/predictions/ah_v2_api_kimi2_pred.json \
  --api-model deepseek-v4-flash \
  --max-samples 100 \
  --max-concurrency 4 \
  --resume \
  --output results/judge/ah_judge_kimi2_deepseek_v4_binary24_100.json
```

## Output interpretation

The report stores per-row scores, raw judge responses, and summary statistics.
For binary-24, `hard_fail=true` means at least one meaning-safety dimension failed.
Use the block-level `issue_rate` fields to diagnose whether a model mainly fails
on semantic safety, vocabulary, structure, discourse, or editing/formatting.
