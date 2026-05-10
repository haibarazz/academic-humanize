# Data

The full corpus and generated training data are not included in this repository.
The original project was built from academic-paper paragraphs, and those source
materials should not be redistributed without checking copyright and licensing.

This repository provides schema-compatible toy examples only:

- `data/examples/sample_train.json`: tiny SFT-style training split.
- `data/examples/sample_val.json`: tiny validation split.
- `data/examples/sample_dpo_pairs.jsonl`: tiny DPO pair file.

Expected SFT row schema:

```json
{
  "sample_id": "unique_id",
  "paper_id": "source_or_group_id",
  "task_type": "ah",
  "instruction": "Rewrite instruction",
  "input": "AI-like academic draft",
  "output": "human or high-quality reference rewrite"
}
```

Expected DPO row schema:

```json
{
  "pair_id": "unique_pair_id",
  "sample_id": "source_sample_id",
  "paper_id": "source_or_group_id",
  "prompt": "instruction + input text",
  "chosen": "preferred response",
  "rejected": "less preferred response",
  "metadata": {}
}
```

For a real run, prepare your own data under the paths expected by the configs:

```text
cloud_data/ah_v2/train/final_train_v2.json
cloud_data/ah_v2/val/final_val_v2.json
cloud_data/ah_v2/dpo/train/ah_dpo_pairs_train.jsonl
cloud_data/ah_v2/dpo/val/ah_dpo_pairs_val.jsonl
```

For a toy smoke test, copy the examples into the config paths manually.
