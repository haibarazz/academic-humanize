"""
Build TRL-style prompt/chosen/rejected DPO pairs for Academic Humanize.

The script uses the SFT train split as the source of prompts and human
references. It can combine cheap raw-input rejected pairs with API-generated
AI-like rejected candidates.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


DEFAULT_SFT_TRAIN_FILE = "cloud_data/ah_v2/train/final_train_v2.json"
DEFAULT_CANDIDATE_FILE = "data/generated/dpo_rejected_candidates_raw.jsonl"
DEFAULT_OUTPUT_ALL = "cloud_data/ah_v2/dpo/ah_dpo_pairs_all.jsonl"
DEFAULT_OUTPUT_TRAIN = "cloud_data/ah_v2/dpo/train/ah_dpo_pairs_train.jsonl"
DEFAULT_OUTPUT_VAL = "cloud_data/ah_v2/dpo/val/ah_dpo_pairs_val.jsonl"
DEFAULT_REPORT_FILE = "cloud_data/ah_v2/dpo/ah_dpo_pairs_report.json"
DEFAULT_DATASET_VERSION = "ah_dpo_v2"

AI_MARKERS = [
    "accentuate",
    "ameliorate",
    "amplify",
    "ascertain",
    "bolster",
    "conceptualize",
    "consolidate",
    "culminate",
    "decipher",
    "delineate",
    "delve",
    "delve into",
    "disseminate",
    "elucidate",
    "endeavor",
    "foster",
    "intricate",
    "leverage",
    "nuanced",
    "perpetuate",
    "pivotal",
    "profound",
    "scrutinize",
    "substantiate",
    "testament",
    "underscore",
    "unveil",
    "vibrant",
    "taken together",
]

STRUCTURAL_PATTERNS = [
    re.compile(r"\bnot only\b[\s\S]{0,120}\bbut also\b", re.I),
    re.compile(r"\bnot\b[\s\S]{0,80}\bbut\b", re.I),
    re.compile(r"—"),
    re.compile(r"\b(taken together|this underscores|this highlights the importance)\b", re.I),
    re.compile(r",\s+(demonstrating|highlighting|enabling|underscoring)\b", re.I),
    re.compile(r"\b(furthermore|moreover|additionally|consequently|subsequently)\b", re.I),
    re.compile(r"\b(the utilization of|the implementation of|the advancement of|the integration of)\b", re.I),
]

CJK_RE = re.compile(r"[\u3400-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AH DPO prompt/chosen/rejected pair files")
    parser.add_argument("--sft-train-file", type=str, default=DEFAULT_SFT_TRAIN_FILE)
    parser.add_argument("--candidate-file", type=str, default=DEFAULT_CANDIDATE_FILE)
    parser.add_argument("--output-all", type=str, default=DEFAULT_OUTPUT_ALL)
    parser.add_argument("--output-train", type=str, default=DEFAULT_OUTPUT_TRAIN)
    parser.add_argument("--output-val", type=str, default=DEFAULT_OUTPUT_VAL)
    parser.add_argument("--report-file", type=str, default=DEFAULT_REPORT_FILE)
    parser.add_argument("--dataset-version", type=str, default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--max-generated-per-sample", type=int, default=2)
    parser.add_argument("--min-len-ratio", type=float, default=0.55)
    parser.add_argument("--max-len-ratio", type=float, default=1.70)
    parser.add_argument("--min-ai-signals", type=int, default=1)
    parser.add_argument("--include-input-pairs", dest="include_input_pairs", action="store_true", default=True)
    parser.add_argument("--no-include-input-pairs", dest="include_input_pairs", action="store_false")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    tmp_path.replace(path)
    return count


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def prompt_text(instruction: str, input_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def looks_english(text: str) -> bool:
    raw = (text or "").strip()
    if not raw or CJK_RE.search(raw):
        return False
    alpha = sum(1 for ch in raw if ch.isalpha())
    ascii_alpha = sum(1 for ch in raw if ch.isascii() and ch.isalpha())
    return alpha > 0 and ascii_alpha / max(alpha, 1) >= 0.60


def count_ai_markers(text: str) -> int:
    lowered = normalize_text(text)
    count = 0
    for marker in AI_MARKERS:
        marker = marker.lower()
        if " " in marker:
            count += int(marker in lowered)
        else:
            count += int(bool(re.search(r"\b" + re.escape(marker) + r"\b", lowered)))
    return count


def count_structural_patterns(text: str) -> int:
    return sum(1 for pattern in STRUCTURAL_PATTERNS if pattern.search(text or ""))


def ai_signal_count(candidate: Mapping[str, Any], rejected: str) -> int:
    lexical = candidate.get("lexical_markers_used")
    structural = candidate.get("structural_patterns_used")
    count = 0
    if isinstance(lexical, list):
        count += len([x for x in lexical if str(x).strip()])
    else:
        count += count_ai_markers(rejected)
    if isinstance(structural, list):
        count += len([x for x in structural if str(x).strip()])
    else:
        count += count_structural_patterns(rejected)
    return count


def build_pair(
    source_row: Mapping[str, Any],
    rejected: str,
    pair_id: str,
    pair_type: str,
    dataset_version: str,
    extra_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    instruction = str(source_row.get("instruction", "")).strip()
    input_text = str(source_row.get("input", "")).strip()
    chosen = str(source_row.get("output", "")).strip()
    return {
        "pair_id": pair_id,
        "sample_id": str(source_row.get("sample_id", "")).strip(),
        "paper_id": str(source_row.get("paper_id", "unknown")).strip() or "unknown",
        "prompt": prompt_text(instruction, input_text),
        "chosen": chosen,
        "rejected": rejected.strip(),
        "metadata": {
            "pair_type": pair_type,
            "dataset_version": dataset_version,
            "instruction": instruction,
            "input_text": input_text,
            "source_candidate_id": str(source_row.get("candidate_id", "")).strip(),
            "source_dataset_version": str(source_row.get("dataset_version", "")).strip(),
            **dict(extra_metadata),
        },
    }


def validate_pair(pair: Mapping[str, Any], args: argparse.Namespace, require_ai_signal: bool) -> Tuple[bool, str]:
    chosen = str(pair.get("chosen", "")).strip()
    rejected = str(pair.get("rejected", "")).strip()
    if not chosen or not rejected:
        return False, "empty_chosen_or_rejected"
    if not looks_english(chosen) or not looks_english(rejected):
        return False, "not_english"
    if normalize_text(chosen) == normalize_text(rejected):
        return False, "chosen_equals_rejected"
    length_ratio = len(rejected) / max(len(chosen), 1)
    if not (args.min_len_ratio <= length_ratio <= args.max_len_ratio):
        return False, "length_ratio_out_of_range"
    if require_ai_signal:
        signals = int(pair.get("metadata", {}).get("ai_signal_count", 0))
        if signals < args.min_ai_signals:
            return False, "too_few_ai_signals"
    return True, ""


def group_candidates(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        grouped[sample_id].append(dict(row))
    return grouped


def split_by_paper(rows: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not rows:
        return [], []
    paper_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        paper_id = str(row.get("paper_id", "unknown")).strip() or "unknown"
        paper_to_rows[paper_id].append(row)

    papers = sorted(paper_to_rows)
    rnd = random.Random(seed)
    rnd.shuffle(papers)

    target_val = int(round(len(rows) * max(0.0, min(val_ratio, 0.5))))
    if target_val <= 0 and len(rows) > 1:
        target_val = 1

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    val_count = 0
    for paper_id in papers:
        group = paper_to_rows[paper_id]
        if val_count < target_val:
            val_rows.extend(group)
            val_count += len(group)
        else:
            train_rows.extend(group)

    if not train_rows and val_rows:
        train_rows.append(val_rows.pop())
    return train_rows, val_rows


def difficulty_sort_key(row: Mapping[str, Any]) -> Tuple[int, str]:
    order = {"medium": 0, "hard": 1, "easy": 2}
    return (order.get(str(row.get("difficulty", "")).lower(), 9), str(row.get("candidate_id", "")))


def build_pairs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Counter]:
    sft_rows = read_json(Path(args.sft_train_file))
    if not isinstance(sft_rows, list):
        raise ValueError(f"{args.sft_train_file} must be a JSON list")
    source_by_sample = {str(row.get("sample_id", "")).strip(): row for row in sft_rows if row.get("sample_id")}
    candidate_rows = read_jsonl(Path(args.candidate_file))
    candidates_by_sample = group_candidates(candidate_rows)

    pairs: List[Dict[str, Any]] = []
    reject_counter: Counter[str] = Counter()
    seen_keys = set()

    def add_pair(pair: Dict[str, Any], require_ai_signal: bool) -> None:
        key = (
            normalize_text(str(pair.get("prompt", ""))),
            normalize_text(str(pair.get("chosen", ""))),
            normalize_text(str(pair.get("rejected", ""))),
        )
        if key in seen_keys:
            reject_counter["duplicate_pair"] += 1
            return
        ok, reason = validate_pair(pair, args, require_ai_signal=require_ai_signal)
        if not ok:
            reject_counter[reason] += 1
            return
        seen_keys.add(key)
        pairs.append(pair)

    for sample_id, source_row in source_by_sample.items():
        if args.include_input_pairs:
            pair = build_pair(
                source_row=source_row,
                rejected=str(source_row.get("input", "")).strip(),
                pair_id=f"{sample_id}__raw_input_rejected",
                pair_type="raw_input",
                dataset_version=args.dataset_version,
                extra_metadata={
                    "difficulty": "raw_input",
                    "candidate_id": "",
                    "ai_signal_count": count_ai_markers(str(source_row.get("input", "")))
                    + count_structural_patterns(str(source_row.get("input", ""))),
                },
            )
            add_pair(pair, require_ai_signal=False)

        generated = sorted(candidates_by_sample.get(sample_id, []), key=difficulty_sort_key)
        if args.max_generated_per_sample > 0:
            generated = generated[: args.max_generated_per_sample]
        for candidate in generated:
            rejected = str(candidate.get("rejected", "")).strip()
            signal_count = ai_signal_count(candidate, rejected)
            pair = build_pair(
                source_row=source_row,
                rejected=rejected,
                pair_id=f"{candidate.get('candidate_id', sample_id)}__pair",
                pair_type="generated_rejected",
                dataset_version=args.dataset_version,
                extra_metadata={
                    "candidate_id": str(candidate.get("candidate_id", "")).strip(),
                    "difficulty": str(candidate.get("difficulty", "")).strip(),
                    "generator_model": str(candidate.get("model_id", "")).strip(),
                    "lexical_markers_used": candidate.get("lexical_markers_used") or [],
                    "structural_patterns_used": candidate.get("structural_patterns_used") or [],
                    "ai_signal_count": signal_count,
                    "local_metrics": candidate.get("local_metrics") or {},
                    "json_parse_ok": bool(candidate.get("json_parse_ok", False)),
                },
            )
            add_pair(pair, require_ai_signal=True)

    return pairs, reject_counter


def count_by(rows: Iterable[Mapping[str, Any]], key_path: Tuple[str, ...]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value: Any = row
        for key in key_path:
            if isinstance(value, Mapping):
                value = value.get(key)
            else:
                value = None
                break
        counts[str(value or "unknown")] += 1
    return dict(counts)


def main() -> int:
    args = parse_args()
    pairs, reject_counter = build_pairs(args)
    train_rows, val_rows = split_by_paper(pairs, args.val_ratio, args.seed)

    output_all = Path(args.output_all)
    output_train = Path(args.output_train)
    output_val = Path(args.output_val)
    report_file = Path(args.report_file)

    write_jsonl(output_all, pairs)
    write_jsonl(output_train, train_rows)
    write_jsonl(output_val, val_rows)

    train_papers = {str(row.get("paper_id", "")) for row in train_rows}
    val_papers = {str(row.get("paper_id", "")) for row in val_rows}
    report = {
        "report_type": "ah_dpo_pair_build_report",
        "created_at": datetime.now().isoformat(),
        "dataset_version": args.dataset_version,
        "inputs": {
            "sft_train_file": args.sft_train_file,
            "candidate_file": args.candidate_file,
        },
        "outputs": {
            "all_pairs": args.output_all,
            "train_pairs": args.output_train,
            "val_pairs": args.output_val,
        },
        "counts": {
            "all_pairs": len(pairs),
            "train_pairs": len(train_rows),
            "val_pairs": len(val_rows),
            "train_papers": len(train_papers),
            "val_papers": len(val_papers),
            "paper_overlap": len(train_papers & val_papers),
        },
        "distribution": {
            "pair_type": count_by(pairs, ("metadata", "pair_type")),
            "difficulty": count_by(pairs, ("metadata", "difficulty")),
        },
        "rejections": dict(reject_counter),
        "config": {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "max_generated_per_sample": args.max_generated_per_sample,
            "include_input_pairs": args.include_input_pairs,
            "min_len_ratio": args.min_len_ratio,
            "max_len_ratio": args.max_len_ratio,
            "min_ai_signals": args.min_ai_signals,
        },
        "leakage_note": (
            "This script defaults to splitting DPO train/val from the SFT train file. "
            "It does not use cloud_data/ah_v2/val/final_val_v2.json, so the final evaluation split stays held out."
        ),
    }
    write_json(report_file, report)

    print("=" * 72)
    print("Academic Humanize DPO pair build complete")
    print("=" * 72)
    print(f"all_pairs:   {len(pairs)}")
    print(f"train_pairs: {len(train_rows)}")
    print(f"val_pairs:   {len(val_rows)}")
    print(f"output_train: {output_train}")
    print(f"output_val:   {output_val}")
    print(f"report_file:  {report_file}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

