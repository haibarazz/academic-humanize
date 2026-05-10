"""
Generate controlled AI-like rejected candidates for AH DPO training.

This is a local/API data-construction script. It reads the SFT train split,
calls an OpenAI-compatible API, and appends JSONL candidates that can later be
converted into TRL-style prompt/chosen/rejected pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.api_client import create_client


DEFAULT_INPUT_FILE = "cloud_data/ah_v2/train/final_train_v2.json"
DEFAULT_PROMPT_FILE = "scripts/dpo/prompt.md"
DEFAULT_OUTPUT_FILE = "data/generated/dpo_rejected_candidates_raw.jsonl"
DEFAULT_REPORT_FILE = "data/generated/dpo_rejected_candidates_report.json"
DEFAULT_DATASET_VERSION = "ah_dpo_v2_rejected"

DIFFICULTY_CYCLE = [
    "easy",
    "easy",
    "medium",
    "medium",
    "medium",
    "medium",
    "medium",
    "hard",
    "hard",
    "hard",
]

AI_MARKERS = [
    "accentuate",
    "ameliorate",
    "amplify",
    "ascertain",
    "bolster",
    "bustling",
    "conceptualize",
    "consolidate",
    "convey",
    "culminate",
    "decipher",
    "demonstrate",
    "depict",
    "delineate",
    "delve",
    "delve into",
    "disseminate",
    "elucidate",
    "endeavor",
    "engage",
    "enumerate",
    "envision",
    "enduring",
    "exacerbate",
    "expedite",
    "foster",
    "galvanize",
    "harmonize",
    "hone",
    "intricate",
    "leverage",
    "manifest",
    "mediate",
    "nuanced",
    "obscure",
    "perpetuate",
    "permeate",
    "pivotal",
    "profound",
    "recapitulate",
    "reconcile",
    "rectify",
    "reimagine",
    "scrutinize",
    "substantiate",
    "tailor",
    "testament",
    "transcend",
    "traverse",
    "underscore",
    "unveil",
    "vibrant",
    "taken together",
]

STRUCTURAL_PATTERNS = {
    "not_only_but_also": re.compile(r"\bnot only\b[\s\S]{0,120}\bbut also\b", re.I),
    "not_x_but_y": re.compile(r"\bnot\b[\s\S]{0,80}\bbut\b", re.I),
    "em_dash": re.compile(r"—"),
    "generic_conclusion": re.compile(r"\b(taken together|this underscores|this highlights the importance)\b", re.I),
    "participial_tail": re.compile(r",\s+(demonstrating|highlighting|enabling|underscoring)\b", re.I),
    "formulaic_transition": re.compile(r"\b(furthermore|moreover|additionally|consequently|subsequently)\b", re.I),
    "abstract_nominalization": re.compile(
        r"\b(the utilization of|the implementation of|the advancement of|the integration of)\b",
        re.I,
    ),
}

CJK_RE = re.compile(r"[\u3400-\u9fff]")
META_RE = re.compile(
    r"^\s*(here(?:'s| is)|revised version|rewritten version|output|json|note|explanation)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AH DPO rejected candidates with an API model")
    parser.add_argument("--input-file", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--prompt-file", type=str, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--report-file", type=str, default=DEFAULT_REPORT_FILE)
    parser.add_argument("--api-model", type=str, default=None, help="API model; defaults to env model")
    parser.add_argument("--dataset-version", type=str, default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--num-samples", type=int, default=0, help="0 means all rows")
    parser.add_argument("--num-candidates", type=int, default=1, help="candidates per source row")
    parser.add_argument("--difficulty", choices=["mixed", "easy", "medium", "hard"], default="mixed")
    parser.add_argument("--max-concurrency", type=int, default=int(os.getenv("API_MAX_CONCURRENCY", "1")))
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--min-len-ratio", type=float, default=0.60)
    parser.add_argument("--max-len-ratio", type=float, default=1.60)
    parser.add_argument("--overwrite", action="store_true")
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


def append_jsonl(path: Path, row: Mapping[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def load_prompt_template(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    begin = "<!-- BEGIN_RUNTIME_PROMPT -->"
    end = "<!-- END_RUNTIME_PROMPT -->"
    if begin in text and end in text:
        text = text.split(begin, 1)[1].split(end, 1)[0]
    return text.strip()


def fill_template(template: str, input_text: str, reference_text: str, difficulty: str) -> str:
    return (
        template.replace("{input_text}", input_text.strip())
        .replace("{reference_text}", reference_text.strip())
        .replace("{difficulty}", difficulty.strip())
    )


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def looks_english(text: str) -> bool:
    raw = (text or "").strip()
    if not raw or CJK_RE.search(raw):
        return False
    alpha = sum(1 for ch in raw if ch.isalpha())
    ascii_alpha = sum(1 for ch in raw if ch.isascii() and ch.isalpha())
    return alpha > 0 and ascii_alpha / max(alpha, 1) >= 0.60


def has_meta_output(text: str) -> bool:
    raw = (text or "").strip()
    return bool("```" in raw or META_RE.search(raw))


def detect_markers(text: str) -> List[str]:
    lowered = normalize_text(text)
    found = []
    for marker in AI_MARKERS:
        marker_norm = marker.lower()
        if " " in marker_norm:
            if marker_norm in lowered:
                found.append(marker)
        else:
            if re.search(r"\b" + re.escape(marker_norm) + r"\b", lowered):
                found.append(marker)
    return sorted(set(found))


def detect_structural_patterns(text: str) -> List[str]:
    return sorted(name for name, pattern in STRUCTURAL_PATTERNS.items() if pattern.search(text or ""))


def choose_difficulty(row_index: int, candidate_index: int, mode: str) -> str:
    if mode != "mixed":
        return mode
    return DIFFICULTY_CYCLE[(row_index + candidate_index) % len(DIFFICULTY_CYCLE)]


def candidate_id_for(sample_id: str, candidate_index: int) -> str:
    return f"{sample_id}__dpo_rej_{candidate_index:02d}"


def build_jobs(rows: List[Dict[str, Any]], args: argparse.Namespace, done_ids: set) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    limit = len(rows) if args.num_samples <= 0 else min(args.num_samples, len(rows))
    selected = rows[:limit]
    for row_index, row in enumerate(selected):
        sample_id = str(row.get("sample_id", "")).strip()
        input_text = str(row.get("input", "")).strip()
        reference_text = str(row.get("output", "")).strip()
        if not sample_id or not input_text or not reference_text:
            continue
        for candidate_index in range(max(1, args.num_candidates)):
            candidate_id = candidate_id_for(sample_id, candidate_index)
            if candidate_id in done_ids:
                continue
            jobs.append(
                {
                    "row_index": row_index,
                    "candidate_index": candidate_index,
                    "candidate_id": candidate_id,
                    "difficulty": choose_difficulty(row_index, candidate_index, args.difficulty),
                    "source_row": row,
                }
            )
    return jobs


def validate_rejected(text: str, input_text: str, reference_text: str, args: argparse.Namespace) -> Tuple[bool, str, Dict[str, Any]]:
    text = (text or "").strip()
    reference_text = (reference_text or "").strip()
    length_ratio = len(text) / max(len(reference_text), 1)
    markers = detect_markers(text)
    structures = detect_structural_patterns(text)
    metrics = {
        "length_ratio_to_reference": float(length_ratio),
        "lexical_marker_count": len(markers),
        "structural_pattern_count": len(structures),
        "detected_lexical_markers": markers,
        "detected_structural_patterns": structures,
    }
    if not text:
        return False, "empty_rejected", metrics
    if not looks_english(text):
        return False, "not_english", metrics
    if has_meta_output(text):
        return False, "meta_output", metrics
    if normalize_text(text) == normalize_text(reference_text):
        return False, "same_as_reference", metrics
    if not (args.min_len_ratio <= length_ratio <= args.max_len_ratio):
        return False, "length_ratio_out_of_range", metrics
    if normalize_text(text) == normalize_text(input_text):
        metrics["same_as_input"] = True
    return True, "", metrics


def generate_one(job: Mapping[str, Any], template: str, client: Any, args: argparse.Namespace) -> Tuple[Optional[Dict[str, Any]], str]:
    row = dict(job["source_row"])
    input_text = str(row.get("input", "")).strip()
    reference_text = str(row.get("output", "")).strip()
    difficulty = str(job["difficulty"])
    user_prompt = fill_template(
        template=template,
        input_text=input_text,
        reference_text=reference_text,
        difficulty=difficulty,
    )
    system_prompt = (
        "You generate controlled rejected examples for DPO preference training. "
        "Follow the user prompt exactly and return valid JSON only."
    )
    raw = client.call(system_prompt, user_prompt)
    parsed = extract_json_object(raw) or {}
    rejected = str(parsed.get("rejected", "")).strip()
    if not rejected and raw:
        rejected = raw.strip()

    ok, reason, local_metrics = validate_rejected(rejected, input_text, reference_text, args)
    if not ok:
        return None, reason

    lexical_used = parsed.get("lexical_markers_used")
    structural_used = parsed.get("structural_patterns_used")
    if not isinstance(lexical_used, list):
        lexical_used = local_metrics["detected_lexical_markers"]
    if not isinstance(structural_used, list):
        structural_used = local_metrics["detected_structural_patterns"]

    payload = {
        "candidate_id": str(job["candidate_id"]),
        "sample_id": str(row.get("sample_id", "")).strip(),
        "paper_id": str(row.get("paper_id", "unknown")).strip() or "unknown",
        "source_candidate_id": str(row.get("candidate_id", "")).strip(),
        "instruction": str(row.get("instruction", "")).strip(),
        "input": input_text,
        "chosen": reference_text,
        "rejected": rejected,
        "difficulty": difficulty,
        "lexical_markers_used": [str(x).strip() for x in lexical_used if str(x).strip()],
        "structural_patterns_used": [str(x).strip() for x in structural_used if str(x).strip()],
        "local_metrics": local_metrics,
        "model_id": client.model,
        "dataset_version": args.dataset_version,
        "source_dataset_version": str(row.get("dataset_version", "")).strip(),
        "raw_response": raw,
        "json_parse_ok": bool(parsed),
        "created_at": datetime.now().isoformat(),
    }
    return payload, ""


def main() -> int:
    args = parse_args()
    input_file = Path(args.input_file)
    prompt_file = Path(args.prompt_file)
    output_file = Path(args.output_file)
    report_file = Path(args.report_file)

    if args.overwrite:
        if output_file.exists():
            output_file.unlink()
        if report_file.exists():
            report_file.unlink()

    rows = read_json(input_file)
    if not isinstance(rows, list):
        raise ValueError(f"{input_file} must be a JSON list")

    existing_rows = read_jsonl(output_file)
    done_ids = {str(row.get("candidate_id", "")).strip() for row in existing_rows if row.get("candidate_id")}
    jobs = build_jobs(rows, args, done_ids)
    template = load_prompt_template(prompt_file)
    client = create_client(model=args.api_model)
    client.max_tokens = int(args.max_tokens)
    client.temperature = float(args.temperature)

    print("=" * 72)
    print("Academic Humanize DPO rejected candidate generation")
    print("=" * 72)
    print(f"input_file:     {input_file}")
    print(f"prompt_file:    {prompt_file}")
    print(f"api_model:      {client.model}")
    print(f"selected_rows:  {len(rows) if args.num_samples <= 0 else min(args.num_samples, len(rows))}")
    print(f"existing_rows:  {len(existing_rows)}")
    print(f"pending_jobs:   {len(jobs)}")
    print(f"concurrency:    {max(1, args.max_concurrency)}")
    print(f"output_file:    {output_file}")
    print("=" * 72)

    write_lock = threading.Lock()
    reject_counter: Counter[str] = Counter()
    new_count = 0

    with ThreadPoolExecutor(max_workers=max(1, args.max_concurrency)) as executor:
        futures = [executor.submit(generate_one, job, template, client, args) for job in jobs]
        progress = tqdm(as_completed(futures), total=len(futures), desc="Generating rejected", ncols=100)
        for future in progress:
            try:
                row, reason = future.result()
            except Exception as exc:
                row, reason = None, f"exception:{type(exc).__name__}"
            if row:
                append_jsonl(output_file, row, write_lock)
                new_count += 1
            else:
                reject_counter[reason or "unknown_failure"] += 1
            progress.set_postfix(new=new_count, rejected=sum(reject_counter.values()))

    final_rows = read_jsonl(output_file)
    difficulty_counts = Counter(str(row.get("difficulty", "unknown")) for row in final_rows)
    report = {
        "report_type": "ah_dpo_rejected_candidate_generation",
        "created_at": datetime.now().isoformat(),
        "dataset_version": args.dataset_version,
        "input_file": str(input_file),
        "prompt_file": str(prompt_file),
        "output_file": str(output_file),
        "api_model": client.model,
        "counts": {
            "input_rows": len(rows),
            "existing_before_run": len(existing_rows),
            "pending_jobs": len(jobs),
            "new_rows": new_count,
            "total_output_rows": len(final_rows),
        },
        "difficulty_counts": dict(difficulty_counts),
        "rejections": dict(reject_counter),
        "config": {
            "num_samples": args.num_samples,
            "num_candidates": args.num_candidates,
            "difficulty": args.difficulty,
            "max_concurrency": args.max_concurrency,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "min_len_ratio": args.min_len_ratio,
            "max_len_ratio": args.max_len_ratio,
        },
    }
    write_json(report_file, report)

    print("=" * 72)
    print("DPO rejected candidate generation complete")
    print("=" * 72)
    print(f"new_rows:          {new_count}")
    print(f"total_output_rows: {len(final_rows)}")
    print(f"report_file:       {report_file}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

