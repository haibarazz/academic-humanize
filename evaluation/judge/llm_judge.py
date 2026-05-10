"""
LLM-as-Judge for Academic Humanize prediction reports.

This is the canonical v2 judge. It reads prediction reports, samples rows, and
asks a fixed judge model to score the output using the rubric in
`evaluation/judge/prompts.md`.

The v2 schema uses six dimensions with an 8-point total:
- d1_lexical_markers: 0/1
- d2_structural_patterns: 0/1
- d3_naturalness: 0/1/2
- d4_semantic_faithfulness: 0/1/2
- d5_terminology_accuracy: 0/1
- d6_edit_value: 0/1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.api_client import create_client

load_dotenv()

DEFAULT_PROMPT_FILE = Path(__file__).resolve().with_name("prompts.md")
PROMPT_VERSION = "ah_judge_v2"

JUDGE_DIMENSIONS = [
    "d1_lexical_markers",
    "d2_structural_patterns",
    "d3_naturalness",
    "d4_semantic_faithfulness",
    "d5_terminology_accuracy",
    "d6_edit_value",
]

JUDGE_SCORE_RANGES = {
    "d1_lexical_markers": (0, 1),
    "d2_structural_patterns": (0, 1),
    "d3_naturalness": (0, 2),
    "d4_semantic_faithfulness": (0, 2),
    "d5_terminology_accuracy": (0, 1),
    "d6_edit_value": (0, 1),
}

MAX_TOTAL = 8
_THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    default_judge_model = os.getenv("AH_JUDGE_MODEL") or os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")
    parser = argparse.ArgumentParser(description="LLM-as-Judge v2 for Academic Humanize prediction reports.")
    parser.add_argument("--report-file", type=str, required=True, help="Prediction/scored ah_model_eval_v1 report path.")
    parser.add_argument("--api-model", type=str, default=default_judge_model, help="Judge model id.")
    parser.add_argument("--prompt-file", type=str, default=str(DEFAULT_PROMPT_FILE), help="Judge rubric markdown file.")
    parser.add_argument("--prompt-version", type=str, default=PROMPT_VERSION, help="Prompt version label stored in the report.")
    parser.add_argument("--output", type=str, default=None, help="Output path. Default: results/judge/ah_judge_*.json")
    parser.add_argument("--max-samples", type=int, default=100, help="Rows to judge. Default: 100. Use <=0 for all rows.")
    parser.add_argument("--sample-source", choices=["rows", "badcases"], default="rows", help="Default reads prediction rows.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Judge response max tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge temperature.")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Concurrent judge API calls. Default: 1.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output file and rerun only failed/unparsed rows.")
    parser.add_argument("--save-every", type=int, default=10, help="Save partial report every N judged rows; 0 disables partial saves.")
    return parser.parse_args()


def load_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Report file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Invalid report format: expected JSON object.")
    if payload.get("report_type") != "ah_model_eval_v1":
        raise ValueError(f"Only ah_model_eval_v1 is supported, got: {payload.get('report_type')}")
    return payload


def load_prompt_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Judge prompt file does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Judge prompt file is empty: {path}")
    return text


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
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


def clamp_dimension_score(dimension: str, value: Any) -> Optional[int]:
    if dimension not in JUDGE_SCORE_RANGES:
        return None
    try:
        score = float(value)
    except Exception:
        return None
    low, high = JUDGE_SCORE_RANGES[dimension]
    rounded = int(round(score))
    return int(max(low, min(high, rounded)))


def clamp_total(value: Any) -> Optional[int]:
    try:
        score = float(value)
    except Exception:
        return None
    return int(max(0, min(MAX_TOTAL, round(score))))


def summarize_scores(values: List[Optional[float]]) -> Dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "median": None, "min": None, "max": None, "count": 0}
    return {
        "mean": float(sum(clean) / len(clean)),
        "median": float(median(clean)),
        "min": float(min(clean)),
        "max": float(max(clean)),
        "count": len(clean),
    }


def build_default_output_path(report_file: Path, api_model: str) -> Path:
    output_dir = Path("results/judge")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = api_model.replace("/", "_").replace(" ", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"ah_judge_{report_file.stem}_{safe_model}_{stamp}.json"


def select_items(payload: Dict[str, Any], sample_source: str, max_samples: int, seed: int) -> Tuple[List[Dict[str, Any]], int]:
    raw_items = payload.get(sample_source)
    if not isinstance(raw_items, list):
        raise ValueError(f"The report is missing a valid `{sample_source}` list.")
    items = [item for item in raw_items if isinstance(item, dict)]
    if not items:
        raise ValueError(f"The report `{sample_source}` list is empty.")

    pool_size = len(items)
    if max_samples and max_samples > 0 and len(items) > max_samples:
        rnd = random.Random(seed)
        items = rnd.sample(items, k=max_samples)
    return items, pool_size


def build_judge_prompt(prompt_text: str, input_text: str, prediction: str, reference: str) -> Tuple[str, str]:
    system_prompt = (
        prompt_text.strip()
        + "\n\nIMPORTANT: Use the full rubric above, including the vocabulary and structural pattern reference. "
        + "Return only one valid JSON object matching the requested output format."
    )
    user_prompt = (
        "**INPUT (AI-like draft):**\n"
        f"{input_text}\n\n"
        "**OUTPUT (Model rewrite):**\n"
        f"{prediction}\n\n"
        "**REFERENCE (Human-written original):**\n"
        f"{reference}\n\n"
        "Evaluate OUTPUT on the 6 dimensions. Respond only with JSON."
    )
    return system_prompt, user_prompt


def build_judged_row(item: Dict[str, Any], idx: int, obj: Optional[Dict[str, Any]], raw_response: str) -> Dict[str, Any]:
    scores = {dimension: clamp_dimension_score(dimension, (obj or {}).get(dimension)) for dimension in JUDGE_DIMENSIONS}
    valid_scores = [value for value in scores.values() if value is not None]
    parse_ok = obj is not None and len(valid_scores) == len(JUDGE_DIMENSIONS)
    computed_total = int(sum(valid_scores)) if parse_ok else None
    model_total = clamp_total((obj or {}).get("total")) if obj else None
    total = computed_total
    total_normalized = float(total / MAX_TOTAL) if total is not None else None

    return {
        "sample_id": str(item.get("sample_id", f"row_{idx}")),
        "paper_id": str(item.get("paper_id", "unknown")),
        "input": str(item.get("input", "")).strip(),
        "prediction": str(item.get("prediction", "")).strip(),
        "reference": str(item.get("reference", item.get("output", ""))).strip(),
        "d1_lexical_markers": scores["d1_lexical_markers"],
        "d2_structural_patterns": scores["d2_structural_patterns"],
        "d3_naturalness": scores["d3_naturalness"],
        "d4_semantic_faithfulness": scores["d4_semantic_faithfulness"],
        "d5_terminology_accuracy": scores["d5_terminology_accuracy"],
        "d6_edit_value": scores["d6_edit_value"],
        "model_total": model_total,
        "total": total,
        "total_normalized": total_normalized,
        "parse_ok": bool(parse_ok),
        "rationale": str((obj or {}).get("rationale", (obj or {}).get("reason", ""))).strip(),
        "raw_response": raw_response,
    }


def build_summary(judged_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "total": summarize_scores([row.get("total") for row in judged_rows]),
        "total_normalized": summarize_scores([row.get("total_normalized") for row in judged_rows]),
    }
    for dimension in JUDGE_DIMENSIONS:
        summary[dimension] = summarize_scores([row.get(dimension) for row in judged_rows])
    return summary


def build_report(
    *,
    run_id: str,
    source_payload: Dict[str, Any],
    report_path: Path,
    args: argparse.Namespace,
    prompt_path: Path,
    pool_size: int,
    judged_rows: List[Dict[str, Any]],
    status: str,
) -> Dict[str, Any]:
    parsed_rows = sum(1 for row in judged_rows if row.get("parse_ok"))
    return {
        "report_type": "ah_judge_report_v2",
        "run_id": run_id,
        "status": status,
        "source_report": str(report_path),
        "source_report_type": str(source_payload.get("report_type", "")).strip(),
        "source_run_id": str(source_payload.get("run_id", "")).strip(),
        "source_model_id": str(source_payload.get("model_id", "")).strip(),
        "judge_model": args.api_model,
        "task": "ah",
        "sample_source": args.sample_source,
        "prompt_file": str(prompt_path),
        "prompt_version": args.prompt_version,
        "dimensions": JUDGE_DIMENSIONS,
        "score_schema": {
            "max_total": MAX_TOTAL,
            "ranges": {key: list(value) for key, value in JUDGE_SCORE_RANGES.items()},
            "total_rule": "sum of six dimensions; model-provided total is stored as model_total only",
        },
        "sampling": {
            "pool_size": pool_size,
            "max_samples": args.max_samples,
            "seed": args.seed,
        },
        "settings": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "max_concurrency": args.max_concurrency,
            "resume": args.resume,
            "save_every": args.save_every,
        },
        "counts": {
            "input_candidates": pool_size,
            "judged_rows": len(judged_rows),
            "parsed_rows": parsed_rows,
            "failed_parse_rows": len(judged_rows) - parsed_rows,
        },
        "summary": build_summary(judged_rows),
        "rows": judged_rows,
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def configure_client(client, args: argparse.Namespace):
    client.max_tokens = int(args.max_tokens)
    client.temperature = float(args.temperature)
    return client


def get_thread_client(args: argparse.Namespace):
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = configure_client(create_client(model=args.api_model), args)
        _THREAD_LOCAL.client = client
    return client


def judge_one_item(
    *,
    idx: int,
    item: Dict[str, Any],
    prompt_text: str,
    args: argparse.Namespace,
    client=None,
) -> Tuple[int, Dict[str, Any]]:
    active_client = client or get_thread_client(args)
    input_text = str(item.get("input", "")).strip()
    prediction = str(item.get("prediction", "")).strip()
    reference = str(item.get("reference", item.get("output", ""))).strip()
    system_prompt, user_prompt = build_judge_prompt(prompt_text, input_text, prediction, reference)
    raw_response = active_client.call(system_prompt, user_prompt) or ""
    parsed = extract_json_block(raw_response)
    return idx, build_judged_row(item, idx, parsed, raw_response)


def completed_rows_in_order(judged_rows: List[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [row for row in judged_rows if row is not None]


def is_completed_judge_row(row: Dict[str, Any]) -> bool:
    return bool(row.get("parse_ok")) and bool(str(row.get("raw_response", "")).strip())


def load_resume_judge_rows(output_path: Path, args: argparse.Namespace, source_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Cannot resume: existing output is not a JSON object: {output_path}")
    if payload.get("report_type") != "ah_judge_report_v2":
        raise ValueError(f"Cannot resume: existing output report_type mismatch: {payload.get('report_type')}")

    existing_source_model = str(payload.get("source_model_id", "")).strip()
    current_source_model = str(source_payload.get("model_id", "")).strip()
    if existing_source_model and existing_source_model != current_source_model:
        raise ValueError(
            f"Cannot resume: existing source_model_id={existing_source_model}, "
            f"current source_model_id={current_source_model}"
        )

    existing_judge_model = str(payload.get("judge_model", "")).strip()
    if existing_judge_model and existing_judge_model != args.api_model:
        raise ValueError(f"Cannot resume: existing judge_model={existing_judge_model}, current judge_model={args.api_model}")

    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        return {}

    completed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not is_completed_judge_row(row):
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            completed[sample_id] = row
    return completed


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    report_path = Path(args.report_file)
    prompt_path = Path(args.prompt_file)
    source_payload = load_report(report_path)
    prompt_text = load_prompt_file(prompt_path)
    items, pool_size = select_items(source_payload, args.sample_source, args.max_samples, args.seed)

    output_path = Path(args.output) if args.output else build_default_output_path(report_path, args.api_model)
    if args.resume and not args.output:
        print("⚠️ --resume usually needs --output pointing to an existing judge report; without --output a new timestamped file is used.")
    run_id = datetime.now().strftime("ah_judge_%Y%m%d_%H%M%S")

    print("=" * 72)
    print("Academic Humanize LLM-as-Judge v2")
    print("=" * 72)
    print(f"source_report: {report_path}")
    print(f"source_model:  {source_payload.get('model_id', '')}")
    print(f"judge_model:   {args.api_model}")
    print(f"prompt_file:   {prompt_path}")
    print(f"sample_source: {args.sample_source}")
    print(f"samples:       {len(items)} / {pool_size}")
    print(f"concurrency:   {max(1, int(args.max_concurrency))}")
    print(f"resume:        {args.resume}")
    print(f"output:        {output_path}")
    print("=" * 72)

    max_concurrency = max(1, int(args.max_concurrency))
    indexed_items = list(enumerate(items, start=1))
    judged_rows: List[Optional[Dict[str, Any]]] = [None] * len(indexed_items)
    pending_items: List[Tuple[int, Dict[str, Any]]] = indexed_items

    if args.resume:
        completed_by_sample_id = load_resume_judge_rows(output_path, args, source_payload)
        pending_items = []
        for idx, item in indexed_items:
            sample_id = str(item.get("sample_id", f"row_{idx}")).strip()
            completed_row = completed_by_sample_id.get(sample_id)
            if completed_row is None:
                pending_items.append((idx, item))
                continue
            judged_rows[idx - 1] = completed_row
        reused_count = len(indexed_items) - len(pending_items)
        print(f"resume: reused={reused_count} | pending={len(pending_items)}")

    if max_concurrency == 1:
        client = configure_client(create_client(model=args.api_model), args)
        iterator = tqdm(pending_items, desc="Judging", unit="sample")
        for done_count, (idx, item) in enumerate(iterator, start=1):
            _, row = judge_one_item(idx=idx, item=item, prompt_text=prompt_text, args=args, client=client)
            judged_rows[idx - 1] = row

            if args.save_every > 0 and done_count % args.save_every == 0:
                save_json(
                    output_path,
                    build_report(
                        run_id=run_id,
                        source_payload=source_payload,
                        report_path=report_path,
                        args=args,
                        prompt_path=prompt_path,
                        pool_size=pool_size,
                        judged_rows=completed_rows_in_order(judged_rows),
                        status="running",
                    ),
                )
    else:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(judge_one_item, idx=idx, item=item, prompt_text=prompt_text, args=args): (idx, item)
                for idx, item in pending_items
            }
            iterator = tqdm(as_completed(futures), total=len(futures), desc=f"Judging x{max_concurrency}", unit="sample")
            for done_count, future in enumerate(iterator, start=1):
                idx, item = futures[future]
                try:
                    _, row = future.result()
                except Exception as exc:
                    row = build_judged_row(item, idx, None, f"ERROR: {exc}")
                judged_rows[idx - 1] = row

                if args.save_every > 0 and done_count % args.save_every == 0:
                    save_json(
                        output_path,
                        build_report(
                            run_id=run_id,
                            source_payload=source_payload,
                            report_path=report_path,
                            args=args,
                            prompt_path=prompt_path,
                            pool_size=pool_size,
                            judged_rows=completed_rows_in_order(judged_rows),
                            status="running",
                        ),
                    )

    report = build_report(
        run_id=run_id,
        source_payload=source_payload,
        report_path=report_path,
        args=args,
        prompt_path=prompt_path,
        pool_size=pool_size,
        judged_rows=completed_rows_in_order(judged_rows),
        status="complete",
    )
    save_json(output_path, report)

    print("=" * 72)
    print("Academic Humanize LLM-as-Judge v2 complete")
    print("=" * 72)
    print(f"judged_rows: {report['counts']['judged_rows']}")
    print(f"parsed_rows: {report['counts']['parsed_rows']}")
    print(f"total_mean: {report['summary']['total']['mean']}")
    print(f"total_normalized_mean: {report['summary']['total_normalized']['mean']}")
    print(f"output: {output_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
