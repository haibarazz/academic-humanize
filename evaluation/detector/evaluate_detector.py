"""
Generate a Academic Humanize detector sidecar report from an existing ah_model_eval_v1 report.

当前 backend 先实现 Fast-DetectGPT API 版。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import build_default_output_path


load_dotenv()

DEFAULT_BACKEND = "fast_detect_gpt"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_DETECTOR = "fast-detect(llama3-8b/llama3-8b-instruct)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Academic Humanize detector sidecar report.")
    parser.add_argument("--report-file", type=str, required=True, help="Path to a ah_model_eval_v1 report.")
    parser.add_argument("--output", type=str, default=None, help="Optional output path.")
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND, choices=[DEFAULT_BACKEND], help="Detector backend.")
    parser.add_argument("--api-url", type=str, default=None, help="Full detector API endpoint URL.")
    parser.add_argument("--api-key", type=str, default=None, help="Detector API key.")
    parser.add_argument("--detector", type=str, default=None, help="FastDetect detector name.")
    parser.add_argument("--text-field", type=str, default=DEFAULT_TEXT_FIELD, help="JSON field name used to send text.")
    parser.add_argument(
        "--response-probability-path",
        type=str,
        default=None,
        help="Optional dot-path to probability field in response JSON.",
    )
    parser.add_argument(
        "--response-criterion-path",
        type=str,
        default=None,
        help="Optional dot-path to criterion field in response JSON.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--include-raw", action="store_true", help="Include raw detector response in rows.")
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(sum(values) / len(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def resolve_api_url(cli_value: Optional[str]) -> str:
    value = (
        cli_value
        or os.getenv("FASTDETECT_API_URL")
        or os.getenv("FAST_DETECT_GPT_API_URL")
        or os.getenv("CUSTOM_API_URL")
    )
    if not value:
        raise ValueError("Missing detector API URL. Set --api-url or FASTDETECT_API_URL.")
    return value


def resolve_api_key(cli_value: Optional[str]) -> str:
    value = (
        cli_value
        or os.getenv("FASTDETECT_API_KEY")
        or os.getenv("FAST_DETECT_GPT_API_KEY")
        or os.getenv("CUSTOM_API_KEY")
    )
    if not value:
        raise ValueError("Missing detector API key. Set --api-key or FASTDETECT_API_KEY.")
    return value


def resolve_detector(cli_value: Optional[str]) -> str:
    value = (
        cli_value
        or os.getenv("FASTDETECT_DETECTOR")
        or os.getenv("FAST_DETECT_GPT_DETECTOR")
        or DEFAULT_DETECTOR
    )
    return str(value).strip()


def load_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Report file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Invalid report format: expected a JSON object.")
    if payload.get("report_type") != "ah_model_eval_v1":
        raise ValueError("Only ah_model_eval_v1 is supported.")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("The report is missing `rows`.")
    return payload


def get_nested_value(payload: Any, path: str | None) -> Any:
    if not path:
        return None
    current = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return None
    return current


def normalize_probability(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = safe_float(value, default=-1.0)
    if number < 0:
        return None
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    return max(0.0, min(1.0, float(number)))


def extract_probability(payload: Dict[str, Any], explicit_path: str | None) -> Optional[float]:
    candidate_paths: List[str] = []
    if explicit_path:
        candidate_paths.append(explicit_path)
    candidate_paths.extend(
        [
            "probability",
            "ai_probability",
            "machine_generated_probability",
            "machine_generated_prob",
            "machine_prob",
            "score",
            "data.probability",
            "data.ai_probability",
            "data.machine_generated_probability",
            "data.machine_generated_prob",
            "data.score",
            "data.prob",
            "result.probability",
            "result.ai_probability",
            "result.machine_generated_probability",
            "result.machine_generated_prob",
            "result.score",
        ]
    )

    for path in candidate_paths:
        value = get_nested_value(payload, path)
        normalized = normalize_probability(value)
        if normalized is not None:
            return normalized
    return None


def extract_criterion(payload: Dict[str, Any], explicit_path: str | None) -> str:
    candidate_paths: List[str] = []
    if explicit_path:
        candidate_paths.append(explicit_path)
    candidate_paths.extend(
        [
            "criterion",
            "data.criterion",
            "result.criterion",
            "message",
            "data.message",
            "result.message",
            "data.details.crit",
            "details.crit",
        ]
    )
    for path in candidate_paths:
        value = get_nested_value(payload, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None:
            return str(value)
    return ""


def post_json(url: str, api_key: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {raw}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc

    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Detector response is not valid JSON: {raw[:300]}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError("Detector response must be a JSON object.")
    if "code" in obj and safe_float(obj.get("code"), default=0.0) != 0.0:
        raise RuntimeError(f"Detector returned code={obj.get('code')}, msg={obj.get('msg')}")
    return obj


def call_fast_detect_gpt(
    api_url: str,
    api_key: str,
    detector: str,
    text: str,
    text_field: str,
    timeout: int,
    probability_path: str | None,
    criterion_path: str | None,
) -> Dict[str, Any]:
    response = post_json(
        api_url,
        api_key,
        {
            "detector": detector,
            text_field: text,
        },
        timeout=timeout,
    )
    probability = extract_probability(response, probability_path)
    if probability is None:
        raise RuntimeError("Unable to locate probability field in detector response.")
    return {
        "machine_generated_prob": probability,
        "human_likeness": 1.0 - probability,
        "criterion": extract_criterion(response, criterion_path),
        "raw_response": response,
    }


def build_output_path(report_file: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return build_default_output_path("ah_detector", report_file.stem)


def main() -> int:
    args = parse_args()
    report_path = Path(args.report_file)
    payload = load_report(report_path)
    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    api_url = resolve_api_url(args.api_url)
    api_key = resolve_api_key(args.api_key)
    detector = resolve_detector(args.detector)
    output_path = build_output_path(report_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detector_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    machine_probs: List[float] = []
    human_scores: List[float] = []

    for idx, row in enumerate(rows, start=1):
        prediction = str(row.get("prediction", "")).strip()
        record: Dict[str, Any] = {
            "sample_id": str(row.get("sample_id", f"row_{idx}")),
            "paper_id": str(row.get("paper_id", "unknown")),
            "status": "ok",
            "prediction_length_chars": len(prediction),
        }
        try:
            result = call_fast_detect_gpt(
                api_url=api_url,
                api_key=api_key,
                detector=detector,
                text=prediction,
                text_field=args.text_field,
                timeout=args.timeout,
                probability_path=args.response_probability_path,
                criterion_path=args.response_criterion_path,
            )
            record["machine_generated_prob"] = float(result["machine_generated_prob"])
            record["human_likeness"] = float(result["human_likeness"])
            record["criterion"] = str(result.get("criterion", "")).strip()
            if args.include_raw:
                record["raw_response"] = result.get("raw_response", {})

            machine_probs.append(record["machine_generated_prob"])
            human_scores.append(record["human_likeness"])
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            errors.append(
                {
                    "sample_id": record["sample_id"],
                    "paper_id": record["paper_id"],
                    "error": str(exc),
                }
            )
        detector_rows.append(record)

        if idx % 20 == 0 or idx == len(rows):
            print(f"progress: {idx}/{len(rows)}")

    success_count = sum(1 for row in detector_rows if row.get("status") == "ok")
    report = {
        "report_type": "ah_detector_report_v1",
        "run_id": datetime.now().strftime("ah_detector_%Y%m%d_%H%M%S"),
        "backend": args.backend,
        "source_report": str(report_path),
        "source_report_type": str(payload.get("report_type", "")).strip(),
        "source_run_id": str(payload.get("run_id", "")).strip(),
        "source_model_id": str(payload.get("model_id", "")).strip(),
        "settings": {
            "api_url": api_url,
            "detector": detector,
            "text_field": args.text_field,
            "timeout": args.timeout,
            "response_probability_path": args.response_probability_path,
            "response_criterion_path": args.response_criterion_path,
            "max_rows": args.max_rows,
            "include_raw": args.include_raw,
        },
        "counts": {
            "input_rows": len(rows),
            "detected_rows": success_count,
            "failed_rows": len(rows) - success_count,
        },
        "summary": {
            "detector_coverage_rate": float(success_count / len(rows)) if rows else 0.0,
            "machine_generated_prob": summarize(machine_probs),
            "human_likeness": summarize(human_scores),
        },
        "rows": detector_rows,
        "errors": errors,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Academic Humanize detector sidecar finished")
    print("=" * 72)
    print(f"source_report: {report_path}")
    print(f"backend: {args.backend}")
    print(f"detector: {detector}")
    print(f"input_rows: {len(rows)} | detected_rows: {success_count} | failed_rows: {len(rows) - success_count}")
    print(f"output: {output_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
