"""
SiliconFlow 批量推理工具（OpenAI SDK 兼容接口）。

用途：
1) 写入批量请求 JSONL
2) 上传文件并创建 batch
3) 轮询 batch 状态
4) 下载 output/error 文件
5) 解析输出并按 custom_id 回收结果
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from urllib.request import Request, urlopen
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


load_dotenv()


TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}


def _to_dict(obj: Any) -> Dict[str, Any]:
    """兼容 SDK 对象与 dict。"""
    if isinstance(obj, dict):
        return obj
    for method_name in ("model_dump", "dict"):
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                payload = method()
            except Exception:
                payload = None
            if isinstance(payload, dict):
                return payload
    return {}


def _extract_content_from_choice_message(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
        if texts:
            return "".join(texts).strip()
    return ""


def extract_chat_content_from_batch_item(item: Dict[str, Any]) -> str:
    """
    从 batch 单行结果中提取文本。
    兼容常见返回结构：
    - {"response": {"body": {"choices": [...]}}}
    - {"response": {"choices": [...]}}
    - {"choices": [...]}
    """
    if not isinstance(item, dict):
        return ""

    candidates: List[Any] = []
    response = item.get("response")
    if isinstance(response, dict):
        body = response.get("body")
        if isinstance(body, dict):
            candidates.append(body)
        candidates.append(response)
    candidates.append(item)

    for payload in candidates:
        if not isinstance(payload, dict):
            continue
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                text = _extract_content_from_choice_message(message.get("content"))
                if text:
                    return text
        # 少数网关可能直接在 choice 上给 text 字段
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return ""


@dataclass
class BatchRunResult:
    """一次 batch job 的回收结果。"""

    responses: Dict[str, Dict[str, Any]]
    errors: Dict[str, Dict[str, Any]]
    pending_custom_ids: List[str]
    manifest_path: Path
    work_dir: Path


class SiliconFlowBatchClient:
    """SiliconFlow 批量推理客户端。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if OpenAI is None:
            raise ImportError("未安装 openai，请先安装依赖。")

        resolved_api_key = (api_key or os.getenv("CUSTOM_API_KEY", "")).strip()
        if not resolved_api_key:
            raise ValueError("缺少 CUSTOM_API_KEY，无法调用批量推理 API。")

        raw_base_url = (base_url or os.getenv("CUSTOM_API_URL", "")).strip()
        normalized_base_url = self._normalize_batch_base_url(raw_base_url)
        if not normalized_base_url:
            normalized_base_url = os.getenv("SILICONFLOW_BATCH_BASE_URL", "https://api.siliconflow.cn/v1").strip()

        self.api_key = resolved_api_key
        self.base_url = normalized_base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _normalize_batch_base_url(base_url: str) -> str:
        """
        把 CUSTOM_API_URL 归一化为 SDK 可用的 /v1 基地址。
        兼容：
        - https://xxx/v1
        - https://xxx/v1/chat/completions
        """
        if not base_url:
            return ""
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            normalized = normalized[: -len("/chat/completions")]
        return normalized

    @staticmethod
    def build_request_line(
        custom_id: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 1.0,
        url: str = "/v1/chat/completions",
    ) -> Dict[str, Any]:
        """构建单条 batch 请求行。"""
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": url,
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
            },
        }

    @staticmethod
    def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
        """写入 JSONL，返回行数。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        return count

    @staticmethod
    def read_jsonl(path: Path) -> List[Dict[str, Any]]:
        """读取 JSONL（坏行跳过）。"""
        rows: List[Dict[str, Any]] = []
        if not path.exists():
            return rows

        bad_lines = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)

        if bad_lines > 0:
            print(f"⚠️ 读取 {path} 时跳过坏行: {bad_lines}")
        return rows

    @staticmethod
    def _chunk_rows(rows: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
        if not rows:
            return []
        safe_size = max(1, int(chunk_size))
        shard_count = math.ceil(len(rows) / safe_size)
        shards: List[List[Dict[str, Any]]] = []
        for idx in range(shard_count):
            start = idx * safe_size
            end = min(len(rows), (idx + 1) * safe_size)
            shards.append(rows[start:end])
        return shards

    def upload_batch_file(self, request_file: Path) -> str:
        """上传批量输入文件，返回 file_id。"""
        with request_file.open("rb") as f:
            obj = self.client.files.create(file=f, purpose="batch")
        payload = _to_dict(obj)
        file_id = payload.get("id") or payload.get("data", {}).get("id")
        if not file_id:
            raise RuntimeError(f"上传文件成功但未返回 file_id: {request_file}")
        return str(file_id)

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        replace_model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """创建 batch 任务。"""
        kwargs: Dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }
        if metadata:
            kwargs["metadata"] = metadata
        if replace_model:
            kwargs["extra_body"] = {"replace": {"model": replace_model}}

        obj = self.client.batches.create(**kwargs)
        payload = _to_dict(obj)
        if not payload:
            raise RuntimeError("创建 batch 失败：返回为空")
        return payload

    def retrieve_batch(self, batch_id: str) -> Dict[str, Any]:
        """查询 batch 状态。"""
        obj = self.client.batches.retrieve(batch_id)
        payload = _to_dict(obj)
        if not payload:
            raise RuntimeError(f"查询 batch 失败: {batch_id}")
        return payload

    def download_file(self, file_id: str, output_path: Path) -> Path:
        """
        下载结果文件到本地。
        使用 with_raw_response 兼容二进制内容获取。
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 兼容 SiliconFlow 返回直链 URL 的场景
        if str(file_id).startswith("http://") or str(file_id).startswith("https://"):
            req = Request(
                url=str(file_id),
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "*/*",
                },
            )
            last_error: Optional[Exception] = None
            for attempt in range(3):
                try:
                    with urlopen(req, timeout=60) as resp:
                        content = resp.read()
                    if not content:
                        raise RuntimeError(f"下载文件失败或内容为空: {file_id}")
                    output_path.write_bytes(content)
                    return output_path
                except Exception as exc:
                    last_error = exc
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))
            raise RuntimeError(f"下载 URL 失败: {file_id}; error={last_error}") from last_error

        content = b""
        try:
            raw = self.client.files.with_raw_response.content(file_id)
            if hasattr(raw, "content"):
                content = raw.content
        except Exception:
            content = b""

        # 回退：兼容部分 SDK 版本返回 HttpxBinaryResponseContent
        if not content:
            try:
                raw_content = self.client.files.content(file_id)
                if isinstance(raw_content, bytes):
                    content = raw_content
                elif hasattr(raw_content, "read"):
                    content = raw_content.read()
                elif hasattr(raw_content, "content"):
                    content = raw_content.content
            except Exception:
                content = b""

        if not content:
            raise RuntimeError(f"下载文件失败或内容为空: {file_id}")
        output_path.write_bytes(content)
        return output_path

    @staticmethod
    def _save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _load_manifest(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None

    def run_batch_job(
        self,
        request_rows: List[Dict[str, Any]],
        job_name: str,
        work_dir: Path,
        completion_window: str = "24h",
        endpoint: str = "/v1/chat/completions",
        replace_model: Optional[str] = None,
        max_rows_per_file: int = 5000,
        poll_interval_sec: int = 30,
        max_wait_minutes: int = 24 * 60,
        overwrite: bool = False,
        submit_only: bool = False,
        collect_only: bool = False,
    ) -> BatchRunResult:
        """
        运行批量任务（支持断点续跑）。

        注意：
        - submit_only=True：只提交，不回收
        - collect_only=True：只回收，不新建任务（要求已有 manifest）
        """
        if submit_only and collect_only:
            raise ValueError("submit_only 与 collect_only 不能同时为 True")

        job_root = Path(work_dir) / job_name
        manifest_path = job_root / "manifest.json"
        requests_dir = job_root / "requests"
        outputs_dir = job_root / "outputs"
        errors_dir = job_root / "errors"

        if overwrite and job_root.exists():
            shutil.rmtree(job_root)

        manifest = self._load_manifest(manifest_path)
        if manifest is None:
            if collect_only:
                raise FileNotFoundError(f"collect_only 模式下未找到 manifest: {manifest_path}")
            if not request_rows:
                raise ValueError("request_rows 为空，无法创建 batch 任务。")

            request_shards = self._chunk_rows(request_rows, max_rows_per_file)
            shards_meta: List[Dict[str, Any]] = []
            requests_dir.mkdir(parents=True, exist_ok=True)
            for idx, rows in enumerate(request_shards):
                shard_name = f"shard_{idx:04d}"
                request_file = requests_dir / f"{shard_name}.jsonl"
                self.write_jsonl(request_file, rows)
                shards_meta.append(
                    {
                        "shard_name": shard_name,
                        "request_file": str(request_file),
                        "request_count": len(rows),
                        "input_file_id": None,
                        "batch_id": None,
                        "status": "prepared",
                        "output_file_id": None,
                        "error_file_id": None,
                        "output_path": None,
                        "error_path": None,
                    }
                )

            manifest = {
                "job_name": job_name,
                "created_at": datetime.now().isoformat(),
                "completion_window": completion_window,
                "endpoint": endpoint,
                "replace_model": replace_model,
                "total_requests": len(request_rows),
                "shards": shards_meta,
            }
            self._save_manifest(manifest_path, manifest)

        # 1) 提交未提交的 shard
        if not collect_only:
            for shard in manifest.get("shards", []):
                if shard.get("batch_id"):
                    continue
                request_file = Path(shard["request_file"])
                input_file_id = self.upload_batch_file(request_file)
                batch_info = self.create_batch(
                    input_file_id=input_file_id,
                    endpoint=endpoint,
                    completion_window=completion_window,
                    replace_model=replace_model,
                    metadata={"job_name": job_name, "shard_name": shard["shard_name"]},
                )
                shard["input_file_id"] = input_file_id
                shard["batch_id"] = batch_info.get("id")
                shard["status"] = batch_info.get("status", "submitted")
                shard["submitted_at"] = datetime.now().isoformat()
                self._save_manifest(manifest_path, manifest)

        if submit_only:
            return BatchRunResult(
                responses={},
                errors={},
                pending_custom_ids=[],
                manifest_path=manifest_path,
                work_dir=job_root,
            )

        # 2) 回收：轮询直到全部终态或超时
        deadline = time.time() + max(1, int(max_wait_minutes)) * 60
        while True:
            non_terminal = 0
            for shard in manifest.get("shards", []):
                batch_id = shard.get("batch_id")
                if not batch_id:
                    continue
                status = str(shard.get("status") or "")
                if status in TERMINAL_STATES and shard.get("output_path") is not None:
                    continue

                info = self.retrieve_batch(batch_id)
                status = str(info.get("status", "unknown"))
                shard["status"] = status
                shard["output_file_id"] = info.get("output_file_id")
                shard["error_file_id"] = info.get("error_file_id")

                if status in TERMINAL_STATES:
                    output_file_id = shard.get("output_file_id")
                    error_file_id = shard.get("error_file_id")
                    if output_file_id and not shard.get("output_path"):
                        output_path = outputs_dir / f"{shard['shard_name']}.output.jsonl"
                        self.download_file(str(output_file_id), output_path)
                        shard["output_path"] = str(output_path)
                    if error_file_id and not shard.get("error_path"):
                        error_path = errors_dir / f"{shard['shard_name']}.error.jsonl"
                        self.download_file(str(error_file_id), error_path)
                        shard["error_path"] = str(error_path)
                else:
                    non_terminal += 1

            self._save_manifest(manifest_path, manifest)

            if non_terminal == 0:
                break
            if time.time() >= deadline:
                break
            time.sleep(max(3, int(poll_interval_sec)))

        # 3) 汇总 output/error
        responses: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, Dict[str, Any]] = {}
        requested_custom_ids: set[str] = set()
        for shard in manifest.get("shards", []):
            request_file = Path(shard["request_file"])
            for row in self.read_jsonl(request_file):
                cid = str(row.get("custom_id", "")).strip()
                if cid:
                    requested_custom_ids.add(cid)

            output_path = shard.get("output_path")
            if output_path:
                for item in self.read_jsonl(Path(output_path)):
                    cid = str(item.get("custom_id", "")).strip()
                    if not cid:
                        continue
                    text = extract_chat_content_from_batch_item(item)
                    responses[cid] = {
                        "custom_id": cid,
                        "text": text,
                        "raw": item,
                        "batch_status": shard.get("status"),
                    }

            error_path = shard.get("error_path")
            if error_path:
                for item in self.read_jsonl(Path(error_path)):
                    cid = str(item.get("custom_id", "")).strip()
                    if not cid:
                        continue
                    errors[cid] = item

        pending_custom_ids = sorted(list(requested_custom_ids - set(responses.keys()) - set(errors.keys())))
        return BatchRunResult(
            responses=responses,
            errors=errors,
            pending_custom_ids=pending_custom_ids,
            manifest_path=manifest_path,
            work_dir=job_root,
        )
