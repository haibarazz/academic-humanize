"""
OpenAI-compatible API client for Academic Humanize workflows.

Environment variables are loaded from .env when present.

Retry behavior:
- 自动重试网络错误和速率限制
- 指数退避策略
- 最多重试3次
"""

import os
import time
import threading
from dotenv import load_dotenv

# OpenAI库
try:
    from openai import OpenAI
except ImportError:
    print("⚠️ 未安装openai库，请运行: pip install openai")
    OpenAI = None

# 加载.env文件中的环境变量
load_dotenv()


class TranslationClient:
    """OpenAI-compatible API client. The provider is selected by CUSTOM_API_URL and the model id."""

    _global_retry_lock = threading.Lock()
    _global_retry_until = 0.0

    def __init__(self, model=None):
        """
        初始化客户端

        Args:
            model: 模型名称，如 "gpt-4", "deepseek-chat", "glm-4" 等
        """
        self.api_key = os.getenv("CUSTOM_API_KEY")
        self.base_url = os.getenv("CUSTOM_API_URL", "").strip()
        self.model = model or os.getenv("TRANSLATION_MODEL", "gpt-3.5-turbo")
        self._sdk_base_url = self._normalize_sdk_base_url(self.base_url)
        self.default_headers = self._build_default_headers()

        # Generation parameters
        self.max_tokens = 1000
        self.temperature = 0.3  # 低温度保证一致性

        # 速率限制
        self.requests_per_minute = int(os.getenv("API_RATE_LIMIT", 60))
        self.last_request_time = 0
        self._rate_limit_lock = threading.Lock()

        # Initialize OpenAI-compatible client
        self.client = None
        if OpenAI:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self._sdk_base_url if self._sdk_base_url else None,
                    default_headers=self.default_headers or None,
                )
                print(f"🚀 使用模型: {self.model}")
            except Exception as e:
                print(f"⚠️ OpenAI库初始化失败，使用requests: {e}")
                self.client = None

    @staticmethod
    def _build_default_headers() -> dict:
        """Optional headers for OpenAI-compatible routers such as OpenRouter."""
        headers = {}
        referer = os.getenv("OPENROUTER_SITE_URL", "").strip()
        app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        if app_name:
            headers["X-Title"] = app_name
        return headers

    @staticmethod
    def _normalize_sdk_base_url(base_url: str) -> str:
        """
        规范化给 OpenAI SDK 的 base_url。
        兼容两种输入：
        1) https://xxx/v1
        2) https://xxx/v1/chat/completions
        """
        if not base_url:
            return ""
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            normalized = normalized[: -len("/chat/completions")]
        return normalized

    @staticmethod
    def _resolve_requests_url(base_url: str) -> str:
        """
        规范化 requests 直连地址，兼容：
        1) .../v1
        2) .../v1/chat/completions
        """
        if not base_url:
            return ""
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    def _is_openrouter(self) -> bool:
        """判断当前 OpenAI-compatible endpoint 是否为 OpenRouter。"""
        endpoint = f"{self.base_url} {self._sdk_base_url}".lower()
        return "openrouter.ai" in endpoint

    def _build_extra_body(self) -> dict:
        """
        OpenRouter reasoning 模型默认会输出 thinking/reasoning tokens。
        评测脚本只需要最终答案；默认关闭 reasoning，避免空 content 和额外计费。
        如确实需要 reasoning，可设置 OPENROUTER_DISABLE_REASONING=false。
        """
        if not self._is_openrouter():
            return {}

        disable_reasoning = os.getenv("OPENROUTER_DISABLE_REASONING", "true").strip().lower()
        if disable_reasoning in {"0", "false", "no", "off"}:
            return {}

        return {
            "reasoning": {
                "effort": "none",
                "exclude": True,
            }
        }

    @staticmethod
    def _has_reasoning_payload(message) -> bool:
        """检测响应是否只有 reasoning/thinking 而没有最终 content。"""
        reasoning_keys = ("reasoning_content", "reasoning", "reasoning_details")
        if isinstance(message, dict):
            return any(bool(message.get(key)) for key in reasoning_keys)
        return any(bool(getattr(message, key, None)) for key in reasoning_keys)

    @classmethod
    def _warn_empty_content(cls, source: str, message=None):
        if message is not None and cls._has_reasoning_payload(message):
            print(
                f"⚠️ {source} content为空，但响应包含reasoning/thinking；"
                "已忽略推理内容，请检查模型是否支持关闭reasoning。"
            )
            return
        print(f"⚠️ {source} content为空")

    @staticmethod
    def _extract_content_from_response(response) -> str:
        """
        兼容不同 SDK/网关返回格式，统一抽取文本内容。
        """
        if response is None:
            print("⚠️ API响应为None")
            return ""

        # 某些网关/适配层会直接返回字符串
        if isinstance(response, str):
            content = response.strip()
            if not content:
                print("⚠️ API响应为空字符串")
            return content

        # OpenAI SDK 标准对象
        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            if len(choices) == 0:
                print("⚠️ choices数组为空")
                return ""
            message = getattr(choices[0], "message", None)
            if message is None:
                print("⚠️ message为None")
                return ""
            content = getattr(message, "content", None)
            if content is None:
                self_msg = "SDK响应"
                if TranslationClient._has_reasoning_payload(message):
                    TranslationClient._warn_empty_content(self_msg, message)
                else:
                    print(f"⚠️ content为None，message有: {dir(message)}")
                return ""
            if isinstance(content, str):
                if not content.strip():
                    TranslationClient._warn_empty_content("API返回", message)
                    return ""
                return content.strip()
            if isinstance(content, list):
                # 兼容 content 为分段结构
                texts = []
                for part in content:
                    if isinstance(part, dict):
                        t = part.get("text")
                        if isinstance(t, str):
                            texts.append(t)
                    else:
                        t = getattr(part, "text", None)
                        if isinstance(t, str):
                            texts.append(t)
                if texts:
                    return "".join(texts).strip()
                print("⚠️ content为list但无有效text字段")

        # dict 返回
        if isinstance(response, dict):
            try:
                message = response["choices"][0]["message"]
                content = message.get("content") if isinstance(message, dict) else None
                if content is None:
                    TranslationClient._warn_empty_content("dict响应", message)
                    return ""
                content = str(content).strip()
                if not content:
                    TranslationClient._warn_empty_content("dict响应", message)
                return content
            except Exception as e:
                print(f"⚠️ 无法从dict响应提取content: {e}")

        # Pydantic 对象转 dict 后再尝试
        for method_name in ("model_dump", "dict"):
            method = getattr(response, method_name, None)
            if callable(method):
                try:
                    payload = method()
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    try:
                        message = payload["choices"][0]["message"]
                        content = message.get("content") if isinstance(message, dict) else None
                        if content is None:
                            TranslationClient._warn_empty_content("Pydantic转dict后", message)
                            return ""
                        content = str(content).strip()
                        if not content:
                            TranslationClient._warn_empty_content("Pydantic转dict后", message)
                        return content
                    except Exception as e:
                        print(f"⚠️ Pydantic转dict后仍无法提取content: {e}")

        # 调试：打印无法解析的响应结构
        print(f"⚠️ 无法解析响应格式，响应类型: {type(response)}")
        try:
            print(f"⚠️ 响应内容: {str(response)[:500]}")
        except Exception:
            pass
        # 尝试打印完整响应对象的所有属性
        try:
            print(f"⚠️ 响应属性: {dir(response)}")
        except Exception:
            pass
        return ""

    def _rate_limit(self):
        """线程安全的启动速率限制。"""
        self._wait_for_global_retry_pause()
        with self._rate_limit_lock:
            current_time = time.time()
            min_interval = 60.0 / max(1, self.requests_per_minute)

            if current_time - self.last_request_time < min_interval:
                sleep_time = min_interval - (current_time - self.last_request_time)
                time.sleep(sleep_time)

            self.last_request_time = time.time()

    @classmethod
    def _wait_for_global_retry_pause(cls):
        with cls._global_retry_lock:
            wait_time = cls._global_retry_until - time.time()
        if wait_time > 0:
            print(f"⚠️ 全局限流暂停中，等待 {wait_time:.1f} 秒后继续请求")
            time.sleep(wait_time)

    @classmethod
    def _record_global_retry_pause(cls, wait_seconds: int):
        if wait_seconds <= 0:
            return
        with cls._global_retry_lock:
            cls._global_retry_until = max(cls._global_retry_until, time.time() + wait_seconds)

    def _should_retry(self, error: Exception, status_code: int = None) -> bool:
        """判断是否应该重试"""
        # 空响应也应该重试（可能是临时问题）
        error_msg = str(error).lower()
        if "响应内容为空" in error_msg or "content为空" in error_msg:
            return True

        # OpenAI库异常
        if hasattr(error, 'status_code'):
            status_code = error.status_code

        # 速率限制
        if status_code == 429:
            return True

        # 服务器错误
        if status_code and status_code >= 500:
            return True

        # 网络连接错误
        retryable_errors = [
            "connection",
            "timeout",
            "network",
            "temporary",
            "unavailable",
            "rate limit"
        ]
        return any(err in error_msg for err in retryable_errors)

    @staticmethod
    def _extract_status_code(error: Exception, status_code: int = None):
        if status_code is not None:
            return status_code
        return getattr(error, "status_code", None)

    @staticmethod
    def _retry_wait_seconds(error: Exception, attempt: int, status_code: int = None) -> int:
        status = TranslationClient._extract_status_code(error, status_code)
        if status == 429:
            response = getattr(error, "response", None)
            headers = getattr(response, "headers", {}) or {}
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after:
                try:
                    return max(1, int(float(retry_after)))
                except Exception:
                    pass
            try:
                return max(1, int(float(os.getenv("API_429_SLEEP_SECONDS", "60"))))
            except Exception:
                return 60
        return (2 ** attempt) * 2

    def _call_api(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        """通用API调用方法，带重试机制"""
        self._rate_limit()

        for attempt in range(max_retries):
            # 优先使用OpenAI库
            if self.client:
                try:
                    request_kwargs = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                    }
                    extra_body = self._build_extra_body()
                    if extra_body:
                        request_kwargs["extra_body"] = extra_body

                    response = self.client.chat.completions.create(**request_kwargs)
                    content = self._extract_content_from_response(response)
                    if content:
                        return content
                    # 空响应也算一种失败，触发重试
                    raise ValueError("响应内容为空")

                except Exception as e:
                    if not self._should_retry(e):
                        print(f"❌ 不可重试的错误: {e}")
                        return ""

                    if attempt < max_retries - 1:
                        wait_time = self._retry_wait_seconds(e, attempt)
                        retry_reason = "命中429限流" if self._extract_status_code(e) == 429 else "请求失败"
                        if self._extract_status_code(e) == 429:
                            self._record_global_retry_pause(wait_time)
                        print(f"⚠️ {retry_reason}，{wait_time}秒后重试 ({attempt + 1}/{max_retries}): {e}")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ 达到最大重试次数: {e}")
                    continue

            # 备选方案：使用requests
            try:
                import requests

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                headers.update(self.default_headers)

                data = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
                data.update(self._build_extra_body())

                request_url = self._resolve_requests_url(self.base_url)
                if not request_url:
                    print("❌ CUSTOM_API_URL 未配置，无法使用 requests 兜底请求")
                    return ""

                response = requests.post(
                    request_url,
                    headers=headers,
                    json=data,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    content = self._extract_content_from_response(result)
                    if content:
                        return content
                    # 空响应也算失败，触发重试
                    raise ValueError("响应内容为空")

                # 检查是否需要重试
                if self._should_retry(Exception(), response.status_code):
                    if attempt < max_retries - 1:
                        wait_time = self._retry_wait_seconds(Exception(), attempt, response.status_code)
                        retry_reason = "命中429限流" if response.status_code == 429 else f"API错误 {response.status_code}"
                        if response.status_code == 429:
                            self._record_global_retry_pause(wait_time)
                        print(f"⚠️ {retry_reason}，{wait_time}秒后重试 ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                else:
                    print(f"❌ API错误: {response.status_code} - {response.text}")
                    return ""

            except requests.exceptions.RequestException as e:
                if not self._should_retry(e):
                    print(f"❌ 不可重试的网络错误: {e}")
                    return ""

                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    print(f"⚠️ 网络错误，{wait_time}秒后重试 ({attempt + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)

        print(f"❌ 所有重试都失败了")
        return ""

    def call(self, system_prompt: str, user_prompt: str) -> str:
        """
        基础API调用方法

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词

        Returns:
            API响应内容
        """
        return self._call_api(system_prompt, user_prompt)


def create_client(model=None):
    """
    Create an API client.

    Args:
        model: 模型名称，如 "gpt-4", "deepseek-chat", "glm-4" 等

    Returns:
        TranslationClient instance
    """
    if not os.getenv("CUSTOM_API_KEY"):
        raise ValueError(
            "未找到API密钥！\n"
            "请在.env文件中设置 CUSTOM_API_KEY 环境变量\n"
            "示例：CUSTOM_API_KEY=your_key_here"
        )

    return TranslationClient(model=model)


if __name__ == "__main__":
    # API client smoke test
    print("API client smoke test...")
    client = create_client("deepseek/deepseek-chat")
    # 测试基础调用
    system_prompt = "You are a professional academic writing assistant."
    user_prompt = "Rewrite this sentence in natural academic English: This work endeavors to propose a novel method."
    result = client.call(system_prompt, user_prompt)

    print(f"\nResponse: {result}")
    print("\nSmoke test complete.")
