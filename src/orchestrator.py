"""
AI Orchestrator - Multi-Model Router
=====================================
Intelligently routes queries to the best AI model based on task type.
Features:
- Automatic task classification and model selection
- Secure credential management (no hardcoded API keys)
- Rate limiting and retry logic with exponential backoff
- Input sanitization and validation
- Comprehensive error handling
- Async support for performance
- Audit logging for security

Security Features:
- No API keys in code
- Input validation against injection attacks
- Secure error messages (no credential leakage)
- Rate limiting to prevent abuse
- Audit trail for all API calls
"""

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from google.auth import credentials as auth_credentials
    from google.oauth2 import service_account

# Import our secure credential manager
from .credentials import CONFIG_DIR, get_api_key

# Configure logging with security in mind (no sensitive data in logs)
logger = logging.getLogger(__name__)
LOCAL_PROVIDERS = {
    "ollama",
    "mlx",
}


def format_http_error(exc: httpx.HTTPStatusError) -> str:
    """Format detailed error message from HTTP exception."""
    response = exc.response
    status_code = response.status_code
    message = response.reason_phrase or str(exc)
    retry_after = response.headers.get("Retry-After")

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        error_info = payload.get("error")
        if isinstance(error_info, dict):
            error_message = error_info.get("message")
            if error_message:
                message = error_message
            details = error_info.get("details", [])
            if isinstance(details, list):
                for detail in details:
                    if (
                        isinstance(detail, dict)
                        and detail.get("@type")
                        == "type.googleapis.com/google.rpc.RetryInfo"
                    ):
                        retry_delay = detail.get("retryDelay")
                        if retry_delay:
                            message = f"{message} Suggested retry after {retry_delay}."
                        break

    if retry_after:
        message = f"{message} Retry-After: {retry_after}."

    return f"HTTP {status_code}: {message}"


class TaskType(Enum):
    """Enumeration of task types the orchestrator can handle"""

    GENERAL_NLP = auto()
    LONG_CONTEXT = auto()
    CODE_GENERATION = auto()
    DATA_ANALYSIS = auto()
    REASONING = auto()
    DEEP_REASONING = auto()
    WEB_SEARCH = auto()
    CREATIVE_WRITING = auto()
    TOOL_USE = auto()
    LOCAL_MODEL = auto()
    MULTIMODAL = auto()
    MATH = auto()
    SUMMARIZATION = auto()


@dataclass(frozen=True)
class ModelCapability:
    """Immutable model capability definition"""

    name: str
    provider: str
    model_id: str
    task_types: tuple[TaskType, ...]
    context_window: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    strengths: tuple[str, ...]
    max_output_tokens: int = 4096
    supports_streaming: bool = True
    supports_functions: bool = False
    supports_vision: bool = False


@dataclass(frozen=True)
class ProviderCharacteristics:
    """
    Captures the nuanced strengths and weaknesses of AI providers.
    Used for intelligent model selection based on prompt characteristics.

    Based on comparative analysis of model capabilities:
    - strengths: Areas where this provider excels
    - weaknesses: Known limitations to consider
    - best_for: Specific use cases where this provider shines
    - avoid_for: Use cases where other providers are preferable
    """

    provider: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    best_for: tuple[str, ...]
    avoid_for: tuple[str, ...]
    # Numeric scores (0.0 to 1.0) for fine-grained selection
    contextual_understanding: float = 0.5
    creativity_originality: float = 0.5
    emotional_intelligence: float = 0.5
    speed_efficiency: float = 0.5
    knowledge_breadth: float = 0.5
    reasoning_depth: float = 0.5
    code_quality: float = 0.5
    objectivity: float = 0.5


@dataclass
class RateLimitState:
    """Track rate limiting state per provider"""

    requests: deque[float] = field(default_factory=deque)
    tokens: deque[tuple[float, int]] = field(default_factory=deque)
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 100000


@dataclass
class APIResponse:
    """Standardized API response container"""

    content: str
    model: str
    provider: str
    usage: dict[str, int]
    latency_ms: float
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InputValidator:
    """Security-focused input validation"""

    # Patterns that might indicate injection attempts
    SUSPICIOUS_PATTERNS = [
        r"<\s*script\b",  # Script tags
        r"javascript\s*:",  # JS protocol
        r"\{\{.*\}\}",  # Template injection
        r"\$\{.*\}",  # Template literals
        r"__proto__",  # Prototype pollution
        r"eval\s*\(",  # Eval calls
        r"exec\s*\(",  # Exec calls
    ]

    MAX_PROMPT_LENGTH = 500000  # 500k chars max
    MAX_MESSAGES = 100

    @classmethod
    def validate_prompt(cls, prompt: str) -> tuple[bool, str]:
        """Validate user prompt for security issues"""
        if not prompt:
            return False, "Invalid prompt: must be a non-empty string"

        if len(prompt) > cls.MAX_PROMPT_LENGTH:
            return False, f"Prompt exceeds maximum length of {cls.MAX_PROMPT_LENGTH}"

        # Check for suspicious patterns (log but don't block - could be legitimate code)
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                logger.warning(f"Potentially suspicious pattern in prompt: {pattern}")

        return True, ""

    @classmethod
    def validate_messages(cls, messages: list[dict[str, Any]]) -> tuple[bool, str]:
        """Validate message array"""
        if not messages:
            return False, "Invalid messages: must be a non-empty list"

        if len(messages) > cls.MAX_MESSAGES:
            return False, f"Too many messages: max {cls.MAX_MESSAGES}"

        for i, msg in enumerate(messages):
            if "role" not in msg or "content" not in msg:
                return False, f"Message {i} missing required fields"

            is_valid, error = cls.validate_prompt(msg["content"])
            if not is_valid:
                return False, f"Message {i}: {error}"

        return True, ""

    @classmethod
    def sanitize_for_logging(cls, text: str, max_len: int = 100) -> str:
        """Sanitize text for safe logging (no sensitive data)"""
        if not text:
            return ""
        # Truncate and remove potential sensitive patterns
        sanitized = text[:max_len]
        # Redact anything that looks like an API key
        sanitized = re.sub(
            r"(sk-|api[_-]?key|bearer\s+)[a-zA-Z0-9\-_]{20,}",
            "[REDACTED]",
            sanitized,
            flags=re.IGNORECASE,
        )
        return sanitized + ("..." if len(text) > max_len else "")


class RateLimiter:
    """Token bucket rate limiter with per-provider limits"""

    def __init__(self) -> None:
        self._states: dict[str, RateLimitState] = {}
        self._lock = asyncio.Lock()

    def _get_state(self, provider: str) -> RateLimitState:
        if provider not in self._states:
            self._states[provider] = RateLimitState()
        return self._states[provider]

    async def check_and_wait(self, provider: str, estimated_tokens: int = 1000) -> bool:
        """Check rate limits and wait if necessary"""
        async with self._lock:
            state = self._get_state(provider)
            now = time.time()
            window_start = now - 60

            # Clean old entries
            while state.requests and state.requests[0] < window_start:
                state.requests.popleft()
            while state.tokens and state.tokens[0][0] < window_start:
                state.tokens.popleft()

            # Check request limit
            if len(state.requests) >= state.max_requests_per_minute:
                wait_time = state.requests[0] - window_start
                logger.info(f"Rate limited on {provider}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                return await self.check_and_wait(provider, estimated_tokens)

            # Check token limit
            current_tokens = sum(t[1] for t in state.tokens)
            if current_tokens + estimated_tokens > state.max_tokens_per_minute:
                wait_time = state.tokens[0][0] - window_start if state.tokens else 1
                logger.info(
                    f"Token rate limited on {provider}, waiting {wait_time:.1f}s"
                )
                await asyncio.sleep(wait_time)
                return await self.check_and_wait(provider, estimated_tokens)

            # Record this request
            state.requests.append(now)
            state.tokens.append((now, estimated_tokens))
            return True


class RetryHandler:
    """Exponential backoff retry handler"""

    RETRYABLE_ERRORS = [
        "429",
        "rate_limit",
        "rate limit",
        "timeout",
        "connection",
        "overloaded",
        "resource exhausted",
        "too many requests",
        "quota",
        "503",
        "502",
        "500",
    ]

    @classmethod
    def is_retryable(cls, error: str) -> bool:
        """Check if error is retryable"""
        error_lower = error.lower()
        return any(e in error_lower for e in cls.RETRYABLE_ERRORS)

    @classmethod
    async def execute_with_retry(
        cls,
        func: Callable[[], Coroutine[Any, Any, APIResponse]],
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> APIResponse:
        """Execute function with exponential backoff retry"""
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                last_error = e
                error_str = str(e)

                if attempt == max_retries or not cls.is_retryable(error_str):
                    raise

                delay = min(base_delay * (2**attempt), max_delay)
                # Add jitter
                delay *= 0.5 + 0.5 * (hash(str(time.time())) % 100) / 100

                logger.warning(
                    f"Retrying after error (attempt {attempt + 1}): "
                    f"{InputValidator.sanitize_for_logging(error_str)}"
                )
                await asyncio.sleep(delay)

        if last_error:
            raise last_error

        # This part should ideally not be reached if func always returns or raises.
        # Added for type safety to guarantee a return value.
        raise RuntimeError("Retry logic finished without returning or raising.")


class BaseProvider(ABC):
    """Abstract base class for AI providers"""

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self._client: Any | None = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider client"""
        pass

    @abstractmethod
    async def complete(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> APIResponse:
        """Send completion request"""
        pass


class OpenAIProvider(BaseProvider):
    """OpenAI API provider"""

    @property
    def provider_name(self) -> str:
        return "openai"

    async def initialize(self) -> bool:
        api_key = get_api_key("openai")
        if not api_key:
            logger.error("OpenAI API key not configured")
            return False

        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=api_key)
            return True
        except ImportError:
            logger.error("openai package not installed")
            return False

    async def complete(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> APIResponse:
        if not self._client:
            await self.initialize()
        assert self._client is not None

        start_time = time.time()

        try:
            await self.rate_limiter.check_and_wait(
                self.provider_name,
                kwargs.get("max_tokens", 1000),
            )

            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.7),
            )

            latency = (time.time() - start_time) * 1000

            return APIResponse(
                content=response.choices[0].message.content,
                model=model,
                provider=self.provider_name,
                usage={
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                },
                latency_ms=latency,
                success=True,
            )
        except Exception as e:
            return APIResponse(
                content="",
                model=model,
                provider=self.provider_name,
                usage={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider"""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def initialize(self) -> bool:
        api_key = get_api_key("anthropic")
        if not api_key:
            logger.error("Anthropic API key not configured")
            return False

        try:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=api_key)
            return True
        except ImportError:
            logger.error("anthropic package not installed")
            return False

    async def complete(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> APIResponse:
        if not self._client:
            await self.initialize()
        assert self._client is not None

        start_time = time.time()

        try:
            await self.rate_limiter.check_and_wait(
                self.provider_name,
                kwargs.get("max_tokens", 1000),
            )

            # Extract system message if present
            system = ""
            chat_messages: list[dict[str, Any]] = []
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                else:
                    chat_messages.append(msg)

            response = await self._client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 4096),
                system=system if system else None,
                messages=chat_messages,
            )

            latency = (time.time() - start_time) * 1000

            content = ""
            if response.content:
                content = response.content[0].text

            return APIResponse(
                content=content,
                model=model,
                provider=self.provider_name,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                latency_ms=latency,
                success=True,
            )
        except Exception as e:
            return APIResponse(
                content="",
                model=model,
                provider=self.provider_name,
                usage={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )


class GoogleProvider(BaseProvider):
    """Google Gemini API provider using the new google-genai SDK"""

    @property
    def provider_name(self) -> str:
        return "google"

    async def initialize(self) -> bool:
        api_key = get_api_key("google")
        if not api_key:
            logger.error("Google API key not configured")
            return False

        try:
            from google import genai

            self._client = genai.Client(api_key=api_key)
            return True
        except ImportError:
            logger.error("google-genai package not installed")
            return False

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> APIResponse:
        if not self._client:
            await self.initialize()
        assert self._client is not None

        start_time = time.time()

        try:
            await self.rate_limiter.check_and_wait(self.provider_name, 1000)

            # Convert messages to Gemini format
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    contents.append(
                        {
                            "role": "user",
                            "parts": [{"text": f"System: {msg['content']}"}],
                        }
                    )
                else:
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append({"role": role, "parts": [{"text": msg["content"]}]})

            # Use native async API
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=contents,
            )

            latency = (time.time() - start_time) * 1000

            # Extract usage metadata
            usage: dict[str, Any] = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                }

            return APIResponse(
                content=response.text,
                model=model,
                provider=self.provider_name,
                usage=usage,
                latency_ms=latency,
                success=True,
            )
        except Exception as e:
            return APIResponse(
                content="",
                model=model,
                provider=self.provider_name,
                usage={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )


class VertexAIProvider(BaseProvider):
    """
    Google Vertex AI provider for enterprise deployments.

    Uses service account authentication for third-party platform integration.
    Provides stable REST endpoints that external platforms can call.
    """

    def __init__(
        self,
        rate_limiter: RateLimiter,
        project_id: str | None = None,
        location: str = "global",
    ):
        super().__init__(rate_limiter)
        self.project_id = project_id
        self.location = location
        self._credentials: (
            auth_credentials.Credentials | service_account.Credentials | None
        ) = None

    @property
    def provider_name(self) -> str:
        return "vertex-ai"

    async def initialize(self) -> bool:
        """Initialize Vertex AI with service account credentials"""
        try:
            import os

            import google.auth

            # Get credentials from environment or default
            credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

            if credentials_path and os.path.exists(credentials_path):
                try:
                    creds, project = google.auth.load_credentials_from_file(
                        credentials_path,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                    self._credentials = creds
                    if not self.project_id and project:
                        self.project_id = project
                    logger.info(f"Loaded Vertex AI credentials from {credentials_path}")
                except Exception as e:
                    logger.warning(
                        "Failed to load credentials from %s; falling back to ADC: %s",
                        credentials_path,
                        e,
                    )
                    creds, project = google.auth.default(
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                    self._credentials = creds
                    if not self.project_id:
                        self.project_id = project
                    logger.info("Using application default credentials for Vertex AI")
            else:
                # Try application default credentials
                creds, project = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self._credentials = creds
                if not self.project_id:
                    self.project_id = project
                logger.info("Using application default credentials for Vertex AI")

            if not self.project_id:
                self.project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get(
                    "GOOGLE_CLOUD_PROJECT"
                )

            if not self.project_id:
                logger.error(
                    "GCP project ID not configured. Set GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT"
                )
                return False

            # Initialize HTTP client
            import httpx

            self._client = httpx.AsyncClient(timeout=120.0)

            logger.info(
                f"Vertex AI initialized for project: {self.project_id}, location: {self.location}"
            )
            return True

        except ImportError as e:
            logger.error(f"Missing dependencies for Vertex AI: {e}")
            logger.error(
                "Install with: pip install google-cloud-aiplatform google-auth"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            return False

    async def _get_access_token(self) -> str:
        """Get OAuth2 access token for API calls"""
        from google.auth.transport.requests import Request

        assert self._credentials is not None
        if not self._credentials.valid:
            await asyncio.to_thread(self._credentials.refresh, Request())

        return self._credentials.token  # type: ignore

    def get_endpoint_url(self, model: str) -> str:
        """
        Get the REST endpoint URL for a model.
        This URL can be shared with third-party platforms.
        """
        base_host = (
            "aiplatform.googleapis.com"
            if self.location == "global"
            else f"{self.location}-aiplatform.googleapis.com"
        )
        return (
            f"https://{base_host}/v1/"
            f"projects/{self.project_id}/locations/{self.location}/"
            f"publishers/google/models/{model}:generateContent"
        )

    async def complete(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> APIResponse:
        if not self._client:
            await self.initialize()
        assert self._client is not None

        start_time = time.time()

        try:
            await self.rate_limiter.check_and_wait(
                self.provider_name,
                kwargs.get("max_tokens", 1000),
            )

            # Get access token
            access_token = await self._get_access_token()

            # Convert messages to Vertex AI format
            contents: list[dict[str, Any]] = []
            for msg in messages:
                if msg["role"] == "system":
                    # Vertex AI doesn't have system role, prepend to first user message
                    continue
                contents.append(
                    {
                        "role": "user" if msg["role"] == "user" else "model",
                        "parts": [{"text": msg["content"]}],
                    }
                )

            # Build request
            url = self.get_endpoint_url(model)

            request_body = {
                "contents": contents,
                "generationConfig": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "maxOutputTokens": kwargs.get("max_tokens", 4096),
                },
            }

            # Make request
            response = await self._client.post(
                url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            latency = (time.time() - start_time) * 1000

            # Extract content
            content = ""
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    content = candidate["content"]["parts"][0].get("text", "")

            # Extract usage
            usage_metadata = data.get("usageMetadata", {})

            return APIResponse(
                content=content,
                model=model,
                provider=self.provider_name,
                usage={
                    "input_tokens": usage_metadata.get("promptTokenCount", 0),
                    "output_tokens": usage_metadata.get("candidatesTokenCount", 0),
                },
                latency_ms=latency,
                success=True,
                metadata={"endpoint_url": url},
            )

        except httpx.HTTPStatusError as e:
            return APIResponse(
                content="",
                model=model,
                provider=self.provider_name,
                usage={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=format_http_error(e),
                metadata={"endpoint_url": self.get_endpoint_url(model)},
            )
        except httpx.TimeoutException as e:
            return APIResponse(
                content="",
                model=model,
                provider=self.provider_name,
                usage={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=f"Timeout error: {e}",
                metadata={"endpoint_url": self.get_endpoint_url(model)},
            )
        except httpx.RequestError as e:
            return APIResponse(
                content="",
                model=model,
                provider=self.provider_name,
                usage={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=f"Connection error: {e}",
                metadata={"endpoint_url": self.get_endpoint_url(model)},
            )
        except Exception as e:
            return APIResponse(
                content="",
                model=model,
                provider=self.provider_name,
                usage={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
                metadata={"endpoint_url": self.get_endpoint_url(model)},
            )


class OllamaProvider(BaseProvider):
    """Ollama local model provider"""

    def __init__(
        self, rate_limiter: RateLimiter, base_url: str = "http://localhost:11434"
    ):
        super().__init__(rate_limiter)
        self.base_url = base_url

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def initialize(self) -> bool:
        try:
            import httpx

            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)
            # Test connection
            response = await self._client.get("/api/version")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False

    async def complete(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> APIResponse:
        if not self._client:
            await self.initialize()
        assert self._client is not None

        start_time = time.time()

        try:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

            latency = (time.time() - start_time) * 1000

            return APIResponse(
                content=data.get("message", {}).get("content", ""),
                model=model,
                provider=self.provider_name,
                usage={
                    "input_tokens": data.get("prompt_eval_count", 0),
                    "output_tokens": data.get("eval_count", 0),
                },
                latency_ms=latency,
                success=True,
            )
        except Exception as e:
            return APIResponse(
                content="",
                model=model,
                provider=self.provider_name,
                usage={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )


class MistralProvider(BaseProvider):
    """Mistral AI API provider"""

    @property
    def provider_name(self) -> str:
        return "mistral"

    async def initialize(self) -> bool:
        api_key = get_api_key("mistral")
        if not api_key:
            logger.error("Mistral API key not configured")
            return False

        try:
            import httpx

            self._client = httpx.AsyncClient(
                base_url="https://api.mistral.ai",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Mistral client: {e}")
            return False

    async def complete(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> APIResponse:
        if not self._client:
            await self.initialize()
        assert self._client is not None

        start_time = time.time()

        try:
            await self.rate_limiter.check_and_wait(
                self.provider_name,
                kwargs.get("max_tokens", 1000),
            )

            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0.7),
                },
            )
            response.raise_for_status()
            data = response.json()

            latency = (time.time() - start_time) * 1000

            return APIResponse(
                content=data["choices"][0]["message"]["content"],
                model=model,
                provider=self.provider_name,
                usage={
                    "input_tokens": data["usage"]["prompt_tokens"],
                    "output_tokens": data["usage"]["completion_tokens"],
                },
                latency_ms=latency,
                success=True,
            )
        except Exception as e:
            return APIResponse(
                content="",
                model=model,
                provider=self.provider_name,
                usage={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )


class GroqProvider(BaseProvider):
    """Groq API provider for fast inference"""

    @property
    def provider_name(self) -> str:
        return "groq"

    async def initialize(self) -> bool:
        api_key = get_api_key("groq")
        if not api_key:
            logger.error("Groq API key not configured")
            return False

        try:
            import httpx

            self._client = httpx.AsyncClient(
                base_url="https://api.groq.com/openai",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            return False

    async def complete(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> APIResponse:
        if not self._client:
            await self.initialize()
        assert self._client is not None

        start_time = time.time()

        try:
            await self.rate_limiter.check_and_wait(
                self.provider_name,
                kwargs.get("max_tokens", 1000),
            )

            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0.7),
                },
            )
            response.raise_for_status()
            data = response.json()

            latency = (time.time() - start_time) * 1000

            return APIResponse(
                content=data["choices"][0]["message"]["content"],
                model=model,
                provider=self.provider_name,
                usage={
                    "input_tokens": data["usage"]["prompt_tokens"],
                    "output_tokens": data["usage"]["completion_tokens"],
                },
                latency_ms=latency,
                success=True,
            )
        except Exception as e:
            return APIResponse(
                content="",
                model=model,
                provider=self.provider_name,
                usage={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )


class XAIProvider(BaseProvider):
    """xAI (Grok) API provider"""

    @property
    def provider_name(self) -> str:
        return "xai"

    async def initialize(self) -> bool:
        api_key = get_api_key("xai")
        if not api_key:
            logger.error("xAI API key not configured")
            return False

        try:
            import httpx

            self._client = httpx.AsyncClient(
                base_url="https://api.x.ai",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize xAI client: {e}")
            return False

    async def complete(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> APIResponse:
        if not self._client:
            await self.initialize()
        assert self._client is not None

        start_time = time.time()

        try:
            await self.rate_limiter.check_and_wait(
                self.provider_name,
                kwargs.get("max_tokens", 1000),
            )

            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0.7),
                },
            )
            response.raise_for_status()
            data = response.json()

            latency = (time.time() - start_time) * 1000

            return APIResponse(
                content=data["choices"][0]["message"]["content"],
                model=model,
                provider=self.provider_name,
                usage={
                    "input_tokens": data["usage"]["prompt_tokens"],
                    "output_tokens": data["usage"]["completion_tokens"],
                },
                latency_ms=latency,
                success=True,
            )
        except Exception as e:
            return APIResponse(
                content="",
                model=model,
                provider=self.provider_name,
                usage={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )


class PerplexityProvider(BaseProvider):
    """Perplexity AI API provider"""

    @property
    def provider_name(self) -> str:
        return "perplexity"

    async def initialize(self) -> bool:
        api_key = get_api_key("perplexity")
        if not api_key:
            logger.error("Perplexity API key not configured")
            return False

        try:
            import httpx

            self._client = httpx.AsyncClient(
                base_url="https://api.perplexity.ai",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Perplexity client: {e}")
            return False

    async def complete(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> APIResponse:
        if not self._client:
            await self.initialize()
        assert self._client is not None

        start_time = time.time()

        try:
            await self.rate_limiter.check_and_wait(
                self.provider_name,
                kwargs.get("max_tokens", 1000),
            )

            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0.7),
                },
            )
            response.raise_for_status()
            data = response.json()

            latency = (time.time() - start_time) * 1000

            return APIResponse(
                content=data["choices"][0]["message"]["content"],
                model=model,
                provider=self.provider_name,
                usage={
                    "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
                },
                latency_ms=latency,
                success=True,
            )
        except Exception as e:
            return APIResponse(
                content="",
                model=model,
                provider=self.provider_name,
                usage={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )


class DeepSeekProvider(BaseProvider):
    """DeepSeek API provider"""

    @property
    def provider_name(self) -> str:
        return "deepseek"

    async def initialize(self) -> bool:
        api_key = get_api_key("deepseek")
        if not api_key:
            logger.error("DeepSeek API key not configured")
            return False

        try:
            import httpx

            self._client = httpx.AsyncClient(
                base_url="https://api.deepseek.com",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {e}")
            return False

    async def complete(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> APIResponse:
        if not self._client:
            await self.initialize()
        assert self._client is not None

        start_time = time.time()

        try:
            await self.rate_limiter.check_and_wait(
                self.provider_name,
                kwargs.get("max_tokens", 1000),
            )

            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0.7),
                },
            )
            response.raise_for_status()
            data = response.json()

            latency = (time.time() - start_time) * 1000

            return APIResponse(
                content=data["choices"][0]["message"]["content"],
                model=model,
                provider=self.provider_name,
                usage={
                    "input_tokens": data["usage"]["prompt_tokens"],
                    "output_tokens": data["usage"]["completion_tokens"],
                },
                latency_ms=latency,
                success=True,
            )
        except Exception as e:
            return APIResponse(
                content="",
                model=model,
                provider=self.provider_name,
                usage={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )


class MLXProvider(BaseProvider):
    """
    MLX local model provider for Apple Silicon.

    Uses the official mlx-lm Python package for optimized inference
    on Mac devices with Apple Silicon (M1/M2/M3/M4).
    """

    def __init__(
        self,
        rate_limiter: RateLimiter,
        model_path: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    ):
        super().__init__(rate_limiter)
        self.model_path = model_path
        self._available = False
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def provider_name(self) -> str:
        return "mlx"

    async def initialize(self) -> bool:
        """Initialize MLX model and tokenizer"""
        try:
            import os

            import mlx.core as mx
            from huggingface_hub import snapshot_download, try_to_load_from_cache
            from mlx_lm import load
            from pathlib import Path

            def _find_local_snapshot_dir(repo_id: str) -> str | None:
                """Search common HF cache roots for a snapshot dir with safetensors."""
                owner_repo = repo_id.replace("/", "--")
                model_folder = f"models--{owner_repo}"

                candidates: list[Path] = []
                # Respect HF_HOME if set
                hf_home = os.environ.get("HF_HOME")
                if hf_home:
                    candidates.append(Path(hf_home) / "hub")

                # macOS default
                candidates.append(
                    Path.home() / "Library" / "Caches" / "huggingface" / "hub"
                )
                # Unix default
                candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

                for hub_root in candidates:
                    model_root = hub_root / model_folder / "snapshots"
                    if not model_root.exists():
                        continue
                    # Check newest snapshots first
                    snapshots = sorted(
                        [p for p in model_root.iterdir() if p.is_dir()],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    for snap in snapshots:
                        shards = list(snap.glob("*.safetensors"))
                        if shards:
                            return str(snap)
                return None

            # Smart Cache Detection (robust)
            # 1) Try to locate an existing local snapshot with weights across all known caches
            local_snapshot = _find_local_snapshot_dir(self.model_path)
            if local_snapshot:
                logger.info(f"Using local snapshot: {local_snapshot}")
                os.environ["HF_HUB_OFFLINE"] = "1"
                self.model_path = local_snapshot
            else:
                # 2) Try a strict local-only snapshot resolution via huggingface_hub
                try:
                    snap_dir = snapshot_download(
                        repo_id=self.model_path,
                        local_files_only=True,
                        allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
                    )
                    shards = list(Path(snap_dir).glob("*.safetensors"))
                    if shards:
                        logger.info(
                            f"Model found locally via snapshot_download: {snap_dir}"
                        )
                        os.environ["HF_HUB_OFFLINE"] = "1"
                        self.model_path = snap_dir
                    else:
                        logger.info(
                            "Local snapshot found but no weights; ONLINE mode enabled to fetch shards."
                        )
                        os.environ["HF_HUB_OFFLINE"] = "0"
                        # Keep model_path as repo id so mlx-lm can download
                except Exception:
                    # 3) Fallback to try_to_load_from_cache for config + check shards presence in that snapshot
                    try:
                        cached_config = try_to_load_from_cache(
                            repo_id=self.model_path, filename="config.json"
                        )
                    except Exception:
                        cached_config = None
                    if cached_config:
                        cache_dir = Path(cached_config).parent
                        shards = list(cache_dir.glob("*.safetensors"))
                        if shards:
                            logger.info(f"Model found in cache: {cached_config}")
                            os.environ["HF_HUB_OFFLINE"] = "1"
                            self.model_path = str(cache_dir)
                        else:
                            logger.info(
                                "Model index found but weights are missing; enabling ONLINE mode to fetch shards."
                            )
                            os.environ["HF_HUB_OFFLINE"] = "0"
                    else:
                        logger.info(
                            f"Model {self.model_path} not found locally; ONLINE mode enabled."
                        )
                        os.environ["HF_HUB_OFFLINE"] = "0"

            # Check if we have a GPU available (Metal)
            if not mx.metal.is_available():
                logger.warning("Apple Metal (GPU) not available for MLX")
                # We can still run on CPU, but it will be slower

            logger.info(f"Loading MLX model: {self.model_path}")

            # Run load in a thread to avoid blocking the event loop (GUI freeze fix)
            def _load_model() -> Any:
                return load(self.model_path)

            result = await asyncio.to_thread(_load_model)

            if isinstance(result, tuple) and len(result) >= 2:
                self._model = result[0]
                self._tokenizer = result[1]
            else:
                # Fallback if signature matches expectation exactly
                self._model, self._tokenizer = result

            self._available = True
            return True

        except ImportError as e:
            logger.warning(f"MLX dependencies not installed: {e}")
            logger.warning("Install with: pip install mlx-lm huggingface-hub")
            self._available = False
            return False
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            self._available = False
            return False

    async def complete(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> APIResponse:
        if not self._available or self._model is None or self._tokenizer is None:
            success = await self.initialize()
            if not success or self._model is None or self._tokenizer is None:
                return APIResponse(
                    content="",
                    model=model,
                    provider=self.provider_name,
                    usage={},
                    latency_ms=0,
                    success=False,
                    error="MLX provider not available or failed to initialize",
                )

        start_time = time.time()

        try:
            from mlx_lm import generate

            # Convert messages to prompt using tokenizer's template if available
            if hasattr(self._tokenizer, "apply_chat_template"):
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = self._format_messages(messages)

            # Run generation in a thread to avoid blocking the event loop
            response_text = await asyncio.to_thread(
                generate,
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", 1024),
                verbose=False,
            )

            latency = (time.time() - start_time) * 1000

            return APIResponse(
                content=response_text,
                model=model,
                provider=self.provider_name,
                usage={
                    "input_tokens": len(prompt.split()),
                    "output_tokens": len(response_text.split()),
                },
                latency_ms=latency,
                success=True,
            )
        except Exception as e:
            return APIResponse(
                content="",
                model=model,
                provider=self.provider_name,
                usage={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )

    def _format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Convert chat messages to a single prompt string (fallback)"""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        return "\n\n".join(parts)


# Provider characteristics registry for intelligent model selection
# Based on comparative analysis of model capabilities and behaviors
PROVIDER_CHARACTERISTICS: dict[str, ProviderCharacteristics] = {
    "openai": ProviderCharacteristics(
        provider="openai",
        strengths=(
            "abstract reasoning leader (52.9% ARC-AGI-2)",
            "perfect math (100% AIME 2025)",
            "fast for lightweight queries",
            "function calling",
            "multimodal support",
        ),
        weaknesses=(
            "can be less nuanced in creative tasks",
            "sometimes verbose",
        ),
        best_for=(
            "abstract reasoning",
            "mathematical problems",
            "API integration",
            "quick responses",
        ),
        avoid_for=(
            "highly creative writing",
            "privacy-sensitive tasks",
        ),
        contextual_understanding=0.85,
        creativity_originality=0.7,
        emotional_intelligence=0.6,
        speed_efficiency=0.9,
        knowledge_breadth=0.9,
        reasoning_depth=0.92,  # GPT-5.2 leads ARC-AGI-2
        code_quality=0.88,
        objectivity=0.8,
    ),
    "anthropic": ProviderCharacteristics(
        provider="anthropic",
        strengths=(
            "coding benchmark leader (80.9% SWE-bench)",
            "30+ hour autonomous coding sessions",
            "strongest prompt injection resistance (5%)",
            "nuanced creative writing",
            "excellent instruction following",
        ),
        weaknesses=(
            "can be overly cautious",
            "may refuse edge cases",
        ),
        best_for=(
            "code generation and debugging",
            "complex reasoning",
            "creative writing",
            "long-form content",
            "security-sensitive applications",
        ),
        avoid_for=("simple quick tasks where speed matters most",),
        contextual_understanding=0.95,
        creativity_originality=0.85,
        emotional_intelligence=0.7,
        speed_efficiency=0.75,
        knowledge_breadth=0.85,
        reasoning_depth=0.93,
        code_quality=0.98,  # 80.9% SWE-bench leader
        objectivity=0.85,
    ),
    "google": ProviderCharacteristics(
        provider="google",
        strengths=(
            "top LMArena score (1501 Elo - first to break 1500)",
            "massive 2M context window",
            "multimodal excellence",
            "91.9% GPQA Diamond",
            "cost-effective",
        ),
        weaknesses=(
            "high hallucination rate (88% - gives wrong answers confidently)",
            "slower on complex tasks due to internal planning",
        ),
        best_for=(
            "long document analysis",
            "multimodal tasks",
            "research requiring broad coverage",
            "math and science problems",
        ),
        avoid_for=(
            "tasks requiring factual precision",
            "applications where wrong answers are costly",
        ),
        contextual_understanding=0.85,
        creativity_originality=0.9,
        emotional_intelligence=0.65,
        speed_efficiency=0.85,
        knowledge_breadth=0.95,
        reasoning_depth=0.92,  # Top LMArena
        code_quality=0.82,
        objectivity=0.65,  # Lowered due to 88% hallucination rate
    ),
    "vertex-ai": ProviderCharacteristics(
        provider="vertex-ai",
        strengths=(
            "enterprise-grade stability",
            "OAuth2 authentication",
            "third-party integration",
            "same models as Google AI",
        ),
        weaknesses=(
            "requires GCP setup",
            "more complex authentication",
        ),
        best_for=(
            "enterprise deployments",
            "GCP integration",
            "production workloads",
        ),
        avoid_for=(
            "quick prototyping",
            "personal use",
        ),
        contextual_understanding=0.85,
        creativity_originality=0.85,
        emotional_intelligence=0.65,
        speed_efficiency=0.85,
        knowledge_breadth=0.9,
        reasoning_depth=0.85,
        code_quality=0.8,
        objectivity=0.8,
    ),
    "ollama": ProviderCharacteristics(
        provider="ollama",
        strengths=(
            "completely free",
            "fully private",
            "works offline",
            "no data sharing",
        ),
        weaknesses=(
            "limited to local hardware",
            "smaller models than cloud",
            "slower on less powerful hardware",
        ),
        best_for=(
            "privacy-sensitive tasks",
            "offline work",
            "cost-conscious usage",
            "experimentation",
        ),
        avoid_for=(
            "complex reasoning requiring large models",
            "tasks needing real-time information",
        ),
        contextual_understanding=0.6,
        creativity_originality=0.5,
        emotional_intelligence=0.4,
        speed_efficiency=0.7,  # Depends on hardware
        knowledge_breadth=0.5,
        reasoning_depth=0.5,
        code_quality=0.6,
        objectivity=0.7,
    ),
    "mlx": ProviderCharacteristics(
        provider="mlx",
        strengths=(
            "optimized for Apple Silicon",
            "fast local inference",
            "completely private",
            "works offline",
            "no API costs",
            "good speed and efficiency",
        ),
        weaknesses=(
            "limited common sense compared to large cloud models",
            "less emotional intelligence",
            "contextual understanding can be limited",
            "smaller knowledge base than cloud models",
        ),
        best_for=(
            "privacy-sensitive tasks",
            "offline work",
            "quick local queries",
            "Mac-native workflows",
            "cost-free inference",
        ),
        avoid_for=(
            "complex multi-step reasoning",
            "tasks requiring empathy or emotional nuance",
            "real-time web information",
            "very large context processing",
        ),
        contextual_understanding=0.55,
        creativity_originality=0.5,
        emotional_intelligence=0.35,
        speed_efficiency=0.85,  # Optimized for Apple Silicon
        knowledge_breadth=0.6,
        reasoning_depth=0.5,
        code_quality=0.6,
        objectivity=0.8,
    ),
    "mistral": ProviderCharacteristics(
        provider="mistral",
        strengths=(
            "excellent coding",
            "strong reasoning",
            "multilingual support",
            "efficient architecture",
        ),
        weaknesses=(
            "smaller model lineup",
            "less multimodal support",
        ),
        best_for=(
            "code generation",
            "multilingual tasks",
            "European language content",
        ),
        avoid_for=(
            "multimodal tasks",
            "image analysis",
        ),
        contextual_understanding=0.8,
        creativity_originality=0.75,
        emotional_intelligence=0.55,
        speed_efficiency=0.85,
        knowledge_breadth=0.75,
        reasoning_depth=0.8,
        code_quality=0.9,
        objectivity=0.8,
    ),
    "groq": ProviderCharacteristics(
        provider="groq",
        strengths=(
            "ultra-fast inference",
            "very cost-effective",
            "good for high-throughput",
        ),
        weaknesses=(
            "limited model selection",
            "runs other providers' models",
        ),
        best_for=(
            "speed-critical applications",
            "high-volume processing",
            "real-time responses",
        ),
        avoid_for=("tasks needing unique model capabilities",),
        contextual_understanding=0.75,
        creativity_originality=0.7,
        emotional_intelligence=0.5,
        speed_efficiency=0.98,
        knowledge_breadth=0.75,
        reasoning_depth=0.7,
        code_quality=0.75,
        objectivity=0.75,
    ),
    "xai": ProviderCharacteristics(
        provider="xai",
        strengths=(
            "real-time X/Twitter integration",
            "creative and rebellious personality",
            "fast reasoning",
            "trending topic analysis",
        ),
        weaknesses=(
            "limited versatility beyond X ecosystem",
            "UI inconsistencies",
            "past moderation issues",
            "humor can interfere with clarity",
        ),
        best_for=(
            "current events and trending topics",
            "X conversation analysis",
            "informal/quippy responses",
        ),
        avoid_for=(
            "serious research",
            "professional writing",
            "brand-safe enterprise applications",
        ),
        contextual_understanding=0.75,
        creativity_originality=0.85,
        emotional_intelligence=0.7,
        speed_efficiency=0.8,
        knowledge_breadth=0.8,
        reasoning_depth=0.75,
        code_quality=0.7,
        objectivity=0.65,  # Lowered due to past issues and X-centric design
    ),
    "perplexity": ProviderCharacteristics(
        provider="perplexity",
        strengths=(
            "top Search Arena (1136 Elo)",
            "citation-first design",
            "8x faster than previous versions",
            "128K context window",
            "real-time web grounding",
        ),
        weaknesses=(
            "citation accuracy varies (contextually inaccurate)",
            "weak on multi-step reasoning",
            "accuracy varies by domain",
            "source-dependent quality",
        ),
        best_for=(
            "research queries with citations",
            "news and trending topics",
            "fact-checking (with verification)",
            "time-sensitive information",
        ),
        avoid_for=(
            "creative writing",
            "code generation",
            "complex multi-step reasoning",
            "niche academic topics",
        ),
        contextual_understanding=0.75,
        creativity_originality=0.5,
        emotional_intelligence=0.4,
        speed_efficiency=0.9,  # 8x faster
        knowledge_breadth=0.95,  # With web search
        reasoning_depth=0.6,  # Weak on multi-step reasoning
        code_quality=0.4,
        objectivity=0.75,  # Citation accuracy issues
    ),
    "deepseek": ProviderCharacteristics(
        provider="deepseek",
        strengths=(
            "10-30x cheaper than competitors",
            "beats GPT-4o level on many benchmarks",
            "strong coding and reasoning",
            "frontier-class at budget pricing",
        ),
        weaknesses=(
            "less established platform",
            "smaller context than some competitors (64K)",
        ),
        best_for=(
            "budget-conscious tasks",
            "coding at scale",
            "reasoning problems",
            "cost-optimized production",
        ),
        avoid_for=(
            "tasks requiring very large context",
            "applications needing established vendor support",
        ),
        contextual_understanding=0.8,
        creativity_originality=0.7,
        emotional_intelligence=0.75,
        speed_efficiency=0.85,
        knowledge_breadth=0.75,
        reasoning_depth=0.88,  # Strong reasoning for price
        code_quality=0.88,  # Beats GPT-4o level
        objectivity=0.8,
    ),
}


class TaskClassifier:
    """Classify user prompts into task types"""

    # Keywords for each task type
    TASK_PATTERNS = {
        TaskType.CODE_GENERATION: [
            r"\b(code|program|script|function|class|implement|develop|debug|refactor)\b",
            r"\b(python|javascript|typescript|java|c\+\+|rust|go|ruby)\b",
            r"\b(api|endpoint|database|sql|query|algorithm)\b",
        ],
        TaskType.DATA_ANALYSIS: [
            r"\b(analyze|analysis|data|statistics|chart|graph|visualiz|csv|excel)\b",
            r"\b(trend|pattern|correlation|regression|forecast|predict)\b",
            r"\b(pandas|numpy|matplotlib|dataset)\b",
        ],
        TaskType.DEEP_REASONING: [
            r"\b(prove|theorem|derive|mathematical|logic|reasoning|why|how)\b",
            r"\b(philosophy|ethics|moral|complex|multi-step|deduce)\b",
            r"\b(research|investigate|explore|comprehensive)\b",
        ],
        TaskType.CREATIVE_WRITING: [
            r"\b(write|story|poem|essay|creative|fiction|narrative)\b",
            r"\b(blog|article|content|copywriting|marketing)\b",
        ],
        TaskType.SUMMARIZATION: [
            r"\b(summarize|summary|tldr|brief|condense|main points)\b",
            r"\b(key takeaways|overview|digest)\b",
        ],
        TaskType.LONG_CONTEXT: [
            r"\b(document|book|paper|long|entire|whole|full text)\b",
            r"\b(pdf|report|transcript|lengthy)\b",
        ],
        TaskType.MATH: [
            r"\b(calculate|compute|solve|equation|integral|derivative)\b",
            r"\b(math|algebra|calculus|geometry|trigonometry)\b",
        ],
        TaskType.WEB_SEARCH: [
            r"\b(search|find|look up|current|latest|news|today)\b",
            r"\b(weather|stock|price|update)\b",
        ],
        TaskType.MULTIMODAL: [
            r"\b(image|picture|photo|visual|diagram|screenshot)\b",
            r"\b(describe|analyze|what.+see)\b",
        ],
        TaskType.LOCAL_MODEL: [
            r"\b(private|confidential|offline|local|sensitive)\b",
            r"\b(no cloud|internal|proprietary)\b",
        ],
    }

    @classmethod
    def classify(cls, prompt: str) -> list[tuple[TaskType, float]]:
        """
        Classify prompt into task types with confidence scores.
        Returns list of (TaskType, confidence) tuples, sorted by confidence.
        """
        scores: dict[TaskType, float] = {}
        prompt_lower = prompt.lower()

        for task_type, patterns in cls.TASK_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                matches = re.findall(pattern, prompt_lower)
                score += len(matches) * 0.3

            if score > 0:
                scores[task_type] = min(score, 1.0)

        # Default to general NLP if no patterns matched
        if not scores:
            scores[TaskType.GENERAL_NLP] = 0.5

        return sorted(scores.items(), key=lambda x: -x[1])


class ModelRegistry:
    """Registry of available AI models and their capabilities"""

    # Current models as of November 2025
    MODELS: dict[str, ModelCapability] = {
        # OpenAI Models
        "gpt-4o": ModelCapability(
            name="GPT-4o",
            provider="openai",
            model_id="gpt-4o",
            task_types=(
                TaskType.GENERAL_NLP,
                TaskType.CODE_GENERATION,
                TaskType.REASONING,
                TaskType.MULTIMODAL,
            ),
            context_window=128000,
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            strengths=("speed", "multimodal", "general purpose"),
            supports_functions=True,
            supports_vision=True,
        ),
        "gpt-4o-mini": ModelCapability(
            name="GPT-4o Mini",
            provider="openai",
            model_id="gpt-4o-mini",
            task_types=(TaskType.GENERAL_NLP, TaskType.SUMMARIZATION),
            context_window=128000,
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            strengths=("cost-effective", "fast", "good for simple tasks"),
            supports_functions=True,
        ),
        # GPT-5 Preview / Placeholder (mapped to o1-preview or latest available)
        "gpt-5-preview": ModelCapability(
            name="GPT-5 (Preview)",
            provider="openai",
            model_id="o1",  # Mapping to o1 as closest to "next-gen" currently available
            task_types=(
                TaskType.DEEP_REASONING,
                TaskType.GENERAL_NLP,
                TaskType.CODE_GENERATION,
            ),
            context_window=200000,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.06,
            strengths=("next-gen reasoning", "complex tasks", "preview capability"),
        ),
        "gpt-4.5-preview": ModelCapability(
            name="GPT-4.5 (Preview)",
            provider="openai",
            model_id="gpt-4o",  # Mapping to 4o as placeholder
            task_types=(TaskType.GENERAL_NLP, TaskType.CODE_GENERATION),
            context_window=128000,
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            strengths=("advanced capability", "preview"),
        ),
        "o1": ModelCapability(
            name="o1",
            provider="openai",
            model_id="o1",
            task_types=(
                TaskType.DEEP_REASONING,
                TaskType.MATH,
                TaskType.CODE_GENERATION,
            ),
            context_window=200000,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.06,
            strengths=("deep reasoning", "math", "complex problems"),
            max_output_tokens=100000,
        ),
        "o1-mini": ModelCapability(
            name="o1-mini",
            provider="openai",
            model_id="o1-mini",
            task_types=(TaskType.REASONING, TaskType.CODE_GENERATION),
            context_window=128000,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.012,
            strengths=("reasoning", "coding", "cost-effective"),
        ),
        # Anthropic Models
        "claude-opus-4.5": ModelCapability(
            name="Claude Opus 4.5",
            provider="anthropic",
            model_id="claude-opus-4-5-20251101",
            task_types=(
                TaskType.CODE_GENERATION,
                TaskType.DEEP_REASONING,
                TaskType.CREATIVE_WRITING,
                TaskType.LONG_CONTEXT,
            ),
            context_window=200000,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            strengths=(
                "most intelligent",
                "coding",
                "nuanced writing",
                "complex tasks",
            ),
            max_output_tokens=32000,
        ),
        "claude-sonnet-4.5": ModelCapability(
            name="Claude Sonnet 4.5",
            provider="anthropic",
            model_id="claude-sonnet-4-5-20250929",
            task_types=(
                TaskType.CODE_GENERATION,
                TaskType.GENERAL_NLP,
                TaskType.REASONING,
            ),
            context_window=200000,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            strengths=("balanced", "coding", "fast"),
            max_output_tokens=16000,
        ),
        "claude-haiku-4.5": ModelCapability(
            name="Claude Haiku 4.5",
            provider="anthropic",
            model_id="claude-haiku-4-5-20251001",
            task_types=(TaskType.GENERAL_NLP, TaskType.SUMMARIZATION),
            context_window=200000,
            cost_per_1k_input=0.0008,
            cost_per_1k_output=0.004,
            strengths=("very fast", "cost-effective", "good for simple tasks"),
            max_output_tokens=8000,
        ),
        # Google Models
        "gemini-3-pro": ModelCapability(
            name="Gemini 3.0 Pro (Preview)",
            provider="google",
            model_id="gemini-3-pro-preview",
            task_types=(
                TaskType.GENERAL_NLP,
                TaskType.CODE_GENERATION,
                TaskType.REASONING,
                TaskType.LONG_CONTEXT,
                TaskType.MULTIMODAL,
            ),
            context_window=2000000,
            cost_per_1k_input=0.0015,
            cost_per_1k_output=0.006,
            strengths=(
                "next-gen intelligence",
                "coding",
                "multimodal",
                "massive context",
            ),
            supports_vision=True,
            supports_functions=True,
        ),
        "gemini-3-flash": ModelCapability(
            name="Gemini 3.0 Flash (Preview)",
            provider="google",
            model_id="gemini-3-flash-preview",
            task_types=(
                TaskType.GENERAL_NLP,
                TaskType.CODE_GENERATION,
                TaskType.LONG_CONTEXT,
                TaskType.MULTIMODAL,
            ),
            context_window=1000000,
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0004,
            strengths=("next-gen speed", "multimodal", "cost-effective"),
            supports_vision=True,
        ),
        "gemini-2.5-pro": ModelCapability(
            name="Gemini 2.5 Pro",
            provider="google",
            model_id="gemini-2.5-pro",
            task_types=(
                TaskType.GENERAL_NLP,
                TaskType.CODE_GENERATION,
                TaskType.REASONING,
                TaskType.LONG_CONTEXT,
                TaskType.MULTIMODAL,
            ),
            context_window=2000000,
            cost_per_1k_input=0.00125,
            cost_per_1k_output=0.005,
            strengths=("advanced reasoning", "coding", "multimodal", "2M context"),
            supports_vision=True,
            supports_functions=True,
        ),
        "gemini-2.5-flash": ModelCapability(
            name="Gemini 2.5 Flash",
            provider="google",
            model_id="gemini-2.5-flash",
            task_types=(
                TaskType.GENERAL_NLP,
                TaskType.CODE_GENERATION,
                TaskType.LONG_CONTEXT,
                TaskType.MULTIMODAL,
            ),
            context_window=1000000,
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0004,
            strengths=("very fast", "multimodal", "cost-effective"),
            supports_vision=True,
        ),
        "gemini-2.0-flash": ModelCapability(
            name="Gemini 2.0 Flash",
            provider="google",
            model_id="gemini-2.0-flash",
            task_types=(
                TaskType.GENERAL_NLP,
                TaskType.MULTIMODAL,
                TaskType.LONG_CONTEXT,
            ),
            context_window=1000000,
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0004,
            strengths=("massive context", "speed", "multimodal"),
            supports_vision=True,
        ),
        "gemini-1.5-pro": ModelCapability(
            name="Gemini 1.5 Pro",
            provider="google",
            model_id="gemini-1.5-pro",
            task_types=(
                TaskType.LONG_CONTEXT,
                TaskType.DATA_ANALYSIS,
                TaskType.MULTIMODAL,
            ),
            context_window=2000000,
            cost_per_1k_input=0.00125,
            cost_per_1k_output=0.005,
            strengths=("2M context", "multimodal", "data analysis"),
            supports_vision=True,
        ),
        # Vertex AI Models (Enterprise/Third-Party Integration)
        "vertex-gemini-3-pro": ModelCapability(
            name="Gemini 3.0 Pro (Vertex AI)",
            provider="vertex-ai",
            model_id="gemini-3-pro-preview",
            task_types=(
                TaskType.GENERAL_NLP,
                TaskType.CODE_GENERATION,
                TaskType.REASONING,
                TaskType.LONG_CONTEXT,
                TaskType.MULTIMODAL,
            ),
            context_window=2000000,
            cost_per_1k_input=0.0015,
            cost_per_1k_output=0.006,
            strengths=("enterprise endpoint", "third-party integration", "OAuth2 auth"),
            supports_vision=True,
            supports_functions=True,
        ),
        "vertex-gemini-3-flash": ModelCapability(
            name="Gemini 3.0 Flash (Vertex AI)",
            provider="vertex-ai",
            model_id="gemini-3-flash-preview",
            task_types=(
                TaskType.GENERAL_NLP,
                TaskType.CODE_GENERATION,
                TaskType.LONG_CONTEXT,
                TaskType.MULTIMODAL,
            ),
            context_window=1000000,
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0004,
            strengths=("enterprise endpoint", "fast", "multimodal"),
            supports_vision=True,
        ),
        "vertex-gemini-2.5-flash": ModelCapability(
            name="Gemini 2.5 Flash (Vertex AI)",
            provider="vertex-ai",
            model_id="gemini-2.5-flash",
            task_types=(
                TaskType.GENERAL_NLP,
                TaskType.CODE_GENERATION,
                TaskType.LONG_CONTEXT,
                TaskType.MULTIMODAL,
            ),
            context_window=1000000,
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0004,
            strengths=("enterprise endpoint", "fast", "stable"),
            supports_vision=True,
        ),
        # Local Models (Ollama)
        "llama3.2": ModelCapability(
            name="Llama 3.2",
            provider="ollama",
            model_id="llama3.2",
            task_types=(TaskType.GENERAL_NLP, TaskType.LOCAL_MODEL),
            context_window=128000,
            cost_per_1k_input=0,
            cost_per_1k_output=0,
            strengths=("free", "private", "offline"),
        ),
        "codellama": ModelCapability(
            name="Code Llama",
            provider="ollama",
            model_id="codellama",
            task_types=(TaskType.CODE_GENERATION, TaskType.LOCAL_MODEL),
            context_window=16000,
            cost_per_1k_input=0,
            cost_per_1k_output=0,
            strengths=("coding", "free", "private"),
        ),
        "deepseek-coder-v2": ModelCapability(
            name="DeepSeek Coder V2",
            provider="ollama",
            model_id="deepseek-coder-v2",
            task_types=(TaskType.CODE_GENERATION, TaskType.LOCAL_MODEL),
            context_window=128000,
            cost_per_1k_input=0,
            cost_per_1k_output=0,
            strengths=("excellent coding", "free", "private"),
        ),
        # Mistral Models
        "mistral-large": ModelCapability(
            name="Mistral Large",
            provider="mistral",
            model_id="mistral-large-latest",
            task_types=(
                TaskType.CODE_GENERATION,
                TaskType.REASONING,
                TaskType.GENERAL_NLP,
            ),
            context_window=128000,
            cost_per_1k_input=0.002,
            cost_per_1k_output=0.006,
            strengths=("coding", "reasoning", "multilingual"),
            supports_functions=True,
        ),
        "codestral": ModelCapability(
            name="Codestral",
            provider="mistral",
            model_id="codestral-latest",
            task_types=(TaskType.CODE_GENERATION,),
            context_window=32000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.003,
            strengths=("excellent coding", "fast", "specialized"),
        ),
        "mistral-small": ModelCapability(
            name="Mistral Small",
            provider="mistral",
            model_id="mistral-small-latest",
            task_types=(TaskType.GENERAL_NLP, TaskType.SUMMARIZATION),
            context_window=32000,
            cost_per_1k_input=0.0002,
            cost_per_1k_output=0.0006,
            strengths=("cost-effective", "fast", "efficient"),
        ),
        # Groq Models (Fast inference)
        "groq-llama-3.3-70b": ModelCapability(
            name="Llama 3.3 70B (Groq)",
            provider="groq",
            model_id="llama-3.3-70b-versatile",
            task_types=(
                TaskType.GENERAL_NLP,
                TaskType.CODE_GENERATION,
                TaskType.REASONING,
            ),
            context_window=128000,
            cost_per_1k_input=0.00059,
            cost_per_1k_output=0.00079,
            strengths=("ultra fast", "versatile", "cost-effective"),
        ),
        "groq-mixtral-8x7b": ModelCapability(
            name="Mixtral 8x7B (Groq)",
            provider="groq",
            model_id="mixtral-8x7b-32768",
            task_types=(TaskType.GENERAL_NLP, TaskType.CODE_GENERATION),
            context_window=32768,
            cost_per_1k_input=0.00024,
            cost_per_1k_output=0.00024,
            strengths=("very fast", "cost-effective", "multilingual"),
        ),
        # xAI Models (Grok)
        "grok-2": ModelCapability(
            name="Grok 2",
            provider="xai",
            model_id="grok-2-latest",
            task_types=(
                TaskType.GENERAL_NLP,
                TaskType.REASONING,
                TaskType.CODE_GENERATION,
                TaskType.CREATIVE_WRITING,
            ),
            context_window=131072,
            cost_per_1k_input=0.002,
            cost_per_1k_output=0.010,
            strengths=("real-time knowledge", "reasoning", "creative"),
        ),
        "grok-2-vision": ModelCapability(
            name="Grok 2 Vision",
            provider="xai",
            model_id="grok-2-vision-latest",
            task_types=(TaskType.MULTIMODAL, TaskType.GENERAL_NLP),
            context_window=32768,
            cost_per_1k_input=0.002,
            cost_per_1k_output=0.010,
            strengths=("vision", "real-time knowledge", "multimodal"),
            supports_vision=True,
        ),
        # Perplexity Models (Web search enabled)
        "perplexity-sonar-pro": ModelCapability(
            name="Sonar Pro",
            provider="perplexity",
            model_id="sonar-pro",
            task_types=(TaskType.WEB_SEARCH, TaskType.GENERAL_NLP, TaskType.REASONING),
            context_window=200000,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            strengths=("web search", "citations", "real-time info"),
        ),
        "perplexity-sonar": ModelCapability(
            name="Sonar",
            provider="perplexity",
            model_id="sonar",
            task_types=(TaskType.WEB_SEARCH, TaskType.GENERAL_NLP),
            context_window=128000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.001,
            strengths=("web search", "cost-effective", "fast"),
        ),
        # DeepSeek Cloud Models
        "deepseek-chat": ModelCapability(
            name="DeepSeek Chat",
            provider="deepseek",
            model_id="deepseek-chat",
            task_types=(
                TaskType.GENERAL_NLP,
                TaskType.CODE_GENERATION,
                TaskType.REASONING,
            ),
            context_window=64000,
            cost_per_1k_input=0.00014,
            cost_per_1k_output=0.00028,
            strengths=("very cost-effective", "coding", "reasoning"),
        ),
        "deepseek-reasoner": ModelCapability(
            name="DeepSeek Reasoner",
            provider="deepseek",
            model_id="deepseek-reasoner",
            task_types=(
                TaskType.DEEP_REASONING,
                TaskType.MATH,
                TaskType.CODE_GENERATION,
            ),
            context_window=64000,
            cost_per_1k_input=0.00055,
            cost_per_1k_output=0.00219,
            strengths=("deep reasoning", "math", "problem-solving"),
        ),
        # MLX Local Models (Apple Silicon optimized)
        "mlx-llama8": ModelCapability(
            name="MLX Llama 8B",
            provider="mlx",
            model_id="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            task_types=(
                TaskType.GENERAL_NLP,
                TaskType.LOCAL_MODEL,
                TaskType.CODE_GENERATION,
            ),
            context_window=8192,
            cost_per_1k_input=0,  # Free local inference
            cost_per_1k_output=0,
            strengths=(
                "free",
                "private",
                "offline",
                "fast on Apple Silicon",
                "good speed and efficiency",
                "objective responses",
            ),
            max_output_tokens=4096,
            supports_streaming=True,
        ),
        "mlx-qwen-3-32b": ModelCapability(
            name="MLX Qwen 3 32B (Vision)",
            provider="mlx",
            model_id="mlx-community/Qwen3-VL-32B-Instruct-4bit",
            task_types=(
                TaskType.MULTIMODAL,
                TaskType.LOCAL_MODEL,
                TaskType.GENERAL_NLP,
            ),
            context_window=8192,
            cost_per_1k_input=0,
            cost_per_1k_output=0,
            strengths=("vision", "local", "private", "high quality"),
            supports_vision=True,
            supports_streaming=True,
        ),
        "mlx-qwen-coder-14b": ModelCapability(
            name="MLX Qwen Coder 14B",
            provider="mlx",
            model_id="mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
            task_types=(TaskType.CODE_GENERATION, TaskType.LOCAL_MODEL),
            context_window=32768,
            cost_per_1k_input=0,
            cost_per_1k_output=0,
            strengths=("coding", "local", "private", "fast"),
            supports_streaming=True,
        ),
    }

    @classmethod
    def get_model(cls, model_key: str) -> ModelCapability | None:
        # Map certain Google Gemini registry keys to Vertex variants by default
        vertex_alias_map = {
            "gemini-3-pro": "vertex-gemini-3-pro",
            "gemini-3-flash": "vertex-gemini-3-flash",
        }
        if model_key in vertex_alias_map and vertex_alias_map[model_key] in cls.MODELS:
            return cls.MODELS[vertex_alias_map[model_key]]

        # Allow lookups by either the registry key (recommended) or the
        # underlying provider model id (useful for "preview"/versioned ids).
        model = cls.MODELS.get(model_key)
        if model is not None:
            return model

        candidates: list[tuple[str, ModelCapability]] = [
            (key, candidate)
            for key, candidate in cls.MODELS.items()
            if candidate.model_id == model_key
        ]
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0][1]

        # Preference rule: For Gemini 3 Pro/Flash Preview, prefer Vertex AI variants
        vertex_preferred_ids = {"gemini-3-pro-preview", "gemini-3-flash-preview"}
        if (
            model_key in vertex_preferred_ids
            or candidates[0][1].model_id in vertex_preferred_ids
        ):
            vertex = [
                (key, candidate)
                for key, candidate in candidates
                if candidate.provider == "vertex-ai" or key.startswith("vertex-")
            ]
            if vertex:
                return sorted(vertex, key=lambda item: item[0])[0][1]

        # Otherwise, prefer the non-Vertex (Google) entry by default
        non_vertex = [
            (key, candidate)
            for key, candidate in candidates
            if candidate.provider != "vertex-ai" and not key.startswith("vertex-")
        ]
        if non_vertex:
            return sorted(non_vertex, key=lambda item: item[0])[0][1]

        # Fallback to first by key order if only Vertex options exist
        return sorted(candidates, key=lambda item: item[0])[0][1]

    @classmethod
    def get_models_for_task(
        cls,
        task_type: TaskType,
        require_local: bool = False,
    ) -> list[ModelCapability]:
        """Get models suitable for a task type"""
        suitable = []
        for model in cls.MODELS.values():
            if task_type in model.task_types:
                if require_local and model.provider not in LOCAL_PROVIDERS:
                    continue
                suitable.append(model)

        # Sort by cost (cheapest first, unless it's local which is free)
        return sorted(suitable, key=lambda m: m.cost_per_1k_input)


class AIOrchestrator:
    """
    Main AI Orchestrator that intelligently routes queries to the best model.

    Features:
    - Automatic task classification
    - Optimal model selection based on task type
    - Secure credential management
    - Rate limiting and retry logic
    - Comprehensive logging
    """

    def __init__(
        self,
        prefer_local: bool | None = None,
        cost_optimize: bool | None = None,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose

        self.rate_limiter = RateLimiter()
        self.providers: dict[str, BaseProvider] = {}
        self.conversation_history: list[dict[str, Any]] = []
        self._max_history_messages = 200

        # Load config first so we can use it for logging setup
        self._user_config = self._load_user_config()
        defaults = self._get_defaults_config()
        self.prefer_local = self._resolve_bool_config(
            prefer_local, defaults, "preferLocal"
        )
        self.cost_optimize = self._resolve_bool_config(
            cost_optimize, defaults, "costOptimize"
        )
        self.local_provider_preference = self._resolve_local_provider_preference(
            defaults
        )
        self._setup_logging()

    def _get_defaults_config(self) -> dict[str, Any]:
        defaults = self._user_config.get("defaults", {})
        if isinstance(defaults, dict):
            return defaults
        return {}

    @staticmethod
    def _resolve_bool_config(
        value: bool | None, defaults: dict[str, Any], key: str
    ) -> bool:
        if value is not None:
            return value

        config_value = defaults.get(key)
        if isinstance(config_value, bool):
            return config_value

        return False

    @staticmethod
    def _resolve_local_provider_preference(
        defaults: dict[str, Any],
    ) -> str | None:
        config_value = defaults.get("localProvider")
        if not isinstance(config_value, str):
            return None

        normalized = config_value.strip().lower()
        if normalized in {"mlx", "ollama"}:
            return normalized
        if normalized in {"any", ""}:
            return None

        return None

    def _apply_local_provider_preference(
        self,
        candidates: list[ModelCapability],
    ) -> list[ModelCapability]:
        if not self.local_provider_preference or not candidates:
            return candidates

        if not all(model.provider in LOCAL_PROVIDERS for model in candidates):
            return candidates

        filtered = [
            model
            for model in candidates
            if model.provider == self.local_provider_preference
        ]
        if filtered:
            return filtered

        if self.verbose:
            logger.warning(
                "Preferred local provider '%s' not available; falling back to other local candidates.",
                self.local_provider_preference,
            )
        return candidates

    def _setup_logging(self) -> None:
        level = logging.DEBUG if self.verbose else logging.INFO

        # Check config for logging overrides
        log_config = self._user_config.get("logging", {})
        if not self.verbose and "level" in log_config:
            level_name = log_config["level"].upper()
            level = getattr(logging, level_name, logging.INFO)

        handlers: list[logging.Handler] = [logging.StreamHandler()]

        # Add file handler if configured
        log_file = log_config.get("file")
        if log_file:
            try:
                # Expand user path if necessary
                import os

                expanded_path = os.path.expanduser(log_file)
                handlers.append(logging.FileHandler(expanded_path, encoding="utf-8"))
            except Exception as e:
                # Fallback to console only if file setup fails
                print(f"Failed to setup log file {log_file}: {e}")

        # Configure root logger to capture all events
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers,
            force=True,  # Force reconfiguration
        )

    def _load_user_config(self) -> dict[str, Any]:
        config_path = CONFIG_DIR / "config.json"
        if not config_path.exists():
            return {}

        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                loaded = json.load(config_file)
        except Exception as exc:
            # Can't log yet as logging isn't set up, use print
            print(f"Warning: Failed to load config from {config_path}: {exc}")
            return {}

        if not isinstance(loaded, dict):
            print(f"Warning: Config file {config_path} did not contain an object.")
            return {}

        return loaded

    def _get_provider_config(self, provider_name: str) -> dict[str, Any]:
        providers = self._user_config.get("providers", {})
        if not isinstance(providers, dict):
            return {}

        provider_config = providers.get(provider_name, {})
        if not isinstance(provider_config, dict):
            return {}

        return provider_config

    async def _get_provider(self, provider_name: str) -> BaseProvider | None:
        """Get or initialize a provider"""
        if provider_name not in self.providers:
            provider: BaseProvider | None = None
            if provider_name == "openai":
                provider = OpenAIProvider(self.rate_limiter)
            elif provider_name == "anthropic":
                provider = AnthropicProvider(self.rate_limiter)
            elif provider_name == "google":
                provider = GoogleProvider(self.rate_limiter)
            elif provider_name == "vertex-ai":
                provider_config = self._get_provider_config("vertex-ai")
                location = provider_config.get("location")
                if isinstance(location, str) and location:
                    provider = VertexAIProvider(self.rate_limiter, location=location)
                else:
                    provider = VertexAIProvider(self.rate_limiter)
            elif provider_name == "ollama":
                provider = OllamaProvider(self.rate_limiter)
            elif provider_name == "mistral":
                provider = MistralProvider(self.rate_limiter)
            elif provider_name == "groq":
                provider = GroqProvider(self.rate_limiter)
            elif provider_name == "xai":
                provider = XAIProvider(self.rate_limiter)
            elif provider_name == "perplexity":
                provider = PerplexityProvider(self.rate_limiter)
            elif provider_name == "deepseek":
                provider = DeepSeekProvider(self.rate_limiter)
            elif provider_name.startswith("mlx-"):
                # Initialize specific MLX model
                model_key = provider_name
                model_def = ModelRegistry.MODELS.get(model_key)
                if model_def:
                    provider = MLXProvider(
                        self.rate_limiter, model_path=model_def.model_id
                    )
                else:
                    logger.error(f"Unknown MLX provider/model: {provider_name}")
            elif provider_name == "mlx":
                provider = MLXProvider(self.rate_limiter)
            else:
                logger.error(f"Unknown provider: {provider_name}")
                return None

            if provider is not None:
                if await provider.initialize():
                    self.providers[provider_name] = provider
                else:
                    # If initialization fails, log it but don't return None from here.
                    # The get method below will handle the case where the provider is not in the dict.
                    logger.error(f"Failed to initialize provider: {provider_name}")

        return self.providers.get(provider_name)

    def _get_config_task_key(self, task_type: TaskType) -> str:
        """Map TaskType enum to config string keys"""
        mapping = {
            TaskType.CODE_GENERATION: "code",
            TaskType.REASONING: "reasoning",
            TaskType.DEEP_REASONING: "reasoning",
            TaskType.CREATIVE_WRITING: "creative",
            TaskType.SUMMARIZATION: "summarization",
            TaskType.LOCAL_MODEL: "local",
            TaskType.LONG_CONTEXT: "long-context",
            TaskType.WEB_SEARCH: "websearch",
            TaskType.MATH: "math",
            TaskType.DATA_ANALYSIS: "data-analysis",
            TaskType.MULTIMODAL: "multimodal",
            TaskType.GENERAL_NLP: "general",
        }
        return mapping.get(task_type, "general")

    def select_model(
        self,
        task_types: list[tuple[TaskType, float]],
    ) -> ModelCapability | None:
        """
        Select the best model for the given task types.

        Uses a multi-factor scoring system that considers:
        1. Task type match (primary factor)
        2. Provider characteristics (strengths/weaknesses)
        3. Cost optimization (if enabled)
        4. Context window size (for long context tasks)
        5. User configuration (routing and priority)
        """
        if not task_types:
            task_types = [(TaskType.GENERAL_NLP, 0.5)]

        primary_task = task_types[0][0]
        all_tasks = [t[0] for t in task_types]

        # 1. Check User Config for Task Routing
        # This overrides standard selection if a valid route is defined
        task_key = self._get_config_task_key(primary_task)
        routing_config = self._user_config.get("taskRouting", {})
        preferred_models = routing_config.get(task_key)

        candidates = []

        if preferred_models and isinstance(preferred_models, list):
            # If user specified models for this task, restrict to those
            for model_key in preferred_models:
                model = ModelRegistry.get_model(model_key)
                if model:
                    candidates.append(model)

            if self.verbose and candidates:
                logger.info(
                    f"Using custom routing for '{task_key}': {[m.name for m in candidates]}"
                )

        # If no custom routing or no valid models found in routing, use standard selection
        if not candidates:
            candidates = ModelRegistry.get_models_for_task(
                primary_task, require_local=self.prefer_local
            )

        if not candidates:
            # Fallback to any available model
            candidates = list(ModelRegistry.MODELS.values())

        # 2. Filter by User Config (Enabled/Disabled)
        model_config = self._user_config.get("models", {})
        filtered_candidates = []
        for model in candidates:
            # Check if model is explicitly disabled in config
            # Try matching by specific model ID or registry key
            specific_config = model_config.get(model.model_id)

            # If not found by model_id, try to find by registry key (more expensive)
            if not specific_config:
                for reg_key, reg_model in ModelRegistry.MODELS.items():
                    if reg_model == model:
                        specific_config = model_config.get(reg_key)
                        break

            if specific_config and specific_config.get("enabled") is False:
                if self.verbose:
                    logger.info(f"Skipping disabled model: {model.name}")
                continue

            filtered_candidates.append(model)

        candidates = filtered_candidates
        if not candidates:
            return None

        candidates = self._apply_local_provider_preference(candidates)

        # Score candidates using provider characteristics
        scored = []
        for model in candidates:
            score = 0.0

            # 1. Task match score (base scoring)
            for task, confidence in task_types:
                if task in model.task_types:
                    score += confidence * 10

            # 2. Provider characteristics scoring
            provider_chars = PROVIDER_CHARACTERISTICS.get(model.provider)
            if provider_chars:
                # Map task types to relevant provider scores
                task_score_weights = self._get_task_score_weights(all_tasks)

                # Apply weighted provider characteristics
                score += (
                    provider_chars.contextual_understanding
                    * task_score_weights.get("contextual", 0)
                )
                score += provider_chars.creativity_originality * task_score_weights.get(
                    "creativity", 0
                )
                score += provider_chars.emotional_intelligence * task_score_weights.get(
                    "emotional", 0
                )
                score += provider_chars.speed_efficiency * task_score_weights.get(
                    "speed", 0
                )
                score += provider_chars.knowledge_breadth * task_score_weights.get(
                    "knowledge", 0
                )
                score += provider_chars.reasoning_depth * task_score_weights.get(
                    "reasoning", 0
                )
                score += provider_chars.code_quality * task_score_weights.get("code", 0)
                score += provider_chars.objectivity * task_score_weights.get(
                    "objectivity", 0
                )

                # Penalty for tasks in avoid_for list
                for avoid_case in provider_chars.avoid_for:
                    avoid_lower = avoid_case.lower()
                    for task in all_tasks:
                        if task.name.lower().replace(
                            "_", " "
                        ) in avoid_lower or avoid_lower in task.name.lower().replace(
                            "_", " "
                        ):
                            score -= 2.0  # Penalty for mismatched use case

            # 3. Cost optimization
            if self.cost_optimize:
                score -= model.cost_per_1k_input * 100

            # 4. Context window bonus for long context tasks
            if TaskType.LONG_CONTEXT in all_tasks:
                score += model.context_window / 100000

            # 5. Prefer local models when privacy keywords detected
            if TaskType.LOCAL_MODEL in all_tasks:
                if model.provider in LOCAL_PROVIDERS:
                    score += 5.0  # Significant bonus for local/private

            # 6. User Priority Bonus
            # Add bonus based on user priority config (default 50)
            specific_config = model_config.get(model.model_id)
            # Try lookup by registry key if needed
            if not specific_config:
                for reg_key, reg_model in ModelRegistry.MODELS.items():
                    if reg_model == model:
                        specific_config = model_config.get(reg_key)
                        break

            if specific_config:
                priority = specific_config.get("priority", 50)
                # Map 0-100 priority to roughly -5 to +5 score adjustment
                priority_bonus = (priority - 50) / 10.0
                score += priority_bonus

            # 7. Prefer Vertex AI for Gemini 3 Pro/Flash Preview
            if model.model_id in {"gemini-3-pro-preview", "gemini-3-flash-preview"}:
                if model.provider == "vertex-ai":
                    score += 1.0
                elif model.provider == "google":
                    score -= 0.2

            scored.append((model, score))

        if not scored:
            return None

        scored.sort(key=lambda x: -x[1])

        if self.verbose:
            logger.info(f"Model scores: {[(m.name, s) for m, s in scored[:5]]}")

        return scored[0][0]

    def _get_task_score_weights(
        self,
        task_types: list[TaskType],
    ) -> dict[str, float]:
        """
        Map task types to weighted provider characteristic scores.

        Returns weights for each provider characteristic based on
        which characteristics are most relevant for the detected tasks.
        """
        weights: dict[str, float] = {
            "contextual": 0.5,  # Base weight for all tasks
            "creativity": 0.0,
            "emotional": 0.0,
            "speed": 0.5,  # Speed is generally important
            "knowledge": 0.5,
            "reasoning": 0.0,
            "code": 0.0,
            "objectivity": 0.5,
        }

        for task in task_types:
            if task == TaskType.CODE_GENERATION:
                weights["code"] += 3.0
                weights["reasoning"] += 1.0
            elif task == TaskType.CREATIVE_WRITING:
                weights["creativity"] += 3.0
                weights["emotional"] += 1.5
            elif task == TaskType.DEEP_REASONING or task == TaskType.REASONING:
                weights["reasoning"] += 3.0
                weights["contextual"] += 1.0
            elif task == TaskType.DATA_ANALYSIS:
                weights["reasoning"] += 2.0
                weights["knowledge"] += 1.0
            elif task == TaskType.MATH:
                weights["reasoning"] += 2.5
                weights["code"] += 0.5
            elif task == TaskType.SUMMARIZATION:
                weights["contextual"] += 2.0
            elif task == TaskType.LONG_CONTEXT:
                weights["contextual"] += 2.0
            elif task == TaskType.WEB_SEARCH:
                weights["knowledge"] += 3.0
            elif task == TaskType.MULTIMODAL:
                weights["contextual"] += 1.0
            elif task == TaskType.LOCAL_MODEL:
                weights["speed"] += 1.0  # Local models prioritize speed

        return weights

    async def query(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_override: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> APIResponse:
        """
        Send a query to the AI orchestrator.

        Args:
            prompt: The user's prompt
            system_prompt: Optional system prompt
            model_override: Force a specific model
            max_tokens: Maximum response tokens
            temperature: Sampling temperature

        Returns:
            APIResponse with the result
        """
        # Validate input
        is_valid, error = InputValidator.validate_prompt(prompt)
        if not is_valid:
            return APIResponse(
                content="",
                model="",
                provider="",
                usage={},
                latency_ms=0,
                success=False,
                error=error,
            )

        # Classify task
        task_types = TaskClassifier.classify(prompt)

        if self.verbose:
            logger.info(f"Classified task types: {task_types}")

        # Select model
        if model_override:
            model = ModelRegistry.get_model(model_override)
            if not model:
                return APIResponse(
                    content="",
                    model=model_override,
                    provider="unknown",
                    usage={},
                    latency_ms=0,
                    success=False,
                    error=f"Unknown model: '{model_override}'.",
                )
        else:
            model = self.select_model(task_types)

        if not model:
            return APIResponse(
                content="",
                model="N/A",
                provider="N/A",
                usage={},
                latency_ms=0,
                success=False,
                error="Could not select a suitable model for the task.",
            )

        # Get provider for the selected model
        # MLX models need a distinct provider instance per model since they load weights locally.
        provider_key = model.provider
        if model.provider == "mlx":
            # Try to resolve the registry key for this model to support per-model MLX providers
            reg_key: str | None = None
            for key, reg_model in ModelRegistry.MODELS.items():
                if reg_model == model:
                    reg_key = key
                    break
            if reg_key and reg_key.startswith("mlx-"):
                provider_key = reg_key

        provider = await self._get_provider(provider_key)
        if not provider:
            return APIResponse(
                content="",
                model=model.name,
                provider=model.provider,
                usage={},
                latency_ms=0,
                success=False,
                error=f"Provider '{model.provider}' is not available or failed to initialize.",
            )

        # Build messages
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Add conversation history
        self.conversation_history.extend(messages)

        # Query the provider with retry logic
        last_response: APIResponse | None = None

        async def _attempt() -> APIResponse:
            nonlocal last_response
            response = await provider.complete(
                self.conversation_history,
                model=model.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            last_response = response
            if (
                not response.success
                and response.error
                and RetryHandler.is_retryable(response.error)
            ):
                raise RuntimeError(response.error)
            return response

        try:
            response = await RetryHandler.execute_with_retry(_attempt)
            # Add successful response to history
            if response.success and response.content:
                self.conversation_history.append(
                    {"role": "assistant", "content": response.content}
                )
                # Prevent unbounded history growth
                if len(self.conversation_history) > self._max_history_messages:
                    self.conversation_history = self.conversation_history[
                        -self._max_history_messages :
                    ]
            return response
        except Exception as e:
            logger.error(f"Query failed after retries: {e}")
            if last_response is not None:
                return last_response
            return APIResponse(
                content="",
                model=model.name,
                provider=model.provider,
                usage={},
                latency_ms=0,
                success=False,
                error=f"An unexpected error occurred: {e}",
            )

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history.clear()

    async def multi_model_query(
        self,
        prompt: str,
        models: list[str],
        system_prompt: str | None = None,
    ) -> dict[str, APIResponse]:
        """
        Query multiple models in parallel for comparison.
        """
        tasks = []
        for model_key in models:
            tasks.append(
                self.query(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model_override=model_key,
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        result_map: dict[str, APIResponse] = {}
        for model, result in zip(models, results, strict=False):
            if isinstance(result, BaseException):
                result_map[model] = APIResponse(
                    content="",
                    model=model,
                    provider="",
                    usage={},
                    latency_ms=0,
                    success=False,
                    error=str(result),
                )
            else:
                # result is APIResponse
                result_map[model] = result

        return result_map


async def main() -> None:
    """CLI interface for the orchestrator"""
    import argparse

    parser = argparse.ArgumentParser(description="AI Orchestrator CLI")
    parser.add_argument("prompt", nargs="?", help="The prompt to send")
    parser.add_argument("--model", "-m", help="Override model selection")
    parser.add_argument(
        "--local",
        "-l",
        action="store_true",
        default=None,
        help="Prefer local models",
    )
    parser.add_argument(
        "--cheap",
        "-c",
        action="store_true",
        default=None,
        help="Optimize for cost",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--configure", action="store_true", help="Configure API keys")
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )

    args = parser.parse_args()

    if args.configure:
        from .credentials import configure_credentials_interactive

        configure_credentials_interactive()
        return

    if args.list_models:
        print("\nAvailable Models:")
        print("=" * 60)
        for key, model in ModelRegistry.MODELS.items():
            print(f"\n{key}:")
            print(f"  Name: {model.name}")
            print(f"  Provider: {model.provider}")
            print(f"  Context: {model.context_window:,} tokens")
            print(
                f"  Cost: ${model.cost_per_1k_input}/1k input, ${model.cost_per_1k_output}/1k output"
            )
            print(f"  Strengths: {', '.join(model.strengths)}")
        return

    if not args.prompt:
        parser.print_help()
        return

    orchestrator = AIOrchestrator(
        prefer_local=args.local,
        cost_optimize=args.cheap,
        verbose=args.verbose,
    )

    response = await orchestrator.query(
        prompt=args.prompt,
        model_override=args.model,
    )

    if response.success:
        print(f"\n[{response.model}] ({response.latency_ms:.0f}ms)")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print(
            f"Tokens: {response.usage.get('input_tokens', 0)} in / {response.usage.get('output_tokens', 0)} out"
        )
    else:
        print(f"\nError: {response.error}")


if __name__ == "__main__":
    asyncio.run(main())
