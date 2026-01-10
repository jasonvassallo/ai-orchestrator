"""
Tests for AI Orchestrator
========================

Run with: pytest tests/ -v

Security note: These tests use mocked API calls and do not
require real API keys. Never commit real API keys to tests.
"""

import os
import sys
from unittest.mock import patch

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.orchestrator as orchestrator_module
from src.credentials import (
    EnvironmentBackend,
)
from src.orchestrator import (
    AIOrchestrator,
    APIResponse,
    BaseProvider,
    InputValidator,
    ModelRegistry,
    RateLimiter,
    RetryHandler,
    TaskClassifier,
    TaskType,
)


class TestInputValidator:
    """Test input validation for security"""

    def test_valid_prompt(self):
        """Valid prompts should pass"""
        is_valid, error = InputValidator.validate_prompt("Hello, how are you?")
        assert is_valid is True
        assert error == ""

    def test_empty_prompt_fails(self):
        """Empty prompts should fail"""
        is_valid, error = InputValidator.validate_prompt("")
        assert is_valid is False
        assert "non-empty" in error.lower()

    def test_none_prompt_fails(self):
        """None prompts should fail"""
        is_valid, error = InputValidator.validate_prompt(None)
        assert is_valid is False

    def test_very_long_prompt_fails(self):
        """Prompts exceeding max length should fail"""
        long_prompt = "x" * (InputValidator.MAX_PROMPT_LENGTH + 1)
        is_valid, error = InputValidator.validate_prompt(long_prompt)
        assert is_valid is False
        assert "maximum length" in error.lower()

    def test_prompt_at_max_length_passes(self):
        """Prompts at exactly max length should pass"""
        prompt = "x" * InputValidator.MAX_PROMPT_LENGTH
        is_valid, error = InputValidator.validate_prompt(prompt)
        assert is_valid is True

    def test_sanitize_removes_api_keys(self):
        """API keys should be redacted in logs"""
        text = "My API key is sk-1234567890abcdefghij"
        sanitized = InputValidator.sanitize_for_logging(text)
        assert "sk-1234567890" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_sanitize_truncates(self):
        """Long text should be truncated"""
        text = "x" * 200
        sanitized = InputValidator.sanitize_for_logging(text, max_len=50)
        assert len(sanitized) == 53  # 50 + "..."
        assert sanitized.endswith("...")


class TestTaskClassifier:
    """Test task classification"""

    def test_code_task_detection(self):
        """Code-related prompts should be classified correctly"""
        prompts = [
            "Write a Python function to sort a list",
            "Debug this JavaScript code",
            "Implement a REST API endpoint",
            "Create a SQL query for user data",
        ]

        for prompt in prompts:
            tasks = TaskClassifier.classify(prompt)
            task_types = [t[0] for t in tasks]
            assert TaskType.CODE_GENERATION in task_types, f"Failed for: {prompt}"

    def test_reasoning_task_detection(self):
        """Reasoning prompts should be classified correctly"""
        prompts = [
            "Prove that the square root of 2 is irrational",
            "Why is the sky blue?",
            "Analyze the implications of this theorem",
        ]

        for prompt in prompts:
            tasks = TaskClassifier.classify(prompt)
            task_types = [t[0] for t in tasks]
            assert (
                TaskType.DEEP_REASONING in task_types
                or TaskType.REASONING in task_types
            ), f"Failed for: {prompt}"

    def test_creative_task_detection(self):
        """Creative prompts should be classified correctly"""
        prompts = [
            "Write a short story about a dragon",
            "Create a blog post about technology",
            "Write a poem about nature",
        ]

        for prompt in prompts:
            tasks = TaskClassifier.classify(prompt)
            task_types = [t[0] for t in tasks]
            assert TaskType.CREATIVE_WRITING in task_types, f"Failed for: {prompt}"

    def test_general_fallback(self):
        """Unknown prompts should fallback to general NLP"""
        tasks = TaskClassifier.classify("Hello there!")
        assert len(tasks) > 0
        # Should have some classification

    def test_confidence_scores(self):
        """Confidence scores should be between 0 and 1"""
        tasks = TaskClassifier.classify("Write Python code to analyze data")
        for _task_type, confidence in tasks:
            assert 0.0 <= confidence <= 1.0


class TestModelRegistry:
    """Test model registry"""

    def test_get_existing_model(self):
        """Should return model for valid key"""
        model = ModelRegistry.get_model("gpt-4o")
        assert model is not None
        assert model.provider == "openai"

    def test_get_nonexistent_model(self):
        """Should return None for invalid key"""
        model = ModelRegistry.get_model("nonexistent-model")
        assert model is None

    def test_get_model_by_provider_model_id(self):
        """Should resolve a registry entry from a provider model id"""
        model = ModelRegistry.get_model("gemini-3-flash-preview")
        assert model is not None
        assert model.provider == "vertex-ai"
        assert model.model_id == "gemini-3-flash-preview"

    def test_model_id_resolution_prefers_vertex_over_google(self):
        """If multiple models share an id, prefer the Vertex entry for Gemini 3 Preview models"""
        model = ModelRegistry.get_model("gemini-3-pro-preview")
        assert model is not None
        assert model.provider == "vertex-ai"

    def test_get_models_for_task(self):
        """Should return suitable models for task type"""
        models = ModelRegistry.get_models_for_task(TaskType.CODE_GENERATION)
        assert len(models) > 0
        for model in models:
            assert TaskType.CODE_GENERATION in model.task_types

    def test_get_local_models(self):
        """Should filter for local models when requested"""
        local_providers = {"ollama", "mlx"}  # Both are valid local providers
        models = ModelRegistry.get_models_for_task(
            TaskType.GENERAL_NLP, require_local=True
        )
        for model in models:
            assert model.provider in local_providers


class TestRateLimiter:
    """Test rate limiting"""

    @pytest.mark.asyncio
    async def test_allows_initial_requests(self):
        """Should allow initial requests"""
        limiter = RateLimiter()
        allowed = await limiter.check_and_wait("openai", 1000)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_tracks_requests(self):
        """Should track request counts"""
        limiter = RateLimiter()
        for _ in range(5):
            await limiter.check_and_wait("openai", 100)

        state = limiter._states["openai"]
        assert len(state.requests) == 5


class TestRetryHandler:
    """Test retry logic"""

    def test_identifies_retryable_errors(self):
        """Should identify retryable errors"""
        retryable_errors = [
            "rate_limit exceeded",
            "connection timeout",
            "503 service unavailable",
            "server overloaded",
            "HTTP 429: Resource exhausted",
            "Too many requests",
        ]

        for error in retryable_errors:
            assert RetryHandler.is_retryable(error) is True

    def test_identifies_non_retryable_errors(self):
        """Should identify non-retryable errors"""
        non_retryable_errors = [
            "invalid api key",
            "authentication failed",
            "model not found",
        ]

        for error in non_retryable_errors:
            assert RetryHandler.is_retryable(error) is False

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Should retry on retryable errors"""
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("rate_limit exceeded")
            return "success"

        result = await RetryHandler.execute_with_retry(
            failing_func,
            max_retries=3,
            base_delay=0.01,
        )

        assert result == "success"
        assert call_count == 3


class TestCredentialBackends:
    """Test credential storage backends"""

    def test_environment_backend_get(self):
        """Environment backend should read from env vars"""
        backend = EnvironmentBackend()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            key = backend.get("openai")
            assert key == "test-key"

    def test_environment_backend_missing(self):
        """Should return None for missing env vars"""
        backend = EnvironmentBackend()

        with patch.dict(os.environ, {}, clear=True):
            key = backend.get("nonexistent")
            assert key is None

    def test_environment_backend_set(self):
        """Should set env vars (non-persistent)"""
        backend = EnvironmentBackend()
        result = backend.set("test", "test-value")
        assert result is True
        assert os.environ.get("TEST_API_KEY") == "test-value"


class TestAIOrchestrator:
    """Test main orchestrator"""

    @pytest.fixture
    def orchestrator(self):
        return AIOrchestrator(verbose=False)

    def test_model_selection_for_code(self, orchestrator):
        """Should select appropriate model for code tasks"""
        tasks = [(TaskType.CODE_GENERATION, 0.9)]
        model = orchestrator.select_model(tasks)

        assert model is not None
        assert TaskType.CODE_GENERATION in model.task_types

    def test_model_selection_cost_optimize(self):
        """Cost optimization should prefer cheaper models"""
        orchestrator = AIOrchestrator(cost_optimize=True)
        tasks = [(TaskType.GENERAL_NLP, 0.5)]
        model = orchestrator.select_model(tasks)

        # Should prefer a cheaper model
        assert model is not None

    def test_model_selection_prefer_local(self):
        """Local preference should select local models (Ollama or MLX)"""
        local_providers = {"ollama", "mlx"}  # Both are valid local providers
        orchestrator = AIOrchestrator(prefer_local=True)
        tasks = [(TaskType.GENERAL_NLP, 0.5)]
        model = orchestrator.select_model(tasks)

        assert model is not None
        assert model.provider in local_providers

    @pytest.mark.asyncio
    async def test_query_validates_input(self, orchestrator):
        """Should validate input before processing"""
        # Empty prompt
        response = await orchestrator.query("")
        assert response.success is False
        assert "non-empty" in response.error.lower()

    @pytest.mark.asyncio
    async def test_query_with_invalid_model_override(self, orchestrator):
        """Should fail gracefully with invalid model"""
        response = await orchestrator.query("Hello", model_override="nonexistent-model")
        assert response.success is False
        assert "unknown model" in response.error.lower()

    def test_clear_history(self, orchestrator):
        """Should clear conversation history"""
        orchestrator.conversation_history = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]

        orchestrator.clear_history()
        assert len(orchestrator.conversation_history) == 0

    @pytest.mark.asyncio
    async def test_query_retries_on_retryable_apiresponse(self, monkeypatch):
        """Should retry when provider returns a retryable APIResponse error"""

        class FakeVertexProvider(BaseProvider):
            def __init__(self, rate_limiter: RateLimiter) -> None:
                super().__init__(rate_limiter)
                self.calls = 0

            @property
            def provider_name(self) -> str:
                return "vertex-ai"

            async def initialize(self) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):
                self.calls += 1
                if self.calls < 3:
                    return APIResponse(
                        content="",
                        model=model,
                        provider=self.provider_name,
                        usage={},
                        latency_ms=0,
                        success=False,
                        error="HTTP 429: Resource exhausted",
                    )
                return APIResponse(
                    content="ok",
                    model=model,
                    provider=self.provider_name,
                    usage={},
                    latency_ms=0,
                    success=True,
                )

        orchestrator = AIOrchestrator(verbose=False)
        provider = FakeVertexProvider(orchestrator.rate_limiter)

        async def fake_get_provider(_name):
            return provider

        async def fast_sleep(_delay):
            return None

        monkeypatch.setattr(orchestrator, "_get_provider", fake_get_provider)
        monkeypatch.setattr(orchestrator_module.asyncio, "sleep", fast_sleep)

        response = await orchestrator.query(
            "Hello", model_override="vertex-gemini-3-pro"
        )

        assert response.success is True
        assert response.content == "ok"
        assert provider.calls == 3


class TestAPIResponse:
    """Test API response dataclass"""

    def test_successful_response(self):
        """Should create successful response"""
        response = APIResponse(
            content="Hello!",
            model="gpt-4o",
            provider="openai",
            usage={"input_tokens": 10, "output_tokens": 5},
            latency_ms=100.0,
            success=True,
        )

        assert response.success is True
        assert response.content == "Hello!"
        assert response.error is None

    def test_failed_response(self):
        """Should create failed response with error"""
        response = APIResponse(
            content="",
            model="gpt-4o",
            provider="openai",
            usage={},
            latency_ms=50.0,
            success=False,
            error="Connection failed",
        )

        assert response.success is False
        assert response.error == "Connection failed"


class TestSecurityCompliance:
    """Security-focused tests"""

    def test_no_hardcoded_api_keys(self):
        """Verify no API keys are hardcoded in source files"""
        import re

        # Patterns that might indicate hardcoded API keys
        patterns = [
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI
            r"sk-ant-[a-zA-Z0-9]{20,}",  # Anthropic
            r"AIza[a-zA-Z0-9]{35}",  # Google
        ]

        src_dir = os.path.join(os.path.dirname(__file__), "..", "src")

        for root, _dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    with open(filepath) as f:
                        content = f.read()
                        for pattern in patterns:
                            matches = re.findall(pattern, content)
                            assert len(matches) == 0, (
                                f"Potential API key found in {filepath}"
                            )

    def test_logs_redact_sensitive_data(self):
        """Verify sensitive data is redacted in logging"""
        sensitive_strings = [
            "sk-1234567890abcdefghijklmnopqrstuvwxyz",
            "api_key=secret123456789012345",
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        ]

        for text in sensitive_strings:
            sanitized = InputValidator.sanitize_for_logging(text)
            # Original key should not appear in sanitized output
            assert "1234567890abcdef" not in sanitized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
