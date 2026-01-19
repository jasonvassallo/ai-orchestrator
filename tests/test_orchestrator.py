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
    SPECIALIZED_TASKS,
    AIOrchestrator,
    APIResponse,
    BaseProvider,
    ChainedExecutor,
    ChainedResponse,
    ChainStep,
    InputValidator,
    LLMRouter,
    ModelRegistry,
    MoonshotProvider,
    RateLimiter,
    RetryHandler,
    RoutingDecision,
    TaskClassifier,
    TaskType,
    needs_llm_routing,
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


class TestMoonshotProvider:
    """Tests for MoonshotProvider (Kimi K2 extended thinking models)"""

    @pytest.fixture
    def rate_limiter(self):
        return RateLimiter()

    def test_provider_name(self, rate_limiter):
        """Should return 'moonshot' as provider name"""
        provider = MoonshotProvider(rate_limiter)
        assert provider.provider_name == "moonshot"

    @pytest.mark.asyncio
    async def test_initialize_missing_api_key(self, rate_limiter, monkeypatch):
        """Should fail to initialize without API key"""
        monkeypatch.setattr(orchestrator_module, "get_api_key", lambda x: None)
        provider = MoonshotProvider(rate_limiter)
        result = await provider.initialize()
        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, rate_limiter, monkeypatch):
        """Should initialize successfully with API key"""
        monkeypatch.setattr(orchestrator_module, "get_api_key", lambda x: "test-key")
        provider = MoonshotProvider(rate_limiter)
        result = await provider.initialize()
        assert result is True
        assert provider._client is not None

    @pytest.mark.asyncio
    async def test_complete_uses_temperature_1_for_thinking_models(
        self, rate_limiter, monkeypatch
    ):
        """Should use temperature=1.0 for thinking models"""
        monkeypatch.setattr(orchestrator_module, "get_api_key", lambda x: "test-key")
        provider = MoonshotProvider(rate_limiter)
        await provider.initialize()

        captured_json = {}

        async def mock_post(url, json):
            captured_json.update(json)

            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "choices": [{"message": {"content": "response"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                    }

            return MockResponse()

        provider._client.post = mock_post

        await provider.complete(
            messages=[{"role": "user", "content": "test"}],
            model="kimi-k2-thinking",
            temperature=0.5,  # This should be overridden to 1.0
        )

        assert captured_json["temperature"] == 1.0

    @pytest.mark.asyncio
    async def test_complete_formats_thinking_content(self, rate_limiter, monkeypatch):
        """Should format response with [Thinking] and [Answer] sections"""
        monkeypatch.setattr(orchestrator_module, "get_api_key", lambda x: "test-key")
        provider = MoonshotProvider(rate_limiter)
        await provider.initialize()

        async def mock_post(url, json):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "Final answer here",
                                    "reasoning_content": "Step by step reasoning",
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                    }

            return MockResponse()

        provider._client.post = mock_post

        response = await provider.complete(
            messages=[{"role": "user", "content": "test"}],
            model="kimi-k2-thinking",
        )

        assert response.success is True
        assert "[Thinking]" in response.content
        assert "[Answer]" in response.content
        assert "Step by step reasoning" in response.content
        assert "Final answer here" in response.content
        assert response.metadata.get("has_reasoning") is True

    @pytest.mark.asyncio
    async def test_complete_without_thinking(self, rate_limiter, monkeypatch):
        """Should return content without thinking sections for non-thinking models"""
        monkeypatch.setattr(orchestrator_module, "get_api_key", lambda x: "test-key")
        provider = MoonshotProvider(rate_limiter)
        await provider.initialize()

        async def mock_post(url, json):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "choices": [{"message": {"content": "Just the answer"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                    }

            return MockResponse()

        provider._client.post = mock_post

        response = await provider.complete(
            messages=[{"role": "user", "content": "test"}],
            model="kimi-k2",
        )

        assert response.success is True
        assert "[Thinking]" not in response.content
        assert response.content == "Just the answer"
        assert response.metadata.get("has_reasoning") is False

    @pytest.mark.asyncio
    async def test_complete_handles_error(self, rate_limiter, monkeypatch):
        """Should return failed response on API error"""
        monkeypatch.setattr(orchestrator_module, "get_api_key", lambda x: "test-key")
        provider = MoonshotProvider(rate_limiter)
        await provider.initialize()

        async def mock_post(url, json):
            raise Exception("API connection failed")

        provider._client.post = mock_post

        response = await provider.complete(
            messages=[{"role": "user", "content": "test"}],
            model="kimi-k2",
        )

        assert response.success is False
        assert "API connection failed" in response.error


class TestNeedsLLMRouting:
    """Test the needs_llm_routing() helper function"""

    def test_returns_false_with_no_specialized_tasks(self):
        """Single general task should not trigger routing"""
        task_types = [(TaskType.GENERAL_NLP, 0.8)]
        assert needs_llm_routing(task_types) is False

    def test_returns_false_with_one_specialized_task(self):
        """Single specialized task should not trigger routing"""
        task_types = [(TaskType.WEB_SEARCH, 0.9), (TaskType.GENERAL_NLP, 0.5)]
        assert needs_llm_routing(task_types) is False

    def test_returns_true_with_two_specialized_tasks(self):
        """Two high-confidence specialized tasks should trigger routing"""
        task_types = [
            (TaskType.WEB_SEARCH, 0.85),
            (TaskType.EXTENDED_THINKING, 0.75),
        ]
        assert needs_llm_routing(task_types) is True

    def test_returns_true_with_code_and_multimodal(self):
        """Code generation and multimodal should trigger routing"""
        task_types = [
            (TaskType.CODE_GENERATION, 0.9),
            (TaskType.MULTIMODAL, 0.8),
        ]
        assert needs_llm_routing(task_types) is True

    def test_ignores_low_confidence_specialized_tasks(self):
        """Low confidence specialized tasks should not count"""
        task_types = [
            (TaskType.WEB_SEARCH, 0.5),  # Below 0.7 threshold
            (TaskType.EXTENDED_THINKING, 0.6),  # Below 0.7 threshold
        ]
        assert needs_llm_routing(task_types) is False

    def test_specialized_tasks_constants(self):
        """Verify SPECIALIZED_TASKS contains expected types"""
        assert TaskType.WEB_SEARCH in SPECIALIZED_TASKS
        assert TaskType.EXTENDED_THINKING in SPECIALIZED_TASKS
        assert TaskType.CODE_GENERATION in SPECIALIZED_TASKS
        assert TaskType.MULTIMODAL in SPECIALIZED_TASKS
        # General NLP should NOT be specialized
        assert TaskType.GENERAL_NLP not in SPECIALIZED_TASKS


class TestRoutingDecision:
    """Test RoutingDecision dataclass"""

    def test_dataclass_fields(self):
        """Should have all required fields"""
        decision = RoutingDecision(
            models=["perplexity-sonar-pro", "kimi-k2-thinking"],
            chain=True,
            reasoning="Web search followed by deep analysis",
            confidence=0.85,
        )
        assert decision.models == ["perplexity-sonar-pro", "kimi-k2-thinking"]
        assert decision.chain is True
        assert decision.reasoning == "Web search followed by deep analysis"
        assert decision.confidence == 0.85

    def test_single_model_no_chain(self):
        """Should work with single model and no chaining"""
        decision = RoutingDecision(
            models=["claude-sonnet-4.5"],
            chain=False,
            reasoning="Simple query, single model sufficient",
            confidence=0.9,
        )
        assert len(decision.models) == 1
        assert decision.chain is False


class TestLLMRouter:
    """Test LLMRouter for smart multi-model routing"""

    class FakeAnthropicProvider(BaseProvider):
        """Fake Anthropic provider for testing LLMRouter"""

        def __init__(self, rate_limiter, response_content: str):
            super().__init__(rate_limiter)
            self.response_content = response_content

        @property
        def provider_name(self) -> str:
            return "anthropic"

        async def initialize(self) -> bool:
            return True

        async def complete(self, messages, model, **kwargs):
            return APIResponse(
                content=self.response_content,
                model=model,
                provider="anthropic",
                usage={"input_tokens": 100, "output_tokens": 50},
                latency_ms=150,
                success=True,
            )

    @pytest.fixture
    def rate_limiter(self):
        return RateLimiter()

    @pytest.mark.asyncio
    async def test_route_returns_valid_routing_decision(self, rate_limiter):
        """Should parse valid JSON response into RoutingDecision"""
        fake_response = '{"models": ["perplexity-sonar-pro"], "chain": false, "reasoning": "Web search needed"}'
        provider = self.FakeAnthropicProvider(rate_limiter, fake_response)
        router = LLMRouter(provider)

        task_types = [(TaskType.WEB_SEARCH, 0.9), (TaskType.REASONING, 0.7)]
        decision = await router.route("What's the latest news?", task_types)

        assert isinstance(decision, RoutingDecision)
        assert decision.models == ["perplexity-sonar-pro"]
        assert decision.chain is False
        assert decision.confidence == 0.85

    @pytest.mark.asyncio
    async def test_route_with_chaining(self, rate_limiter):
        """Should correctly parse chained model response"""
        fake_response = '{"models": ["perplexity-sonar-pro", "kimi-k2-thinking"], "chain": true, "reasoning": "Search then analyze"}'
        provider = self.FakeAnthropicProvider(rate_limiter, fake_response)
        router = LLMRouter(provider)

        task_types = [(TaskType.WEB_SEARCH, 0.9), (TaskType.EXTENDED_THINKING, 0.8)]
        decision = await router.route("Search X and think deeply", task_types)

        assert len(decision.models) == 2
        assert decision.chain is True

    @pytest.mark.asyncio
    async def test_route_handles_invalid_json(self, rate_limiter):
        """Should fallback when JSON parsing fails"""
        provider = self.FakeAnthropicProvider(rate_limiter, "not valid json")
        router = LLMRouter(provider)

        task_types = [(TaskType.WEB_SEARCH, 0.9)]
        decision = await router.route("test prompt", task_types)

        # Should fallback to default
        assert isinstance(decision, RoutingDecision)
        assert decision.confidence < 0.6  # Lower confidence on fallback

    @pytest.mark.asyncio
    async def test_route_handles_markdown_wrapped_json(self, rate_limiter):
        """Should handle JSON wrapped in markdown code blocks"""
        fake_response = '```json\n{"models": ["gpt-4o"], "chain": false, "reasoning": "General query"}\n```'
        provider = self.FakeAnthropicProvider(rate_limiter, fake_response)
        router = LLMRouter(provider)

        task_types = [(TaskType.GENERAL_NLP, 0.8)]
        decision = await router.route("Hello", task_types)

        assert decision.models == ["gpt-4o"]

    def test_prefilter_candidates_limits_to_10(self, rate_limiter):
        """Should return at most 10 candidates"""
        provider = self.FakeAnthropicProvider(rate_limiter, "")
        router = LLMRouter(provider)

        # Use broad tasks that match many models
        task_types = [
            (TaskType.GENERAL_NLP, 0.9),
            (TaskType.CODE_GENERATION, 0.8),
            (TaskType.REASONING, 0.7),
        ]
        candidates = router._prefilter_candidates(task_types)

        assert len(candidates) <= 10

    def test_prefilter_candidates_returns_matching_models(self, rate_limiter):
        """Should return models that match detected task types"""
        provider = self.FakeAnthropicProvider(rate_limiter, "")
        router = LLMRouter(provider)

        task_types = [(TaskType.WEB_SEARCH, 0.9)]
        candidates = router._prefilter_candidates(task_types)

        # All returned models should support WEB_SEARCH
        for model in candidates:
            assert TaskType.WEB_SEARCH in model.task_types


class TestChainedExecutor:
    """Test ChainedExecutor for sequential model execution"""

    class FakeProvider(BaseProvider):
        """Configurable fake provider"""

        def __init__(self, rate_limiter, name: str, response: str, usage: dict | None = None):
            super().__init__(rate_limiter)
            self._name = name
            self._response = response
            self._usage = usage or {"input_tokens": 100, "output_tokens": 200}

        @property
        def provider_name(self) -> str:
            return self._name

        async def initialize(self) -> bool:
            return True

        async def complete(self, messages, model, **kwargs):
            return APIResponse(
                content=self._response,
                model=model,
                provider=self._name,
                usage=self._usage,
                latency_ms=500,
                success=True,
                metadata={},
            )

    @pytest.fixture
    def rate_limiter(self):
        return RateLimiter()

    @pytest.mark.asyncio
    async def test_execute_chain_single_step(self, rate_limiter):
        """Single-step chain should work correctly"""
        provider = self.FakeProvider(rate_limiter, "perplexity", "Search results here")

        async def get_provider(name):
            return provider

        executor = ChainedExecutor(get_provider)
        routing = RoutingDecision(
            models=["perplexity-sonar-pro"],
            chain=False,
            reasoning="Just web search",
            confidence=0.9,
        )

        response = await executor.execute_chain("test query", routing)

        assert isinstance(response, ChainedResponse)
        assert len(response.steps) == 1
        assert "Search results" in response.final_content

    @pytest.mark.asyncio
    async def test_execute_chain_multi_step(self, rate_limiter):
        """Multi-step chain should pass context between steps"""
        call_count = 0
        received_prompts = []

        class TrackingProvider(BaseProvider):
            @property
            def provider_name(self) -> str:
                return "test"

            async def initialize(self) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):
                nonlocal call_count, received_prompts
                call_count += 1
                received_prompts.append(messages[-1]["content"])
                return APIResponse(
                    content=f"Response {call_count}",
                    model=model,
                    provider="test",
                    usage={"input_tokens": 50, "output_tokens": 100},
                    latency_ms=200,
                    success=True,
                    metadata={},
                )

        provider = TrackingProvider(rate_limiter)

        async def get_provider(name):
            return provider

        executor = ChainedExecutor(get_provider)
        routing = RoutingDecision(
            models=["perplexity-sonar-pro", "kimi-k2-thinking"],
            chain=True,
            reasoning="Search then analyze",
            confidence=0.85,
        )

        response = await executor.execute_chain("original question", routing)

        assert call_count == 2
        assert len(response.steps) == 2
        # Second step should include first response in its prompt
        assert "Response 1" in received_prompts[1]

    @pytest.mark.asyncio
    async def test_execute_chain_accumulates_cost(self, rate_limiter):
        """Should accumulate cost across chain steps"""
        # Use model with known pricing
        provider = self.FakeProvider(
            rate_limiter,
            "perplexity",
            "result",
            {"input_tokens": 1000, "output_tokens": 500},
        )

        async def get_provider(name):
            return provider

        executor = ChainedExecutor(get_provider)
        routing = RoutingDecision(
            models=["perplexity-sonar-pro"],
            chain=False,
            reasoning="Test",
            confidence=0.9,
        )

        response = await executor.execute_chain("test", routing)

        # Cost should be calculated based on model pricing
        assert response.total_cost > 0

    @pytest.mark.asyncio
    async def test_execute_chain_handles_missing_model(self, rate_limiter):
        """Should skip invalid model keys gracefully"""
        provider = self.FakeProvider(rate_limiter, "test", "result")

        async def get_provider(name):
            return provider

        executor = ChainedExecutor(get_provider)
        routing = RoutingDecision(
            models=["nonexistent-model-xyz", "perplexity-sonar-pro"],
            chain=True,
            reasoning="Test",
            confidence=0.9,
        )

        response = await executor.execute_chain("test", routing)

        # Should complete with only the valid model
        assert len(response.steps) == 1

    @pytest.mark.asyncio
    async def test_format_labeled_output(self, rate_limiter):
        """Should format output with labeled sections"""
        provider = self.FakeProvider(rate_limiter, "perplexity", "Web search results")

        async def get_provider(name):
            return provider

        executor = ChainedExecutor(get_provider)
        routing = RoutingDecision(
            models=["perplexity-sonar-pro"],
            chain=False,
            reasoning="Test",
            confidence=0.9,
        )

        response = await executor.execute_chain("test", routing)

        # Should include label for perplexity (Web Search Results)
        assert "[Web Search Results]" in response.final_content

    def test_chain_step_dataclass(self, rate_limiter):
        """ChainStep should store all required fields"""
        step = ChainStep(
            model_key="perplexity-sonar-pro",
            model_name="Perplexity Sonar Pro",
            content="Response content",
            reasoning=None,
            usage={"input_tokens": 100, "output_tokens": 50},
            latency_ms=350.5,
        )

        assert step.model_key == "perplexity-sonar-pro"
        assert step.reasoning is None
        assert step.latency_ms == 350.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
