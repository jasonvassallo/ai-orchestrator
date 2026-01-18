# Multi-Model Routing and Kimi K2 Integration Design

**Date**: 2026-01-18
**Status**: Approved
**Author**: AI Orchestrator Team

## Overview

This design adds intelligent multi-model routing to the AI Orchestrator, enabling:
1. Sequential model chaining (e.g., web search → deep analysis)
2. Moonshot Kimi K2 Thinking model integration for extended reasoning
3. Smart tiered routing using regex + LLM fallback
4. Comprehensive model metadata for routing decisions
5. Model currency auditing with passive update suggestions

## 1. New TaskType

Add `EXTENDED_THINKING` to the `TaskType` enum:

```python
class TaskType(Enum):
    # ... existing types ...
    EXTENDED_THINKING = auto()  # Deep reasoning with visible thought process
```

### TaskClassifier Patterns

```python
TaskType.EXTENDED_THINKING: [
    r"think.*(through|deeply|step.by.step|carefully)",
    r"(reason|explain).*(your|the).*(thought|reasoning|logic)",
    r"show.*your.*(work|thinking|reasoning)",
    r"(analyze|consider).*all.*(aspects|angles|possibilities)",
    r"work.*this.*out",
]
```

## 2. MoonshotProvider Implementation

### Provider Class

```python
class MoonshotProvider(BaseProvider):
    """Moonshot AI provider for Kimi K2 models."""

    provider_name = "moonshot"
    BASE_URL = "https://api.moonshot.ai/v1"

    async def initialize(self) -> bool:
        api_key = self._credential_manager.get_api_key("moonshot")
        if not api_key:
            return False

        # 25-minute timeout for thinking models
        timeout = httpx.Timeout(1500.0)
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        return True

    async def complete(self, messages, model, max_tokens, temperature) -> APIResponse:
        # For thinking models, use recommended temperature
        if "thinking" in model:
            temperature = 1.0

        response = await self._client.post("/chat/completions", json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        data = response.json()
        message = data["choices"][0]["message"]
        reasoning = message.get("reasoning_content")
        content = message.get("content", "")

        # Format with labeled sections
        if reasoning:
            content = f"[Thinking]\n{reasoning}\n\n[Answer]\n{content}"

        return APIResponse(content=content, ...)
```

### Model Registry Entries

```python
"kimi-k2-thinking": ModelCapability(
    name="Kimi K2 Thinking",
    provider="moonshot",
    model_id="kimi-k2-thinking",
    task_types=(TaskType.EXTENDED_THINKING, TaskType.DEEP_REASONING, TaskType.MATH, TaskType.CODE_GENERATION),
    context_window=256000,
    cost_per_1k_input=0.0006,
    cost_per_1k_output=0.0025,
    strengths=("extended thinking", "reasoning traces", "math", "complex analysis"),
    supports_extended_thinking=True,
    reasoning_token_limit=128000,
    latency_class="slow",
    knowledge_cutoff="2025-01",
),
"kimi-k2": ModelCapability(
    name="Kimi K2",
    provider="moonshot",
    model_id="kimi-k2",
    task_types=(TaskType.REASONING, TaskType.CODE_GENERATION, TaskType.GENERAL_NLP),
    context_window=256000,
    cost_per_1k_input=0.0003,
    cost_per_1k_output=0.0012,
    strengths=("reasoning", "coding", "general tasks"),
    latency_class="standard",
    knowledge_cutoff="2025-01",
),
```

## 3. Multi-Model Routing Architecture

### RoutingDecision Dataclass

```python
@dataclass
class RoutingDecision:
    """Result of the routing analysis."""
    models: list[str]           # Ordered list of model registry keys
    chain: bool                 # True = sequential execution
    reasoning: str              # Why this routing was chosen
    confidence: float           # 0.0-1.0
```

### Tiered Routing Flow

```
User Prompt
    │
    ▼
Tier 1: TaskClassifier.classify(prompt)
    │
    ├─── Single high-confidence task ──→ Standard select_model()
    │
    └─── Multiple specialized tasks ──→ Tier 2: LLM Router (Haiku)
                                              │
                                              ▼
                                        RoutingDecision
                                        (models, chain, reasoning)
```

### Escalation Logic

```python
def needs_llm_routing(task_types: list[tuple[TaskType, float]]) -> bool:
    SPECIALIZED_TASKS = {
        TaskType.WEB_SEARCH,
        TaskType.EXTENDED_THINKING,
        TaskType.CODE_GENERATION,
        TaskType.MULTIMODAL
    }

    high_confidence = [(t, c) for t, c in task_types if c >= 0.7]
    specialized_count = sum(1 for t, _ in high_confidence if t in SPECIALIZED_TASKS)

    return specialized_count >= 2
```

## 4. Enhanced ModelCapability Metadata

```python
@dataclass(frozen=True)
class ModelCapability:
    # Existing fields...
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

    # NEW: Enhanced metadata for smart routing
    knowledge_cutoff: str = "unknown"
    supports_web_search: bool = False
    supports_extended_thinking: bool = False
    reasoning_token_limit: int = 0
    latency_class: str = "standard"  # "instant", "standard", "slow"
    best_for: tuple[str, ...] = ()
    avoid_for: tuple[str, ...] = ()

    # NEW: For auto-update feature
    model_family: str = ""
    version_date: str = ""
    is_preview: bool = False
    successor_model: str | None = None
```

## 5. LLM Router with Pre-Filtered Candidates

```python
class LLMRouter:
    """Uses Claude Haiku with pre-filtered model candidates."""

    def _prefilter_candidates(
        self,
        task_types: list[tuple[TaskType, float]]
    ) -> list[ModelCapability]:
        """Pre-filter to ~5-10 relevant candidates based on task types."""
        candidates = []
        all_tasks = {t for t, _ in task_types}

        for model in ModelRegistry.MODELS.values():
            # Include if model supports any detected task
            if any(t in model.task_types for t in all_tasks):
                candidates.append(model)

        # Limit to top candidates by relevance
        return candidates[:10]

    async def route(
        self,
        prompt: str,
        task_types: list[tuple[TaskType, float]]
    ) -> RoutingDecision:
        candidates = self._prefilter_candidates(task_types)
        candidates_json = self._format_candidates_json(candidates)

        response = await self._haiku_client.complete(
            messages=[
                {"role": "system", "content": self.ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Models:\n{candidates_json}\n\nPrompt: {prompt[:500]}"}
            ],
            model="claude-3-5-haiku-latest",
            max_tokens=150,
        )

        return self._parse_routing_response(response)
```

## 6. Chained Execution

### ChainedExecutor Class

```python
@dataclass
class ChainStep:
    model_key: str
    model_name: str
    content: str
    reasoning: str | None
    usage: dict
    latency_ms: float

@dataclass
class ChainedResponse:
    steps: list[ChainStep]
    final_content: str
    total_latency_ms: float
    total_cost: float

class ChainedExecutor:
    async def execute_chain(
        self,
        prompt: str,
        routing: RoutingDecision,
        system_prompt: str | None = None,
    ) -> ChainedResponse:
        steps = []
        accumulated_context = prompt

        for i, model_key in enumerate(routing.models):
            model = ModelRegistry.get_model(model_key)
            provider = await self._get_provider(model.provider)

            # Build prompt with previous context
            if i > 0:
                step_prompt = self._build_chain_prompt(
                    original_prompt=prompt,
                    previous_output=steps[-1].content,
                    step_number=i + 1,
                )
            else:
                step_prompt = accumulated_context

            response = await provider.complete(...)
            steps.append(ChainStep(...))
            accumulated_context = response.content

        return ChainedResponse(
            steps=steps,
            final_content=self._format_labeled_output(steps),
            total_latency_ms=sum(s.latency_ms for s in steps),
            total_cost=self._calculate_total_cost(steps),
        )
```

### Labeled Output Format

```
[Web Search Results]
According to recent sources...
[Citations: ...]

---

[Deep Analysis]
<thinking>
The user wants implications analyzed...
</thinking>

Based on the latest developments...
```

## 7. Auto-Update Mechanism (Passive)

```python
class ModelRegistry:
    @classmethod
    def check_for_updates(cls) -> list[str]:
        """Return list of models that have successors available."""
        warnings = []
        for key, model in cls.MODELS.items():
            if model.successor_model and model.successor_model in cls.MODELS:
                warnings.append(
                    f"Model '{key}' has successor '{model.successor_model}' available. "
                    f"Consider updating config."
                )
        return warnings

    @classmethod
    def log_update_suggestions(cls) -> None:
        """Log warnings on startup for outdated models."""
        for warning in cls.check_for_updates():
            logger.warning(f"⚠️ {warning}")
```

## 8. Credentials Integration

### Add Moonshot to ENV_VAR_MAP

```python
# In EnvironmentBackend
ENV_VAR_MAP = {
    # ... existing ...
    "moonshot": "MOONSHOT_API_KEY",
}
```

### Add to Interactive Configuration

```python
# In configure_credentials_interactive()
providers = [
    # ... existing ...
    ("moonshot", "Moonshot (Kimi K2 Thinking)"),
]
```

### Storage Location

When using `python -m src.credentials`:
- **macOS**: Stored in Keychain under service `ai_orchestrator`, account `moonshot`
- **Fallback**: Environment variable `MOONSHOT_API_KEY`

## 9. Model Registry Updates Needed

Based on research (January 2026), the following models should be added or updated:

### New Models to Add

| Provider | Model | Priority |
|----------|-------|----------|
| OpenAI | GPT-5.2 (Instant/Thinking/Pro) | High |
| OpenAI | o3, o3-pro, o4-mini | High |
| xAI | Grok 3, Grok 4, Grok 4 Fast | Medium |
| Groq | Llama 4 Scout, Llama 4 Maverick | Medium |
| Perplexity | Sonar Reasoning, Sonar Reasoning Pro | High |
| DeepSeek | DeepSeek V3.2-exp | Medium |
| Mistral | Mistral Large 3, Codestral 2 | Low |
| Ollama | llama4:scout, qwen3 | Low |

### Existing Models - Pricing Updates

Several models have updated pricing that should be reflected in the registry.

## 10. Implementation Plan

1. **Phase 1: Core Infrastructure**
   - Add `EXTENDED_THINKING` TaskType
   - Add `MoonshotProvider` class
   - Add Moonshot to credentials system
   - Add enhanced `ModelCapability` fields

2. **Phase 2: Routing System**
   - Implement `RoutingDecision` dataclass
   - Implement `needs_llm_routing()` function
   - Implement `LLMRouter` with pre-filtering
   - Integrate into main `process()` method

3. **Phase 3: Chained Execution**
   - Implement `ChainedExecutor`
   - Implement labeled output formatting
   - Handle context passing between steps

4. **Phase 4: Model Updates**
   - Add new models to registry
   - Update pricing for existing models
   - Implement passive update warnings

5. **Phase 5: Testing**
   - Unit tests for new components
   - Integration tests for chained execution
   - End-to-end routing tests

## Appendix: Key Configuration Values

| Setting | Value |
|---------|-------|
| Thinking model timeout | 25 minutes (1500 seconds) |
| Kimi K2 Thinking temperature | 1.0 (recommended) |
| Router escalation threshold | 2+ specialized tasks at ≥0.7 confidence |
| Pre-filter candidate limit | 10 models |
| Router max tokens | 150 |
