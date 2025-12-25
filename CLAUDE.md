# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Orchestrator is a multi-model AI router that intelligently routes queries to the best model based on task type. It supports 10 providers (OpenAI, Anthropic, Google, Mistral, Groq, xAI, Perplexity, DeepSeek, Ollama, MLX) with 25+ models, featuring secure credential management and multiple UI options.

## Common Commands

### Development

```bash
# Install with all features
pip install -e ".[all]"

# Install minimal (CLI only)
pip install -e "."

# Install dev dependencies
pip install -e ".[dev]"
```

### Running Applications

```bash
# CLI usage
python -m src.orchestrator "Your prompt here"
python -m src.orchestrator "Prompt" --model claude-sonnet-4.5  # Model override
python -m src.orchestrator "Prompt" --local                    # Prefer local (Ollama/MLX)
python -m src.orchestrator "Prompt" --cheap                    # Cost optimize

# GUI app
python -m src.gui.app      # or: ai-app

# Terminal UI
python -m src.tui.app      # or: ai-chat

# Menu bar app
python -m src.menubar.app  # or: ai-menubar

# Configure credentials
python -m src.credentials  # or: ai-configure
```

### Testing & Quality

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_orchestrator.py -v

# Run a specific test
pytest tests/test_orchestrator.py::TestInputValidator::test_valid_prompt -v

# Type checking (only checks core modules)
mypy src

# Linting
ruff check src

# Format check
black --check src
```

## Architecture

### Core Module Design

The codebase follows a provider abstraction pattern where all AI providers inherit from `BaseProvider`:

```
src/orchestrator.py
├── TaskType (Enum)              - Classification categories (CODE_GENERATION, REASONING, etc.)
├── ModelCapability (dataclass)  - Model definition with task_types, costs, strengths
├── ProviderCharacteristics      - Provider-level strengths/weaknesses for intelligent selection
├── TaskClassifier               - Regex-based prompt classification → TaskType
├── ModelRegistry                - Static registry of 25+ models with capabilities
├── PROVIDER_CHARACTERISTICS     - Registry mapping providers to their characteristics
├── BaseProvider (ABC)           - Abstract provider interface
│   ├── OpenAIProvider
│   ├── AnthropicProvider
│   ├── GoogleProvider
│   ├── VertexAIProvider
│   ├── OllamaProvider
│   ├── MLXProvider              - Apple Silicon local inference via CLI
│   ├── MistralProvider
│   ├── GroqProvider
│   ├── XAIProvider
│   ├── PerplexityProvider
│   └── DeepSeekProvider
├── RateLimiter                  - Token bucket rate limiting per provider
├── RetryHandler                 - Exponential backoff with jitter
├── InputValidator               - Security validation and sanitization
└── AIOrchestrator               - Main orchestrator class
```

### Credential Management (`src/credentials.py`)

Three-tier fallback chain with priority order:
1. **System Keyring** (macOS Keychain, Windows Credential Manager, Linux Secret Service)
2. **Encrypted File** (`~/.ai_orchestrator/credentials.enc`) using Fernet/PBKDF2
3. **Environment Variables** (fallback)

### Data Flow

1. User prompt → `InputValidator.validate_prompt()` → Security checks
2. Prompt → `TaskClassifier.classify()` → List of (TaskType, confidence) tuples
3. Task types → `AIOrchestrator.select_model()` → Best `ModelCapability`
4. Model → `AIOrchestrator._get_provider()` → Provider instance (lazy init)
5. Provider → `RetryHandler.execute_with_retry()` → `APIResponse`

### UI Layer

All UIs use the same `AIOrchestrator` core:
- **GUI** (`src/gui/`): PySide6 with async via qasync
- **TUI** (`src/tui/`): Textual framework
- **Menu Bar** (`src/menubar/`): rumps for macOS

### Storage

SQLite-based conversation persistence in `~/.ai_orchestrator/conversations.db`:
- `ConversationStorage` class manages CRUD operations
- Messages linked to conversations via foreign key

## Key Patterns

### Adding a New Provider

1. Create a class extending `BaseProvider` in `src/orchestrator.py`
2. Implement `provider_name`, `initialize()`, and `complete()` methods
3. Add provider to `_get_provider()` factory method
4. Add model entries to `ModelRegistry.MODELS` dict
5. Add `ProviderCharacteristics` entry to `PROVIDER_CHARACTERISTICS` dict
6. If local provider, add to `local_providers` set in `get_models_for_task()`
7. Add env var mapping to `EnvironmentBackend.ENV_VAR_MAP` in credentials.py (if cloud-based)

### Model Selection Algorithm

Models are scored using a multi-factor system in `select_model()`:

1. **Task type match** (primary factor, +10 per matched task × confidence)
2. **Provider characteristics** - Weighted scores from `ProviderCharacteristics`:
   - `contextual_understanding`, `creativity_originality`, `emotional_intelligence`
   - `speed_efficiency`, `knowledge_breadth`, `reasoning_depth`, `code_quality`, `objectivity`
3. **Avoid-for penalties** (-2.0 when task matches provider's known weaknesses)
4. **Cost optimization** (if enabled, penalizes expensive models)
5. **Context window bonus** (for LONG_CONTEXT tasks)
6. **Local model bonus** (+5.0 when LOCAL_MODEL task detected and provider is local)

The `_get_task_score_weights()` method maps TaskTypes to relevant characteristic weights.

### Local Providers

Both `ollama` and `mlx` are recognized as local providers:
- `prefer_local=True` filters to only these providers
- `TaskType.LOCAL_MODEL` detection gives them a scoring bonus
- MLX is optimized for Apple Silicon via command-line invocation

### Async Pattern

All provider calls are async. The orchestrator uses:
- `asyncio` for async operations
- `httpx.AsyncClient` for HTTP-based providers
- Provider SDKs' async clients where available
- `asyncio.to_thread()` for MLX subprocess calls

## Configuration

- User config: `~/.ai_orchestrator/config.json`
- Schema: `config/config-schema.json`
- Sample: `config/config.sample.json`

## Testing Notes

- Tests use mocking; no real API keys needed
- `pytest-asyncio` with `asyncio_mode = "auto"` for async tests
- Security compliance tests verify no hardcoded API keys in source
- Local provider tests accept both "ollama" and "mlx" as valid
