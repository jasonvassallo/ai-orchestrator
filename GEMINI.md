# GEMINI.md

This file provides context and guidance for Google's Gemini models when working with this repository.

## Project Overview

AI Orchestrator is an intelligent multi-model AI router written in Python. It automatically routes user queries to the best AI model based on task classification (e.g., coding, reasoning, creative writing, extended thinking). It supports 11 providers (OpenAI, Anthropic, Google/Vertex AI, Mistral, Groq, xAI, Perplexity, DeepSeek, Moonshot, Ollama, MLX) with 25+ models, featuring secure credential management, multi-model chaining, and multiple UI options (CLI, GUI, TUI, Menu Bar, VS Code Extension).

## Common Commands

### Setup & Installation

```bash
# Create and activate virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install with all features (GUI, TUI, all providers)
pip install -e ".[all]"

# Install minimal (CLI only)
pip install -e "."

# Install dev dependencies (test, lint, type-check)
pip install -e ".[dev]"
```

> **Note:** Always ensure your virtual environment is activated (`source .venv/bin/activate`) before running commands, or use the direct path (e.g., `./.venv/bin/python`, `./.venv/bin/ruff`).

### Running Applications

```bash
# CLI usage
python -m src.orchestrator "Your prompt here"
python -m src.orchestrator "Prompt" --model claude-sonnet-4.5  # Model override
python -m src.orchestrator "Prompt" --local                    # Prefer local (Ollama/MLX)
python -m src.orchestrator "Prompt" --cheap                    # Cost optimize
python -m src.orchestrator "Prompt" --verbose                  # Debug mode

# GUI app
python -m src.gui.app      # or: ai-app

# Terminal UI
python -m src.tui.app      # or: ai-chat

# Menu bar app
python -m src.menubar.app  # or: ai-menubar

# Manage local models (cleanup/download)
python -m src.manage_models
python -m src.manage_models --yes           # Non-interactive download/cleanup
python -m src.manage_models --yes --no-clean  # Skip cache cleanup

# Configure credentials
python -m src.credentials  # or: ai-configure
```

### Testing & Quality

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run a single test file
pytest tests/test_orchestrator.py -v

# Run a specific test
pytest tests/test_orchestrator.py::TestInputValidator::test_valid_prompt -v

# Type checking (strict for core modules)
mypy src

# Linting
ruff check src

# Formatting
ruff format .          # Apply formatting
ruff format --check .  # Check only
```

### Building Distributables

```bash
# Build macOS .app bundle
python setup_app.py py2app

# Build VS Code extension
cd vscode-extension && npm install && npm run package
```

## Architecture

### Core Components

| Path | Description |
| :--- | :--- |
| `src/orchestrator.py` | Main logic: routing, providers, task classification |
| `src/credentials.py` | Security-critical: API key management |
| `src/storage.py` | SQLite persistence for conversation history |
| `src/music.py` | MIDI and audio file generation (MusicGen with selectable models) |
| `src/manage_models.py` | Utility for managing local AI models (disk space) |
| `src/gui/` | PySide6 (Qt) desktop application |
| `src/tui/` | Textual terminal interface |
| `src/menubar/` | rumps macOS menu bar utility |
| `vscode-extension/` | JavaScript VS Code extension |
| `config/` | Configuration schema and samples |
| `tests/` | Unit and integration tests |

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
│   ├── DeepSeekProvider
│   └── MoonshotProvider         - Kimi K2 extended thinking models
├── LLMRouter                    - Smart multi-model routing decisions
├── ChainedExecutor              - Sequential multi-model execution
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
3. **Smart Cache Detection** → Check if local model (MLX/MusicGen) exists → Enable `HF_HUB_OFFLINE`
4. Task types → `needs_llm_routing()` → Check if multiple specialized tasks detected
5. If single task: `AIOrchestrator.select_model()` → Best `ModelCapability`
6. If multi-task: `LLMRouter.route()` → `RoutingDecision` (models list, chain flag)
7. If chaining: `ChainedExecutor.execute_chain()` → Sequential model execution with context passing
8. Model → `AIOrchestrator._get_provider()` → Provider instance (lazy init)
9. Provider → `RetryHandler.execute_with_retry()` → `APIResponse` (or `ChainedResponse` for chains)

### UI Layer

All UIs use the same `AIOrchestrator` core:
- **GUI** (`src/gui/`): PySide6 (Qt) with async via qasync
- **TUI** (`src/tui/`): Textual framework
- **Menu Bar** (`src/menubar/`): rumps for macOS
- **VS Code** (`vscode-extension/`): JavaScript extension with keyboard shortcuts

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
7. Update `src/credentials.py` to handle the new provider's API key (if cloud-based):
   - Add env var mapping to `EnvironmentBackend.ENV_VAR_MAP`

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

### Multi-Model Routing and Chaining

The orchestrator supports intelligent multi-model routing for complex tasks:

1. **Tiered Routing**: Regex-based `TaskClassifier` runs first (free, instant). If multiple specialized tasks are detected at ≥0.7 confidence, escalates to `LLMRouter` (Claude Haiku) for smart routing decisions.

2. **Specialized Task Types**: `WEB_SEARCH`, `EXTENDED_THINKING`, `CODE_GENERATION`, `MULTIMODAL` trigger multi-model consideration.

3. **Chaining**: When `RoutingDecision.chain=True`, `ChainedExecutor` runs models sequentially, passing context between steps. Example: Perplexity (web search) → Kimi K2 (deep analysis).

4. **Labeled Output**: Chained responses use section labels like `[Web Search Results]`, `[Thinking]`, `[Answer]` for transparency.

### Extended Thinking (Kimi K2)

Moonshot's Kimi K2 Thinking model provides visible reasoning traces:
- `reasoning_content` field in API response contains the thinking process
- 25-minute timeout for complex reasoning tasks
- Temperature 1.0 recommended for thinking models
- Supports `TaskType.EXTENDED_THINKING` classification

### Local Model Optimization (Smart Cache)

To prevent unwanted 5GB+ downloads and ensure privacy, the orchestrator implements **Smart Cache Detection**:
- **MLX/MusicGen:** Checks the Hugging Face cache for the model ID before loading
- **Offline Mode:** If found locally, sets `HF_HUB_OFFLINE=1` to bypass network checks and load instantly from disk
- **On-Demand:** If the model is missing, reverts to online mode to allow download after informing the user

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

## Coding Style

- **Python:** 3.10+ required, 4-space indentation
- **Formatting:** Ruff (compatible with Black)
- **Linting:** Ruff
- **Type Checking:** MyPy (strict for `orchestrator.py`, `credentials.py`, `storage.py`)
- **Naming:** `snake_case` for modules/functions, `CamelCase` for classes
- **Tests:** `test_*.py` files with `Test*` classes

## Testing Notes

- Tests use mocking; no real API keys needed
- `pytest-asyncio` with `asyncio_mode = "auto"` for async tests
- Security compliance tests verify no hardcoded API keys in source
- Local provider tests accept both "ollama" and "mlx" as valid

## Commit & PR Guidelines

- **Messages:** Short, imperative, present tense with component prefix
  - Example: `orchestrator: fix model selection bug`
- **PRs:** Mention test results, link relevant issues
- **UI Changes:** Include screenshots or short clips

## Security

- **Never commit API keys or secrets**
- Use `python -m src.credentials` or environment variables to configure providers
- Refer to `config/config.sample.json` for expected configuration shape

## Related Files

- `AGENTS.md` - General context for AI agents (OpenAI/Generic)
- `GEMINI.md` - Context specific to Google's Gemini (this file)
- `CLAUDE.md` - Context specific to Anthropic's Claude

## Gemini Added Memories
- Always update GEMINI.md with any code changes or documentation updates, and remind the user to commit files at appropriate times.
- Current date context: End of December 2025.
- The user's domain is djvassallo.com, intended for electronic music production and sales.
- User is setting up Cloudflare Zero Trust for djvassallo.com and encountered a conflict between Cloudflare WARP and iCloud Private Relay on macOS.
- User intends to host djvassallo.com on a different machine than the one currently being configured with WARP.
- The user plans to use an old Mac Mini as a dedicated server for hosting djvassallo.com.
