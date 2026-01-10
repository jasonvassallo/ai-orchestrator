# GEMINI.md

This file provides context and guidance for Google's Gemini models when working with this repository.

## Project Overview

**AI Orchestrator** is an intelligent multi-model AI router written in Python. It automatically routes user queries to the best AI model based on task classification (e.g., coding, reasoning, creative writing). It supports **11 providers** (OpenAI, Anthropic, Google, Vertex AI, Mistral, Groq, xAI, Perplexity, DeepSeek, Ollama, MLX) with 25+ models, featuring secure credential management and multiple UI options.

Vertex preference for Gemini 3 Preview:
- The orchestrator maps `gemini-3-pro` → `vertex-gemini-3-pro` and `gemini-3-flash` → `vertex-gemini-3-flash` by default, leveraging Vertex AI limits when you have ADC configured.
- You can still explicitly call Google endpoints (`gemini-3-pro-preview`, `gemini-3-flash-preview`) if needed.

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

> **Vertex Credentials:** Vertex AI uses Google Application Default Credentials, not a `GEMINI_API_KEY`. Run:
>
> ```bash
> gcloud auth application-default login
> export GOOGLE_CLOUD_PROJECT="your-gcp-project"
> export GOOGLE_CLOUD_LOCATION="global"
> ```

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

### MLX Cache Consistency

To avoid redundant downloads and ensure MLX loads fully offline, set a single cache home:

```bash
export HF_HOME="$HOME/Library/Caches/huggingface"
```
The orchestrator searches both `~/.cache/huggingface/hub` and `~/Library/Caches/huggingface/hub` (and respects `HF_HOME`) and enables offline mode when `*.safetensors` are present.
```

### Testing & Quality

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run a single test file
pytest tests/test_orchestrator.py -v

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
| `src/music.py` | MIDI and audio file generation |
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
3. **Smart Cache Detection** → Check if local model (MLX/MusicGen) exists → Enable `HF_HUB_OFFLINE`
4. Task types → `AIOrchestrator.select_model()` → Best `ModelCapability`
5. Model → `AIOrchestrator._get_provider()` → Provider instance (lazy init)
6. Provider → `RetryHandler.execute_with_retry()` → `APIResponse`

### Local Model Optimization (Smart Cache)

To prevent unwanted 5GB+ downloads and ensure privacy, the orchestrator implements **Smart Cache Detection**:
- **MLX/MusicGen:** Checks the Hugging Face cache for the model ID before loading.
- **Offline Mode:** If found locally, it sets `HF_HUB_OFFLINE=1` to bypass network checks and load instantly from disk.
- **On-Demand:** If the model is missing, it reverts to online mode to allow the download after informing the user.

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
1. **Task type match** (primary factor)
2. **Provider characteristics** (e.g., creativity, reasoning depth)
3. **Avoid-for penalties** (weakness avoidance)
4. **Cost optimization** (if enabled)
5. **Context window bonus** (for long contexts)
6. **Local model bonus** (if local preference enabled)

### Async Pattern
All provider calls are async using `asyncio` and `httpx`.

## Coding Style & Standards

- **Python:** 3.10+ required.
- **Formatting:** Ruff (compatible with Black).
- **Linting:** Ruff.
- **Type Checking:** MyPy (strict for core modules: `orchestrator.py`, `credentials.py`, `storage.py`).
- **Naming:** `snake_case` for modules/functions, `CamelCase` for classes.
- **Tests:** `test_*.py` files with `Test*` classes.

## Commit & PR Guidelines

- **Messages:** Short, imperative, present tense with component prefix.
  - Example: `orchestrator: fix model selection bug`
- **PRs:** Mention test results, link relevant issues.
- **UI Changes:** Include screenshots or short clips.

## Related Documentation

For additional context, refer to:
- `CLAUDE.md`: Context specific to Anthropic's Claude.
- `AGENTS.md`: General context for generic AI agents.

## Gemini Added Memories
- Always update GEMINI.md with any code changes or documentation updates, and remind the user to commit files at appropriate times.
- Current date context: January 2026.