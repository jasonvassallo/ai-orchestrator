# Repository Guidelines

## Project Structure & Module Organization

The core Python package lives in `src/`, with `src/orchestrator.py` as the main entry point, `src/credentials.py` for secure key handling, `src/storage.py` for persistence, `src/music.py` for MIDI and MusicGen audio generation, and `src/manage_models.py` for local model cleanup/download management. UI clients are organized under `src/gui/` (PySide6), `src/tui/` (Textual), and `src/menubar/` (rumps). Configuration lives in `config/` (`config-schema.json`, `config.sample.json`), tests in `tests/`, helper scripts in `scripts/`, and the VS Code extension under `vscode-extension/`. Build artifacts land in `dist/` and `build/`.

## Build, Test, and Development Commands

- `pip install -e "."` installs the minimal CLI-only dependencies.
- `pip install -e ".[dev]"` installs test, lint, and type-check tooling.
- `pip install -e ".[all]"` installs all providers and UI dependencies.
- `python -m src.orchestrator "..."` or `ai-orchestrator "..."` runs the CLI.
- `ai-app`, `ai-chat`, and `ai-menubar` run the GUI, TUI, and menu bar apps.
- `python -m src.gui.app`, `python -m src.tui.app`, `python -m src.menubar.app` run UI apps directly.
- `python -m src.manage_models` manages local model cache (disk cleanup).
- `python -m src.credentials` or `ai-configure` configures API credentials.
- `python setup_app.py py2app` builds the macOS `.app` bundle.
- `cd vscode-extension && npm install && npm run package` builds the VS Code extension.
- `pytest` runs all tests; `pytest --cov=src --cov-report=html` generates coverage.
- `mypy src` runs type checks (strict for core modules).
- `ruff check src` lints; `ruff format .` formats.

> Note: Activate the virtual environment (`source .venv/bin/activate`) before running CLI tools, or use the direct venv paths (e.g., `./.venv/bin/python`).

## Architecture Overview

The orchestrator supports 11 providers: OpenAI, Anthropic, Google (Gemini & Vertex AI), Mistral, Groq, xAI, Perplexity, DeepSeek, Moonshot, Ollama, and MLX (Apple Silicon local inference). All providers inherit from `BaseProvider` with `provider_name`, `initialize()`, and `complete()` methods.

Key components in `src/orchestrator.py`:
- `TaskType` enum classifies prompts (CODE_GENERATION, REASONING, CREATIVE_WRITING, etc.)
- `TaskClassifier` classifies prompts into task types with confidence scores
- `ModelCapability` dataclass defines model attributes (task_types, costs, context_window)
- `ProviderCharacteristics` dataclass captures provider strengths/weaknesses for intelligent selection
- `PROVIDER_CHARACTERISTICS` registry maps providers to numeric scores (contextual_understanding, creativity_originality, code_quality, etc.)
- `ModelRegistry.MODELS` dict contains 25+ model definitions
- `AIOrchestrator.select_model()` uses multi-factor scoring: task match, provider characteristics, cost optimization, context window bonus, and local model bonus
- `LLMRouter` and `ChainedExecutor` enable multi-model routing and sequential chaining
- `RateLimiter`, `RetryHandler`, and `InputValidator` enforce throttling, retries, and prompt safety

Local providers (`ollama`, `mlx`) are recognized via `local_providers` set in `get_models_for_task()`. `prefer_local=True` filters to these providers and `TaskType.LOCAL_MODEL` gives them a scoring bonus.

## UI Layer

All UI clients share the same `AIOrchestrator` core:
- `src/gui/` uses PySide6 (Qt)
- `src/tui/` uses Textual
- `src/menubar/` uses rumps
- `vscode-extension/` provides VS Code integration

## Storage

Conversation history is persisted via SQLite in `~/.ai_orchestrator/conversations.db` using `ConversationStorage` in `src/storage.py`.

## Credential Management

Credentials fall back in this order:
1. System keyring (macOS Keychain, Windows Credential Manager, Linux Secret Service)
2. Encrypted file (`~/.ai_orchestrator/credentials.enc`)
3. Environment variables

## Data Flow

1. Prompt validation via `InputValidator.validate_prompt()`
2. Task classification via `TaskClassifier.classify()`
3. Optional smart cache detection for local models
4. If multi-task, `needs_llm_routing()` uses `LLMRouter.route()` and optional `ChainedExecutor` sequencing
5. If single-task, `AIOrchestrator.select_model()` chooses the best `ModelCapability`
6. Provider initialization via `AIOrchestrator._get_provider()`
7. Request execution with retries via `RetryHandler.execute_with_retry()` (retryable APIResponse errors and Vertex AI HTTP failures are surfaced for backoff)

## Local Model Optimization (Smart Cache)
For MLX/MusicGen, the orchestrator checks the local Hugging Face cache first across multiple roots:

- Respects `HF_HOME` (uses `HF_HOME/hub`)
- Scans `~/Library/Caches/huggingface/hub` (macOS)
- Scans `~/.cache/huggingface/hub` (Unix default)

If a snapshot containing `*.safetensors` exists, it sets `HF_HUB_OFFLINE=1` and loads directly from disk. If only an index is present (missing shards), it enables ONLINE mode to allow fetching missing files.

Tip: set `HF_HOME="$HOME/Library/Caches/huggingface"` to keep a single consistent cache on macOS.

## Model Selection Algorithm

Scoring factors include task match, provider characteristics, avoid-for penalties, cost optimization, context window bonuses, a local model bonus when `prefer_local` is enabled, and a small bias favoring Vertex AI for Gemini 3 Preview models when available.

Auto routing also supports subscription-first preferences via `defaults.preferSubscriptionProviders` and `defaults.preferSubscriptionModels`.
Web search defaults to Perplexity Sonar (reasoning-pro when reasoning is detected), and advanced math/logic/coding prefers Kimi K2 Thinking while long-context general reasoning prefers Gemini 3 Pro.
Auto-routing logs include prompt length + hash only (no prompt/response content).

## Async Pattern

Providers use async calls with `asyncio` and `httpx`, with SDK async clients where available.

## Configuration

User config: `~/.ai_orchestrator/config.json`. Schema and samples live in `config/`.

## Coding Style & Naming Conventions

Python 3.10+ is required. Use 4-space indentation and format with Ruff (compatible with Black). Lint with Ruff and type-check with MyPy (strict for `src/orchestrator.py`, `src/credentials.py`, `src/storage.py`). Use `snake_case` for modules/functions and `CamelCase` for classes. Tests follow `test_*.py` naming with `Test*` classes.

## Testing Guidelines

Tests run with Pytest and pytest-asyncio and are located in `tests/`. Run `pytest` for the full suite, or `pytest --cov=src --cov-report=html` for coverage reports. Type checking and linting are run via `mypy src` and `ruff check src`. Local provider tests accept both "ollama" and "mlx" as valid providers.

## Adding a New Provider

1. Create class extending `BaseProvider` in `src/orchestrator.py`
2. Implement `provider_name`, `initialize()`, and `complete()` methods
3. Add to `_get_provider()` factory method
4. Add model entries to `ModelRegistry.MODELS`
5. Add `ProviderCharacteristics` entry to `PROVIDER_CHARACTERISTICS`
6. If local, add to `local_providers` set in `get_models_for_task()`
7. If cloud-based, add env var mapping to `EnvironmentBackend.ENV_VAR_MAP` in credentials.py

## Commit & Pull Request Guidelines

Commit messages in this repo are short and imperative; component prefixes like `orchestrator:` are common. Keep commits focused, and mention test results in the PR description. Link relevant issues, and include screenshots or short clips for UI changes.

## Security & Configuration Tips

Never commit API keys or secrets. Use `python -m src.credentials` or environment variables to configure providers. Refer to `config/config.sample.json` and `config/config-schema.json` for expected configuration shape.
