# AI Orchestrator

**Intelligent Multi-Model AI Router with Secure Credential Management**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Security: Hardened](https://img.shields.io/badge/security-hardened-green.svg)](https://github.com/jasonvassallo/ai-orchestrator/security)

AI Orchestrator automatically routes your queries to the best AI model based on task type. It supports **10 providers** including OpenAI, Anthropic Claude, Google (Gemini & Vertex AI), Mistral, Groq, xAI (Grok), Perplexity, DeepSeek, Moonshot (Kimi K2), and MLX (Apple Silicon optimized) - all with secure credential management and production-ready features.

## Features

- **Native Mac App**: Beautiful ChatGPT-like desktop application
- **Menu Bar App**: Quick access from your Mac's menu bar
- **Terminal UI**: Gorgeous terminal-based interface
- **Intelligent Routing**: Automatically selects the best model based on task classification
- **Multi-Model Chaining**: Sequential model execution for complex tasks (e.g., web search → deep analysis)
- **Extended Thinking**: Kimi K2 Thinking integration with visible reasoning traces
- **Secure Credentials**: API keys stored in OS keychain or encrypted file - NEVER in code
- **High Performance**: Async operations, rate limiting, and retry logic with exponential backoff
- **Vertex AI Resilience**: Structured retryable 429/resource-exhausted handling with clearer error messages
- **Security Hardened**: Input validation, audit logging, no credential leakage
- **VS Code Extension**: Full-featured extension with keyboard shortcuts
- **Local Models**: Privacy-first option with MLX on Apple Silicon
- **Cost Optimization**: Route to cheaper models when appropriate
- **Web Search**: Perplexity integration for real-time information
- **Image Generation**: DALL-E integration for creating images
- **Music Generation**: Create MIDI and audio files with AI (separate drums/bass/chords tracks, 90s tech-house patterns)
- **Animated Status**: Real-time processing indicators across all UIs (validating, routing, generating, etc.)
- **Conversation Export**: Export to markdown or JSON from all interfaces (CLI: `-o`, TUI: Ctrl+E, GUI: Ctrl/Cmd+E)
- **Model Attribution**: Every response includes which model generated it for transparency
- **Incognito Mode**: Disable history saving while preserving existing context
- **Conversation Compaction**: Summarize long conversations to save context tokens (TUI: Ctrl+K, GUI: Ctrl+K, Menu Bar: Compact History)
- **Audit Timestamps**: All messages and responses include ISO timestamps for security analysis
- **Security Analyzer**: Post-incident detection of prompt injection, jailbreaks, and response leakage with severity-based logging

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jasonvassallo/ai-orchestrator.git
cd ai-orchestrator

# Install with all providers and GUI apps
pip install -e ".[all]"

# Or install specific components
pip install -e "."                      # Minimal (CLI only)
pip install -e ".[openai,anthropic]"    # Specific providers
pip install -e ".[gui]"                 # Native Mac app
pip install -e ".[tui]"                 # Terminal UI
pip install -e ".[mlx]"                 # MLX local models (Apple Silicon)
pip install -e ".[ui]"                  # All UI options

# Using uv (recommended for fast Python + venv management)
# Uses .python-version for the interpreter
uv venv .venv
uv pip install -e ".[all]"
```

### Configure Credentials (REQUIRED)

**NEVER store API keys in code or config files!**

```bash
# Interactive configuration (recommended)
python -m src.credentials

# Or set environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export MISTRAL_API_KEY="..."
export GROQ_API_KEY="..."
export XAI_API_KEY="..."
export PERPLEXITY_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export MOONSHOT_API_KEY="..."
```

Vertex AI models use Application Default Credentials (ADC) instead of an API key:
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-gcp-project"
export GOOGLE_CLOUD_LOCATION="global"  # or a region like us-central1
```

### Basic Usage

```bash
# CLI usage
python -m src.orchestrator "Explain quantum computing"

# With model override
python -m src.orchestrator "Debug this Python code" --model claude-sonnet-4.5

# Prefer local models (MLX)
python -m src.orchestrator "Summarize this text" --local

# Cost optimization mode
python -m src.orchestrator "Write a haiku" --cheap

# Export response to file (markdown or JSON)
python -m src.orchestrator "Summarize AI news" -o response.md
python -m src.orchestrator "Explain quantum computing" -o response.json

# Verbose mode for debugging
python -m src.orchestrator "Complex analysis task" --verbose
```

### Python API

```python
import asyncio
from src.orchestrator import AIOrchestrator

async def main():
    orchestrator = AIOrchestrator(
        prefer_local=False,
        cost_optimize=False,
        verbose=True
    )

    # Simple query
    response = await orchestrator.query(
        "Write a Python function to sort a list"
    )
    print(response.content)

    # With system prompt
    response = await orchestrator.query(
        "Review this code for bugs",
        system_prompt="You are an expert code reviewer focusing on security.",
        model_override="claude-opus-4.5"
    )
    print(response.content)

    # Multi-model comparison
    responses = await orchestrator.multi_model_query(
        "Explain machine learning",
        models=["gpt-4o", "claude-sonnet-4.5", "gemini-2.0-flash"]
    )
    for model, resp in responses.items():
        print(f"\n{model}:\n{resp.content[:200]}...")

asyncio.run(main())
```

## Utilities

Helper scripts for Gemini CLI troubleshooting live in `scripts/`:

- `scripts/gemini-retry.sh`: Wraps `gemini` to retry 429/resource-exhausted responses with backoff.
- `scripts/gemini-429-diagnose.sh`: Parses Gemini debug logs to classify rate-limit vs quota exhaustion signals.

## Model Management

Use the model manager to check/download recommended local models and clean cache:

```bash
python -m src.manage_models
python -m src.manage_models --yes
python -m src.manage_models --yes --no-clean
```

It includes MLX Qwen3 4B, MLX Qwen 2.5 Coder 14B, MLX Llama 3.2 11B Vision, MLX Ministral 14B Reasoning, plus MusicGen. If `hf-transfer` is installed, downloads will use it automatically. For targeted removals, use `hf cache rm <repo_id>`.

## Mac Applications

AI Orchestrator includes three beautiful interface options:

### Native Mac App (ChatGPT-like)

A full-featured desktop application with:
- Conversation history sidebar
- Model selector dropdown
- Feature toggles (Think, Web, Research, Image, Music)
- Streaming responses
- Markdown rendering with syntax highlighting

```bash
# Run the GUI app
ai-app
# or
python -m src.gui.app
```

### Menu Bar App

Quick access from your Mac's menu bar:
- Always accessible
- Quick query popup
- Model switching
- Launch other apps

```bash
# Run the menu bar app
ai-menubar
# or
python -m src.menubar.app
```

### Terminal UI

Beautiful terminal-based interface:
- Modern, stylish design
- Keyboard-driven
- Works over SSH

```bash
# Run the terminal UI
ai-chat
# or
python -m src.tui.app
```

### Building the .app Bundle

Create a standalone macOS application:

```bash
# Install build dependencies
pip install -e ".[dev]"

# Build the .app bundle
python setup_app.py py2app

# Output: dist/AI Orchestrator.app
```

## Supported Models (25+)

### Anthropic (Claude)

| Model | Best For | Context | Strengths |
|-------|----------|---------|-----------|
| `claude-opus-4.5` | Complex tasks, coding | 200K | Most intelligent, nuanced writing |
| `claude-sonnet-4.5` | Everyday coding | 200K | Balanced, fast, great for code |
| `claude-haiku-4.5` | Simple tasks | 200K | Very fast, cost-effective |

### OpenAI (GPT)

| Model | Best For | Context | Strengths |
|-------|----------|---------|-----------|
| `gpt-4o` | General purpose | 128K | Multimodal, fast |
| `gpt-4o-mini` | Simple tasks | 128K | Cost-effective |
| `o1` | Math, reasoning | 200K | Deep reasoning |
| `o1-mini` | Coding, reasoning | 128K | Balanced reasoning |

### Google (Gemini)

| Model | Best For | Context | Strengths |
|-------|----------|---------|-----------|
| `gemini-3-pro` | Complex tasks, coding | 2M | Next-gen intelligence, massive context |
| `gemini-3-flash` | Speed, multimodal | 1M | Next-gen speed, cost-effective |
| `gemini-2.5-pro` | Reasoning, coding | 2M | Advanced reasoning, stable |
| `gemini-2.5-flash` | Speed, production | 1M | Fast, stable, efficient |
| `gemini-2.0-flash` | Speed, long docs | 1M | Massive context, fast |
| `gemini-1.5-pro` | Very long docs | 2M | Largest context window |

### Vertex AI (Google Cloud)

| Model | Best For | Context | Strengths |
|-------|----------|---------|-----------|
| `vertex-gemini-3-pro` | Enterprise, complex tasks | 1M | Enterprise-grade, latest Gemini |
| `vertex-gemini-3-flash` | Enterprise, speed | 1M | Fast enterprise inference |
| `vertex-gemini-2.5-flash` | Enterprise, speed | 1M | Fast enterprise inference |

> Note:
> - Vertex AI uses Google Cloud ADC instead of API keys. See [Configure Credentials](#configure-credentials-required).
> - The orchestrator now prefers Vertex for the Gemini 3 Preview models by default:
>   - gemini-3-pro → vertex-gemini-3-pro
>   - gemini-3-flash → vertex-gemini-3-flash
>   This provides higher limits using your Vertex subscription. You can still explicitly select the Google Gemini endpoints if desired.

### Mistral

| Model | Best For | Context | Strengths |
|-------|----------|---------|-----------|
| `mistral-large` | Coding, reasoning | 128K | Multilingual, function calling |
| `codestral` | Code generation | 32K | Specialized coding model |
| `mistral-small` | Simple tasks | 32K | Cost-effective, fast |

### Groq (Ultra-Fast Inference)

| Model | Best For | Context | Strengths |
|-------|----------|---------|-----------|
| `groq-llama-3.3-70b` | General, coding | 128K | Ultra-fast, versatile |
| `groq-mixtral-8x7b` | General use | 32K | Very fast, multilingual |

### xAI (Grok)

| Model | Best For | Context | Strengths |
|-------|----------|---------|-----------|
| `grok-2` | General, creative | 131K | Real-time knowledge |
| `grok-2-vision` | Vision tasks | 32K | Multimodal |

### Perplexity (Web Search)

| Model | Best For | Context | Strengths |
|-------|----------|---------|-----------|
| `perplexity-sonar-pro` | Research, citations | 200K | Web search, real-time info |
| `perplexity-sonar` | Quick lookups | 128K | Fast web search |

### DeepSeek (Cost-Effective)

| Model | Best For | Context | Strengths |
|-------|----------|---------|-----------|
| `deepseek-chat` | General, coding | 64K | Very cost-effective |
| `deepseek-reasoner` | Math, reasoning | 64K | Deep reasoning, problem-solving |

### Moonshot (Kimi K2 - Extended Thinking)

| Model | Best For | Context | Strengths |
|-------|----------|---------|-----------|
| `kimi-k2-thinking` | Complex reasoning, math | 256K | Extended thinking with visible reasoning traces |
| `kimi-k2` | General, coding | 256K | Fast, cost-effective |

### Local Models (MLX)

| Model | Provider | Best For | Strengths |
|-------|----------|----------|-----------|
| `mlx-llama-vision-11b` | MLX | Apple Silicon, Vision | Vision, documents, charts, writing, 128K context |
| `mlx-qwen3-4b` | MLX | Apple Silicon, Daily/Coding | Fast, efficient, local, private |
| `mlx-qwen2.5-coder-14b` | MLX | Apple Silicon, Coding | Strong coding, debugging, refactoring, local |
| `mlx-ministral-14b-reasoning` | MLX | Apple Silicon, Deep Thinking | Reasoning, math, STEM, local |

### Smart Routing Configuration

Enable LLM-based routing for every prompt (uses your routing model subscription):

```json
{
  "defaults": {
    "enableLLMRouting": true,
    "routerAllTasks": true,
    "routingModel": "gemini-3-flash-preview"
  }
}
```

Routing preferences (subscription + web search):

```json
{
  "defaults": {
    "preferSubscriptionProviders": ["vertex-ai"],
    "preferSubscriptionModels": ["vertex-gemini-3-pro"]
  },
  "models": {
    "gemini-2.0-flash": { "enabled": false }
  },
  "taskRouting": {
    "websearch": ["perplexity-sonar-pro", "perplexity-sonar-reasoning-pro"]
  }
}
```

- Auto mode prefers local MLX for simple prompts, Vertex Gemini 3 Flash for standard tasks, Kimi K2 Thinking for advanced math/logic/coding, and Vertex Gemini 3 Pro for advanced long-context/general reasoning.
- Web search defaults to Perplexity Sonar Pro; when reasoning is required it uses Sonar Reasoning Pro.

## API Keys Required

Based on which providers you want to use:

| Provider | Get API Key | Required? |
|----------|-------------|-----------|
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Recommended |
| Anthropic | [console.anthropic.com](https://console.anthropic.com/) | Recommended |
| Google | [aistudio.google.com](https://aistudio.google.com/apikey) | Optional |
| Vertex AI | ADC via `gcloud auth` (see above) | Optional |
| Mistral | [console.mistral.ai](https://console.mistral.ai/) | Optional |
| Groq | [console.groq.com](https://console.groq.com/) | Optional |
| xAI | [console.x.ai](https://console.x.ai/) | Optional |
| Perplexity | [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api) | Optional |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com/) | Optional |
| Moonshot | [platform.moonshot.ai](https://platform.moonshot.ai/) | Optional |
| MLX | N/A (local, requires `mlx-lm` or `mlx-vlm`) | Optional |

**You already have:** Claude (Anthropic), OpenAI, and Perplexity - these are ready to use!

## Security Features

### Secure Credential Storage

The credential manager uses a priority-based fallback chain:

1. **System Keychain** (Most Secure)
   - macOS: Keychain
   - Windows: Credential Manager
   - Linux: Secret Service (GNOME Keyring, KWallet)

2. **Encrypted File** (Secure, Portable)
   - Machine-specific key derivation (PBKDF2)
   - Fernet encryption (AES-128)
   - File permissions restricted to owner

3. **Environment Variables** (Fallback)
   - Least secure, but always available
   - Useful for CI/CD and containers

### Input Validation

- Maximum prompt length enforcement (500K chars)
- Suspicious pattern detection (logged, not blocked)
- Auto-routing logs store prompt length + hash only (no prompt/response content)
- Rate limiting to prevent abuse

### Audit Trail

All API calls are logged with:
- Timestamp
- Model/provider used
- Token usage
- Latency
- Success/failure status
- No prompt/response content (only hashes for routing diagnostics)

## Music Generation

AI Orchestrator includes a powerful music generation module optimized for electronic music production:

### Features

- **Separate MIDI Tracks**: Generates individual files for drums, bass, and chords
- **Combined MIDI**: Full arrangement with all tracks in one file
- **90s Tech-House Patterns**: Authentic patterns with:
  - Syncopated kicks (4-on-the-floor with groove)
  - Offbeat open hi-hats
  - Minor chord progressions with 7ths
  - Default 124-128 BPM range
- **AI Audio Generation**: MusicGen integration with selectable models for audio file creation

### Usage

In the GUI app, click the **Music** toggle and configure:
- Key signature (default: G Minor)
- BPM (default: 124-128 for tech house)
- Genre/style
- Energy level
- Duration
- **AI Model** (MusicGen variant: small/medium/large, stereo, melody)

Generated files are saved to `~/Music/AI Orchestrator/` and can be opened in:
- Logic Pro
- Ableton Live
- GarageBand
- Any DAW that supports MIDI

### Dependencies

```bash
# For MIDI generation
pip install midiutil

# For AI audio generation (optional)
# Recommended in a separate venv; see MUSICGEN.md
pip install torch "transformers>=4.45,<5.0" scipy accelerate

If you see a MusicGen error like `MusicgenDecoderConfig has no attribute 'decoder'`,
see MUSICGEN.md for compatible version combos and dedicated venv setup.

### MLX Smart Cache Tips

- To keep a single consistent cache location, set:

```bash
export HF_HOME="$HOME/Library/Caches/huggingface"
```

- The orchestrator automatically scans both `~/.cache/huggingface/hub` and
  `~/Library/Caches/huggingface/hub` (and respects `HF_HOME`) for MLX snapshots
  that include `*.safetensors` shards. If found, it enables offline mode to prevent network usage.

### Using the Project Virtual Environment

Always prefer the project venv for running and testing:

```bash
./.venv/bin/python -m src.orchestrator "Hello"
```

This ensures consistent dependencies (e.g., MLX, Transformers) and avoids mixing with user-level Python.
```

## Configuration

Copy `config/config.sample.json` to `~/.ai_orchestrator/config.json`:

```json
{
  "version": "2.0.0",
  "defaults": {
    "preferLocal": false,
    "localProvider": "mlx",
    "costOptimize": true,
    "enableLLMRouting": true,
    "routerAllTasks": true,
    "routingModel": "gemini-3-flash-preview",
    "preferSubscriptionProviders": ["vertex-ai"],
    "preferSubscriptionModels": ["vertex-gemini-3-pro"]
  },
  "models": {
    "gemini-2.0-flash": { "enabled": false }
  },
  "taskRouting": {
    "general": ["mlx-qwen3-4b", "vertex-gemini-3-flash", "vertex-gemini-3-pro"],
    "code": ["mlx-qwen2.5-coder-14b", "kimi-k2-thinking", "vertex-gemini-3-pro"],
    "reasoning": ["kimi-k2-thinking", "vertex-gemini-3-pro"],
    "websearch": ["perplexity-sonar-pro", "perplexity-sonar-reasoning-pro"],
    "long-context": ["vertex-gemini-3-pro"]
  }
}
```

To force MLX for local routing, set `defaults.preferLocal` to `true`, keep
`defaults.localProvider` as `mlx`, and point `taskRouting.local` at
`mlx-llama-vision-11b`.

To override the Vertex AI location (defaults to `global`), add:

```json
{
  "providers": {
    "vertex-ai": {
      "location": "global"
    }
  }
}
```

## VS Code Extension

### Installation

```bash
cd vscode-extension
npm install
npm run package
# Install the .vsix file in VS Code
```

### Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| Ask AI | `Ctrl+Shift+A` | Open query input |
| Explain Code | `Ctrl+Shift+E` | Explain selected code |
| Improve Code | - | Suggest improvements |
| Configure Credentials | - | Set up API keys |
| Select Model | - | Override model selection |
| Clear History | - | Clear conversation |

### Context Menu

Right-click selected code for:
- "AI Orchestrator: Explain Selected Code"
- "AI Orchestrator: Improve Selected Code"

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Type checking
mypy src

# Linting
ruff check src
ruff format --check .
```

## Project Structure

```
ai-orchestrator/
├── src/
│   ├── __init__.py
│   ├── orchestrator.py    # Main orchestrator logic (25+ models, 11 providers)
│   ├── credentials.py     # Secure credential management
│   ├── storage.py         # Conversation storage (SQLite)
│   ├── music.py           # Music generation (MIDI + MusicGen audio)
│   ├── gui/               # Native Mac GUI app (PySide6)
│   │   ├── app.py         # Main entry point
│   │   ├── main_window.py # Primary window
│   │   ├── chat_widget.py # Chat display
│   │   ├── input_widget.py# Input area with toggles
│   │   ├── sidebar.py     # Conversation history
│   │   └── styles.py      # macOS styling
│   ├── tui/               # Terminal UI (Textual)
│   │   └── app.py         # Terminal interface
│   └── menubar/           # Menu bar app (rumps)
│       └── app.py         # Menu bar interface
├── config/
│   ├── config-schema.json # JSON schema for config
│   └── config.sample.json # Sample configuration
├── tests/
│   ├── test_orchestrator.py
│   └── __init__.py
├── vscode-extension/
│   ├── src/
│   │   └── extension.js   # VS Code extension
│   └── package.json
├── setup_app.py           # py2app build script
├── pyproject.toml         # Python project config
├── LICENSE                # MIT License
└── README.md
```

## Troubleshooting

### "API key not configured"

```bash
# Check if credentials are set
python -c "from src.credentials import get_api_key; print(get_api_key('openai'))"

# Configure interactively
python -m src.credentials
```

### "Rate limited"

The orchestrator handles rate limiting automatically with exponential backoff. If you consistently hit limits, consider:
- Using a cheaper model for simple tasks
- Spreading requests over time
- Upgrading your API tier

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Anthropic for Claude
- OpenAI for GPT models
- Google for Gemini and Vertex AI
- Mistral AI for Codestral
- Groq for fast inference
- xAI for Grok
- Perplexity for web search
- DeepSeek for cost-effective models
- Moonshot AI for Kimi K2 extended thinking
- Apple for MLX framework
