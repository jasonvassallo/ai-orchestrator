# AI Orchestrator

**Intelligent Multi-Model AI Router with Secure Credential Management**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Security: Hardened](https://img.shields.io/badge/security-hardened-green.svg)](https://github.com/jasonvassallo/ai-orchestrator/security)

AI Orchestrator automatically routes your queries to the best AI model based on task type. It supports **9 providers** including OpenAI, Anthropic Claude, Google Gemini, Mistral, Groq, xAI (Grok), Perplexity, DeepSeek, and local models via Ollama - all with secure credential management and production-ready features.

## Features

- **Intelligent Routing**: Automatically selects the best model based on task classification
- **Secure Credentials**: API keys stored in OS keychain or encrypted file - NEVER in code
- **High Performance**: Async operations, rate limiting, and retry logic with exponential backoff
- **Security Hardened**: Input validation, audit logging, no credential leakage
- **VS Code Extension**: Full-featured extension with keyboard shortcuts
- **Local Models**: Privacy-first option with Ollama support
- **Cost Optimization**: Route to cheaper models when appropriate
- **Web Search**: Perplexity integration for real-time information

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jasonvassallo/ai-orchestrator.git
cd ai-orchestrator

# Install with all providers
pip install -e ".[all]"

# Or install minimal + specific providers
pip install -e "."
pip install -e ".[openai,anthropic]"
```

### Configure Credentials (REQUIRED)

**NEVER store API keys in code or config files!**

```bash
# Interactive configuration (recommended)
python -m src.credentials

# Or set environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export MISTRAL_API_KEY="..."
export GROQ_API_KEY="..."
export XAI_API_KEY="..."
export PERPLEXITY_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

### Basic Usage

```bash
# CLI usage
python -m src.orchestrator "Explain quantum computing"

# With model override
python -m src.orchestrator "Debug this Python code" --model claude-sonnet-4.5

# Prefer local models (Ollama)
python -m src.orchestrator "Summarize this text" --local

# Cost optimization mode
python -m src.orchestrator "Write a haiku" --cheap

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
| `gemini-2.0-flash` | Speed, long docs | 1M | Massive context, fast |
| `gemini-1.5-pro` | Very long docs | 2M | Largest context window |

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

### Local Models (Ollama)

| Model | Best For | Strengths |
|-------|----------|-----------|
| `llama3.2` | Private, offline | Free, privacy-first |
| `codellama` | Private coding | Free, code-focused |
| `deepseek-coder-v2` | Complex code | Excellent coding, free |

## API Keys Required

Based on which providers you want to use:

| Provider | Get API Key | Required? |
|----------|-------------|-----------|
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Recommended |
| Anthropic | [console.anthropic.com](https://console.anthropic.com/) | Recommended |
| Google | [aistudio.google.com](https://aistudio.google.com/apikey) | Optional |
| Mistral | [console.mistral.ai](https://console.mistral.ai/) | Optional |
| Groq | [console.groq.com](https://console.groq.com/) | Optional |
| xAI | [console.x.ai](https://console.x.ai/) | Optional |
| Perplexity | [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api) | Optional |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com/) | Optional |
| Ollama | N/A (local) | Optional |

**You already have:** Claude (Anthropic), OpenAI, Ollama, and Perplexity - these are ready to use!

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
- API key redaction in logs
- Rate limiting to prevent abuse

### Audit Trail

All API calls are logged with:
- Timestamp
- Model/provider used
- Token usage
- Latency
- Success/failure status

## Configuration

Copy `config/config.sample.json` to `~/.ai_orchestrator/config.json`:

```json
{
  "version": "2.0.0",
  "defaults": {
    "preferLocal": false,
    "costOptimize": false,
    "maxTokens": 4096,
    "temperature": 0.7
  },
  "taskRouting": {
    "code": ["claude-sonnet-4.5", "codestral", "deepseek-chat"],
    "reasoning": ["o1", "claude-opus-4.5", "deepseek-reasoner"],
    "websearch": ["perplexity-sonar-pro", "perplexity-sonar"],
    "local": ["llama3.2", "deepseek-coder-v2"]
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
black --check src
```

## Project Structure

```
ai-orchestrator/
├── src/
│   ├── __init__.py
│   ├── orchestrator.py    # Main orchestrator logic (25+ models, 9 providers)
│   └── credentials.py     # Secure credential management
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

### "Ollama not available"

```bash
# Ensure Ollama is running
ollama serve

# Test connection
curl http://localhost:11434/api/version
```

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
- Google for Gemini
- Mistral AI for Codestral
- Groq for fast inference
- xAI for Grok
- Perplexity for web search
- DeepSeek for cost-effective models
- Ollama for local model support
