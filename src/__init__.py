"""
AI Orchestrator - Intelligent Multi-Model AI Router
====================================================

A production-ready AI orchestrator that automatically routes queries
to the best AI model based on task type, with secure credential
management and comprehensive error handling.

Security Features:
- No API keys stored in code
- Secure credential storage (keyring/encrypted file)
- Input validation and sanitization
- Audit logging

Example Usage:
    >>> from src import AIOrchestrator, get_api_key, set_api_key
    >>>
    >>> # Configure credentials (run once)
    >>> set_api_key("openai", "sk-...")
    >>> set_api_key("anthropic", "sk-ant-...")
    >>>
    >>> # Use the orchestrator
    >>> import asyncio
    >>>
    >>> async def main():
    ...     orchestrator = AIOrchestrator()
    ...     response = await orchestrator.query("Explain quantum computing")
    ...     print(response.content)
    >>>
    >>> asyncio.run(main())
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Jason Vassallo"

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .credentials import (
        CredentialManager,
        configure_credentials_interactive,
        get_api_key,
        get_credential_manager,
        set_api_key,
    )
    from .orchestrator import (
        AIOrchestrator,
        APIResponse,
        InputValidator,
        ModelCapability,
        ModelRegistry,
        TaskClassifier,
        TaskType,
    )

__all__ = [
    # Version
    "__version__",
    # Credential management
    "get_api_key",
    "set_api_key",
    "get_credential_manager",
    "CredentialManager",
    "configure_credentials_interactive",
    # Orchestrator
    "AIOrchestrator",
    "TaskType",
    "TaskClassifier",
    "ModelCapability",
    "ModelRegistry",
    "APIResponse",
    "InputValidator",
]

_CREDENTIALS_EXPORTS = {
    "get_api_key",
    "set_api_key",
    "get_credential_manager",
    "CredentialManager",
    "configure_credentials_interactive",
}

_ORCHESTRATOR_EXPORTS = {
    "AIOrchestrator",
    "TaskType",
    "TaskClassifier",
    "ModelCapability",
    "ModelRegistry",
    "APIResponse",
    "InputValidator",
}


def __getattr__(name: str) -> Any:
    if name in _CREDENTIALS_EXPORTS:
        module = import_module(".credentials", __name__)
    elif name in _ORCHESTRATOR_EXPORTS:
        module = import_module(".orchestrator", __name__)
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
