"""
Secure Credentials Manager for AI Orchestrator
==============================================
Provides secure API key storage and retrieval using multiple backends:
1. System keyring (most secure - uses OS credential store)
2. Encrypted file with machine-specific key
3. Environment variables (fallback)

NEVER stores API keys in plain text or in code.
"""

import os
import sys
import json
import hashlib
import base64
import logging
import getpass
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from functools import lru_cache
from abc import ABC, abstractmethod

# Cryptography imports with fallback
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Keyring import with fallback
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

logger = logging.getLogger(__name__)

# Constants
SERVICE_NAME = "ai_orchestrator"
CONFIG_DIR = Path.home() / ".ai_orchestrator"
ENCRYPTED_CREDS_FILE = CONFIG_DIR / "credentials.enc"


@dataclass(frozen=True)
class APICredential:
    """Immutable credential container with secure string handling"""
    provider: str
    _key: str  # Private - never exposed directly
    
    def get_key(self) -> str:
        """Returns the API key. Log warning on access for audit trail."""
        logger.debug(f"API key accessed for provider: {self.provider}")
        return self._key
    
    def __repr__(self) -> str:
        """Prevent accidental key exposure in logs"""
        return f"APICredential(provider={self.provider}, key=****)"
    
    def __str__(self) -> str:
        return self.__repr__()


class CredentialBackend(ABC):
    """Abstract base class for credential storage backends"""
    
    @abstractmethod
    def get(self, provider: str) -> Optional[str]:
        """Retrieve API key for provider"""
        pass
    
    @abstractmethod
    def set(self, provider: str, api_key: str) -> bool:
        """Store API key for provider"""
        pass
    
    @abstractmethod
    def delete(self, provider: str) -> bool:
        """Remove API key for provider"""
        pass
    
    @abstractmethod
    def list_providers(self) -> list:
        """List all stored providers"""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the system"""
        pass


class KeyringBackend(CredentialBackend):
    """Uses OS keychain/keyring for secure storage (most secure option)"""
    
    @property
    def is_available(self) -> bool:
        if not KEYRING_AVAILABLE:
            return False
        try:
            # Test if keyring is functional
            keyring.get_password(SERVICE_NAME, "__test__")
            return True
        except Exception:
            return False
    
    def get(self, provider: str) -> Optional[str]:
        if not self.is_available:
            return None
        try:
            return keyring.get_password(SERVICE_NAME, provider)
        except Exception as e:
            logger.warning(f"Keyring get failed for {provider}: {e}")
            return None
    
    def set(self, provider: str, api_key: str) -> bool:
        if not self.is_available:
            return False
        try:
            keyring.set_password(SERVICE_NAME, provider, api_key)
            logger.info(f"Stored credential in keyring for: {provider}")
            return True
        except Exception as e:
            logger.error(f"Keyring set failed for {provider}: {e}")
            return False
    
    def delete(self, provider: str) -> bool:
        if not self.is_available:
            return False
        try:
            keyring.delete_password(SERVICE_NAME, provider)
            return True
        except Exception as e:
            logger.warning(f"Keyring delete failed for {provider}: {e}")
            return False
    
    def list_providers(self) -> list:
        # Keyring doesn't support listing, return empty
        return []


class EncryptedFileBackend(CredentialBackend):
    """Encrypted file storage using machine-specific key derivation"""
    
    def __init__(self):
        self._fernet: Optional[Fernet] = None
        self._init_encryption()
    
    @property
    def is_available(self) -> bool:
        return CRYPTO_AVAILABLE and self._fernet is not None
    
    def _get_machine_id(self) -> bytes:
        """Generate machine-specific identifier for key derivation"""
        identifiers = []
        
        # Collect stable machine identifiers
        if sys.platform == "darwin":
            try:
                import subprocess
                result = subprocess.run(
                    ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                    capture_output=True, text=True
                )
                for line in result.stdout.split("\n"):
                    if "IOPlatformUUID" in line:
                        identifiers.append(line.split('"')[-2])
                        break
            except Exception:
                pass
        elif sys.platform == "linux":
            try:
                with open("/etc/machine-id", "r") as f:
                    identifiers.append(f.read().strip())
            except Exception:
                pass
        
        # Add username and hostname as fallback
        identifiers.extend([
            getpass.getuser(),
            os.uname().nodename if hasattr(os, 'uname') else "unknown"
        ])
        
        combined = ":".join(identifiers)
        return hashlib.sha256(combined.encode()).digest()
    
    def _init_encryption(self):
        """Initialize Fernet encryption with machine-specific key"""
        if not CRYPTO_AVAILABLE:
            return
        
        try:
            # Derive key from machine ID
            salt = b"ai_orchestrator_v1"  # Static salt for key derivation
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=480000,  # OWASP recommended minimum
            )
            key = base64.urlsafe_b64encode(kdf.derive(self._get_machine_id()))
            self._fernet = Fernet(key)
            
            # Ensure config directory exists
            CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self._fernet = None
    
    def _load_credentials(self) -> Dict[str, str]:
        """Load and decrypt credentials file"""
        if not self.is_available or not ENCRYPTED_CREDS_FILE.exists():
            return {}
        
        try:
            with open(ENCRYPTED_CREDS_FILE, "rb") as f:
                encrypted = f.read()
            decrypted = self._fernet.decrypt(encrypted)
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return {}
    
    def _save_credentials(self, creds: Dict[str, str]) -> bool:
        """Encrypt and save credentials file"""
        if not self.is_available:
            return False
        
        try:
            encrypted = self._fernet.encrypt(json.dumps(creds).encode())
            
            # Write atomically with restricted permissions
            temp_file = ENCRYPTED_CREDS_FILE.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                f.write(encrypted)
            os.chmod(temp_file, 0o600)  # Owner read/write only
            temp_file.rename(ENCRYPTED_CREDS_FILE)
            return True
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            return False
    
    def get(self, provider: str) -> Optional[str]:
        creds = self._load_credentials()
        return creds.get(provider)
    
    def set(self, provider: str, api_key: str) -> bool:
        creds = self._load_credentials()
        creds[provider] = api_key
        success = self._save_credentials(creds)
        if success:
            logger.info(f"Stored credential in encrypted file for: {provider}")
        return success
    
    def delete(self, provider: str) -> bool:
        creds = self._load_credentials()
        if provider in creds:
            del creds[provider]
            return self._save_credentials(creds)
        return True
    
    def list_providers(self) -> list:
        return list(self._load_credentials().keys())


class EnvironmentBackend(CredentialBackend):
    """Environment variable fallback (least secure, but always available)"""
    
    # Mapping of provider names to environment variable names
    ENV_VAR_MAP = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "cohere": "COHERE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "xai": "XAI_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "groq": "GROQ_API_KEY",
        "ollama": "OLLAMA_API_KEY",
    }
    
    @property
    def is_available(self) -> bool:
        return True
    
    def _get_env_var(self, provider: str) -> str:
        return self.ENV_VAR_MAP.get(
            provider.lower(),
            f"{provider.upper()}_API_KEY"
        )
    
    def get(self, provider: str) -> Optional[str]:
        return os.environ.get(self._get_env_var(provider))
    
    def set(self, provider: str, api_key: str) -> bool:
        # Can't persistently set environment variables
        os.environ[self._get_env_var(provider)] = api_key
        logger.warning(
            f"Set API key in environment (non-persistent) for: {provider}"
        )
        return True
    
    def delete(self, provider: str) -> bool:
        env_var = self._get_env_var(provider)
        if env_var in os.environ:
            del os.environ[env_var]
        return True
    
    def list_providers(self) -> list:
        return [p for p, v in self.ENV_VAR_MAP.items() if os.environ.get(v)]


class CredentialManager:
    """
    Main credential manager with fallback chain:
    1. System keyring (most secure)
    2. Encrypted file (secure, portable)
    3. Environment variables (fallback)
    """
    
    def __init__(self):
        self._backends = [
            KeyringBackend(),
            EncryptedFileBackend(),
            EnvironmentBackend(),
        ]
        self._cache: Dict[str, APICredential] = {}
        self._validate_security()
    
    def _validate_security(self):
        """Log security posture on initialization"""
        available = [
            type(b).__name__ for b in self._backends if b.is_available
        ]
        logger.info(f"Available credential backends: {available}")
        
        if not any(isinstance(b, (KeyringBackend, EncryptedFileBackend)) 
                   and b.is_available for b in self._backends):
            logger.warning(
                "⚠️  No secure credential storage available. "
                "Install 'keyring' or 'cryptography' for secure storage."
            )
    
    def get_credential(self, provider: str) -> Optional[APICredential]:
        """
        Retrieve API credential, checking backends in priority order.
        Results are cached for performance.
        """
        provider = provider.lower()
        
        # Check cache first
        if provider in self._cache:
            return self._cache[provider]
        
        # Try each backend in order
        for backend in self._backends:
            if not backend.is_available:
                continue
            
            api_key = backend.get(provider)
            if api_key:
                credential = APICredential(provider=provider, _key=api_key)
                self._cache[provider] = credential
                logger.debug(
                    f"Retrieved credential for {provider} from "
                    f"{type(backend).__name__}"
                )
                return credential
        
        logger.warning(f"No credential found for provider: {provider}")
        return None
    
    def set_credential(self, provider: str, api_key: str) -> bool:
        """
        Store API credential in the most secure available backend.
        """
        provider = provider.lower()
        
        # Validate API key format (basic security check)
        if not api_key or len(api_key) < 10:
            logger.error("Invalid API key: too short")
            return False
        
        # Use first available secure backend
        for backend in self._backends:
            if backend.is_available and not isinstance(backend, EnvironmentBackend):
                if backend.set(provider, api_key):
                    # Clear cache
                    self._cache.pop(provider, None)
                    return True
        
        # Fall back to environment
        env_backend = self._backends[-1]
        return env_backend.set(provider, api_key)
    
    def delete_credential(self, provider: str) -> bool:
        """Remove credential from all backends"""
        provider = provider.lower()
        self._cache.pop(provider, None)
        
        success = True
        for backend in self._backends:
            if backend.is_available:
                success = backend.delete(provider) and success
        return success
    
    def list_configured_providers(self) -> list:
        """List all providers with stored credentials"""
        providers = set()
        for backend in self._backends:
            if backend.is_available:
                providers.update(backend.list_providers())
        return sorted(providers)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Convenience method to get raw API key string"""
        cred = self.get_credential(provider)
        return cred.get_key() if cred else None
    
    def clear_cache(self):
        """Clear the credential cache"""
        self._cache.clear()


# Global singleton instance
_manager: Optional[CredentialManager] = None


def get_credential_manager() -> CredentialManager:
    """Get the global credential manager instance"""
    global _manager
    if _manager is None:
        _manager = CredentialManager()
    return _manager


# Convenience functions
def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider"""
    return get_credential_manager().get_api_key(provider)


def set_api_key(provider: str, api_key: str) -> bool:
    """Store API key for a provider"""
    return get_credential_manager().set_credential(provider, api_key)


def configure_credentials_interactive():
    """Interactive CLI for configuring API credentials"""
    print("\n🔐 AI Orchestrator Credential Configuration\n")
    print("=" * 50)
    
    manager = get_credential_manager()
    
    providers = [
        ("openai", "OpenAI (GPT-4o, o1, o3)"),
        ("anthropic", "Anthropic (Claude Opus 4.5, Sonnet 4.5)"),
        ("google", "Google (Gemini 2.0, 2.5)"),
        ("mistral", "Mistral (Codestral, Mistral Large)"),
        ("xai", "xAI (Grok 2)"),
        ("groq", "Groq (Fast inference - Llama, Mixtral)"),
        ("perplexity", "Perplexity (Sonar - Web Search)"),
        ("deepseek", "DeepSeek (Chat, Reasoner)"),
        ("ollama", "Ollama (Local models - no API key needed)"),
    ]
    
    for provider_id, provider_name in providers:
        existing = manager.get_credential(provider_id)
        status = "✓ configured" if existing else "✗ not set"
        print(f"\n{provider_name}: [{status}]")
        
        response = input(f"Configure {provider_id}? (y/N/clear): ").strip().lower()
        
        if response == 'clear':
            manager.delete_credential(provider_id)
            print(f"  → Cleared {provider_id} credentials")
        elif response == 'y':
            api_key = getpass.getpass(f"  Enter API key for {provider_id}: ")
            if api_key:
                if manager.set_credential(provider_id, api_key):
                    print(f"  → Saved {provider_id} credentials securely")
                else:
                    print(f"  → Failed to save {provider_id} credentials")
    
    print("\n" + "=" * 50)
    print("Configuration complete!")
    print(f"Configured providers: {manager.list_configured_providers()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    configure_credentials_interactive()
