
from .base_client import BaseLLMClient, ModelConfig, LLMResponse
from .azure_client import AzureAIFoundryClient
from .huggingface_client import HuggingFaceClient

import logging
from typing import Optional, Dict, Any

__version__ = "1.0.0"
__all__ = [
    "BaseLLMClient",
    "ModelConfig",
    "LLMResponse",
    "AzureAIFoundryClient",
    "HuggingFaceClient",
    "create_client",
    "get_available_providers"
]

logger = logging.getLogger(__name__)

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def create_client(
    provider: str,
    config: ModelConfig,
    **kwargs
) -> BaseLLMClient:
    """
    Factory-Funktion zum Erstellen eines LLM-Clients.
    
    Args:
        provider: Provider-Name ('azure' oder 'huggingface')
        config: Modellkonfiguration
        **kwargs: Provider-spezifische Parameter

    """
    provider = provider.lower().strip()
    
    if provider in ["azure", "azure_ai_foundry", "openai"]:
        logger.info(f"Erstelle Azure OpenAI Client für Modell: {config.model_name}")
        return AzureAIFoundryClient(config, **kwargs)
    
    elif provider in ["huggingface", "hf", "hugging_face"]:
        logger.info(f"Erstelle Hugging Face Client für Modell: {config.model_name}")
        return HuggingFaceClient(config, **kwargs)
    
    else:
        available = get_available_providers()
        raise ValueError(
            f"Unbekannter Provider: '{provider}'. "
            f"Verfügbare Provider: {', '.join(available)}"
        )


def get_available_providers() -> list:
    """
    Gibt eine Liste der verfügbaren LLM-Provider zurück.

    """
    return ["azure", "huggingface"]


def validate_credentials() -> Dict[str, bool]:
    """
    Überprüft, ob die erforderlichen Credentials gesetzt sind.

    """
    import os
    
    status = {
        "azure": bool(
            os.getenv("AZURE_API_KEY") and 
            os.getenv("AZURE_PROJECT_ENDPOINT")
        ),
        "huggingface": bool(os.getenv("HF_API_KEY"))
    }
    
    return status


def get_recommended_config(use_case: str = "summarization") -> ModelConfig:
    """
    Gibt empfohlene Konfigurationen für verschiedene Anwendungsfälle.
    
    Args:
        use_case: Anwendungsfall ('summarization', 'creative', 'factual')

    """
    configs = {
        "summarization": ModelConfig(
            model_name="gpt-4",
            temperature=0.3,
            max_tokens=1000,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0
        ),
        "creative": ModelConfig(
            model_name="gpt-4",
            temperature=0.8,
            max_tokens=1500,
            top_p=0.95,
            frequency_penalty=0.5,
            presence_penalty=0.3
        ),
        "factual": ModelConfig(
            model_name="gpt-4",
            temperature=0.1,
            max_tokens=800,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
    }
    
    if use_case not in configs:
        logger.warning(
            f"Unbekannter use_case: '{use_case}'. "
            f"Verwende 'summarization' als Standard."
        )
        use_case = "summarization"
    
    return configs[use_case]


# Package-Initialisierung
logger.debug(f"LLM Interface Package v{__version__} geladen")

# Zeige Credential-Status beim Import (nur im DEBUG-Modus)
if logger.isEnabledFor(logging.DEBUG):
    cred_status = validate_credentials()
    for provider, available in cred_status.items():
        status_str = "✓" if available else "✗"
        logger.debug(f"Provider {provider}: {status_str}")