from .base_client import BaseLLMClient, ModelConfig, LLMResponse
from .azure_client import AzureAIFoundryClient
from .huggingface_client import HuggingFaceClient
from .azure_client import AzureAIFoundryClient

import logging
from typing import Optional, Dict, Any

__version__ = "2.0.0"
__all__ = [
    "BaseLLMClient",
    "ModelConfig",
    "LLMResponse",
    "AzureAIFoundryClient",
    "HuggingFaceClient",
    "create_client",
    "get_available_providers",
    "AzureAIFoundryClient"
]

    """
    Überprüft, ob die erforderlichen Credentials gesetzt sind.
    
    Returns:
        Dictionary mit Provider-Namen als Keys und Verfügbarkeit als Values
        
    Beispiel:
        >>> creds = validate_credentials()
        >>> print(creds)
        {'azure_ai_foundry': True, 'huggingface': True}
    """
    import os
    
    result = {
        'azure_ai_foundry': bool(os.getenv('AZURE_API_KEY') and os.getenv('AZURE_PROJECT_ENDPOINT')),
        'huggingface': bool(os.getenv('HF_API_KEY'))
    }
    
    return result