"""
Abstract Base Class für LLM-Clients.

"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
@dataclass
class ModelConfig:
    """Konfiguration für LLM-Anfragen."""
    model_name: str
    temperature: float = 0.7
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_tokens: int = 5000
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def validate(self) -> None:
        """Validiert die Modellkonfiguration."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature muss zwischen 0.0 und 2.0 liegen, nicht {self.temperature}")
        
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k muss >= 1 sein, nicht {self.top_k}")
        
        if self.top_p is not None and not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p muss zwischen 0.0 und 1.0 liegen, nicht {self.top_p}")
        
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens muss >= 1 sein, nicht {self.max_tokens}")

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
@dataclass
class LLMResponse:
    """Standardisierte Antwort von LLM-Anfragen."""
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert die Antwort in ein Dictionary."""
        return {
            'text': self.text,
            'model': self.model,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'finish_reason': self.finish_reason,
            'metadata': self.metadata
        }

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class BaseLLMClient(ABC):
    """Abstract Base Class für alle LLM-Clients."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialisiert den LLM-Client.
        
        Args:
            config: Modellkonfiguration
        """
        config.validate()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generiert Text basierend auf einem Prompt.
        
        Args:
            prompt: Eingabe-Prompt
            **kwargs: Zusätzliche modellspezifische Parameter
        """
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[LLMResponse]:
        """
        Generiert Text für mehrere Prompts (Batch-Verarbeitung).
        
        Args:
            prompts: Liste von Eingabe-Prompts
            **kwargs: Zusätzliche modellspezifische Parameter

        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Überprüft, ob der Service verfügbar ist.

        """
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """
        Schätzt die Anzahl der Tokens in einem Text.
        
        Args:
            text: Zu analysierender Text

        """
        # Einfache Heuristik: ~4 Zeichen pro Token
        # Für präzisere Schätzungen sollte tiktoken verwendet werden
        return len(text) // 4
    
    def update_config(self, **kwargs) -> None:
        """
        Aktualisiert die Konfiguration des Clients.
        
        Args:
            **kwargs: Zu aktualisierende Konfigurationsparameter
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.logger.warning(f"Unbekannter Konfigurationsparameter: {key}")
        
        self.config.validate()
        self.logger.info(f"Konfiguration aktualisiert: {kwargs}")
    
    def get_config(self) -> ModelConfig:
        """
        Gibt die aktuelle Konfiguration zurück.

        """
        return self.config
    
    def __str__(self) -> str:
        """String-Repräsentation des Clients."""
        return f"{self.__class__.__name__}(model={self.config.model_name})"
    
    def __repr__(self) -> str:
        """Detaillierte String-Repräsentation."""
        return (f"{self.__class__.__name__}("
                f"model={self.config.model_name}, "
                f"temperature={self.config.temperature}, "
                f"max_tokens={self.config.max_tokens})")
