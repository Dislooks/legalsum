"""
Azure AI Foundry Client für die Zusammenfassungs-Pipeline.

Implementiert die Kommunikation mit Azure AI Foundry
"""

import os
import time
from typing import List, Optional, Dict, Any
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
from openai import AzureOpenAI
import logging

from .base_client import BaseLLMClient, ModelConfig, LLMResponse

logger = logging.getLogger(__name__)

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class AzureAIFoundryClient(BaseLLMClient):
    """
    Hybrid Client für Azure OpenAI Service + Azure AI Foundry.

    """
    
    # Modelle, die Azure OpenAI API verwenden
    AZURE_OPENAI_MODELS = [
        'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo',
        'gpt-5', 'gpt-5-mini',
        'gpt-35-turbo', 'gpt-3.5-turbo', "gpt-oss-120b"
    ]
    
    # Modelle, die AI Foundry / MaaS API verwenden
    AI_FOUNDRY_MODELS = [
        'Llama', 'Mistral', 'Phi', 'Gemma', 'Command'
    ]
    
    def __init__(
        self,
        config: ModelConfig,
        # Azure OpenAI Credentials
        azure_openai_key: Optional[str] = None,
        azure_openai_endpoint: Optional[str] = None,
        azure_openai_api_version: str = "2024-12-01-preview",
        # AI Foundry Credentials
        ai_foundry_key: Optional[str] = None,
        ai_foundry_endpoint: Optional[str] = None,
        # Deployment Name
        deployment_name: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialisiert den Hybrid Client.
        
        Args:
            config: Modellkonfiguration
            azure_openai_key: Azure OpenAI API-Key
            azure_openai_endpoint: Azure OpenAI Endpoint
            azure_openai_api_version: API Version für Azure OpenAI
            ai_foundry_key: AI Foundry API-Key (für Llama, Mistral etc.)
            ai_foundry_endpoint: AI Foundry Endpoint
            deployment_name: Name des Deployments
            retry_attempts: Anzahl Wiederholungsversuche
            retry_delay: Wartezeit zwischen Versuchen
        """
        super().__init__(config)
        
        self.deployment_name = deployment_name or config.model_name
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Erkenne welche API verwendet werden soll
        self.api_type = self._detect_api_type(config.model_name)
        
        # Azure OpenAI Setup
        self.azure_openai_key = azure_openai_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = azure_openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_version = azure_openai_api_version
        
        # AI Foundry Setup
        self.ai_foundry_key = ai_foundry_key or os.getenv("AZURE_AI_KEY")
        self.ai_foundry_endpoint = ai_foundry_endpoint or os.getenv("AZURE_AI_ENDPOINT")
        
        # Initialisiere den entsprechenden Client
        if self.api_type == "azure_openai":
            if not self.azure_openai_key or not self.azure_openai_endpoint:
                raise ValueError(
                    "Azure OpenAI Credentials fehlen. Setze AZURE_OPENAI_API_KEY "
                    "und AZURE_OPENAI_ENDPOINT"
                )
            
            self.client = AzureOpenAI(
                api_key=self.azure_openai_key,
                api_version=self.azure_openai_api_version,
                azure_endpoint=self.azure_openai_endpoint
            )
            self.logger.info(f"Azure OpenAI Client initialisiert: {self.deployment_name}")
            
        else:  # ai_foundry
            if not self.ai_foundry_key or not self.ai_foundry_endpoint:
                raise ValueError(
                    "AI Foundry Credentials fehlen. Setze AZURE_AI_FOUNDRY_KEY "
                    "und AZURE_AI_ENDPOINT"
                )
            
            self.client = OpenAI(
                api_key=self.ai_foundry_key,
                base_url=self.ai_foundry_endpoint
            )
            self.logger.info(f"AI Foundry Client initialisiert: {self.deployment_name}")


    
    def _detect_api_type(self, model_name: str) -> str:
        """
        Erkennt automatisch welche API verwendet werden soll.
        
        Args:
            model_name: Name des Modells oder Deployments

        """
        model_lower = model_name.lower()
        
        # Prüfe ob es ein Azure OpenAI Modell ist
        for pattern in self.AZURE_OPENAI_MODELS:
            if pattern in model_lower:
                return "azure_openai"
        
        # Prüfe ob es ein AI Foundry Modell ist
        for pattern in self.AI_FOUNDRY_MODELS:
            if pattern in model_lower:
                return "ai_foundry"
        
        # Default: Wenn unklar, nehme Azure OpenAI
        self.logger.warning(
            f"Konnte API-Typ für Modell '{model_name}' nicht automatisch erkennen. "
            f"Verwende Azure OpenAI als Standard."
        )
        return "azure_openai"
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generiert Text mit dem entsprechenden API-Client.
        
        Args:
            prompt: Benutzerprompt
            system_message: Optionale System-Nachricht
            **kwargs: Zusätzliche Parameter

        """
        # Erstelle Nachrichten-Array
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Merge config mit kwargs
        params = self._prepare_params(kwargs)
        
        # Retry-Logik
        last_exception = None
        for attempt in range(self.retry_attempts):
            try:
                self.logger.debug(
                    f"API-Aufruf Versuch {attempt + 1}/{self.retry_attempts} "
                    f"(API: {self.api_type})"
                )
                
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    **params
                )
                
                return self._parse_response(response)
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Versuch {attempt + 1} fehlgeschlagen: {str(e)}")
                
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    self.logger.error(f"Alle {self.retry_attempts} Versuche fehlgeschlagen")
                    raise last_exception
    
    def batch_generate(
        self,
        prompts: List[str],
        system_message: Optional[str] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """Verarbeitet mehrere Prompts sequenziell."""
        responses = []
        total = len(prompts)
        
        for idx, prompt in enumerate(prompts, 1):
            self.logger.info(f"Verarbeite Prompt {idx}/{total}")
            try:
                response = self.generate(prompt, system_message, **kwargs)
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Fehler bei Prompt {idx}: {str(e)}")
                error_response = LLMResponse(
                    text=f"[FEHLER: {str(e)}]",
                    model=self.config.model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    finish_reason="error",
                    metadata={"error": str(e)}
                )
                responses.append(error_response)
        
        return responses
    
    def is_available(self) -> bool:
        """Überprüft Verfügbarkeit des Services."""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            self.logger.error(f"Service nicht verfügbar: {str(e)}")
            return False
    
    def _prepare_params(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Bereitet API-Parameter vor."""
        params = {
            "temperature": overrides.get("temperature", self.config.temperature),
        }

        if self.api_type == "azure_openai":
            # neuere GPT-Modelle erwarten max_completion_tokens
            if "gpt-4o" in self.config.model_name.lower() or "gpt-5" in self.config.model_name.lower():
                params["max_completion_tokens"] = overrides.get("max_tokens", self.config.max_tokens)
            else:
                params["max_tokens"] = overrides.get("max_tokens", self.config.max_tokens)
        else:
            # Foundry-Modelle (Mistral, Llama) bleiben bei max_tokens
            params["max_tokens"] = overrides.get("max_tokens", self.config.max_tokens)

        
        # Top-P (von beiden APIs unterstützt)
        if self.config.top_p is not None:
            params["top_p"] = overrides.get("top_p", self.config.top_p)
        
        # Frequency/Presence Penalty (hauptsächlich Azure OpenAI)
        if self.api_type == "azure_openai":
            if self.config.frequency_penalty is not None:
                params["frequency_penalty"] = overrides.get(
                    "frequency_penalty", self.config.frequency_penalty
                )
            if self.config.presence_penalty is not None:
                params["presence_penalty"] = overrides.get(
                    "presence_penalty", self.config.presence_penalty
                )
        
        # Top-K (nur für AI Foundry Modelle)
        if self.api_type == "ai_foundry" and self.config.top_k is not None:
            # Hinweis: OpenAI SDK unterstützt top_k nicht direkt
            # Manche Modelle akzeptieren es über extra_body
            self.logger.debug(
                f"top_k Parameter wird ignoriert (nicht unterstützt von OpenAI SDK)"
            )
        
        return params
    
    def _parse_response(self, response) -> LLMResponse:
        """Parst die API-Antwort."""
        choice = response.choices[0]
        usage = response.usage
        
        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "unknown",
            metadata={
                "response_id": response.id,
                "api_type": self.api_type,
                "created": getattr(response, 'created', None)
            }
        )
    
    def count_tokens(self, text: str) -> int:
        """Zählt Tokens (approximativ oder mit tiktoken für GPT)."""
        if self.api_type == "azure_openai":
            try:
                import tiktoken
                if "gpt-4" in self.config.model_name.lower() or "gpt-5" in self.config.model_name.lower():
                    encoding = tiktoken.encoding_for_model("gpt-4")
                else:
                    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                return len(encoding.encode(text))
            except ImportError:
                self.logger.warning("tiktoken nicht installiert, verwende Approximation")
                return self.estimate_tokens(text)
            except Exception as e:
                self.logger.warning(f"Fehler beim Token-Counting: {str(e)}")
                return self.estimate_tokens(text)
        else:
            # Für Llama, Mistral etc. verwende Approximation
            return self.estimate_tokens(text)
