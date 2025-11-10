"""
Hugging Face Client für die Zusammenfassungs-Pipeline.

"""

import os
import time
import requests
from typing import List, Optional, Dict, Any
import logging

from .base_client import BaseLLMClient, ModelConfig, LLMResponse

logger = logging.getLogger(__name__)

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class HuggingFaceClient(BaseLLMClient):
    def __init__(
        self,
        config: ModelConfig,
        api_key: Optional[str] = None,
        inference_endpoint_url: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 120
    ):
        """
        Initialisiert den Client.

        Args:
            config: Modellkonfiguration
            api_key: HuggingFace API-Token (oder über HF_API_KEY env)
            inference_endpoint_url: Basis-URL des Endpoints (z.B. https://xxx.endpoints.huggingface.cloud/v1)
            retry_attempts: Anzahl der Wiederholungsversuche
            retry_delay: Wartezeit zwischen Versuchen
            timeout: Request-Timeout in Sekunden
        """
        super().__init__(config)

        self.api_key = api_key or os.getenv("HF_API_KEY")
        if not self.api_key:
            raise ValueError("API-Key fehlt. Setzen Sie HF_API_KEY oder übergeben Sie api_key.")

        self.api_url = (inference_endpoint_url or os.getenv("HF_INFERENCE_ENDPOINT_URL"))
        if not self.api_url:
            raise ValueError("HuggingFace Inference Endpoint URL fehlt!")
        
        self.api_url = self.api_url.rstrip("/")
        if self.api_url.endswith("/v1"):
            self.api_url = self.api_url[:-3]
        
        self.api_url = self.api_url + "/v1"

        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.timeout = timeout

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        self.logger.info(f"OpenAI-kompatibler Client initialisiert: {config.model_name}")
        self.logger.info(f"Nutze Endpoint: {self.api_url}")

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generiert Text über /chat/completions oder /generate.
        Versucht zuerst /chat/completions, dann /generate als Fallback.
        """
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": kwargs.get("stream", False)
        }

        last_exception = None
        
        # Versuche zuerst /chat/completions (OpenAI-kompatibel)
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return self._parse_response(response.json(), prompt)
            except requests.exceptions.HTTPError as e:
               
                if e.response.status_code == 404:
                    self.logger.info("Chat completions endpoint nicht gefunden (404), versuche /generate Fallback")
                    try:
                        return self._generate_tgi_fallback(prompt, **kwargs)
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback zu /generate fehlgeschlagen: {fallback_error}")
                        raise e  # Original 404 Fehler werfen
                # Andere HTTP-Fehler: Normal retry
                last_exception = e
                self.logger.warning(f"Versuch {attempt+1} fehlgeschlagen: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Versuch {attempt+1} fehlgeschlagen: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise last_exception
    
    def _generate_tgi_fallback(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Fallback für HuggingFace TGI (Text Generation Inference) Endpoints.
        Verwendet direkten /generate Endpoint ohne /v1 Präfix.
        """
        # Entferne /v1 aus URL für direkten HF Endpoint
        base_url = self.api_url.replace("/v1", "")
        
        # TGI-kompatibles Payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "return_full_text": False
            }
        }
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    f"{base_url}/generate",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return self._parse_tgi_response(response.json(), prompt)
            except Exception as e:
                self.logger.warning(f"TGI Fallback Versuch {attempt+1} fehlgeschlagen: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
    
    def _parse_tgi_response(self, data: Dict[str, Any], original_prompt: str) -> LLMResponse:
        """
        Parst die TGI-Response.
        """
        try:
            generated_text = data.get("generated_text", "")
            
            # Token counts schätzen
            prompt_tokens = self.estimate_tokens(original_prompt)
            completion_tokens = self.estimate_tokens(generated_text)
            
            return LLMResponse(
                text=generated_text.strip(),
                model=self.config.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                finish_reason="completed",
                metadata={"raw": data, "endpoint_type": "tgi"}
            )
        except Exception as e:
            self.logger.error(f"Fehler beim Parsen der TGI Response: {e}")
            raise
        except Exception as e:
            last_exception = e
            self.logger.warning(f"Versuch {attempt+1} fehlgeschlagen: {e}")
            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        raise last_exception

    def batch_generate(self, prompts: List[str], **kwargs) -> List[LLMResponse]:
        responses = []
        for idx, prompt in enumerate(prompts, 1):
            self.logger.info(f"Verarbeite Prompt {idx}/{len(prompts)}")
            try:
                responses.append(self.generate(prompt, **kwargs))
            except Exception as e:
                self.logger.error(f"Fehler bei Prompt {idx}: {e}")
                responses.append(
                    LLMResponse(
                        text=f"[FEHLER: {e}]",
                        model=self.config.model_name,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        finish_reason="error",
                        metadata={"error": str(e)}
                    )
                )
        return responses

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.api_url}/models", headers=self.headers, timeout=10)
            return r.status_code == 200
        except Exception as e:
            self.logger.error(f"Service nicht verfügbar: {e}")
            return False

    def _parse_response(self, data: Dict[str, Any], original_prompt: str) -> LLMResponse:
        """
        Parst die OpenAI-kompatible Antwort.
        """
        try:
            choice = data["choices"][0]
            generated_text = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "completed")
        except Exception:
            generated_text = str(data)
            finish_reason = "unknown"

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", self.estimate_tokens(original_prompt))
        completion_tokens = usage.get("completion_tokens", self.estimate_tokens(generated_text))
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        return LLMResponse(
            text=generated_text.strip(),
            model=data.get("model", self.config.model_name),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            metadata={"raw": data}
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Ruft Informationen über das Modell ab.
        
        Returns:
            Dictionary mit Modell-Metadaten
        """
        try:
            # HF Model Hub API
            model_api_url = f"https://huggingface.co/api/models/{self.config.model_name}"
            response = requests.get(model_api_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Konnte Modell-Info nicht abrufen: {str(e)}")
            return {}
    
    def warm_up(self) -> bool:
        """
        Aktiviert das Modell durch eine Test-Anfrage.
        Nützlich, um Ladezeiten bei ersten echten Anfragen zu vermeiden.
        
        Returns:
            True wenn erfolgreich, sonst False
        """
        try:
            self.logger.info("Wärme Modell auf...")
            self.generate("Hello", max_tokens=5)
            self.logger.info("Modell ist bereit")
            return True
        except Exception as e:
            self.logger.warning(f"Warm-up fehlgeschlagen: {str(e)}")
            return False
