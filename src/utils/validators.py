"""
Input Validation Utilities für Legal Summarization Pipeline

Validiert Konfigurationen, Parameter, Dateien und API-Antworten.
Verhindert Common Errors und gibt hilfreiche Fehlermeldungen.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import re
import os

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class ValidationError(Exception):
    """Custom Exception für Validierungsfehler."""
    pass

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class ConfigValidator:
    """Validiert YAML-Konfigurationsdateien."""
    
    VALID_PROVIDERS = ['azure_ai_foundry', 'huggingface']
    VALID_METRICS = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    
    @staticmethod
    def validate_models_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validiert models_config.yaml.
        
        Args:
            config: Geladene Config als Dictionary
        """
        errors = []
        
        # Azure OpenAI Check
        if 'azure_ai_foundry' in config:
            azure = config['azure_ai_foundry']
            
            if 'endpoint_url' not in azure:
                errors.append("azure_ai_foundry.endpoint_url fehlt")
            elif not azure['endpoint_url'].startswith(('http://', 'https://')):
                errors.append("azure_ai_foundry.endpoint_url muss mit http:// oder https:// beginnen")
            
            if 'models' not in azure or not azure['models']:
                errors.append("azure_ai_foundry.models ist leer oder fehlt")
            else:
                for model_name, model_config in azure['models'].items():
                    if 'deployment_name' not in model_config:
                        errors.append(f"azure_ai_foundry.models.{model_name}.deployment_name fehlt")
        
        # HuggingFace Check
        if 'huggingface' in config:
            hf = config['huggingface']
            
            if 'models' not in hf or not hf['models']:
                errors.append("huggingface.models ist leer oder fehlt")
            else:
                for model_name, model_config in hf['models'].items():
                    if 'model_id' not in model_config:
                        errors.append(f"huggingface.models.{model_name}.model_id fehlt")
        
        # Default Parameters Check
        if 'default_parameters' in config:
            params = config['default_parameters']
            
            if 'temperature' in params:
                temp = params['temperature']
                if not 0 <= temp <= 2:
                    errors.append(f"temperature muss zwischen 0 und 2 liegen, ist: {temp}")
            
            if 'top_k' in params:
                top_k = params['top_k']
                if not isinstance(top_k, int) or top_k < 1:
                    errors.append(f"top_k muss positive Integer sein, ist: {top_k}")
            
            if 'top_p' in params:
                top_p = params['top_p']
                if not 0 <= top_p <= 1:
                    errors.append(f"top_p muss zwischen 0 und 1 liegen, ist: {top_p}")
        
        if errors:
            raise ValidationError(
                "Fehler in models_config.yaml:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        
        return config
    
    @staticmethod
    def validate_evaluation_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validiert evaluation_config.yaml.
        
        Args:
            config: Geladene Config als Dictionary

        """
        errors = []
        
        # ROUGE Metrics Check
        if 'rouge_metrics' in config and config['rouge_metrics'].get('enabled'):
            rouge = config['rouge_metrics']
            
            if 'metrics' not in rouge:
                errors.append("rouge_metrics.metrics fehlt")
            else:
                for metric in rouge['metrics']:
                    if metric not in ConfigValidator.VALID_METRICS:
                        errors.append(
                            f"Ungültige ROUGE-Metrik: {metric}. "
                            f"Gültig: {ConfigValidator.VALID_METRICS}"
                        )
        
        # BERTScore Check
        if 'bertscore' in config and config['bertscore'].get('enabled'):
            bert = config['bertscore']
            
            if 'model' not in bert:
                errors.append("bertscore.model fehlt")
        
        # LLM Judge Check
        if 'llm_judge' in config and config['llm_judge'].get('enabled'):
            judge = config['llm_judge']
            
            if 'judge_model' not in judge:
                errors.append("llm_judge.judge_model fehlt")
            else:
                judge_model = judge['judge_model']
                
                if 'provider' not in judge_model:
                    errors.append("llm_judge.judge_model.provider fehlt")
                elif judge_model['provider'] not in ConfigValidator.VALID_PROVIDERS:
                    errors.append(
                        f"Ungültiger Judge Provider: {judge_model['provider']}. "
                        f"Gültig: {ConfigValidator.VALID_PROVIDERS}"
                    )
            
            # Evaluation Criteria Check
            if 'evaluation_criteria' in judge:
                total_weight = 0
                for criterion, details in judge['evaluation_criteria'].items():
                    if 'weight' not in details:
                        errors.append(f"llm_judge.evaluation_criteria.{criterion}.weight fehlt")
                    else:
                        total_weight += details['weight']
                
                if abs(total_weight - 1.0) > 0.01:
                    errors.append(
                        f"Summe der Kriterien-Gewichte muss 1.0 sein, ist: {total_weight}"
                    )
        
        if errors:
            raise ValidationError(
                "Fehler in evaluation_config.yaml:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        
        return config

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class ParameterValidator:
    """Validiert LLM-Parameter für API-Calls."""
    
    @staticmethod
    def validate_generation_params(
        temperature: float,
        top_k: int,
        top_p: float,
        max_tokens: int
    ) -> None:
        """
        Validiert Generation-Parameter.
        
        Args:
            temperature: Sampling-Temperatur
            top_k: Top-K Sampling
            top_p: Nucleus Sampling
            max_tokens: Maximale Output-Länge

        """
        errors = []
        
        if not 0 <= temperature <= 2:
            errors.append(f"temperature muss zwischen 0 und 2 liegen, ist: {temperature}")
        
        if not isinstance(top_k, int) or top_k < 1:
            errors.append(f"top_k muss positive Integer sein, ist: {top_k}")
        
        if not 0 <= top_p <= 1:
            errors.append(f"top_p muss zwischen 0 und 1 liegen, ist: {top_p}")
        
        if not isinstance(max_tokens, int) or max_tokens < 1:
            errors.append(f"max_tokens muss positive Integer sein, ist: {max_tokens}")
        
        if max_tokens > 8192:
            errors.append(
                f"max_tokens sehr hoch ({max_tokens}). "
                "Empfohlen: 500-2000 für Zusammenfassungen"
            )
        
        if errors:
            raise ValidationError("\n".join(errors))
    
    @staticmethod
    def validate_experiment_config(config: Dict[str, Any]) -> None:
        """
        Validiert Experiment-Konfiguration.
        
        Args:
            config: Experiment-Config Dictionary

        """
        required_keys = ['experiment_id', 'model', 'provider', 'temperature', 'top_k']
        missing = [key for key in required_keys if key not in config]
        
        if missing:
            raise ValidationError(
                f"Fehlende Pflichtfelder in Experiment-Config: {', '.join(missing)}"
            )
        
        # Validiere Generation-Parameter
        ParameterValidator.validate_generation_params(
            temperature=config['temperature'],
            top_k=config['top_k'],
            top_p=config.get('top_p'),
            max_tokens=config.get('max_tokens', 500)
        )

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class FileValidator:
    """Validiert Dateien und Verzeichnisse."""
    
    SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt']
    MAX_FILE_SIZE_MB = 50
    
    @staticmethod
    def validate_input_file(file_path: Union[str, Path]) -> Path:
        """
        Validiert Input-Datei.
        
        Args:
            file_path: Pfad zur Datei
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ValidationError(f"Datei existiert nicht: {path}")
        
        if not path.is_file():
            raise ValidationError(f"Pfad ist keine Datei: {path}")
        
        if path.suffix.lower() not in FileValidator.SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"Nicht unterstütztes Dateiformat: {path.suffix}. "
                f"Unterstützt: {', '.join(FileValidator.SUPPORTED_EXTENSIONS)}"
            )
        
        # Dateigröße prüfen
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > FileValidator.MAX_FILE_SIZE_MB:
            raise ValidationError(
                f"Datei zu groß: {size_mb:.1f} MB. "
                f"Maximum: {FileValidator.MAX_FILE_SIZE_MB} MB"
            )
        
        return path
    
    @staticmethod
    def validate_directory(dir_path: Union[str, Path], create: bool = False) -> Path:
        """
        Validiert Verzeichnis.
        
        Args:
            dir_path: Pfad zum Verzeichnis
            create: Erstelle Verzeichnis falls nicht existent

        """
        path = Path(dir_path)
        
        if not path.exists():
            if create:
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise ValidationError(f"Verzeichnis existiert nicht: {path}")
        
        if not path.is_dir():
            raise ValidationError(f"Pfad ist kein Verzeichnis: {path}")
        
        return path
    
    @staticmethod
    def scan_input_directory(
        dir_path: Union[str, Path],
        recursive: bool = False
    ) -> List[Path]:
        """
        Scannt Verzeichnis nach unterstützten Dokumenten.
        
        Args:
            dir_path: Eingabe-Verzeichnis
            recursive: Rekursiv in Unterverzeichnisse

        """
        directory = FileValidator.validate_directory(dir_path)
        
        pattern = "**/*" if recursive else "*"
        files = []
        
        for ext in FileValidator.SUPPORTED_EXTENSIONS:
            files.extend(directory.glob(f"{pattern}{ext}"))
        
        if not files:
            raise ValidationError(
                f"Keine unterstützten Dokumente in {directory} gefunden. "
                f"Unterstützte Formate: {', '.join(FileValidator.SUPPORTED_EXTENSIONS)}"
            )
        
        return sorted(files)

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class APIResponseValidator:
    """Validiert API-Antworten von LLMs."""
    
    @staticmethod
    def validate_summary_response(
        response: Any,
        min_length: int = 50,
        max_length: int = 5000
    ) -> str:
        """
        Validiert generierte Zusammenfassung.
        
        Args:
            response: API-Response
            min_length: Minimale Zeichenlänge
            max_length: Maximale Zeichenlänge

        """
        if not response:
            raise ValidationError("Leere API-Response")
        
        # Extrahiere Text aus verschiedenen Response-Formaten
        if isinstance(response, dict):
            summary = response.get('text') or response.get('summary') or response.get('content')
            if not summary:
                raise ValidationError(
                    f"Keine Zusammenfassung in Response gefunden. Keys: {list(response.keys())}"
                )
        elif isinstance(response, str):
            summary = response
        else:
            raise ValidationError(f"Unbekanntes Response-Format: {type(response)}")
        
        summary = summary.strip()
        
        # Längen-Check
        if len(summary) < min_length:
            raise ValidationError(
                f"Zusammenfassung zu kurz: {len(summary)} Zeichen. Minimum: {min_length}"
            )
        
        if len(summary) > max_length:
            raise ValidationError(
                f"Zusammenfassung zu lang: {len(summary)} Zeichen. Maximum: {max_length}"
            )
        
        # Plausibilitäts-Checks
        if summary.count('\n') > 50:
            raise ValidationError("Zu viele Zeilenumbrüche in Zusammenfassung")
        
        # Prüfe auf verdächtige Patterns
        suspicious_patterns = [
            r'<\|im_end\|>',  # HuggingFace Artefakte
            r'\[INST\]',       # Llama Prompt-Marker
            r'<\|endoftext\|>',
            r'###\s*Instruction',  # Prompt-Leakage
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, summary):
                raise ValidationError(
                    f"Verdächtiges Pattern in Zusammenfassung gefunden: {pattern}"
                )
        
        return summary
    
    @staticmethod
    def validate_judge_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validiert LLM-as-a-Judge Response.
        
        Args:
            response: Judge-Response als Dictionary

        """
        required_criteria = [
            'correctness',
            'completeness',
            'coherence',
            'conciseness',
            'legal_terminology'
        ]
        
        missing = [c for c in required_criteria if c not in response]
        if missing:
            raise ValidationError(
                f"Fehlende Bewertungskriterien: {', '.join(missing)}"
            )
        
        # Validiere jeden Score
        for criterion, details in response.items():
            if criterion == 'overall_assessment':
                continue
            
            if not isinstance(details, dict):
                raise ValidationError(
                    f"Kriterium '{criterion}' muss Dictionary sein"
                )
            
            if 'score' not in details:
                raise ValidationError(
                    f"Kriterium '{criterion}' fehlt 'score'"
                )
            
            score = details['score']
            if not isinstance(score, (int, float)) or not 1 <= score <= 5:
                raise ValidationError(
                    f"Score für '{criterion}' muss zwischen 1 und 5 liegen, ist: {score}"
                )
        
        return response

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class TextValidator:
    """Validiert Text-Content."""
    
    @staticmethod
    def validate_text_length(
        text: str,
        min_words: int = 100,
        max_words: int = 50000,
        context_name: str = "Text"
    ) -> None:
        """
        Validiert Textlänge.
        
        Args:
            text: Zu validierender Text
            min_words: Minimale Wortanzahl
            max_words: Maximale Wortanzahl
            context_name: Name für Fehlermeldung

        """
        word_count = len(text.split())
        
        if word_count < min_words:
            raise ValidationError(
                f"{context_name} zu kurz: {word_count} Wörter. Minimum: {min_words}"
            )
        
        if word_count > max_words:
            raise ValidationError(
                f"{context_name} zu lang: {word_count} Wörter. Maximum: {max_words}"
            )
    
    @staticmethod
    def detect_language(text: str, expected: str = 'de') -> None:
        """
        Einfache Spracherkennung (für deutsche Texte).
        
        Args:
            text: Zu prüfender Text
            expected: Erwartete Sprache

        """
        # Einfacher Heuristik-basierter Check für Deutsch
        german_indicators = [
            'der', 'die', 'das', 'und', 'ist', 'nicht', 'haben',
            'werden', 'kann', 'durch', 'für', 'von', 'auf'
        ]
        
        text_lower = text.lower()
        match_count = sum(
            1 for indicator in german_indicators
            if f' {indicator} ' in text_lower
        )
        
        if match_count < 3:
            raise ValidationError(
                f"Text scheint nicht deutsch zu sein. "
                f"Nur {match_count} deutsche Indikator-Wörter gefunden."
            )

def validate_config_files(
    models_config_path: Union[str, Path],
    evaluation_config_path: Union[str, Path]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Lädt und validiert beide Config-Dateien.
    
    Args:
        models_config_path: Pfad zu models_config.yaml
        evaluation_config_path: Pfad zu evaluation_config.yaml

    """    
    def resolve_env_vars(value):
        """Ersetzt ${VAR_NAME} mit Umgebungsvariablen."""
        if not isinstance(value, str):
            return value
        
        # Ersetze ${VAR_NAME}
        pattern = r'\$\{([^}]+)\}'
        
        def replacer(match):
            var_name = match.group(1)
            return os.getenv(var_name, '')
        
        return re.sub(pattern, replacer, value)
    
    def process_dict(d):
        """Verarbeitet Dictionary rekursiv und ersetzt Env-Variablen."""
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = process_dict(value)
            elif isinstance(value, list):
                d[key] = [process_dict(item) if isinstance(item, dict) else resolve_env_vars(item) for item in value]
            else:
                d[key] = resolve_env_vars(value)
        return d
    
    try:
        with open(models_config_path, 'r', encoding='utf-8') as f:
            models_config = yaml.safe_load(f)
        
        with open(evaluation_config_path, 'r', encoding='utf-8') as f:
            evaluation_config = yaml.safe_load(f)
    
    except FileNotFoundError as e:
        raise ValidationError(f"Config-Datei nicht gefunden: {e}")
    except yaml.YAMLError as e:
        raise ValidationError(f"Fehler beim Parsen der YAML-Datei: {e}")
    
    models_config = process_dict(models_config)
    evaluation_config = process_dict(evaluation_config)
    
    # Validiere
    models_config = ConfigValidator.validate_models_config(models_config)
    evaluation_config = ConfigValidator.validate_evaluation_config(evaluation_config)
    
    return models_config, evaluation_config

def quick_validate(
    text: str,
    min_words: int = 100,
    max_words: int = 50000
) -> bool:
    """
    Schnelle Validierung ob Text verarbeitbar ist.
    
    Args:
        text: Zu prüfender Text
        min_words: Minimale Wortanzahl
        max_words: Maximale Wortanzahl

    """
    try:
        TextValidator.validate_text_length(text, min_words, max_words)
        return True
    except ValidationError:
        return False