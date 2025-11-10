"""
Logging Utilities für strukturiertes Logging mit Rotation, verschiedenen Log-Levels

"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional
import json

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class PipelineLogger:
    """
    Zentraler Logger für die gesamte Pipeline.

    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        max_bytes: int = 10_485_760,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialisiert Logger-System.
        
        Args:
            log_dir: Verzeichnis für Log-Dateien
            log_level: Logging-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            max_bytes: Maximale Größe pro Log-File
            backup_count: Anzahl Backup-Files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_level = getattr(logging, log_level.upper())
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # Hauptlogger
        self.main_logger = self._create_logger(
            "pipeline",
            self.log_dir / "pipeline.log"
        )
        
        # Spezialisierte Logger
        self.api_logger = self._create_logger(
            "api_calls",
            self.log_dir / "api_calls.log"
        )
        
        self.evaluation_logger = self._create_logger(
            "evaluation",
            self.log_dir / "evaluation.log"
        )
        
        self.error_logger = self._create_logger(
            "errors",
            self.log_dir / "errors.log",
            level=logging.ERROR
        )
        
    def _create_logger(
        self,
        name: str,
        log_file: Path,
        level: Optional[int] = None
    ) -> logging.Logger:
        """Erstellt konfigurierten Logger mit File- und Console-Handler."""
        logger = logging.getLogger(name)
        logger.setLevel(level or self.log_level)
        logger.handlers.clear()
        
        # Formatter mit Zeitstempel und Kontext
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File Handler mit Rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console Handler (nur für INFO und höher)
        if level != logging.ERROR:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def log_api_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration: float,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[dict] = None
    ):
        """
        Strukturiertes Logging von API-Calls für Kostenanalyse.
        
        Args:
            provider: azure_ai_foundry oder huggingface
            model: Modellname
            input_tokens: Anzahl Input-Tokens
            output_tokens: Anzahl Output-Tokens
            duration: Dauer in Sekunden
            success: Erfolg des Calls
            error: Fehlermeldung falls vorhanden
            metadata: Zusätzliche Metadaten (temperature, top_p, etc.)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            "duration_seconds": round(duration, 3),
            "success": success,
            "error": error,
            "metadata": metadata or {}
        }
        
        self.api_logger.info(json.dumps(log_entry, ensure_ascii=False))
        
        # Zusätzlich zum Hauptlog bei Fehler
        if not success:
            self.error_logger.error(
                f"API Call failed: {provider}/{model} - {error}"
            )
    
    def log_experiment_start(
        self,
        experiment_id: str,
        config: dict,
        num_documents: int
    ):
        """Loggt Start eines Experiments."""
        self.main_logger.info(
            f"Starting Experiment {experiment_id} | "
            f"Documents: {num_documents} | "
            f"Config: {json.dumps(config, ensure_ascii=False)}"
        )
    
    def log_experiment_end(
        self,
        experiment_id: str,
        duration: float,
        success_count: int,
        total_count: int
    ):
        """Loggt Ende eines Experiments."""
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        self.main_logger.info(
            f"Completed Experiment {experiment_id} | "
            f"Duration: {duration:.2f}s | "
            f"Success Rate: {success_rate:.1f}% ({success_count}/{total_count})"
        )
    
    def log_evaluation(
        self,
        document_id: str,
        model: str,
        metrics: dict
    ):
        """Loggt Evaluationsergebnisse."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id,
            "model": model,
            "metrics": metrics
        }
        self.evaluation_logger.info(json.dumps(log_entry, ensure_ascii=False))
    
    def info(self, message: str):
        """Standard Info-Log."""
        self.main_logger.info(message)
    
    def warning(self, message: str):
        """Warning-Log."""
        self.main_logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        """Error-Log mit optionalem Exception-Traceback."""
        self.main_logger.error(message, exc_info=exc_info)
        self.error_logger.error(message, exc_info=exc_info)
    
    def debug(self, message: str):
        """Debug-Log."""
        self.main_logger.debug(message)
    
    def get_api_statistics(self) -> dict:
        """
        Analysiert API-Call Logs und gibt Statistiken zurück.
        
        Returns:
            Dictionary mit Token-Counts, Kosten, Fehlerrate
        """
        api_log_file = self.log_dir / "api_calls.log"
        
        if not api_log_file.exists():
            return {}
        
        stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "by_provider": {},
            "by_model": {}
        }
        
        with open(api_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Extrahiere JSON aus Log-Zeile
                    json_start = line.find('{')
                    if json_start == -1:
                        continue
                    
                    entry = json.loads(line[json_start:])
                    
                    stats["total_calls"] += 1
                    
                    if entry["success"]:
                        stats["successful_calls"] += 1
                    else:
                        stats["failed_calls"] += 1
                    
                    stats["total_input_tokens"] += entry["tokens"]["input"]
                    stats["total_output_tokens"] += entry["tokens"]["output"]
                    
                    # Provider-Stats
                    provider = entry["provider"]
                    if provider not in stats["by_provider"]:
                        stats["by_provider"][provider] = {
                            "calls": 0,
                            "tokens": 0
                        }
                    stats["by_provider"][provider]["calls"] += 1
                    stats["by_provider"][provider]["tokens"] += entry["tokens"]["total"]
                    
                    # Model-Stats
                    model = entry["model"]
                    if model not in stats["by_model"]:
                        stats["by_model"][model] = {
                            "calls": 0,
                            "tokens": 0,
                            "avg_duration": []
                        }
                    stats["by_model"][model]["calls"] += 1
                    stats["by_model"][model]["tokens"] += entry["tokens"]["total"]
                    stats["by_model"][model]["avg_duration"].append(
                        entry["duration_seconds"]
                    )
                    
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Durchschnittliche Dauer berechnen
        for model in stats["by_model"]:
            durations = stats["by_model"][model]["avg_duration"]
            if durations:
                stats["by_model"][model]["avg_duration"] = sum(durations) / len(durations)
            else:
                stats["by_model"][model]["avg_duration"] = 0
        
        return stats


# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
_global_logger: Optional[PipelineLogger] = None

def get_logger() -> PipelineLogger:
    """
    Gibt globalen Logger zurück (Singleton Pattern).

    """
    global _global_logger
    if _global_logger is None:
        _global_logger = PipelineLogger()
    return _global_logger

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def init_logger(
    log_dir: str = "logs",
    log_level: str = "INFO"
) -> PipelineLogger:
    """
    Initialisiert globalen Logger mit Custom-Config.
    
    Args:
        log_dir: Log-Verzeichnis
        log_level: Logging-Level

    """
    global _global_logger
    _global_logger = PipelineLogger(log_dir=log_dir, log_level=log_level)
    return _global_logger