"""
Zentrale Pipeline zur Steuerung des Zusammenfassungsprozesses
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from dataclasses import dataclass, asdict

from ..llm_interface.base_client import BaseLLMClient
from ..document_processing.readers import DocumentReader
from .prompt_templates import PromptTemplate
from .output_handler import OutputHandler

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
@dataclass
class SummarizationConfig:
    """Konfiguration für einen Summarization-Durchlauf."""
    model_name: str
    endpoint_type: str  # 'azure' oder 'huggingface'
    temperature: float
    top_p: float
    max_response_length: int
    prompt_template_name: str = "default"
    
    def validate(self) -> None:
        """Validiert die Konfigurationsparameter."""
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature muss zwischen 0 und 2 liegen")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p muss zwischen 0 und 1 liegen")
        if self.max_response_length < 50:
            raise ValueError("max_response_length muss mindestens 50 sein")

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
@dataclass
class SummarizationResult:
    """Ergebnis einer einzelnen Zusammenfassung."""
    source_file: str
    summary: str
    model_config: Dict[str, Any]
    timestamp: str
    processing_time: float
    metadata: Dict[str, Any]

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class SummarizationPipeline:
    """
    Hauptklasse für die Dokumentenzusammenfassung.

    """
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        experiment_params: Dict[str, Any],
        output_dir: Path,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialisiert die Pipeline.
        
        Args:
            llm_client: Client für LLM-Kommunikation (Azure/HuggingFace)
            output_handler: Handler für Ausgabespeicherung
            logger: Optional Logger-Instanz
        """
        self.llm_client = llm_client
        self.experiment_params = experiment_params
        self.logger = logger or logging.getLogger(__name__)
        self.prompt_template = PromptTemplate()

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        
        self.output_handler = OutputHandler(
            base_output_dir=output_dir,
            create_dirs=True,
            logger=self.logger
        )
        
    def process_documents(
        self,
        documents: List[Tuple[str, Dict]]
    ) -> List[SummarizationResult]:
        """
        Verarbeitet eine Liste von bereits eingelesenen Dokumenten.
        """
        results = []
        
        for i, (text_content, metadata) in enumerate(documents, 1):
            self.logger.info(f"Verarbeite Dokument {i}/{len(documents)}")
            
            try:
                start_time = datetime.now()
                
                # Prompt erstellen
                prompt = self.prompt_template.create_prompt(
                    template_name=self.experiment_params.get('prompt_template_name', 'default'),
                    text=text_content,
                    max_length=self.experiment_params.get('max_response_length', 7000)
                )
                
                # LLM aufrufen - gibt LLMResponse Objekt zurück
                llm_response = self.llm_client.generate(
                    prompt=prompt,
                    temperature=self.experiment_params.get('temperature', 0.7),
                    top_p=self.experiment_params.get('top_p', 0.9),
                    max_tokens=self.experiment_params.get('max_response_length', 7000)
                )
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                # Ergebnis erstellen - WICHTIG: .text extrahieren!
                result = SummarizationResult(
                    source_file=metadata.get('filename', f'document_{i}'),
                    summary=llm_response.text,  # .text hinzufügen!
                    model_config=self.experiment_params,
                    timestamp=datetime.now().isoformat(),
                    processing_time=processing_time,
                    metadata={
                        **metadata,
                        "original_length": len(text_content),
                        "summary_length": len(llm_response.text),  # .text
                        "compression_ratio": len(llm_response.text) / len(text_content) if text_content else 0,
                        "llm_metadata": llm_response.to_dict()  # Speichere vollständige LLM-Info
                    }
                )
                
                results.append(result)
                self.output_handler.save_summary(result)
                
            except Exception as e:
                self.logger.error(f"Fehler bei Dokument {i}: {e}", exc_info=True)
                continue
        
        return results
    
    def process_directory(
        self,
        input_dir: Path,
        config: SummarizationConfig,
        file_extensions: Optional[List[str]] = None
    ) -> List[SummarizationResult]:
        """
        Verarbeitet alle Dokumente in einem Verzeichnis.
        
        Args:
            input_dir: Pfad zum Eingabeverzeichnis
            config: Konfiguration für die Zusammenfassung
            file_extensions: Liste erlaubter Dateierweiterungen (z.B. ['.pdf', '.docx'])

        """
        config.validate()
        
        if file_extensions is None:
            file_extensions = ['.pdf', '.docx', '.txt']
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Eingabeverzeichnis nicht gefunden: {input_dir}")
        
        # Sammle alle relevanten Dateien
        files = []
        for ext in file_extensions:
            files.extend(input_path.glob(f"**/*{ext}"))
        
        if not files:
            self.logger.warning(f"Keine Dateien mit Erweiterungen {file_extensions} in {input_dir} gefunden")
            return []
        
        self.logger.info(f"Starte Verarbeitung von {len(files)} Dokumenten mit Konfiguration: {config}")
        
        results = []
        for i, file_path in enumerate(files, 1):
            self.logger.info(f"Verarbeite Dokument {i}/{len(files)}: {file_path.name}")
            try:
                result = self._process_single_document(file_path, config)
                results.append(result)
                
                # Speichere Zwischenergebnis
                self.output_handler.save_summary(result)
                
            except Exception as e:
                self.logger.error(f"Fehler bei Verarbeitung von {file_path.name}: {str(e)}", exc_info=True)
                continue
        
        # Speichere Batch-Metadaten
        self.output_handler.save_batch_metadata(results, config)
        
        self.logger.info(f"Verarbeitung abgeschlossen. {len(results)}/{len(files)} Dokumente erfolgreich zusammengefasst")
        return results
    
    def _process_single_document(
        self,
        file_path: Path,
        config: SummarizationConfig
    ) -> SummarizationResult:
        """
        Verarbeitet ein einzelnes Dokument.
        
        Args:
            file_path: Pfad zur Eingabedatei
            config: Konfiguration für die Zusammenfassung

        """
        start_time = datetime.now()
        
        # Dokument einlesen
        text_content, metadata = self.document_reader.read_document(file_path)
        
        # Prompt erstellen
        prompt = self.prompt_template.create_prompt(
            template_name=config.prompt_template_name,
            text=text_content,
            max_length=config.max_response_length
        )
        
        # LLM aufrufen
        summary = self.llm_client.generate(
            prompt=prompt,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_response_length
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Ergebnis erstellen
        result = SummarizationResult(
            source_file=file_path.name,
            summary=summary,
            model_config=asdict(config),
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            metadata={
                **metadata,
                "original_length": len(text_content),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text_content) if text_content else 0
            }
        )
        
        return result
    
    def process_single_file(
        self,
        file_path: Path,
        config: SummarizationConfig
    ) -> SummarizationResult:
        """
        Öffentliche Methode zur Verarbeitung einer einzelnen Datei.
        
        Args:
            file_path: Pfad zur Datei
            config: Summarization-Konfiguration

        """
        config.validate()
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        
        self.logger.info(f"Verarbeite einzelne Datei: {file_path}")
        result = self._process_single_document(Path(file_path), config)
        self.output_handler.save_summary(result)
        
        return result
    
    def run_experiment(
        self,
        input_dir: Path,
        experiment_config: Dict[str, Any],
        experiment_name: str
    ) -> Dict[str, List[SummarizationResult]]:
        """
        Führt ein vollständiges Experiment mit mehreren Parameterkombinationen durch.
        
        Args:
            input_dir: Pfad zu den Eingabedokumenten
            experiment_config: Dict mit Listen von Parametern zum Testen
            experiment_name: Name des Experiments für die Ausgabe

        """
        self.logger.info(f"Starte Experiment: {experiment_name}")
        
        temperatures = experiment_config.get("temperatures", [0.7])
        top_p_values = experiment_config.get("top_p_values", [0.9])
        
        all_results = {}
        total_runs = len(temperatures) * len(top_p_values)
        current_run = 0
        
        for temp in temperatures:
            for top_p in top_p_values:
                current_run += 1
                config_name = f"temp_{temp}_topp_{top_p}"
                
                self.logger.info(f"Durchlauf {current_run}/{total_runs}: {config_name}")
                
                config = SummarizationConfig(
                    model_name=experiment_config["model_name"],
                    endpoint_type=experiment_config["endpoint_type"],
                    temperature=temp,
                    top_p=top_p,
                    max_response_length=experiment_config.get("max_response_length", 500),
                    prompt_template_name=experiment_config.get("prompt_template_name", "default")
                )
                
                # Setze Ausgabeverzeichnis für diese Konfiguration
                self.output_handler.set_experiment_context(experiment_name, config_name)
                
                results = self.process_directory(input_dir, config)
                all_results[config_name] = results
        
        # Speichere Experiment-Übersicht
        self._save_experiment_summary(experiment_name, all_results, experiment_config)
        
        self.logger.info(f"Experiment {experiment_name} abgeschlossen")
        return all_results
    
    def _save_experiment_summary(
        self,
        experiment_name: str,
        results: Dict[str, List[SummarizationResult]],
        config: Dict[str, Any]
    ) -> None:
        """Speichert eine Zusammenfassung des Experiments."""
        summary = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "configuration": config,
            "results_summary": {
                config_name: {
                    "num_documents": len(result_list),
                    "avg_processing_time": sum(r.processing_time for r in result_list) / len(result_list) if result_list else 0,
                    "avg_compression_ratio": sum(r.metadata.get("compression_ratio", 0) for r in result_list) / len(result_list) if result_list else 0
                }
                for config_name, result_list in results.items()
            }
        }
        
        output_path = self.output_handler.get_experiment_path(experiment_name) / "experiment_summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Experiment-Zusammenfassung gespeichert: {output_path}")
