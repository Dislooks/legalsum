"""
Verwaltung und Speicherung von Zusammenfassungsergebnissen mit Metadaten.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from dataclasses import asdict

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class OutputHandler:
    """
    Handler für die strukturierte Speicherung von Zusammenfassungsergebnissen.

    """
    
    def __init__(
        self,
        base_output_dir: Path,
        create_dirs: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialisiert den OutputHandler.
        
        Args:
            base_output_dir: Basis-Ausgabeverzeichnis
            create_dirs: Automatisch Verzeichnisse erstellen
            logger: Optional Logger-Instanz
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.current_experiment = None
        self.current_config = None
        
        if create_dirs:
            self._create_directory_structure()
    
    def _create_directory_structure(self) -> None:
        """Erstellt die grundlegende Verzeichnisstruktur."""
        directories = [
            self.base_output_dir / "summaries" / "json",
            self.base_output_dir / "summaries" / "txt",
            self.base_output_dir / "metadata",
            self.base_output_dir / "experiments",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Verzeichnisstruktur erstellt in: {self.base_output_dir}")
    
    def set_experiment_context(self, experiment_name: str, config_name: str) -> None:
        """
        Setzt den Kontext für ein Experiment (für organisierte Ausgabe).
        
        Args:
            experiment_name: Name des Experiments
            config_name: Name der Konfiguration (z.B. "temp_0.7_topp_0.9")
        """
        self.current_experiment = experiment_name
        self.current_config = config_name
        
        # Erstelle Experiment-spezifische Verzeichnisse
        exp_dir = self.get_experiment_path(experiment_name) / config_name
        (exp_dir / "summaries").mkdir(parents=True, exist_ok=True)
        (exp_dir / "metadata").mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Experiment-Kontext gesetzt: {experiment_name}/{config_name}")
    
    def get_experiment_path(self, experiment_name: str) -> Path:
        """Gibt den Pfad für ein spezifisches Experiment zurück."""
        return self.base_output_dir / "experiments" / experiment_name
    
    def save_summary(
        self,
        result,  # Type: SummarizationResult (vermeidet zirkuläre Imports)
        output_format: str = "both"
    ) -> Dict[str, Path]:
        """
        Speichert eine Zusammenfassung in verschiedenen Formaten.
        
        Args:
            result: SummarizationResult-Objekt
            output_format: "json", "txt" oder "both"

        """
        saved_paths = {}
        
        # Bestimme Ausgabeverzeichnis
        if self.current_experiment and self.current_config:
            base_dir = self.get_experiment_path(self.current_experiment) / self.current_config / "summaries"
        else:
            base_dir = self.base_output_dir / "summaries"
        
        # Erstelle Dateinamen (ohne Erweiterung des Originaldokuments)
        source_name = Path(result.source_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Speichere als JSON (strukturiert mit allen Metadaten)
        if output_format in ["json", "both"]:
            json_dir = base_dir / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            json_path = json_dir / f"{source_name}_{timestamp}.json"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            
            saved_paths['json'] = json_path
            self.logger.debug(f"JSON gespeichert: {json_path}")
        
        # Speichere als TXT (lesbare Version)
        if output_format in ["txt", "both"]:
            txt_dir = base_dir / "txt"
            txt_dir.mkdir(parents=True, exist_ok=True)
            txt_path = txt_dir / f"{source_name}_{timestamp}.txt"
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(self._format_summary_as_text(result))
            
            saved_paths['txt'] = txt_path
            self.logger.debug(f"TXT gespeichert: {txt_path}")
        
        return saved_paths
    
    def _format_summary_as_text(self, result) -> str:
        """
        Formatiert ein SummarizationResult als lesbaren Text.
        
        Args:
            result: SummarizationResult-Objekt
 
        """
        lines = [
            f"ZUSAMMENFASSUNG: {result.source_file}",
            "",
            result.summary,
            "",
        ]
                
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_batch_metadata(
        self,
        results: List,  # Type: List[SummarizationResult]
        config  # Type: SummarizationConfig
    ) -> Path:
        """
        Speichert Metadaten für einen Batch von Zusammenfassungen.
        
        Args:
            results: Liste von SummarizationResult-Objekten
            config: Verwendete SummarizationConfig

        """
        # Bestimme Ausgabepfad
        if self.current_experiment and self.current_config:
            metadata_dir = self.get_experiment_path(self.current_experiment) / self.current_config / "metadata"
        else:
            metadata_dir = self.base_output_dir / "metadata"
        
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_path = metadata_dir / f"batch_metadata_{timestamp}.json"
        
        # Erstelle Metadaten-Struktur
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "configuration": asdict(config),
            "total_documents": len(results),
            "statistics": self._calculate_batch_statistics(results),
            "results": [
                {
                    "source_file": r.source_file,
                    "timestamp": r.timestamp,
                    "processing_time": r.processing_time,
                    "metadata": r.metadata
                }
                for r in results
            ]
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Batch-Metadaten gespeichert: {metadata_path}")
        return metadata_path
    
    def _calculate_batch_statistics(self, results: List) -> Dict[str, Any]:
        """
        Berechnet statistische Kennzahlen für einen Batch.
        
        Args:
            results: Liste von SummarizationResult-Objekten

        """
        if not results:
            return {}
        
        processing_times = [r.processing_time for r in results]
        compression_ratios = [r.metadata.get("compression_ratio", 0) for r in results]
        summary_lengths = [r.metadata.get("summary_length", 0) for r in results]
        
        return {
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "avg_compression_ratio": sum(compression_ratios) / len(compression_ratios),
            "avg_summary_length": sum(summary_lengths) / len(summary_lengths),
            "total_processing_time": sum(processing_times)
        }
    
    def export_results_to_csv(
        self,
        results: List,  # Type: List[SummarizationResult]
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Exportiert Ergebnisse als CSV-Datei für weitere Analyse.
        
        Args:
            results: Liste von SummarizationResult-Objekten
            output_path: Optional spezifischer Ausgabepfad

        """
        if output_path is None:
            if self.current_experiment and self.current_config:
                csv_dir = self.get_experiment_path(self.current_experiment) / self.current_config
            else:
                csv_dir = self.base_output_dir
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = csv_dir / f"results_{timestamp}.csv"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Definiere CSV-Felder
        fieldnames = [
            "source_file",
            "timestamp",
            "model_name",
            "temperature",
            "top_p",
            "processing_time",
            "original_length",
            "summary_length",
            "compression_ratio"
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    "source_file": result.source_file,
                    "timestamp": result.timestamp,
                    "model_name": result.model_config.get("model_name", "N/A"),
                    "temperature": result.model_config.get("temperature", "N/A"),
                    "top_p": result.model_config.get("top_p", "N/A"),
                    "processing_time": result.processing_time,
                    "original_length": result.metadata.get("original_length", 0),
                    "summary_length": result.metadata.get("summary_length", 0),
                    "compression_ratio": result.metadata.get("compression_ratio", 0)
                }
                writer.writerow(row)
        
        self.logger.info(f"Ergebnisse als CSV exportiert: {output_path}")
        return output_path
    
    def create_consolidated_output(
        self,
        results: List,  # Type: List[SummarizationResult]
        output_filename: str = "consolidated_summaries.txt"
    ) -> Path:
        """
        Erstellt eine konsolidierte Datei mit allen Zusammenfassungen.
        Nützlich für manuelle Überprüfung oder LLM-as-a-Judge.
        
        Args:
            results: Liste von SummarizationResult-Objekten
            output_filename: Name der Ausgabedatei

        """
        if self.current_experiment and self.current_config:
            output_dir = self.get_experiment_path(self.current_experiment) / self.current_config
        else:
            output_dir = self.base_output_dir
        
        output_path = output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("KONSOLIDIERTE ZUSAMMENFASSUNGEN\n")
            f.write(f"Anzahl Dokumente: {len(results)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"DOKUMENT {i}/{len(results)}: {result.source_file}\n")
                f.write("=" * 80 + "\n\n")
                f.write(result.summary)
                f.write("\n\n")
        
        self.logger.info(f"Konsolidierte Ausgabe erstellt: {output_path}")
        return output_path
    
    def save_comparison_file(
        self,
        original_text: str,
        summary: str,
        source_file: str,
        gold_standard: Optional[str] = None
    ) -> Path:
        """
        Erstellt eine Vergleichsdatei mit Originaltext, Zusammenfassung und optional Gold-Standard.
        Ideal für LLM-as-a-Judge Evaluation.
        
        Args:
            original_text: Originaltext
            summary: Generierte Zusammenfassung
            source_file: Name der Quelldatei
            gold_standard: Optional manuell erstellte Referenzzusammenfassung

        """
        if self.current_experiment and self.current_config:
            output_dir = self.get_experiment_path(self.current_experiment) / self.current_config / "comparisons"
        else:
            output_dir = self.base_output_dir / "comparisons"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        source_name = Path(source_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{source_name}_comparison_{timestamp}.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"VERGLEICHSDOKUMENT: {source_file}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("ORIGINALTEXT:\n")
            f.write("-" * 80 + "\n")
            f.write(original_text)
            f.write("\n\n")
            
            f.write("GENERIERTE ZUSAMMENFASSUNG:\n")
            f.write("-" * 80 + "\n")
            f.write(summary)
            f.write("\n\n")
            
            if gold_standard:
                f.write("GOLD-STANDARD (REFERENZ):\n")
                f.write("-" * 80 + "\n")
                f.write(gold_standard)
                f.write("\n\n")
            
            f.write("=" * 80 + "\n")
        
        self.logger.debug(f"Vergleichsdatei erstellt: {output_path}")
        return output_path
    
    def get_output_structure_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über die aktuelle Ausgabestruktur zurück.
        
        Returns:
            Dictionary mit Struktur-Informationen
        """
        info = {
            "base_directory": str(self.base_output_dir),
            "current_experiment": self.current_experiment,
            "current_config": self.current_config,
            "directories": {}
        }
        
        # Liste alle Verzeichnisse auf
        for path in self.base_output_dir.rglob("*"):
            if path.is_dir():
                rel_path = path.relative_to(self.base_output_dir)
                file_count = len(list(path.glob("*")))
                info["directories"][str(rel_path)] = file_count
        
        return info
    
    def cleanup_old_files(self, days_to_keep: int = 30) -> int:
        """
        Löscht alte Ausgabedateien (optional für Produktivumgebungen).
        
        Args:
            days_to_keep: Anzahl Tage, die Dateien behalten werden sollen

        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0
        
        for file_path in self.base_output_dir.rglob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    file_path.unlink()
                    deleted_count += 1
                    self.logger.debug(f"Gelöscht: {file_path}")
        
        self.logger.info(f"{deleted_count} alte Dateien gelöscht (älter als {days_to_keep} Tage)")
        return deleted_count
