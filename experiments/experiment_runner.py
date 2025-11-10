
import json
import yaml
import logging
import sys
import unicodedata
from pathlib import Path
from dotenv import load_dotenv
import os, yaml

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / "config" / "credentials.env")

from typing import Dict, List, Optional, Union, Tuple, Any 
from datetime import datetime
from dataclasses import dataclass, asdict
import itertools
from tqdm import tqdm

from src.llm_interface.azure_client import AzureAIFoundryClient
from src.llm_interface.huggingface_client import HuggingFaceClient
from src.llm_interface.base_client import ModelConfig
from src.summarization.pipeline import SummarizationPipeline
from src.evaluation.automated_metrics import AutomatedMetrics
from src.evaluation.llm_judge import LLMJudge
from src.utils.logger import PipelineLogger
from src.document_processing.readers import read_documents_from_directory
from src.document_processing.document_cache import DocumentCache


# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
@dataclass
class ExperimentConfig:
    """Konfiguration eines einzelnen Experiments"""
    experiment_id: str
    model_provider: str  # 'azure_ai_foundry' oder 'huggingface'
    model_name: str
    temperature: float
    top_k: int
    top_p_values: float
    max_output_length: int = 3000
    repetition_penalty: float = 1.1
    timestamp: str = None
    document_type: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentConfig':
        return cls(**data)

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class ExperimentRunner:
    """
    Führt systematische Experimente durch und verwaltet Ergebnisse.

    """
    
    def __init__(
        self,
        config_path: str = "config/models_config.yaml",
        eval_config_path: str = "config/evaluation_config.yaml",
        experiments_dir: str = "experiments",
        results_dir: str = "results",
        log_level: str = "INFO"
    ):
        self.config_path = Path(config_path)
        self.eval_config_path = Path(eval_config_path)
        self.experiments_dir = Path(experiments_dir)
        self.results_dir = Path(results_dir)
        
        # Verzeichnisse erstellen
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Konfigurationen laden
        self.model_config = self._load_yaml(self.config_path)
        self.eval_config = self._load_yaml(self.eval_config_path)
                
        # Logger
        self.pipeline_logger = PipelineLogger(
        log_dir=str(PROJECT_ROOT / "logs"),
        log_level=log_level
        )
        
        self.logger = self.pipeline_logger.main_logger
        self.document_cache = DocumentCache(logger=self.logger)
        
        # LLM Clients initialisieren
        self.azure_client = None
        self.hf_client = None
    
    def preload_documents(
        self,
        input_dir: Path,
        extensions: List[str],
        max_documents: Optional[int] = None
    ) -> int:
        """Lädt Dokumente einmal in den Cache."""
        if extensions is None:
            extensions = ['.pdf', '.docx', '.txt']
        
        self.logger.info("=" * 70)
        self.logger.info("ðŸ“¦ PRELOADING DOCUMENTS")
        self.logger.info("=" * 70)
        
        num_docs = self.document_cache.load_from_directory(
            directory=input_dir,
            extensions=extensions,
            max_documents=max_documents
        )
        
        self.logger.info(f"Cache bereit: {num_docs} Dokumente")
        self.logger.info("=" * 70)
        
        return num_docs
    
    @staticmethod
    def _extract_group_prefix(experiment_id: str) -> str:
        """Extrahiert den Gruppenpräfix aus einer experiment_id"""
        parts = experiment_id.split("_")
        if parts[-1].isdigit():
            return "_".join(parts[:-1])
        return experiment_id
    
    def _load_yaml(self, path: Path) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
        # Environment-Variablen expandieren
        expanded = os.path.expandvars(raw)
        return yaml.safe_load(expanded)
    
    def _init_clients(self):
        """LLM Clients lazy initialisieren"""
        if self.azure_client is None:
            self.azure_client = AzureAIFoundryClient()
        if self.hf_client is None:
            self.hf_client = HuggingFaceClient()
            
    def _create_llm_client(self, provider: str, model_name: str, config: ExperimentConfig):
        """Erstellt LLM-Client basierend auf Provider und Config"""
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        if provider == "azure_openai":
            azure_config = self.model_config['azure_openai']
            model_info = azure_config.get('models', {}).get(model_name, {})

            model_config = ModelConfig(
                model_name=model_name,
                temperature=config.temperature,
                max_tokens=model_info.get('max_tokens', 8192),
                top_p=config.top_p_values,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            return AzureAIFoundryClient(
                config=model_config,
                azure_openai_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_openai_endpoint=azure_config.get("endpoint_url"),
                azure_openai_api_version=azure_config.get("api_version"),
                deployment_name=model_info.get("deployment_name", model_name)
            )

        elif provider == "azure_ai_foundry":
            azure_config = self.model_config['azure_ai_foundry']
            model_info = azure_config.get('models', {}).get(model_name, {})

            model_config = ModelConfig(
                model_name=model_name,
                temperature=config.temperature,
                max_tokens=model_info.get('max_tokens', 8192),
                top_p=config.top_p_values,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )

            return AzureAIFoundryClient(
                config=model_config,
                ai_foundry_key=os.getenv("AZURE_AI_KEY"),
                ai_foundry_endpoint=azure_config.get("endpoint_url"),
                deployment_name=model_info.get("deployment_name", model_name)
            )
            
        elif provider == 'huggingface':
            hf_config = self.model_config['huggingface']
            model_info = hf_config.get('models', {}).get(model_name, {})
            
            model_config = ModelConfig(
                model_name=model_info.get('model_id', model_name),
                temperature=config.temperature,
                max_tokens=model_info.get('max_new_tokens', config.max_output_length),
                top_k=config.top_k,
                top_p=config.top_p_values,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            inference_endpoint = model_info.get('inference_endpoint_url')

            return HuggingFaceClient(
                config=model_config,
                api_key=os.getenv('HF_API_KEY'),
                inference_endpoint_url=inference_endpoint
            )
        else:
            raise ValueError(f"Unbekannter Provider: {provider}")

    def create_grid_search_experiments(
        self,
        models: List[str],
        temperatures: List[float],
        top_k: int,
        top_p_values: List[float],
        experiment_name: str = "grid_search",
        document_type: Optional[str] = None
    ) -> List[ExperimentConfig]:
        """
        Erstellt Grid Search Experimente für alle Kombinationen.
        
        Args:
            models: Liste von Modellnamen
            temperatures: Liste von Temperature-Werten
            top_k: Top-K Wert
            top_p_values: Liste von Top-P Werten
            experiment_name: Name für Experiment-Gruppe
            document_type: "court_ruling" oder "contract" (optional)
            
        """
        experiments = []
        experiment_id_base = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        combinations = list(itertools.product(models, temperatures, top_k, top_p_values))
        
        self.logger.info(
            f"Erstelle {len(combinations)} Experimente für Grid Search "
            f"(Dokumententyp: {document_type or 'default'})"
        )
        
        for idx, (model, temp, top_k, top_p_values) in enumerate(combinations, 1):
            provider = self._get_provider_for_model(model)
            
            config = ExperimentConfig(
                experiment_id=f"{experiment_id_base}_{idx:03d}",
                model_provider=provider,
                model_name=model,
                temperature=temp,
                top_k=top_k,
                top_p_values=top_p_values,
                document_type=document_type
            )
            
            experiments.append(config)
        
        # Experiments als JSON speichern
        self._save_experiment_configs(experiments, experiment_name)
        
        return experiments
    
    def _get_prompt_template_for_type(self, document_type: Optional[str]) -> str:
        """
        Bestimmt den Prompt-Template-Namen basierend auf Dokumententyp.
        
        Args:
            document_type: "court_ruling", "contract", oder None

        """
        if document_type == "court_ruling":
            return "court_ruling"
        elif document_type == "contract":
            return "contract"
        else:
            return "legal"
        
    def _get_provider_for_model(self, model_name: str) -> str:
        """Bestimmt Provider basierend auf Modellnamen"""
        azure_openai_models = self.model_config.get('azure_openai', {}).get('models', {})
        azure_foundry_models = self.model_config.get('azure_ai_foundry', {}).get('models', {})
        hf_models = self.model_config.get('huggingface', {}).get('models', {})

        if model_name in azure_openai_models:
            return 'azure_openai'
        elif model_name in azure_foundry_models:
            return 'azure_ai_foundry'
        elif model_name in hf_models:
            return 'huggingface'
        else:
            raise ValueError(f"Modell '{model_name}' nicht in Konfiguration gefunden")

    
    def _save_experiment_configs(self, experiments: List[ExperimentConfig], name: str):
        """Speichert Experiment-Konfigurationen"""
        config_file = self.experiments_dir / "experiment_configs" / f"{name}_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        configs_dict = [exp.to_dict() for exp in experiments]
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(configs_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Experiment-Konfigurationen gespeichert: {config_file}")
    
    def load_experiment_configs(self, config_name: str) -> List[ExperimentConfig]:
        """Lädt gespeicherte Experiment-Konfigurationen"""
        config_file = self.experiments_dir / "experiment_configs" / f"{config_name}_config.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Konfiguration nicht gefunden: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            configs_dict = json.load(f)
        
        experiments = [ExperimentConfig.from_dict(cfg) for cfg in configs_dict]
        
        self.logger.info(f"Geladen: {len(experiments)} Experiment-Konfigurationen")
        return experiments

    def run_experiment(
        self,
        config: ExperimentConfig,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        evaluate: bool = True,
        use_llm_judge: bool = False,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Führt ein einzelnes Experiment durch.
        
        Args:
            config: Experiment-Konfiguration
            input_dir: Verzeichnis mit Eingabedokumenten
            output_dir: Ausgabeverzeichnis (optional)
            evaluate: Automatische Evaluation durchführen
            use_llm_judge: LLM-as-a-Judge verwenden
            document_type: Urteile oder Verträge

        """
        self.logger.info(f"Starte Experiment: {config.experiment_id}")
        self.logger.info(f"  Modell: {config.model_name}")
        self.logger.info(f"  Temperature: {config.temperature}, Top-P: {config.top_p_values}")
        self.logger.info(f"Dokumententyp: {config.document_type or 'default'}")
        
        # Output-Verzeichnis
        if output_dir is None:
            output_dir = self.results_dir / "summaries" / config.experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM-Client erstellen
        llm_client = self._create_llm_client(
            provider=config.model_provider,
            model_name=config.model_name,
            config=config
        )
        
        # Experiment-Parameter vorbereiten
        experiment_params = {
            'model_name': config.model_name,
            'model_provider': config.model_provider, 
            'temperature': config.temperature,
            'top_k': config.top_k,
            'top_p': config.top_p_values,
            'max_response_length': config.max_output_length,
            'repetition_penalty': config.repetition_penalty,
            'prompt_template_name': self._get_prompt_template_for_type(config.document_type)
        }
        
        # Pipeline mit korrekten Parametern erstellen
        pipeline = SummarizationPipeline(
            llm_client=llm_client,
            experiment_params=experiment_params,
            output_dir=output_dir,
            logger=self.logger
        )
        if use_cache and not self.document_cache.is_empty():
            self.logger.info("Verwende Dokumente aus Cache")
            documents = self.document_cache.get_documents()
            documents_raw = [
                {'text': text, 'metadata': metadata}
                for text, metadata in documents
            ]
        else:        
            if input_dir is None:
                raise ValueError("input_dir erforderlich wenn use_cache=False")
        
        # Konvertiere zu Tupel-Format (text, metadata)
        documents = [
            (doc['text'], doc['metadata']) 
            for doc in documents_raw
        ]
        
        self.logger.info(f"Verarbeite {len(documents)} Dokumente...")
        
        # Verwende process_documents statt Datei-Iteration
        summaries = pipeline.process_documents(documents)
        
        def sanitize_string(s: str) -> str:
            """Entfernt problematische Unicode-Zeichen"""
            # Normalisiere Unicode (NFD = decomposed form)
            s = unicodedata.normalize('NFKD', s)
            # Encode/Decode um fehlerhafte Zeichen zu entfernen
            s = s.encode('utf-8', errors='ignore').decode('utf-8')
            return s

        # Metadaten sammeln
        all_meta = []
        for idx, (summary_result, doc_raw) in enumerate(zip(summaries, documents_raw)):
            meta = {
                'document_id': sanitize_string(doc_raw['metadata'].get('filename', f'doc_{idx}')),
                'document_path': str(doc_raw['metadata'].get('filepath', '')),
                'document_type': config.document_type or 'default',
                'prompt_template': experiment_params['prompt_template_name'],
                'experiment_id': config.experiment_id,
                'model': config.model_name,
                'temperature': config.temperature,
                'top_k': config.top_k,
                'top_p': config.top_p_values,
                'summary_length': len(summary_result.summary.split()) if hasattr(summary_result, 'summary') else 0,
                'timestamp': datetime.now().isoformat()
            }
            all_meta.append(meta)
        
        # Metadaten speichern
        metadata_file = output_dir / "experiment_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_meta, f, indent=2, ensure_ascii=False)
        
        # Evaluation
        eval_results = {}
        if evaluate:
            eval_results = self._run_evaluation(
                summaries=summaries,
                config=config,
                use_llm_judge=use_llm_judge,
                output_dir=output_dir,
                document_type=config.document_type
            )
        
        # Experiment-Report erstellen
        report = {
            'experiment_id': config.experiment_id,
            'config': config.to_dict(),
            'document_type': config.document_type or 'default',
            'prompt_template': experiment_params['prompt_template_name'],
            'num_documents': len(summaries),
            'output_dir': str(output_dir),
            'evaluation_results': eval_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Report speichern
        report_file = output_dir / "experiment_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Experiment abgeschlossen: {config.experiment_id}")
        return report
    
    def _save_summary(
        self,
        summary_result: Dict,
        output_path: Path,
        config: ExperimentConfig
    ):
        """Speichert einzelne Zusammenfassung mit Metadaten"""
        
        # Bereinige alle String-Werte
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            elif isinstance(d, str):
                return d.encode('utf-8', errors='ignore').decode('utf-8')
            elif isinstance(d, list):
                return [clean_dict(item) for item in d]
            return d
        
        output_data = {
            'summary': summary_result['summary'],
            'source_document': summary_result.get('source_document', ''),
            'experiment_config': config.to_dict(),
            'metadata': summary_result.get('metadata', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        # Bereinige vor dem Speichern
        output_data = clean_dict(output_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def _run_evaluation(
        self,
        summaries: List[Any],
        config: ExperimentConfig,
        use_llm_judge: bool = False,
        output_dir: Path = None,
        document_type: Optional[str] = None
    ) -> Dict[str, Any]:
        
        """
        Führt Evaluation durch mit typ-spezifischen Kriterien.
        
        Args:
            summaries: Generierte Zusammenfassungen
            config: Experiment-Config
            use_llm_judge: LLM Judge aktivieren
            output_dir: Output-Verzeichnis
            document_type: "court_ruling" oder "contract"
        """

        if output_dir is None:
            output_dir = self.results_dir / "evaluations" / config.experiment_id
        
        eval_dir = output_dir / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
       
        gold_standards_dir = Path("data/gold_standards")
        if gold_standards_dir.exists():
            
            evaluator = AutomatedMetrics(
                config=self.eval_config
            )
            
            
            auto_results = evaluator.evaluate_directory(
                summaries_dir=output_dir,
                gold_standards_dir=gold_standards_dir,
                metrics=['rouge', 'bertscore']
            )
            
            results['automated_metrics'] = auto_results
            
            # Speichern
            with open(eval_dir / "automated_metrics.json", 'w', encoding='utf-8') as f:
                json.dump(auto_results, f, indent=2)
        
        
        if use_llm_judge:
            self.logger.info(
                f"Starte LLM-Judge Evaluation (Dokumententyp: {document_type or 'default'})"
            )
            
            # Judge-Client erstellen
            judge_config = self.eval_config.get('llm_judge', {})
            judge_model_info = judge_config.get('judge_model', {})
            
            judge_client = self._create_llm_client(
                provider=judge_model_info.get('provider', 'azure_ai_foundry'),
                model_name=judge_model_info.get('name', 'gpt-4o-mini'),
                config=ExperimentConfig(
                    experiment_id='judge',
                    model_provider=judge_model_info.get('provider', 'azure_ai_foundry'),
                    model_name=judge_model_info.get('name', 'gpt-4o-mini'),
                    temperature=0.3,
                    top_k=50,
                    top_p_values=0.95
                )
            )
            
            # Judge mit Dokumententyp initialisieren
            judge = LLMJudge(
                config={'llm_judge': judge_config},
                llm_client=judge_client,
                document_type=document_type 
            )
            
            # Logge verwendete Kriterien für Transparenz
            criteria_info = judge.get_criteria_info()
            self.logger.info(f"Verwendete Evaluationskriterien: {criteria_info}")
            
            judge_results = judge.evaluate_directory(
                summaries_dir=output_dir,
                gold_summaries_dir=Path("data/gold_standards")
            )
            
            # Speichere auch Kriterien-Info in den Ergebnissen
            results['llm_judge'] = {
                'evaluation_criteria': criteria_info,
                'results': judge_results
            }
            
            # Speichern
            with open(eval_dir / "llm_judge_results.json", 'w', encoding='utf-8') as f:
                json.dump(results['llm_judge'], f, indent=2, ensure_ascii=False)
        
        return results

    def run_batch_experiments(
        self,
        experiments: List[ExperimentConfig],
        input_dir: Path = None,
        evaluate: bool = True,
        use_llm_judge: bool = False,
        resume_from: Optional[str] = None,
        include_manual_evaluation: bool = False,
        preload_documents: bool = True,   
    ) -> List[Dict[str, Any]]:
        """
        Führt mehrere Experimente nacheinander durch.
        
        Args:
            experiments: Liste von Experiment-Konfigurationen
            input_dir: Eingabeverzeichnis
            evaluate: Evaluation durchführen
            use_llm_judge: LLM Judge verwenden
            resume_from: Experiment-ID zum Fortsetzen

        """
        if preload_documents:
            self.preload_documents(input_dir,None,100)       
        
        results = []
        
        # Resume: Bereits durchgeführte Experimente überspringen
        start_idx = 0
        if resume_from:
            # Lade bisherige Ergebnisse aus Checkpoint
            checkpoint_file = self.results_dir / "checkpoint.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    results = checkpoint.get('results', [])
                    self.logger.info(f"Lade {len(results)} bisherige Ergebnisse aus Checkpoint")
            
            # Finde Position des letzten abgeschlossenen Experiments
            for idx, exp in enumerate(experiments):
                if exp.experiment_id == resume_from:
                    start_idx = idx + 1  # Starte beim NÄCHSTEN Experiment (nicht beim letzten)
                    self.logger.info(f"Setze fort ab Experiment {start_idx + 1}/{len(experiments)}")
                    break
            
            if start_idx == 0:
                self.logger.warning(
                    f"Experiment-ID '{resume_from}' nicht gefunden. "
                    f"Starte von Anfang an."
                )
        
        # Batch-Verarbeitung mit Progress Bar
        for exp in tqdm(experiments[start_idx:], desc="Experimente"):
            try:
                report = self.run_experiment(
                    config=exp,  
                    input_dir=input_dir,
                    use_cache=True,
                    evaluate=evaluate,
                    use_llm_judge=use_llm_judge
                )
                results.append(report)
                
                # Checkpoint: Zwischenergebnisse speichern
                self._save_checkpoint(results, experiments)
                
            except Exception as e:
                self.logger.error(f"Fehler bei Experiment {exp.experiment_id}: {e}")
                continue
            
        # Finale Aggregation
        self._create_aggregated_report(results, experiments)

        if results:
            first_id = results[0]['experiment_id']
            group_prefix = self._extract_group_prefix(first_id)

            from experiments.experiment_analyzer import ExperimentAnalyzer
            analyzer = ExperimentAnalyzer()
            analyzer.create_comprehensive_report(
                experiment_name=f"Analysis for {group_prefix}",
                group_prefix=group_prefix            
                )
        
        if results and include_manual_evaluation:
            from experiments.experiment_utils import ExperimentUtils
            group_ids = [r['experiment_id'] for r in results]
            try:
                ExperimentUtils.export_for_manual_evaluation(experiment_ids=group_ids)
            except Exception as e:
                self.logger.warning(
                    f"âš ï¸ Fehler beim Export der manuellen Evaluation für {group_ids}: {e}"
                )
        
        print(f"âž¡ï¸ Verwende group_prefix: {group_prefix}")

        return results
    
    def _save_checkpoint(self, results: List[Dict], experiments: List[ExperimentConfig]):
        """Speichert Checkpoint für Resume"""
        checkpoint_file = self.results_dir / "checkpoint.json"
        
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'completed_experiments': len(results),
            'total_experiments': len(experiments),
            'last_experiment_id': results[-1]['experiment_id'] if results else None,
            'results': results,
            'experiment_configs': [exp.to_dict() for exp in experiments]  # Speichere alle Experiment-Configs
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    
    def _create_aggregated_report(
        self,
        results: List[Dict],
        experiments: List[ExperimentConfig]
    ):
        """Erstellt aggregierten Bericht über alle Experimente"""
        report_dir = self.results_dir / "aggregated_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f"batch_report_{timestamp}.json"
        
        # Statistiken berechnen
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(experiments),
            'completed_experiments': len(results),
            'success_rate': len(results) / len(experiments) if experiments else 0,
            'experiments': results,
            'summary_statistics': self._compute_statistics(results)
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Aggregierter Bericht erstellt: {report_file}")
    
    def _compute_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Berechnet Statistiken über alle Experimente"""
        if not results:
            return {}
        
        # Hier kÃ¶nnen detaillierte Statistiken berechnet werden
        # z.B. durchschnittliche ROUGE-Scores pro Modell, etc.
        stats = {
            'total_documents_processed': sum(r['num_documents'] for r in results),
            'models_tested': list(set(r['config']['model_name'] for r in results)),
            'temperature_range': {
                'min': min(r['config']['temperature'] for r in results),
                'max': max(r['config']['temperature'] for r in results)
            }
        }
        
        return stats
