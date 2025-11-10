import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import shutil

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class ExperimentUtils:
    """Hilfs-Tools für Experiment-Management"""
        
    @staticmethod
    def estimate_duration(
        num_api_calls: int,
        avg_time_per_call: float = 5.0,
        parallel_requests: int = 1
    ) -> Dict[str, float]:
        """
        Schätzt Dauer für Experiment.
        
        Args:
            num_api_calls: Gesamtzahl API-Calls
            avg_time_per_call: Durchschnittliche Zeit pro Call (Sekunden)
            parallel_requests: Anzahl paralleler Requests
            
        Returns:
            Dictionary mit Zeitschätzungen
        """
        total_seconds = (num_api_calls / parallel_requests) * avg_time_per_call
        
        duration = {
            'total_seconds': total_seconds,
            'total_minutes': total_seconds / 60,
            'total_hours': total_seconds / 3600,
            'estimated_completion': datetime.now() + timedelta(seconds=total_seconds)
        }
        
        return duration
    
    @staticmethod
    def validate_experiment_data(
        summaries_dir: str = "results/summaries"
    ) -> Dict[str, any]:
        """
        Validiert Experiment-Daten auf Vollständigkeit und Konsistenz.
        
        Returns:
            Validierungs-Report
        """
        summaries_path = Path(summaries_dir)
        
        if not summaries_path.exists():
            return {'status': 'error', 'message': 'Summaries-Verzeichnis nicht gefunden'}
        
        report = {
            'status': 'ok',
            'experiments': [],
            'issues': [],
            'statistics': {}
        }
        
        # Alle Experiment-Verzeichnisse
        exp_dirs = [d for d in summaries_path.iterdir() if d.is_dir()]
        
        for exp_dir in exp_dirs:
            exp_report = {
                'experiment_id': exp_dir.name,
                'has_metadata': False,
                'num_summaries': 0,
                'missing_summaries': []
            }
            
            # Metadata prüfen
            metadata_file = exp_dir / "experiment_metadata.json"
            if metadata_file.exists():
                exp_report['has_metadata'] = True
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                exp_report['num_summaries'] = len(metadata)
                
                # Prüfe ob alle Summaries vorhanden
                for meta in metadata:
                    doc_id = meta['document_id']
                    summary_file = exp_dir / f"{doc_id}.json"
                    
                    if not summary_file.exists():
                        exp_report['missing_summaries'].append(doc_id)
            else:
                report['issues'].append(f"Keine experiment_metadata.json in {exp_dir.name}")
            
            report['experiments'].append(exp_report)
        
        # Statistiken
        report['statistics'] = {
            'total_experiments': len(exp_dirs),
            'total_summaries': sum(e['num_summaries'] for e in report['experiments']),
            'experiments_with_issues': sum(1 for e in report['experiments'] if e['missing_summaries'])
        }
        
        return report
    
    @staticmethod
    def cleanup_incomplete_experiments(
        summaries_dir: str = "results/summaries",
        dry_run: bool = True
    ):
        """
        Entfernt unvollständige Experimente.
        
        Args:
            summaries_dir: Pfad zu Summaries
            dry_run: Wenn True, nur anzeigen ohne löschen
        """
        validation = ExperimentUtils.validate_experiment_data(summaries_dir)
        
        incomplete = [
            exp for exp in validation['experiments']
            if exp['missing_summaries'] or not exp['has_metadata']
        ]
        
        if not incomplete:
            print("✅ Keine unvollständigen Experimente gefunden")
            return
        
        print(f"\n⚠️  Gefunden: {len(incomplete)} unvollständige Experimente")
        
        for exp in incomplete:
            exp_dir = Path(summaries_dir) / exp['experiment_id']
            
            if dry_run:
                print(f"[DRY RUN] Würde löschen: {exp_dir}")
                print(f"  Fehlende Summaries: {len(exp['missing_summaries'])}")
            else:
                print(f"Lösche: {exp_dir}")
                shutil.rmtree(exp_dir)
        
        if dry_run:
            print("\nFühre erneut aus mit dry_run=False zum tatsächlichen Löschen")
    
    @staticmethod
    def merge_experiment_results(
        experiment_ids: List[str],
        output_file: str = "results/merged_results.xlsx"
    ):
        """
        Merged mehrere Experimente in eine Excel-Datei.
        
        Args:
            experiment_ids: Liste von Experiment-IDs
            output_file: Pfad zur Ausgabedatei
        """
        all_summaries = []
        all_evaluations = []
        
        for exp_id in experiment_ids:
            # Summaries laden
            metadata_file = Path(f"results/summaries/{exp_id}/experiment_metadata.json")
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                all_summaries.extend(metadata)
            
            # Evaluations laden
            eval_file = Path(f"results/evaluations/{exp_id}/automated_metrics.json")
            if eval_file.exists():
                with open(eval_file, 'r', encoding='utf-8') as f:
                    evals = json.load(f)
                
                for doc_id, metrics in evals.items():
                    eval_row = {
                        'experiment_id': exp_id,
                        'document_id': doc_id,
                        **metrics
                    }
                    all_evaluations.append(eval_row)
        
        # DataFrames erstellen
        df_summaries = pd.DataFrame(all_summaries)
        df_evaluations = pd.DataFrame(all_evaluations)
        
        # Excel schreiben
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_summaries.to_excel(writer, sheet_name='Summaries', index=False)
            df_evaluations.to_excel(writer, sheet_name='Evaluations', index=False)
        
        print(f"✅ Merged Ergebnisse gespeichert: {output_file}")
        print(f"   Summaries: {len(df_summaries)} Einträge")
        print(f"   Evaluations: {len(df_evaluations)} Einträge")
    
    @staticmethod
    def create_progress_report(
        experiments_total: int,
        experiments_completed: int,
        start_time: datetime
    ) -> str:
        """
        Erstellt Fortschritts-Report.
        
        Returns:
            Formatierter Report-String
        """
        elapsed = datetime.now() - start_time
        progress_pct = (experiments_completed / experiments_total) * 100
        
        if experiments_completed > 0:
            avg_time_per_exp = elapsed / experiments_completed
            remaining_time = avg_time_per_exp * (experiments_total - experiments_completed)
            eta = datetime.now() + remaining_time
        else:
            eta = None
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    EXPERIMENT PROGRESS                        ║
╠══════════════════════════════════════════════════════════════╣
║ Abgeschlossen: {experiments_completed}/{experiments_total} ({progress_pct:.1f}%)                      
║ Verstrichene Zeit: {str(elapsed).split('.')[0]}                    
║ ETA: {eta.strftime('%Y-%m-%d %H:%M:%S') if eta else 'Berechnung...'}         
╚══════════════════════════════════════════════════════════════╝
        """
        
        return report
    
    @staticmethod
    def export_for_manual_evaluation(
        experiment_ids: Union[str, List[str]],
        output_file: Optional[str] = None,
        gold_standards_dir: Union[str, Path] = "data/gold_standards"
    ):
        """
        Exportiert ALLE Dokumente für manuelle Evaluation mit typspezifischen Bewertungskriterien.
        Nutzt experiment_report.json aus allen Unterordnern einer Gruppe.

        Args:
            experiment_ids: Einzelne ID oder Liste von IDs. Wenn nur eine ID übergeben wird,
                            werden automatisch alle Unterordner mit demselben group_prefix eingesammelt.
            output_file: Pfad zur Ausgabedatei (optional)
            gold_standards_dir: Verzeichnis mit Gold-Standard-Zusammenfassungen
        """

        if isinstance(experiment_ids, str):
            experiment_ids = [experiment_ids]

        # Group Prefix bestimmen
        def _extract_group_prefix(experiment_id: str) -> str:
            parts = experiment_id.split("_")
            if parts[-1].isdigit():
                return "_".join(parts[:-1])
            return experiment_id

        group_prefix = _extract_group_prefix(experiment_ids[0])

        # Alle Unterordner mit diesem Prefix einsammeln
        summaries_root = Path("results/summaries")
        all_exp_dirs = [d.name for d in summaries_root.iterdir() if d.is_dir()]
        group_ids = [d for d in all_exp_dirs if d.startswith(group_prefix)]

        court_rulings, contracts = [], []
        gold_standards_path: Path = Path(gold_standards_dir)

        for exp_id in group_ids:
            report_file = summaries_root / exp_id / "experiment_report.json"
            if not report_file.exists():
                print(f"⚠️ Kein experiment_report.json für {exp_id} gefunden")
                continue

            with open(report_file, "r", encoding="utf-8") as f:
                report = json.load(f)

            model = report.get("config", {}).get("model_name", "unknown")
            temperature = report.get("config", {}).get("temperature", "")
            top_p = report.get("config", {}).get("top_p_values", "")
            doc_type = report.get("document_type", "default")

            summaries = report.get("evaluation_results", {}).get("automated_metrics", [])
            for entry in summaries:
                doc_id = entry.get("document_id", "unknown")
                summary_text = entry.get("summary", "[Summary nicht gefunden]")

                # Gold-Standard laden
                gold_file = gold_standards_path / f"{doc_id}.txt"
                if not gold_file.exists():
                    base_name = doc_id.rsplit(".", 1)[0]
                    gold_file = gold_standards_path / f"{base_name}.txt"
                if gold_file.exists():
                    gold_standard = gold_file.read_text(encoding="utf-8")
                else:
                    gold_standard = "[Gold-Standard nicht gefunden]"

                eval_row_base = {
                    "filename": doc_id,
                    "summary": summary_text,
                    "gold_standard": gold_standard,
                    "model": model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "experiment_id": exp_id,
                }

                if doc_type == "court_ruling":
                    eval_row = {
                        **eval_row_base,
                        "correctness": "",
                        "completeness": "",
                        "legal_conciseness": "",
                        "legal_terminology": "",
                        "coherence": "",
                        "overall_rating": "",
                        "comments": "",
                    }
                    court_rulings.append(eval_row)

                elif doc_type == "contract":
                    eval_row = {
                        **eval_row_base,
                        "correctness": "",
                        "completeness": "",
                        "contractual_coherence": "",
                        "conciseness": "",
                        "legal_terminology": "",
                        "overall_rating": "",
                        "comments": "",
                    }
                    contracts.append(eval_row)

                else:
                    eval_row = {
                        **eval_row_base,
                        "correctness": "",
                        "completeness": "",
                        "legal_conciseness": "",
                        "legal_terminology": "",
                        "coherence": "",
                        "overall_rating": "",
                        "comments": "",
                    }
                    court_rulings.append(eval_row)

        # Excel-Datei schreiben
        if output_file is None:
            output_file = f"results/evaluations/manual_evaluation_{group_prefix}.xlsx"

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            if court_rulings:
                pd.DataFrame(court_rulings).to_excel(writer, sheet_name="court_ruling", index=False)
            if contracts:
                pd.DataFrame(contracts).to_excel(writer, sheet_name="contract", index=False)

        print(f"✅ Manuelle Evaluation erstellt: {output_file}")
        print(f"   Gerichtsurteile: {len(court_rulings)} Dokumente")
        print(f"   Verträge: {len(contracts)} Dokumente")
        print(f"   Gesamt: {len(court_rulings) + len(contracts)} Dokumente")
    
    @staticmethod
    def summarize_checkpoint(
        checkpoint_file: str = "results/checkpoint.json"
    ):
        """Gibt Checkpoint-Status aus"""
        checkpoint_path = Path(checkpoint_file)
        
        if not checkpoint_path.exists():
            print("❌ Kein Checkpoint gefunden")
            return
        
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        
        print("\n" + "="*70)
        print("CHECKPOINT STATUS")
        print("="*70)
        print(f"Timestamp: {checkpoint['timestamp']}")
        print(f"Abgeschlossen: {checkpoint['completed_experiments']}/{checkpoint['total_experiments']}")
        print(f"Letztes Experiment: {checkpoint['last_experiment_id']}")
        print(f"Fortschritt: {(checkpoint['completed_experiments']/checkpoint['total_experiments']*100):.1f}%")
        print("="*70 + "\n")
        
        remaining = checkpoint['total_experiments'] - checkpoint['completed_experiments']
        print(f"Zum Fortsetzen: python run_experiment.py --config XYZ --resume")
        print(f"Verbleibend: {remaining} Experimente\n")
