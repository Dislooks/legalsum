"""
automated_metrics.py

Berechnung automatisierter Evaluationsmetriken (ROUGE, BERTScore) für 
die Bewertung von generierten Zusammenfassungen gegen Goldstandards.

Autor: Masterarbeits-Pipeline
Datum: 2025-10-03
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import csv

import logging
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import pandas as pd

logger = logging.getLogger(__name__)

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class AutomatedMetrics:
    """
    Klasse zur Berechnung automatisierter Evaluationsmetriken.
    
    Unterstützt:
    - ROUGE (1, 2, L)
    - BERTScore (Precision, Recall, F1)
    """
    
    def __init__(self, config: Dict):
        """
        Initialisiert die Metriken mit Konfiguration.
        
        Args:
            config: Dictionary mit Evaluation-Konfiguration
        """
        self.config = config
        self.rouge_config = config.get('rouge_metrics', {})
        self.bertscore_config = config.get('bertscore', {})
        self.logger = logging.getLogger(__name__)
        
        # ROUGE Scorer initialisieren
        if self.rouge_config.get('enabled', True):
            rouge_types = self.rouge_config.get('metrics', ['rouge1', 'rouge2', 'rougeL'])
            use_stemmer = self.rouge_config.get('parameters', {}).get('use_stemmer', True)
            
            self.rouge_scorer = rouge_scorer.RougeScorer(
                rouge_types=rouge_types,
                use_stemmer=use_stemmer
            )
            logger.info(f"ROUGE Scorer initialisiert mit Metriken: {rouge_types}")
        
        # BERTScore Konfiguration
        if self.bertscore_config.get('enabled', True):
            self.bert_model = self.bertscore_config.get('model', 'bert-base-german-cased')
            self.bert_num_layers = self.bertscore_config.get('parameters', {}).get('num_layers', 9)
            self.bert_idf = self.bertscore_config.get('parameters', {}).get('idf', True)
            logger.info(f"BERTScore konfiguriert mit Modell: {self.bert_model}")
    
    def _read_json_robust(self, filepath: Path) -> Dict:
        """
        Liest JSON-Datei mit mehreren Encoding-Versuchen.
        """
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    data = json.load(f)
                    # Handle sowohl Dict als auch List
                    return data
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        
        # Fallback: Ignoriere fehlerhafte Bytes
        try:
            logger.warning(f"Verwende Fallback-Encoding für {filepath}")
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Konnte Datei nicht lesen: {filepath}") from e

    def _read_text_robust(self, filepath: Path) -> str:
        """Liest Textdatei mit mehreren Encoding-Versuchen."""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        logger.warning(f"Verwende Fallback-Encoding für {filepath}")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def calculate_rouge(
        self, 
        hypothesis: str, 
        reference: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Berechnet ROUGE-Scores zwischen Hypothese und Referenz.
        
        Args:
            hypothesis: Generierte Zusammenfassung
            reference: Goldstandard-Zusammenfassung
        """
        if not self.rouge_config.get('enabled', True):
            logger.warning("ROUGE ist deaktiviert in der Konfiguration")
            return {}
        
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            
            # Konvertiere zu serialisierbarem Format
            result = {}
            for metric, score_obj in scores.items():
                result[metric] = {
                    'precision': score_obj.precision,
                    'recall': score_obj.recall,
                    'fmeasure': score_obj.fmeasure
                }
            
            logger.debug(f"ROUGE berechnet: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei ROUGE-Berechnung: {e}")
            return {}
    
    def calculate_bertscore(
        self,
        hypotheses: Union[str, List[str]],
        references: Union[str, List[str]]
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Berechnet BERTScore zwischen Hypothesen und Referenzen.
        
        Args:
            hypotheses: Generierte Zusammenfassung(en)
            references: Goldstandard-Zusammenfassung(en)

        """
        if not self.bertscore_config.get('enabled', True):
            logger.warning("BERTScore ist deaktiviert in der Konfiguration")
            return {}
        
        # Sicherstellen, dass Inputs Listen sind
        if isinstance(hypotheses, str):
            hypotheses = [hypotheses]
        if isinstance(references, str):
            references = [references]
        
        if len(hypotheses) != len(references):
            logger.error("Anzahl der Hypothesen und Referenzen stimmt nicht überein")
            return {}
        
        try:
            P, R, F1 = bert_score(
                cands=hypotheses,
                refs=references,
                model_type=self.bert_model,
                num_layers=self.bert_num_layers,
                idf=self.bert_idf,
                lang=self.bertscore_config.get('parameters', {}).get('lang', 'de'),
                rescale_with_baseline=self.bertscore_config.get('parameters', {}).get('rescale_with_baseline', True),
                verbose=False
            )
            
            result = {
                'precision': P.tolist(),
                'recall': R.tolist(),
                'f1': F1.tolist()
            }
            
            # Für einzelne Texte: Extrahiere Skalare
            if len(hypotheses) == 1:
                result = {
                    'precision': float(P[0]),
                    'recall': float(R[0]),
                    'f1': float(F1[0])
                }
            
            logger.debug(f"BERTScore berechnet: Mittelwert F1 = {np.mean(F1.cpu().numpy()):.4f}")

            return result
            
        except Exception as e:
            logger.error(f"Fehler bei BERTScore-Berechnung: {e}")
            return {}
    
    def evaluate_directory(
        self,
        summaries_dir: Path,
        gold_standards_dir: Path,
        metrics: List[str] = None
    ) -> List[Dict]:
        """
        Evaluiert alle Zusammenfassungen in einem Verzeichnis.
        
        Args:
            summaries_dir: Verzeichnis mit Zusammenfassungen
            gold_standards_dir: Verzeichnis mit Goldstandards
            metrics: Liste der zu berechnenden Metriken
        """
        # Lade Goldstandards
        gold_standards = load_gold_standards(gold_standards_dir)
        
        # Sammle Zusammenfassungen
        summaries = []
        for json_file in summaries_dir.glob("**/*.json"):
            try:
                # Skip experiment_metadata.json und evaluation results
                if json_file.name in ['experiment_metadata.json', 'experiment_report.json', 'automated_metrics.json', 'llm_judge_results.json']:
                    logger.debug(f"Überspringe Metadaten-Datei: {json_file}")
                    continue
                
                # Robustes Lesen
                summary_data = self._read_json_robust(json_file)
                
                # Prüfe ob es eine Liste ist (z.B. experiment_metadata.json)
                if isinstance(summary_data, list):
                    logger.warning(f"Datei {json_file} enthält eine Liste, überspringe")
                    continue
                    
                # Extrahiere relevante Informationen
                doc_id = Path(summary_data.get('source_file', '')).stem
            
                model_name = 'unknown'

                if 'model_config' in summary_data:
                    model_config = summary_data['model_config']
                    if isinstance(model_config, dict):
                        model_name = model_config.get('model_name', 'unknown')

                if model_name == 'unknown' and 'metadata' in summary_data:
                    metadata = summary_data.get('metadata', {})
                    if 'llm_metadata' in metadata:
                        llm_metadata = metadata['llm_metadata']
                        if isinstance(llm_metadata, dict):
                            model_name = llm_metadata.get('model_name', 'unknown')

                if model_name == 'unknown':
                    model_name = summary_data.get('model_name', 'unknown')

                self.logger.debug(f"Extrahierter Modellname für {doc_id}: {model_name}")

                summaries.append({
                    'document_id': doc_id,
                    'summary': summary_data.get('summary', ''),
                    'model': model_name
                })                             
               
            except Exception as e:
                logger.error(f"Fehler beim Laden von {json_file}: {e}")
                continue
        
        # Batch-Evaluation durchführen
        return self.batch_evaluate(summaries, gold_standards)
    
    def batch_evaluate(
        self,
        summaries: List[Dict],
        gold_standards: Dict[str, str]
    ) -> List[Dict]:
        """
        Evaluiert eine Liste von Zusammenfassungen gegen Goldstandards.
        
        Args:
            summaries: Liste von Dicts mit 'document_id', 'summary', 'model', etc.
            gold_standards: Dict mapping document_id -> goldstandard_text

        """
        logger.info(f"Starte Batch-Evaluation für {len(summaries)} Zusammenfassungen")
        
        results = []
        
        for summary_data in summaries:
            doc_id = summary_data.get('document_id')
            summary_text = summary_data.get('summary', '')
            
            if doc_id not in gold_standards:
                logger.warning(f"Kein Goldstandard für Dokument {doc_id} gefunden")
                continue
            
            gold_text = gold_standards[doc_id]
            
            # ROUGE berechnen
            rouge_scores = self.calculate_rouge(summary_text, gold_text)
            
            # BERTScore berechnen
            bert_scores = self.calculate_bertscore(summary_text, gold_text)
            
            # Ergebnisse zusammenführen
            result = {
                **summary_data,
                'rouge_scores': rouge_scores,
                'bertscore': bert_scores,
                'gold_standard_length': len(gold_text.split()),
                'summary_length': len(summary_text.split())
            }
            
            results.append(result)
        
        logger.info(f"Batch-Evaluation abgeschlossen: {len(results)} Ergebnisse")
        return results
    
    def aggregate_scores(
        self,
        evaluation_results: List[Dict],
        group_by: Optional[str] = None
    ) -> Dict:
        """
        Aggregiert Evaluationsergebnisse über mehrere Dokumente.
        
        Args:
            evaluation_results: Liste von Evaluationsergebnissen
            group_by: Optional - Gruppierung nach 'model', 'temperature', etc.

        """
        if not evaluation_results:
            logger.warning("Keine Evaluationsergebnisse zum Aggregieren")
            return {}
        
        df = pd.DataFrame(evaluation_results)
        
        aggregated = {}
        
        if group_by and group_by in df.columns:
            groups = df.groupby(group_by)
            
            for group_name, group_df in groups:
                aggregated[group_name] = self._calculate_aggregate_stats(group_df)
        else:
            aggregated['overall'] = self._calculate_aggregate_stats(df)
        
        return aggregated
    
    def _calculate_aggregate_stats(self, df: pd.DataFrame) -> Dict:
        """
        Berechnet aggregierte Statistiken für einen DataFrame.
        
        Args:
            df: DataFrame mit Evaluationsergebnissen

        """
        stats = {
            'n_documents': len(df),
            'rouge': {},
            'bertscore': {}
        }
        
        # ROUGE-Aggregation
        if 'rouge_scores' in df.columns:
            rouge_data = df['rouge_scores'].tolist()
            
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                fmeasures = [
                    scores.get(metric, {}).get('fmeasure', np.nan)
                    for scores in rouge_data
                    if scores and metric in scores
                ]
                
                if fmeasures:
                    stats['rouge'][metric] = {
                        'mean': float(np.nanmean(fmeasures)),
                        'std': float(np.nanstd(fmeasures)),
                        'min': float(np.nanmin(fmeasures)),
                        'max': float(np.nanmax(fmeasures)),
                        'median': float(np.nanmedian(fmeasures))
                    }
        
        # BERTScore-Aggregation
        if 'bertscore' in df.columns:
            bert_data = df['bertscore'].tolist()
            
            for metric in ['precision', 'recall', 'f1']:
                values = [
                    scores.get(metric, np.nan)
                    for scores in bert_data
                    if scores and metric in scores
                ]
                
                if values:
                    stats['bertscore'][metric] = {
                        'mean': float(np.nanmean(values)),
                        'std': float(np.nanstd(values)),
                        'min': float(np.nanmin(values)),
                        'max': float(np.nanmax(values)),
                        'median': float(np.nanmedian(values))
                    }
        
        return stats
    
    def interpret_scores(self, scores: Dict) -> Dict[str, str]:
        """
        Interpretiert Evaluations-Scores anhand von Thresholds.
        
        Args:
            scores: Dictionary mit ROUGE/BERTScore-Ergebnissen
            
        Returns:
            Dictionary mit Interpretationen ('excellent', 'good', 'acceptable', 'poor')
        """
        interpretations = {}
        
        # ROUGE Interpretation
        rouge_thresholds = self.rouge_config.get('thresholds', {})
        if 'rouge_scores' in scores:
            for metric, values in scores['rouge_scores'].items():
                fmeasure = values.get('fmeasure', 0)
                interpretation = self._threshold_interpret(fmeasure, rouge_thresholds)
                interpretations[f'{metric}_interpretation'] = interpretation
        
        # BERTScore Interpretation
        bert_thresholds = self.bertscore_config.get('thresholds', {})
        if 'bertscore' in scores:
            f1_score = scores['bertscore'].get('f1', 0)
            interpretation = self._threshold_interpret(f1_score, bert_thresholds)
            interpretations['bertscore_interpretation'] = interpretation
        
        return interpretations
    
    def _threshold_interpret(self, value: float, thresholds: Dict) -> str:
        """
        Interpretiert einen Wert anhand von Thresholds.
        
        Args:
            value: Zu interpretierender Wert
            thresholds: Dictionary mit Schwellenwerten

        """
        if value >= thresholds.get('excellent', 0.5):
            return 'excellent'
        elif value >= thresholds.get('good', 0.4):
            return 'good'
        elif value >= thresholds.get('acceptable', 0.3):
            return 'acceptable'
        else:
            return 'poor'
    
    def save_results(
        self,
        results: List[Dict],
        output_path: Union[str, Path],
        format: str = 'json'
    ) -> None:
        """
        Speichert Evaluationsergebnisse in verschiedenen Formaten.
        
        Args:
            results: Liste von Evaluationsergebnissen
            output_path: Pfad zur Output-Datei
            format: 'json', 'csv', oder 'excel'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                    
            elif format == 'csv':
                # Flatten nested structures für CSV
                flattened = self._flatten_results(results)
                df = pd.DataFrame(flattened)
                df.to_csv(output_path, index=False, encoding='utf-8', sep=';', decimal=',')
                
            elif format == 'excel':
                df = pd.DataFrame(self._flatten_results(results))
                df.to_excel(output_path, index=False, engine='openpyxl')
                
            logger.info(f"Ergebnisse gespeichert: {output_path}")
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Ergebnisse: {e}")
    
    def _flatten_results(self, results: List[Dict]) -> List[Dict]:
        """
        Flacht verschachtelte Dictionaries für tabellarische Formate.
        
        Args:
            results: Liste von verschachtelten Dictionaries

        """
        flattened = []
        
        for result in results:
            flat = {}
            
            # Basis-Felder kopieren
            for key, value in result.items():
                if not isinstance(value, dict):
                    flat[key] = value
            
            # ROUGE Scores auflösen
            if 'rouge_scores' in result:
                for metric, scores in result['rouge_scores'].items():
                    for score_type, score_value in scores.items():
                        flat[f'{metric}_{score_type}'] = score_value
            
            # BERTScore auflösen
            if 'bertscore' in result:
                for metric, value in result['bertscore'].items():
                    flat[f'bertscore_{metric}'] = value
            
            flattened.append(flat)
        
        return flattened

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def load_gold_standards(gold_standards_path: Union[str, Path]) -> Dict[str, str]:
    """
    Lädt Goldstandards aus einem Verzeichnis oder einer Datei.
    
    Args:
        gold_standards_path: Pfad zu Goldstandards (Verzeichnis oder JSON-Datei)

    """
    path = Path(gold_standards_path)
    gold_standards = {}
    
    if path.is_file() and path.suffix == '.json':
        # Einzelne JSON-Datei mit allen Goldstandards
        with open(path, 'r', encoding='utf-8') as f:
            gold_standards = json.load(f)
            
    elif path.is_dir():
        # Verzeichnis mit einzelnen Goldstandard-Dateien
        for file in path.glob('*.txt'):
            doc_id = file.stem
            with open(file, 'r', encoding='utf-8') as f:
                gold_standards[doc_id] = f.read().strip()
    
    logger.info(f"Geladen: {len(gold_standards)} Goldstandards aus {path}")
    return gold_standards
