"""

Framework für qualitative Evaluation von Zusammenfassungen.
Verwendet ein LLM zur Bewertung anhand definierter Kriterien.
Bewertet generierte Summaries im Vergleich zu Gold-Standard-Zusammenfassungen.

"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt 
class LLMJudge:
    """
    LLM-as-a-Judge Framework mit dokumententyp-spezifischer Evaluation.
    """
    
    def __init__(
        self, 
        config: Dict, 
        llm_client=None, 
        document_type: Optional[str] = None
    ):
        """
        Args:
            config: Dictionary mit LLM-Judge-Konfiguration
            llm_client: LLM-Client für Judge-Modell
            document_type: "court_ruling", "contract", etc.
        """
        self.base_config = config.get('llm_judge', {})
        self.judge_config = self.base_config.get('judge_model', {})
        self.llm_client = llm_client
        self.document_type = document_type
        
        # Lade typ-spezifische oder Default-Konfiguration
        self._load_type_specific_config()
        
        logger.info(
            f"LLM-Judge initialisiert für Dokumententyp: {document_type or 'default'}"
        )
        logger.info(f"Evaluationskriterien: {list(self.criteria.keys())}")
        logger.info(f"Gewichtungen: {self.weights}")
    
    def _load_type_specific_config(self):
        """Lädt typ-spezifische Konfiguration oder fällt auf Default zurück."""
        type_configs = self.base_config.get('document_type_configs', {})
        
        # Falls Dokumententyp angegeben und Config existiert
        if self.document_type and self.document_type in type_configs:
            type_config = type_configs[self.document_type]
            
            # Typ-spezifische Kriterien
            self.criteria = type_config.get(
                'evaluation_criteria',
                self.base_config.get('evaluation_criteria', {})
            )
            
            # Typ-spezifischer Prompt
            self.prompt_template = type_config.get(
                'judge_prompt_template',
                self.base_config.get('judge_prompt_template', '')
            )
            
            logger.info(f"✅ Verwende typ-spezifische Konfiguration: {self.document_type}")
        else:
            # Default-Konfiguration
            self.criteria = self.base_config.get('evaluation_criteria', {})
            self.prompt_template = self.base_config.get('judge_prompt_template', '')
            
            if self.document_type:
                logger.warning(
                    f"Keine typ-spezifische Config für '{self.document_type}' gefunden. "
                    f"Verwende Default-Konfiguration."
                )
        
        # Gewichtungen extrahieren
        self.weights = {
            criterion: details.get('weight', 0.2)
            for criterion, details in self.criteria.items()
        }
        
        # Validiere Gewichtungen
        total_weight = sum(self.weights.values())
        if not (0.99 <= total_weight <= 1.01):  # Toleranz für Rundungsfehler
            logger.warning(
                f"⚠️ Gewichtungen summieren sich zu {total_weight:.2f}, nicht zu 1.0!"
            )
    
    def get_criteria_info(self) -> Dict:
        """Gibt Informationen über die verwendeten Kriterien zurück."""
        return {
            'document_type': self.document_type or 'default',
            'criteria': {
                name: {
                    'weight': details.get('weight'),
                    'description': details.get('description'),
                    'scale': details.get('scale')
                }
                for name, details in self.criteria.items()
            },
            'total_weight': sum(self.weights.values())
        }
    
    def _read_json_robust(self, filepath: Path) -> Dict:
        """
        Liest JSON-Datei mit mehreren Encoding-Versuchen.
        """
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    data = json.load(f)
                    return data
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        
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

    def evaluate_single(
        self,
        gold_summary: str,
        summary: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Bewertet eine einzelne Zusammenfassung mit LLM-as-a-Judge.
        
        Args:
            gold_summary: Gold-Standard-Zusammenfassung als Referenz
            summary: Generierte Zusammenfassung
            document_id: ID des Dokuments (optional)
            metadata: Zusätzliche Metadaten (optional)

        """
        logger.info(f"Starte LLM-Judge Evaluation für Dokument: {document_id or 'unknown'}")
        
        if not gold_summary:
            error_msg = f"Keine Gold-Standard-Zusammenfassung für {document_id} vorhanden"
            logger.error(error_msg)
            return {
                'document_id': document_id,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Prompt erstellen
            prompt = self._create_evaluation_prompt(gold_summary, summary)
            
            # LLM-Anfrage
            response = self._call_judge_llm(prompt)
            
            # Response parsen
            evaluation = self._parse_judge_response(response)
            
            # Gesamtscore berechnen
            overall_score = self._calculate_weighted_score(evaluation)
            
            result = {
                'document_id': document_id,
                'timestamp': datetime.now().isoformat(),
                'criteria_scores': evaluation,
                'overall_score': overall_score,
                'metadata': metadata or {},
                'judge_model': self.judge_config.get('name', self.judge_config.get('model', 'unknown')),
                'raw_response': response
            }
            
            logger.info(f"Evaluation abgeschlossen. Gesamtscore: {overall_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei LLM-Judge Evaluation: {e}")
            return {
                'document_id': document_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def evaluate_directory(
        self,
        summaries_dir: Path,
        gold_summaries_dir: Path
    ) -> List[Dict]:
        """
        Evaluiert alle Zusammenfassungen in einem Verzeichnis.
        
        Args:
            summaries_dir: Verzeichnis mit generierten Zusammenfassungen
            gold_summaries_dir: Verzeichnis mit Gold-Standard-Zusammenfassungen

        """
        evaluation_items = []
        
        for json_file in summaries_dir.glob("**/*.json"):
            try:
                # Skip experiment_metadata.json und evaluation results
                if json_file.name in ['experiment_metadata.json', 'experiment_report.json', 
                                      'automated_metrics.json', 'llm_judge_results.json']:
                    logger.debug(f"Überspringe Metadaten-Datei: {json_file}")
                    continue
                
                # Robustes Lesen der generierten Summary
                summary_data = self._read_json_robust(json_file)
                
                # Prüfe ob es eine Liste ist
                if isinstance(summary_data, list):
                    logger.warning(f"Datei {json_file} enthält eine Liste, überspringe")
                    continue
                
                # Extrahiere source_file
                source_file = summary_data.get('source_file', '')
                if not source_file:
                    source_file = summary_data.get('metadata', {}).get('source_file', '')
                
                if not source_file:
                    logger.warning(f"Keine source_file in {json_file}, überspringe")
                    continue
                
                doc_id = Path(source_file).stem
                
                # Suche Gold-Standard-Zusammenfassung
                gold_summary = self._find_gold_summary(doc_id, gold_summaries_dir)
                
                if not gold_summary:
                    logger.warning(f"Keine Gold-Standard-Zusammenfassung für {doc_id} gefunden")
                    continue
                
                evaluation_items.append({
                    'document_id': doc_id,
                    'gold_summary': gold_summary,
                    'summary': summary_data.get('summary', ''),
                    'metadata': summary_data.get('model_config', {})
                })
                
            except Exception as e:
                logger.error(f"Fehler beim Verarbeiten von {json_file}: {e}")
                continue
        
        if not evaluation_items:
            logger.warning(f"Keine gültigen Summaries in {summaries_dir} gefunden")
            return []
        
        logger.info(f"Gefunden: {len(evaluation_items)} Summaries zur Evaluation mit Gold-Standards")
        return self.batch_evaluate(evaluation_items)

    def _find_gold_summary(self, doc_id: str, gold_summaries_dir: Path) -> Optional[str]:
        """
        Sucht nach Gold-Standard-Zusammenfassung für ein Dokument.
        
        Args:
            doc_id: Dokument-ID
            gold_summaries_dir: Verzeichnis mit Gold-Standard-Zusammenfassungen

        """
        # Mögliche Dateiformate
        for ext in ['.txt', '.json']:
            gold_file = gold_summaries_dir / f"{doc_id}{ext}"
            
            if gold_file.exists():
                try:
                    if ext == '.json':
                        data = self._read_json_robust(gold_file)
                        # Erwarte 'summary' oder 'gold_summary' Key
                        return data.get('summary') or data.get('gold_summary', '')
                    else:
                        return self._read_text_robust(gold_file)
                except Exception as e:
                    logger.warning(f"Fehler beim Lesen von {gold_file}: {e}")
                    continue
        
        logger.debug(f"Keine Gold-Summary für {doc_id} gefunden")
        return None

    def batch_evaluate(
        self,
        evaluation_items: List[Dict],
        rate_limit_delay: float = 2.0
    ) -> List[Dict]:
        """
        Bewertet mehrere Zusammenfassungen mit LLM-as-a-Judge.
        
        Args:
            evaluation_items: Liste von Dicts mit 'gold_summary', 'summary', 'document_id'
            rate_limit_delay: Verzögerung zwischen Anfragen (Sekunden)
            
        """
        logger.info(f"Starte Batch-Evaluation für {len(evaluation_items)} Dokumente")
        
        results = []
        
        for i, item in enumerate(evaluation_items, 1):
            logger.info(f"Evaluiere Dokument {i}/{len(evaluation_items)}")
            
            result = self.evaluate_single(
                gold_summary=item.get('gold_summary', ''),
                summary=item.get('summary', ''),
                document_id=item.get('document_id'),
                metadata=item.get('metadata', {})
            )
            
            results.append(result)
            
            # Rate Limiting
            if i < len(evaluation_items):
                time.sleep(rate_limit_delay)
        
        logger.info(f"Batch-Evaluation abgeschlossen: {len(results)} Ergebnisse")
        return results
    
    def _create_evaluation_prompt(self, gold_summary: str, summary: str) -> str:
        """
        Erstellt den Evaluation-Prompt aus Template.
        
        Args:
            gold_summary: Gold-Standard-Zusammenfassung
            summary: Generierte Zusammenfassung

        """
        # Truncate sehr lange Gold-Summaries (optional)
        max_gold_length = 8000
        if len(gold_summary) > max_gold_length:
            gold_summary = gold_summary[:max_gold_length] + "\n[... Text gekürzt ...]"
        
        prompt = self.prompt_template.format(
            gold_summary=gold_summary,
            summary=summary
        )
        
        return prompt
    
    def _call_judge_llm(self, prompt: str) -> str:
        """
        Ruft das Judge-LLM mit dem Evaluation-Prompt auf.
        
        Args:
            prompt: Formatierter Evaluation-Prompt
            
        Returns:
            Raw Response vom LLM
        """
        if self.llm_client is None:
            logger.warning("Kein LLM-Client konfiguriert. Verwende Mock-Response.")
            return self._mock_judge_response()
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=self.judge_config.get('temperature', 0.1),
                max_tokens=8192
            )
        
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                logger.error(f"Unerwarteter Response-Typ: {type(response)}")
                raise ValueError(f"Konnte Text nicht aus Response extrahieren: {type(response)}")
                    
        except Exception as e:
            logger.error(f"Fehler beim LLM-Aufruf: {e}")
            raise
    
    def _parse_judge_response(self, response: str) -> Dict:
        """
        Parst die JSON-Response des Judge-LLMs.
        
        Args:
            response: Raw Response vom LLM
            
        Returns:
            Dictionary mit strukturierten Bewertungen
        """
        try:
            # Versuche JSON zu extrahieren
            if '```json' in response:
                json_start = response.find('```json') + 7
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
            elif '{' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
            else:
                json_str = response
            
            evaluation = json.loads(json_str)
            
            # Validiere Struktur
            self._validate_evaluation_structure(evaluation)
            
            return evaluation
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON-Parsing-Fehler: {e}")
            logger.debug(f"Response war: {response}")
            return self._fallback_parse(response)
    
    def _validate_evaluation_structure(self, evaluation: Dict) -> None:
        """
        Validiert die Struktur der Evaluation.
        
        Args:
            evaluation: Geparste Evaluation

        """
        required_criteria = list(self.criteria.keys())
        
        for criterion in required_criteria:
            if criterion not in evaluation:
                logger.warning(f"Kriterium '{criterion}' fehlt in Evaluation")
            else:
                if 'score' not in evaluation[criterion]:
                    raise ValueError(f"Score fehlt für Kriterium '{criterion}'")
                
                score = evaluation[criterion]['score']
                if not isinstance(score, (int, float)) or score < 1 or score > 5:
                    raise ValueError(f"Ungültiger Score für '{criterion}': {score}")
    
    def _fallback_parse(self, response: str) -> Dict:
        """
        Fallback-Parser wenn JSON-Parsing fehlschlägt.
        
        Args:
            response: Raw Response

        """
        logger.warning("Verwende Fallback-Parsing für LLM-Response")
        
        evaluation = {}
        
        for criterion in self.criteria.keys():
            import re
            pattern = rf"{criterion}[:\s]+(?:score[:\s]+)?(\d+)"
            match = re.search(pattern, response, re.IGNORECASE)
            
            if match:
                score = int(match.group(1))
                evaluation[criterion] = {
                    'score': score,
                    'reasoning': 'Automatisch extrahiert (Fallback)'
                }
            else:
                evaluation[criterion] = {
                    'score': 3,
                    'reasoning': 'Konnte nicht geparst werden'
                }
        
        return evaluation
    
    def _calculate_weighted_score(self, evaluation: Dict) -> float:
        """
        Berechnet gewichteten Gesamtscore.
        
        Args:
            evaluation: Dictionary mit Kriterien-Bewertungen

        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for criterion, details in evaluation.items():
            if criterion in self.weights and 'score' in details:
                score = details['score']
                weight = self.weights[criterion]
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            logger.warning("Gesamtgewicht ist 0. Verwende Durchschnitt.")
            scores = [d['score'] for d in evaluation.values() if 'score' in d]
            return np.mean(scores) if scores else 0.0
        
        return weighted_sum / total_weight
    
    def _mock_judge_response(self) -> str:
        """
        Mock-Response für Entwicklung/Testing.
        
        Returns:
            Beispiel JSON-Response
        """
        mock = {
            "correctness": {
                "score": 4,
                "reasoning": "Alle wesentlichen Fakten korrekt wiedergegeben."
            },
            "completeness": {
                "score": 4,
                "reasoning": "Meiste Kernpunkte erfasst, kleinere Details fehlen."
            },
            "coherence": {
                "score": 5,
                "reasoning": "Hervorragend strukturiert und verständlich."
            },
            "conciseness": {
                "score": 4,
                "reasoning": "Gut komprimiert, minimal redundant."
            },
            "legal_terminology": {
                "score": 4,
                "reasoning": "Meist korrekte Fachbegriffe verwendet."
            },
            "overall_assessment": "Eine qualitativ hochwertige Zusammenfassung."
        }
        
        return json.dumps(mock, indent=2)
    
    def aggregate_results(
        self,
        results: List[Dict],
        group_by: Optional[str] = None
    ) -> Dict:
        """
        Aggregiert LLM-Judge Ergebnisse über mehrere Dokumente.
        
        Args:
            results: Liste von Evaluationsergebnissen
            group_by: Optional - Gruppierung nach Metadaten-Feld

        """
        if not results:
            logger.warning("Keine Ergebnisse zum Aggregieren")
            return {}
        
        # Filtere fehlerhafte Ergebnisse
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            logger.error("Keine validen Ergebnisse zum Aggregieren")
            return {}
        
        df = pd.DataFrame(valid_results)
        
        aggregated = {}
        
        if group_by and f'metadata.{group_by}' in df.columns:
            groups = df.groupby(f'metadata.{group_by}')
            
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
            'criteria': {},
            'overall_score': {}
        }
        
        # Extrahiere Scores pro Kriterium
        for criterion in self.criteria.keys():
            scores = []
            
            for criteria_scores in df['criteria_scores']:
                if criterion in criteria_scores and 'score' in criteria_scores[criterion]:
                    scores.append(criteria_scores[criterion]['score'])
            
            if scores:
                stats['criteria'][criterion] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'median': float(np.median(scores))
                }
        
        # Gesamtscore-Statistiken
        if 'overall_score' in df.columns:
            overall_scores = df['overall_score'].dropna().tolist()
            
            if overall_scores:
                stats['overall_score'] = {
                    'mean': float(np.mean(overall_scores)),
                    'std': float(np.std(overall_scores)),
                    'min': float(np.min(overall_scores)),
                    'max': float(np.max(overall_scores)),
                    'median': float(np.median(overall_scores))
                }
        
        return stats
    
    def generate_report(
        self,
        results: List[Dict],
        output_path: Union[str, Path],
        include_details: bool = True
    ) -> None:
        """
        Generiert einen umfassenden Evaluation-Report.
        
        Args:
            results: Liste von Evaluationsergebnissen
            output_path: Pfad zur Output-Datei
            include_details: Detaillierte Reasoning einschließen
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_evaluations': len(results),
                'judge_model': self.judge_config.get('name', self.judge_config.get('model', 'unknown')),
                'criteria': list(self.criteria.keys()),
                'weights': self.weights,
                'evaluation_type': 'gold_standard_comparison'
            },
            'aggregate_statistics': self.aggregate_results(results),
            'individual_results': []
        }
        
        # Individuelle Ergebnisse
        for result in results:
            item = {
                'document_id': result.get('document_id'),
                'overall_score': result.get('overall_score'),
                'criteria_scores': {}
            }
            
            if 'criteria_scores' in result:
                for criterion, details in result['criteria_scores'].items():
                    item['criteria_scores'][criterion] = details.get('score')
                    
                    if include_details and 'reasoning' in details:
                        item['criteria_scores'][criterion] = {
                            'score': details['score'],
                            'reasoning': details['reasoning']
                        }
            
            report['individual_results'].append(item)
        
        # Speichern
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation-Report gespeichert: {output_path}")
    
    def compare_with_automated_metrics(
        self,
        llm_judge_results: List[Dict],
        automated_results: List[Dict]
    ) -> Dict:
        """
        Vergleicht LLM-Judge Scores mit automatisierten Metriken.
        
        Args:
            llm_judge_results: Ergebnisse von LLM-as-a-Judge
            automated_results: Ergebnisse von ROUGE/BERTScore

        """
        from scipy.stats import pearsonr, spearmanr
        
        llm_df = pd.DataFrame(llm_judge_results)
        auto_df = pd.DataFrame(automated_results)
        
        merged = pd.merge(
            llm_df[['document_id', 'overall_score']],
            auto_df[['document_id', 'rouge_scores', 'bertscore']],
            on='document_id',
            how='inner'
        )
        
        if len(merged) == 0:
            logger.warning("Keine übereinstimmenden Dokumente für Vergleich gefunden")
            return {}
        
        correlations = {}
        
        llm_scores = merged['overall_score'].values
        
        # ROUGE-1 F1 extrahieren
        rouge1_scores = [
            r.get('rouge1', {}).get('fmeasure', np.nan)
            for r in merged['rouge_scores']
        ]
        
        if not all(np.isnan(rouge1_scores)):
            pearson_r, pearson_p = pearsonr(llm_scores, rouge1_scores)
            spearman_r, spearman_p = spearmanr(llm_scores, rouge1_scores)
            
            correlations['llm_vs_rouge1'] = {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p)
            }
        
        # BERTScore F1 extrahieren
        bert_f1_scores = [
            b.get('f1', np.nan)
            for b in merged['bertscore']
        ]
        
        if not all(np.isnan(bert_f1_scores)):
            pearson_r, pearson_p = pearsonr(llm_scores, bert_f1_scores)
            spearman_r, spearman_p = spearmanr(llm_scores, bert_f1_scores)
            
            correlations['llm_vs_bertscore'] = {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p)
            }
        
        logger.info("Korrelationsanalyse abgeschlossen")
        return correlations
    
    def save_results(
        self,
        results: List[Dict],
        output_path: Union[str, Path],
        format: str = 'json'
    ) -> None:
        """
        Speichert LLM-Judge Ergebnisse in verschiedenen Formaten.
        
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
                flattened = self._flatten_results(results)
                df = pd.DataFrame(flattened)
                df.to_csv(output_path, index=False, encoding='utf-8')
                
            elif format == 'excel':
                df = pd.DataFrame(self._flatten_results(results))
                df.to_excel(output_path, index=False, engine='openpyxl')
                
            logger.info(f"Ergebnisse gespeichert: {output_path}")
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern: {e}")
    
    def _flatten_results(self, results: List[Dict]) -> List[Dict]:
        """
        Flacht verschachtelte Dictionaries für CSV/Excel.
        
        Args:
            results: Liste von verschachtelten Dictionaries
            
        """
        flattened = []
        
        for result in results:
            flat = {
                'document_id': result.get('document_id'),
                'timestamp': result.get('timestamp'),
                'overall_score': result.get('overall_score'),
                'judge_model': result.get('judge_model')
            }
            
            # Kriterien-Scores extrahieren
            if 'criteria_scores' in result:
                for criterion, details in result['criteria_scores'].items():
                    flat[f'{criterion}_score'] = details.get('score')
                    flat[f'{criterion}_reasoning'] = details.get('reasoning', '')
            
            # Metadaten
            if 'metadata' in result:
                for key, value in result['metadata'].items():
                    flat[f'metadata_{key}'] = value
            
            flattened.append(flat)
        
        return flattened