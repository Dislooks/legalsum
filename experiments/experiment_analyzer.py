"""
Experiment Analyzer - Analyse und Visualisierung von Experiment-Ergebnissen

"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from dataclasses import dataclass

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
@dataclass
class ComparisonResult:
    """Ergebnis eines statistischen Vergleichs"""
    metric_name: str
    group1_mean: float
    group2_mean: float
    group1_std: float
    group2_std: float
    t_statistic: float
    p_value: float
    effect_size: float  # Cohen's d
    significant: bool

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class ExperimentAnalyzer:
    """
    Analysiert Experiment-Ergebnisse und erstellt Visualisierungen.

    """
    
    def __init__(
        self,
        results_dir: str = "results",
        output_dir: str = "results/visualizations"
    ):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot-Style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        self.df_summaries = None
        self.df_evaluations = None
        
    def load_experiment_results(
        self,
        experiment_ids: Optional[List[str]] = None,
        group_prefix: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Lädt Ergebnisse aus Experimenten. 
        Ohne Angabe werden alle Unterordner einer Gruppe geladen.
        """

        summaries_dir = self.results_dir / "summaries"
        if not summaries_dir.exists():
            raise FileNotFoundError(f"Summaries-Verzeichnis nicht gefunden: {summaries_dir}")

        # Ordner bestimmen
        if experiment_ids is None:
            experiment_dirs = [d for d in summaries_dir.iterdir() if d.is_dir()]
            if group_prefix:
                experiment_dirs = [d for d in experiment_dirs if d.name.startswith(group_prefix)]
            if not experiment_dirs:
                print("⚠️ Keine Experimente gefunden")
                self.df_summaries = pd.DataFrame()
                return self.df_summaries
            print(f"ℹ️ Lade {len(experiment_dirs)} Experimente: {[d.name for d in experiment_dirs]}")
        else:
            experiment_dirs = [summaries_dir / exp_id for exp_id in experiment_ids]

        all_data: List[Dict] = []

        for exp_dir in experiment_dirs:
            metadata_file = exp_dir / "experiment_metadata.json"
            if not metadata_file.exists():
                alt_file = exp_dir / "metadata.json"
                metadata_file = alt_file if alt_file.exists() else None

            if not metadata_file:
                print(f"⚠️ Keine Metadaten in {exp_dir}")
                continue

            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)

                if isinstance(content, list):
                    all_data.extend(content)
                elif isinstance(content, dict):
                    all_data.append(content)
            except Exception as e:
                print(f"⚠️ Fehler beim Laden von {metadata_file}: {e}")

        df = pd.DataFrame(all_data) if all_data else pd.DataFrame()

        # Normalisierung
        if 'top_p' not in df.columns and 'top_p_values' in df.columns:
            df = df.rename(columns={'top_p_values': 'top_p'})

        self.df_summaries = df
        return df
    
    def load_evaluation_results(
        self,
        experiment_ids: Optional[List[str]] = None,
        group_prefix: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Lädt Evaluations-Ergebnisse (automated_metrics.json) und mappt Modelle/Parameter
        aus Metadaten und experiment_report.json. Ohne Angabe wird automatisch die gesamte Gruppe geladen.

        Args:
            experiment_ids: Liste von Experiment-IDs (None = alle Experimente der Gruppe)
            group_prefix: Gemeinsamer Präfix für eine Gruppe von Experimenten
        """
        eval_root = self.results_dir / "summaries"
        if not eval_root.exists():
            raise FileNotFoundError(f"Summaries-Verzeichnis nicht gefunden: {eval_root}")

        # Experimente bestimmen
        if experiment_ids is None:
            exp_dirs = [d for d in eval_root.iterdir() if d.is_dir()]
            if group_prefix:
                exp_dirs = [d for d in exp_dirs if d.name.startswith(group_prefix)]
            if not exp_dirs:
                print("⚠️ Keine Experimente gefunden")
                self.df_evaluations = pd.DataFrame()
                return self.df_evaluations
            print(f"ℹ️ Lade Evaluationen für {len(exp_dirs)} Experimente: {[d.name for d in exp_dirs]}")
        else:
            exp_dirs = [eval_root / exp_id for exp_id in experiment_ids]

        rows: List[Dict] = []

        for exp_dir in exp_dirs:
            exp_id = exp_dir.name
            eval_file = exp_dir / "evaluations" / "automated_metrics.json"
            if not eval_file.exists():
                print(f"⚠️ Keine automated_metrics.json in {exp_dir}")
                continue

            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    auto_metrics_list = json.load(f)
                if not isinstance(auto_metrics_list, list):
                    print(f"⚠️ Unerwartetes Format in {eval_file}: {type(auto_metrics_list)}")
                    continue
            except Exception as e:
                print(f"⚠️ Fehler beim Laden von {eval_file}: {e}")
                continue

            # --- Defaults aus experiment_report.json laden ---
            default_model, default_temp, default_top_p = None, None, None
            report_file = exp_dir / "experiment_report.json"
            if report_file.exists():
                try:
                    with open(report_file, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    cfg = report.get("config", {})
                    default_model = cfg.get("model_name") or cfg.get("model")
                    default_temp = cfg.get("temperature")
                    default_top_p = cfg.get("top_p") or cfg.get("top_p_values")
                except Exception as e:
                    print(f"⚠️ Fehler beim Laden von {report_file}: {e}")

            # --- Metadaten laden für Mapping ---
            model_lookup, temp_lookup, top_p_lookup = {}, {}, {}
            meta_file = exp_dir / "experiment_metadata.json"
            if not meta_file.exists():
                alt_file = exp_dir / "metadata.json"
                meta_file = alt_file if alt_file.exists() else None

            if meta_file and meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta_content = json.load(f)
                    if isinstance(meta_content, list):
                        for m in meta_content:
                            did = str(m.get("document_id", "")).strip()
                            if did:
                                model_lookup[did] = m.get("model")
                                temp_lookup[did] = m.get("temperature")
                                top_p_lookup[did] = m.get("top_p") or m.get("top_p_values")
                    elif isinstance(meta_content, dict):
                        default_model = default_model or meta_content.get("model")
                        default_temp = default_temp or meta_content.get("temperature")
                        default_top_p = default_top_p or meta_content.get("top_p") or meta_content.get("top_p_values")
                except Exception as e:
                    print(f"⚠️ Fehler beim Laden der Metadaten {meta_file}: {e}")

            # --- Rows aufbauen ---
            for item in auto_metrics_list:
                doc_id = str(item.get("document_id", "")).strip()
                mdl = item.get("model") or model_lookup.get(doc_id) or default_model or "unknown"
                temp = item.get("temperature") or temp_lookup.get(doc_id) or default_temp
                top_p = item.get("top_p") or top_p_lookup.get(doc_id) or default_top_p

                rouge = item.get("rouge_scores", {})
                rows.append({
                    "experiment_id": exp_id,
                    "document_id": doc_id,
                    "model": mdl,
                    "temperature": temp,
                    "top_p": top_p,
                    "rouge1_f1": rouge.get("rouge1", {}).get("fmeasure"),
                    "rouge2_f1": rouge.get("rouge2", {}).get("fmeasure"),
                    "rougeL_f1": rouge.get("rougeL", {}).get("fmeasure"),
                    "bertscore_f1": (item.get("bertscore") or {}).get("f1"),
                    "summary_length": item.get("summary_length"),
                })

        df = pd.DataFrame(rows)

        # Vereinheitlichung: top_p_values → top_p
        if "top_p_values" in df.columns and "top_p" not in df.columns:
            df = df.rename(columns={"top_p_values": "top_p"})

        # Numerische Typen erzwingen
        for col in ["temperature", "top_p"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self.df_evaluations = df
        return df
    
    def load_llm_judge_results(
        self,
        experiment_ids: Optional[List[str]] = None,
        group_prefix: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Lädt LLM-Judge-Ergebnisse aus llm_judge_results.json.
        
        Args:
            experiment_ids: Liste von Experiment-IDs (optional)
            group_prefix: Präfix zum Filtern von Experimenten

        """
        eval_root = self.results_dir / "summaries"
        if not eval_root.exists():
            print(f"⚠️ Summaries-Verzeichnis nicht gefunden: {eval_root}")
            self.df_llm_judge = pd.DataFrame()
            return self.df_llm_judge
        
        # Experimente bestimmen
        if experiment_ids is None:
            exp_dirs = [d for d in eval_root.iterdir() if d.is_dir()]
            if group_prefix:
                exp_dirs = [d for d in exp_dirs if d.name.startswith(group_prefix)]
            if not exp_dirs:
                print("⚠️ Keine Experimente gefunden")
                self.df_llm_judge = pd.DataFrame()
                return self.df_llm_judge
            print(f"ℹ️ Lade LLM-Judge-Ergebnisse für {len(exp_dirs)} Experimente: {[d.name for d in exp_dirs]}")
        else:
            exp_dirs = [eval_root / exp_id for exp_id in experiment_ids]
        
        rows: List[Dict] = []
        
        for exp_dir in exp_dirs:
            exp_id = exp_dir.name
            judge_file = exp_dir / "evaluations" / "llm_judge_results.json"
            
            if not judge_file.exists():
                print(f"ℹ️ Keine llm_judge_results.json in {exp_dir}")
                continue
            
            try:
                with open(judge_file, "r", encoding="utf-8") as f:
                    judge_data = json.load(f)
                
                # Struktur: {'evaluation_criteria': {...}, 'results': [...]}
                judge_results = judge_data.get('results', [])
                
                if not isinstance(judge_results, list):
                    print(f"⚠️ Unerwartetes Format in {judge_file}")
                    continue
                
            except Exception as e:
                print(f"⚠️ Fehler beim Laden von {judge_file}: {e}")
                continue
            
            # Metadaten aus experiment_report.json laden
            default_model, default_temp, default_top_p = None, None, None
            report_file = exp_dir / "experiment_report.json"
            if report_file.exists():
                try:
                    with open(report_file, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    cfg = report.get("config", {})
                    default_model = cfg.get("model_name") or cfg.get("model")
                    default_temp = cfg.get("temperature")
                    default_top_p = cfg.get("top_p") or cfg.get("top_p_values")
                except Exception as e:
                    print(f"⚠️ Fehler beim Laden von {report_file}: {e}")
            
            # Metadaten-Mapping für document_id
            model_lookup, temp_lookup, top_p_lookup = {}, {}, {}
            meta_file = exp_dir / "experiment_metadata.json"
            if not meta_file.exists():
                alt_file = exp_dir / "metadata.json"
                meta_file = alt_file if alt_file.exists() else None
            
            if meta_file and meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta_content = json.load(f)
                    if isinstance(meta_content, list):
                        for m in meta_content:
                            did = str(m.get("document_id", "")).strip()
                            if did:
                                model_lookup[did] = m.get("model")
                                temp_lookup[did] = m.get("temperature")
                                top_p_lookup[did] = m.get("top_p") or m.get("top_p_values")
                    elif isinstance(meta_content, dict):
                        default_model = default_model or meta_content.get("model")
                        default_temp = default_temp or meta_content.get("temperature")
                        default_top_p = default_top_p or meta_content.get("top_p") or meta_content.get("top_p_values")
                except Exception as e:
                    print(f"⚠️ Fehler beim Laden der Metadaten {meta_file}: {e}")
            
            # Rows aufbauen - flache Struktur erstellen
            for result in judge_results:
                doc_id = str(result.get("document_id", "")).strip()
                
                # Metadaten zuordnen
                mdl = result.get("model") or model_lookup.get(doc_id) or default_model or "unknown"
                temp = result.get("temperature") or temp_lookup.get(doc_id) or default_temp
                top_p = result.get("top_p") or top_p_lookup.get(doc_id) or default_top_p
                
                # Basis-Daten
                row = {
                    "experiment_id": exp_id,
                    "document_id": doc_id,
                    "model": mdl,
                    "temperature": temp,
                    "top_p": top_p,
                    "overall_score": result.get("overall_score")
                }
                
                # Kriterien-Scores extrahieren
                criteria_scores = result.get("criteria_scores", {})
                for criterion, details in criteria_scores.items():
                    if isinstance(details, dict):
                        row[f"{criterion}_score"] = details.get("score")
                        row[f"{criterion}_reasoning"] = details.get("reasoning", "")
                    else:
                        # Fallback falls nur Score als Zahl vorliegt
                        row[f"{criterion}_score"] = details
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Vereinheitlichung: top_p_values → top_p
        if "top_p_values" in df.columns and "top_p" not in df.columns:
            df = df.rename(columns={"top_p_values": "top_p"})
        
        # Numerische Typen erzwingen
        for col in ["temperature", "top_p", "overall_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Score-Spalten numerisch machen
        score_cols = [col for col in df.columns if col.endswith("_score")]
        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        self.df_llm_judge = df
        print(f"✅ LLM-Judge-Ergebnisse geladen: {len(df)} Zeilen")
        return df

    
    def compute_descriptive_statistics(
        self,
        group_by: str = 'model'
    ) -> pd.DataFrame:
        """
        Berechnet deskriptive Statistiken.
        
        Args:
            group_by: Gruppierungsvariable ('model', 'temperature', 'top_p')

        """
        if self.df_evaluations is None:
            raise ValueError("Erst Evaluations-Daten laden mit load_evaluation_results()")
        
        if group_by not in self.df_evaluations.columns:
            print(f"⚠️ Spalte '{group_by}' fehlt. Verfügbare Spalten: {list(self.df_evaluations.columns)}")
            return pd.DataFrame()

        # Metriken die aggregiert werden sollen
        metric_cols = [col for col in self.df_evaluations.columns 
                      if any(m in col.lower() for m in ['rouge', 'bert', 'score'])]
        
        stats_df = self.df_evaluations.groupby(group_by).agg(
            **{f"{col}_mean": (col, "mean") for col in metric_cols},
            **{f"{col}_std": (col, "std") for col in metric_cols},
            **{f"{col}_median": (col, "median") for col in metric_cols},
            **{f"{col}_min": (col, "min") for col in metric_cols},
            **{f"{col}_max": (col, "max") for col in metric_cols},
        )
        
        return stats_df
    
    def compare_models(
        self,
        model1: str,
        model2: str,
        metric: str = 'rouge1_f1'
    ) -> ComparisonResult:
        """
        Vergleicht zwei Modelle statistisch.
        
        Args:
            model1: Name des ersten Modells
            model2: Name des zweiten Modells
            metric: Zu vergleichende Metrik

        """
        if self.df_evaluations is None:
            raise ValueError("Erst Evaluations-Daten laden")
        
        # Daten extrahieren
        group1 = self.df_evaluations[
            self.df_evaluations['model'] == model1
        ][metric].dropna()
        
        group2 = self.df_evaluations[
            self.df_evaluations['model'] == model2
        ][metric].dropna()
        
        # T-Test
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        # Cohen's d (Effektstärke)
        pooled_std = np.sqrt(
            ((len(group1) - 1) * group1.std()**2 + 
             (len(group2) - 1) * group2.std()**2) / 
            (len(group1) + len(group2) - 2)
        )
        cohens_d = (group1.mean() - group2.mean()) / pooled_std
        
        result = ComparisonResult(
            metric_name=metric,
            group1_mean=group1.mean(),
            group2_mean=group2.mean(),
            group1_std=group1.std(),
            group2_std=group2.std(),
            t_statistic=t_stat,
            p_value=p_value,
            effect_size=cohens_d,
            significant=p_value < 0.05
        )
        
        return result
    
    def plot_model_comparison(
        self,
        metric: str = 'rouge1_f1',
        save: bool = True
    ) -> Optional[Figure]:
        """
        Erstellt Bar Chart für Modell-Vergleich.
        
        Args:
            metric: Zu plottende Metrik
            save: Plot speichern?

        """
        if self.df_evaluations is None:
            raise ValueError("Erst Evaluations-Daten laden")
        
        # Daten aggregieren
        model_stats = self.df_evaluations.groupby('model')[metric].agg(['mean', 'std'])
        
        # Plot erstellen
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(model_stats))
        bars = ax.bar(
            x,
            model_stats['mean'],
            yerr=model_stats['std'],
            capsize=5,
            alpha=0.8
        )
        
        ax.set_xlabel('Modell', fontsize=12)
        ax.set_ylabel(f'{metric} Score', fontsize=12)
        ax.set_title(f'Modellvergleich: {metric}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_stats.index, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Werte über Balken anzeigen
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f'model_comparison_{metric}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot gespeichert: {filename}")
        
        return fig
    
    def plot_parameter_heatmap(
        self,
        metric: str = 'rouge1_f1',
        model: Optional[str] = None,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Erstellt Heatmap für Temperature × Top-P Matrix.

        Args:
            metric: Zu plottende Metrik
            model: Spezifisches Modell (None = alle)
            save: Plot speichern?

        """
        if self.df_evaluations is None:
            raise ValueError("Erst Evaluations-Daten laden")

        # DataFrame kopieren und Spaltennamen vereinheitlichen
        df = self.df_evaluations.copy()
        df = df.rename(columns={"top_p_values": "top_p"})

        if df.empty or "temperature" not in df.columns or "top_p" not in df.columns:
            print("⚠️ Keine Parameter-Spalten vorhanden – überspringe Parameter-Heatmap")
            return None

        # Optional nach Modell filtern
        if model:
            df = df[df["model"] == model]

        # Pivot-Tabelle erstellen
        
        print(df[['model','temperature','top_p', metric]].drop_duplicates())

        pivot = df.pivot_table(
            values=metric,
            index="temperature",
            columns="top_p",
            aggfunc="mean"
        )

        if pivot.empty:
            print("⚠️ Pivot-Tabelle leer – keine Heatmap erzeugt")
            return None

        # Heatmap erstellen
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=pivot.mean().mean(),
            ax=ax,
            cbar_kws={'label': f'{metric} Score'}
        )

        title = f'Parameter Heatmap: {metric}'
        if model:
            title += f' ({model})'

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Top-P', fontsize=12)
        ax.set_ylabel('Temperature', fontsize=12)

        plt.tight_layout()

        if save:
            filename = self.output_dir / f'heatmap_{metric}'
            if model:
                filename = self.output_dir / f'heatmap_{model}_{metric}'
            filename = filename.with_suffix('.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Heatmap gespeichert: {filename}")

        return fig

    
    def plot_correlation_matrix(self, save: bool = True) -> Optional[Figure]:
        """Erstellt Korrelationsmatrix zwischen Metriken"""
        if self.df_evaluations is None:
            raise ValueError("Erst Evaluations-Daten laden")

        # Nur numerische Metriken
        metric_cols = [
            col for col in self.df_evaluations.columns
            if any(m in col.lower() for m in ['rouge', 'bert', 'score'])
        ]

        if not metric_cols:
            print("⚠️ Keine numerischen Metriken gefunden – überspringe Korrelationsmatrix")
            return None

        # Immer DataFrame erzwingen
        df = self.df_evaluations.loc[:, metric_cols]
        corr_matrix = df.corr()

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            ax=ax,
            cbar_kws={'label': 'Korrelation'}
        )

        ax.set_title('Korrelation zwischen Evaluationsmetriken',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            filename = self.output_dir / 'correlation_matrix.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Korrelationsmatrix gespeichert: {filename}")

        return fig
        
    def plot_distribution(
        self,
        metric: str = 'rouge1_f1',
        group_by: str = 'model',
        save: bool = True
    ) -> Figure:
        """
        Erstellt Box-Plot für Metrik-Verteilungen.
        
        Args:
            metric: Zu plottende Metrik
            group_by: Gruppierungsvariable
            save: Plot speichern?

        """
        if self.df_evaluations is None:
            raise ValueError("Erst Evaluations-Daten laden")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Box Plot
        self.df_evaluations.boxplot(
            column=metric,
            by=group_by,
            ax=ax,
            grid=True
        )
        
        ax.set_xlabel(group_by.capitalize(), fontsize=12)
        ax.set_ylabel(f'{metric} Score', fontsize=12)
        ax.set_title(f'Verteilung: {metric} nach {group_by}', 
                    fontsize=14, fontweight='bold')
        
        plt.suptitle('')  # Standard-Titel entfernen
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f'distribution_{metric}_by_{group_by}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Verteilungsplot gespeichert: {filename}")
        
        return fig
    
    def export_to_latex(
        self,
        metric: str = 'rouge1_f1',
        group_by: str = 'model',
        output_file: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Exportiert Ergebnistabelle als LaTeX.
        
        Args:
            metric: Metrik für Tabelle
            group_by: Gruppierung
            output_file: Ausgabedatei (optional)

        """
        if self.df_evaluations is None:
            raise ValueError("Erst Evaluations-Daten laden")
        
        # Statistiken berechnen
        stats = self.df_evaluations.groupby(group_by).agg(
            Mean=(metric, 'mean'),
            Std=(metric, 'std'),
            Median=(metric, 'median')
        )
        
        # LaTeX-Tabelle generieren
        latex_code = stats.to_latex(
            float_format='%.3f',
            caption=f'Vergleich nach {group_by}: {metric}',
            label=f'tab:{group_by}_{metric}',
            position='htbp'
        )
        
        # Speichern
        if output_file is None:
            output_file = self.output_dir / f'latex_table_{group_by}_{metric}.tex'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(latex_code)
        
        print(f"LaTeX-Tabelle gespeichert: {output_file}")
        
        return latex_code
    
    def export_to_excel(
        self,
        output_file: Optional[Union[str, Path]] = None
    ):
        """
        Exportiert alle Ergebnisse als Excel-Datei.

        """
        if output_file is None:
            output_file = self.output_dir / 'experiment_results.xlsx'
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summaries
            if self.df_summaries is not None:
                self.df_summaries.to_excel(
                    writer,
                    sheet_name='Summaries',
                    index=False
                )
            
            # Evaluations (automatische Metriken)
            if self.df_evaluations is not None:
                self.df_evaluations.to_excel(
                    writer,
                    sheet_name='Evaluations',
                    index=False
                )
            
            # LLM-Judge-Ergebnisse
            if self.df_llm_judge is not None and not self.df_llm_judge.empty:
                self.df_llm_judge.to_excel(
                    writer,
                    sheet_name='LLM_Judge',
                    index=False
                )
                print(f"✅ LLM-Judge-Ergebnisse exportiert: {len(self.df_llm_judge)} Zeilen")
            
            # Deskriptive Statistik pro Modell
            if self.df_evaluations is not None:
                stats = self.compute_descriptive_statistics(group_by='model')
                stats.to_excel(writer, sheet_name='Model_Statistics')
        
        print(f"Excel-Export gespeichert: {output_file}")
