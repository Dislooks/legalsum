import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class EvaluationVisualizer:
    """
    Klasse zur Visualisierung von Evaluationsergebnissen.

    """
    
    def __init__(self, config: Dict):
        """
        Initialisiert den Visualizer mit Konfiguration.
        
        Args:
            config: Dictionary mit Visualisierungskonfiguration
        """
        self.config = config.get('visualization', {})
        
        # Matplotlib-Style setzen
        style = self.config.get('style', 'seaborn-v0_8')
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' nicht verfügbar. Verwende default.")
        
        # Farbschema
        self.colors = self.config.get('colors', {
            'gpt_models': '#10a37f',
            'llama_models': '#5436da',
            'deepseek_models': '#ff6b6b'
        })
        
        # Ausgabeformat
        self.output_format = self.config.get('format', 'png')
        self.dpi = self.config.get('dpi', 300)
        
        # Seaborn-Palette setzen
        sns.set_palette("husl")
        
        logger.info("EvaluationVisualizer initialisiert")
    
    def create_visualizations(
        self,
        results: Dict,
        output_dir: Path,
        mode: str = "models"
    ) -> Dict[str, List[Path]]:
        """
        Erstellt Visualisierungen basierend auf Modus.
        
        Args:
            results: Evaluationsergebnisse
            output_dir: Ausgabeverzeichnis
            mode: 'models', 'parameters', oder 'criteria'
            
        Returns:
            Dictionary mit Listen von Plot-Pfaden
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {'plots': []}
        
        if mode == 'models':
            # Modellvergleiche
            for metric in ['rouge1', 'rouge2', 'rougeL', 'bertscore']:
                path = output_dir / f'model_comparison_{metric}.{self.output_format}'
                self.plot_model_comparison(results, metric=metric, output_path=path)
                plot_paths['plots'].append(path)
                
        elif mode == 'parameters':
            # Parameter-Analysen
            path = output_dir / f'parameter_heatmap.{self.output_format}'
            self.plot_parameter_heatmap(results, output_path=path)
            plot_paths['plots'].append(path)
            
        elif mode == 'criteria':
            # Kriterien-Analysen (LLM-Judge)
            if 'criteria_scores' in results[0] if results else {}:
                path = output_dir / f'criteria_radar.{self.output_format}'
                self.plot_criteria_radar(results, output_path=path)
                plot_paths['plots'].append(path)
        
        return plot_paths

    def plot_model_comparison(
        self,
        results: List[Dict],
        metric: str = 'rouge1',
        output_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ) -> None:
        """
        Erstellt Bar Chart zum Vergleich verschiedener Modelle.
        
        Args:
            results: Liste von Evaluationsergebnissen
            metric: Zu vergleichende Metrik ('rouge1', 'bertscore', 'overall_score')
            output_path: Pfad zum Speichern (optional)
            title: Plot-Titel (optional)
        """
        logger.info(f"Erstelle Modellvergleich für Metrik: {metric}")
        
        # Daten vorbereiten
        df = pd.DataFrame(results)
        
        if 'model' not in df.columns:
            logger.error("Spalte 'model' nicht in Ergebnissen gefunden")
            return
        
        # Metrik extrahieren
        scores = self._extract_metric_scores(df, metric)
        
        if scores is None or scores.empty:
            logger.error(f"Konnte Scores für Metrik '{metric}' nicht extrahieren")
            return
        
        # Aggregiere nach Modell
        model_stats = scores.groupby('model').agg({
            'score': ['mean', 'std', 'count']
        }).reset_index()
        
        model_stats.columns = ['model', 'mean', 'std', 'count']
        model_stats = model_stats.sort_values('mean', ascending=False)
        
        # Plot erstellen
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(model_stats))
        bars = ax.bar(
            x,
            model_stats['mean'],
            yerr=model_stats['std'],
            capsize=5,
            color=self._get_model_colors(model_stats['model'].tolist()),
            alpha=0.8,
            edgecolor='black',
            linewidth=1.2
        )
        
        # Werte auf Balken anzeigen
        for i, (bar, mean) in enumerate(zip(bars, model_stats['mean'])):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{mean:.3f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        # Styling
        ax.set_xlabel('Modell', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric} Score', fontsize=12, fontweight='bold')
        ax.set_title(
            title or f'Modellvergleich: {metric}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_stats['model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if output_path:
            self._save_plot(fig, output_path)
        else:
            plt.show()
        
        plt.close()
    
    def plot_parameter_heatmap(
        self,
        results: List[Dict],
        metric: str = 'rouge1',
        output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Erstellt Heatmap für Temperature × Top-P Parametermatrix.
        
        Args:
            results: Liste von Evaluationsergebnissen
            metric: Zu visualisierende Metrik
            output_path: Pfad zum Speichern (optional)
        """
        logger.info(f"Erstelle Parameter-Heatmap für {metric}")
        
        df = pd.DataFrame(results)
        
        # Prüfe ob Parameter vorhanden sind
        required_cols = ['temperature', 'top_p']
        if not all(col in df.columns for col in required_cols):
            logger.error("Benötigte Parameter-Spalten nicht gefunden")
            return
        
        # Metrik extrahieren
        scores = self._extract_metric_scores(df, metric)
        
        if scores is None:
            return
        
        # Pivot-Table erstellen
        pivot = scores.pivot_table(
            values='score',
            index='temperature',
            columns='top_p',
            aggfunc='mean'
        )
        
        # Plot erstellen
        fig, ax = plt.subplots(figsize=(10, 7))
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            center=0.5,
            cbar_kws={'label': f'{metric} Score'},
            linewidths=0.5,
            linecolor='white',
            ax=ax
        )
        
        ax.set_title(
            f'Parameter-Optimierung: {metric}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel('Top-P', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            self._save_plot(fig, output_path)
        else:
            plt.show()
        
        plt.close()
    
    def plot_score_distribution(
        self,
        results: List[Dict],
        metric: str = 'rouge1',
        group_by: Optional[str] = 'model',
        output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Erstellt Box Plot für Score-Verteilungen.
        
        Args:
            results: Liste von Evaluationsergebnissen
            metric: Zu visualisierende Metrik
            group_by: Gruppierungskriterium (z.B. 'model', 'temperature')
            output_path: Pfad zum Speichern (optional)
        """
        logger.info(f"Erstelle Box Plot für {metric} gruppiert nach {group_by}")
        
        df = pd.DataFrame(results)
        
        if group_by not in df.columns:
            logger.error(f"Spalte '{group_by}' nicht gefunden")
            return
        
        # Metrik extrahieren
        scores = self._extract_metric_scores(df, metric)
        
        if scores is None:
            return
        
        # Plot erstellen
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sortiere Gruppen nach Median
        group_order = scores.groupby(group_by)['score'].median().sort_values(ascending=False).index
        
        bp = ax.boxplot(
            [scores[scores[group_by] == g]['score'].values for g in group_order],
            labels=group_order,
            patch_artist=True,
            notch=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=8)
        )
        
        # Färbung
        colors = self._get_model_colors(group_order) if group_by == 'model' else None
        if colors:
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        # Styling
        ax.set_xlabel(group_by.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric} Score', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Score-Verteilung: {metric} nach {group_by}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            self._save_plot(fig, output_path)
        else:
            plt.show()
        
        plt.close()
    
    def plot_correlation_scatter(
        self,
        results: List[Dict],
        metric_x: str = 'rouge1',
        metric_y: str = 'bertscore',
        output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Erstellt Scatter Plot zur Korrelationsanalyse zwischen Metriken.
        
        Args:
            results: Liste von Evaluationsergebnissen
            metric_x: X-Achsen Metrik
            metric_y: Y-Achsen Metrik
            output_path: Pfad zum Speichern (optional)
        """
        logger.info(f"Erstelle Korrelations-Scatter: {metric_x} vs {metric_y}")
        
        df = pd.DataFrame(results)
        
        # Scores extrahieren
        scores_x = self._extract_metric_scores(df, metric_x)
        scores_y = self._extract_metric_scores(df, metric_y)
        
        if scores_x is None or scores_y is None:
            return
        
        # Merge auf Index
        merged = pd.merge(
            scores_x[['score']].rename(columns={'score': 'x'}),
            scores_y[['score']].rename(columns={'score': 'y'}),
            left_index=True,
            right_index=True
        )
        
        # Korrelation berechnen
        correlation = merged['x'].corr(merged['y'])
        
        # Plot erstellen
        fig, ax = plt.subplots(figsize=(8, 8))
        
        scatter = ax.scatter(
            merged['x'],
            merged['y'],
            alpha=0.6,
            s=100,
            c=range(len(merged)),
            cmap='viridis',
            edgecolors='black',
            linewidth=0.5
        )
        
        # Regressionslinie
        z = np.polyfit(merged['x'], merged['y'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged['x'].min(), merged['x'].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Regression')
        
        # Styling
        ax.set_xlabel(f'{metric_x} Score', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric_y} Score', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Korrelation: {metric_x} vs {metric_y}\nPearson r = {correlation:.3f}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            self._save_plot(fig, output_path)
        else:
            plt.show()
        
        plt.close()
    
    def plot_criteria_radar(
        self,
        llm_judge_results: List[Dict],
        group_by: str = 'model',
        output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Erstellt Radar Chart für LLM-Judge Kriterien-Profile.
        
        Args:
            llm_judge_results: Ergebnisse von LLM-as-a-Judge
            group_by: Gruppierungskriterium (z.B. 'model')
            output_path: Pfad zum Speichern (optional)
        """
        logger.info(f"Erstelle Radar Chart für Kriterien-Profile")
        
        df = pd.DataFrame(llm_judge_results)
        
        if 'criteria_scores' not in df.columns:
            logger.error("Keine Kriterien-Scores gefunden")
            return
        
        # Extrahiere Kriterien
        criteria = list(df['criteria_scores'].iloc[0].keys())
        criteria = [c for c in criteria if c != 'overall_assessment']
        
        # Aggregiere nach Gruppe
        if group_by in df.columns or f'metadata.{group_by}' in df.columns:
            group_col = f'metadata.{group_by}' if f'metadata.{group_by}' in df.columns else group_by
            groups = df[group_col].unique()
        else:
            groups = ['all']
            df['group'] = 'all'
            group_col = 'group'
        
        # Plot erstellen
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
        angles += angles[:1]  # Schließe den Kreis
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria, size=10)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'], size=8)
        ax.grid(True)
        
        # Plot für jede Gruppe
        for i, group in enumerate(groups):
            group_df = df[df[group_col] == group]
            
            # Durchschnittliche Scores pro Kriterium
            values = []
            for criterion in criteria:
                scores = [
                    cs.get(criterion, {}).get('score', 0)
                    for cs in group_df['criteria_scores']
                    if criterion in cs
                ]
                values.append(np.mean(scores) if scores else 0)
            
            values += values[:1]  # Schließe den Kreis
            
            ax.plot(angles, values, 'o-', linewidth=2, label=str(group))
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_title(
            'LLM-Judge Kriterien-Profile',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        
        if output_path:
            self._save_plot(fig, output_path)
        else:
            plt.show()
        
        plt.close()
    
    def generate_summary_table(
        self,
        results: List[Dict],
        metrics: List[str],
        output_path: Union[str, Path],
        format: str = 'latex'
    ) -> None:
        """
        Generiert Summary-Tabelle für die Masterarbeit.
        
        Args:
            results: Liste von Evaluationsergebnissen
            metrics: Liste der zu inkludierenden Metriken
            output_path: Pfad zur Output-Datei
            format: 'latex', 'markdown', oder 'html'
        """
        logger.info(f"Generiere Summary-Tabelle im Format: {format}")
        
        df = pd.DataFrame(results)
        
        if 'model' not in df.columns:
            logger.error("Spalte 'model' nicht gefunden")
            return
        
        # Aggregiere nach Modell
        summary_data = []
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            row = {'Model': model, 'N': len(model_df)}
            
            for metric in metrics:
                scores = self._extract_metric_scores(model_df, metric)
                if scores is not None and not scores.empty:
                    mean = scores['score'].mean()
                    std = scores['score'].std()
                    row[f'{metric}_mean'] = mean
                    row[f'{metric}_std'] = std
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Format abhängig von Output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'latex':
            latex_str = summary_df.to_latex(
                index=False,
                float_format="%.3f",
                caption="Modellvergleich: Evaluationsmetriken",
                label="tab:model_comparison",
                column_format='l' + 'c' * (len(summary_df.columns) - 1)
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex_str)
                
        elif format == 'markdown':
            md_str = summary_df.to_markdown(index=False, floatfmt=".3f")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_str)
                
        elif format == 'html':
            html_str = summary_df.to_html(index=False, float_format="%.3f")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_str)
        
        logger.info(f"Summary-Tabelle gespeichert: {output_path}")
    
    def create_comprehensive_report(
        self,
        results: Dict,
        output_dir: Union[str, Path]
    ) -> None:
        """
        Erstellt umfassenden Report mit allen Visualisierungen.
        
        Args:
            results: Dictionary mit allen Evaluationsergebnissen
                {
                    'automated': [...],
                    'llm_judge': [...],
                    'config': {...}
                }
            output_dir: Verzeichnis für Output-Dateien
        """
        logger.info("Erstelle umfassenden Evaluation-Report")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        automated_results = results.get('automated', [])
        llm_judge_results = results.get('llm_judge', [])
        
        # 1. Modellvergleiche
        if automated_results:
            for metric in ['rouge1', 'rouge2', 'rougeL', 'bertscore']:
                try:
                    self.plot_model_comparison(
                        automated_results,
                        metric=metric,
                        output_path=output_dir / f'model_comparison_{metric}.{self.output_format}'
                    )
                except Exception as e:
                    logger.error(f"Fehler bei Modellvergleich {metric}: {e}")
        
        # 2. Parameter-Heatmaps
        if automated_results:
            try:
                self.plot_parameter_heatmap(
                    automated_results,
                    metric='rouge1',
                    output_path=output_dir / f'parameter_heatmap.{self.output_format}'
                )
            except Exception as e:
                logger.error(f"Fehler bei Heatmap: {e}")
        
        # 3. Score-Verteilungen
        if automated_results:
            try:
                self.plot_score_distribution(
                    automated_results,
                    metric='bertscore',
                    group_by='model',
                    output_path=output_dir / f'score_distribution.{self.output_format}'
                )
            except Exception as e:
                logger.error(f"Fehler bei Box Plot: {e}")
        
        # 4. Korrelationen
        if automated_results:
            try:
                self.plot_correlation_scatter(
                    automated_results,
                    metric_x='rouge1',
                    metric_y='bertscore',
                    output_path=output_dir / f'correlation_rouge_bert.{self.output_format}'
                )
            except Exception as e:
                logger.error(f"Fehler bei Scatter Plot: {e}")
        
        # 5. Radar Chart (LLM-Judge)
        if llm_judge_results:
            try:
                self.plot_criteria_radar(
                    llm_judge_results,
                    group_by='model',
                    output_path=output_dir / f'criteria_radar.{self.output_format}'
                )
            except Exception as e:
                logger.error(f"Fehler bei Radar Chart: {e}")
        
        # 6. Summary-Tabellen
        if automated_results:
            try:
                self.generate_summary_table(
                    automated_results,
                    metrics=['rouge1', 'rouge2', 'bertscore'],
                    output_path=output_dir / 'summary_table.tex',
                    format='latex'
                )
            except Exception as e:
                logger.error(f"Fehler bei Summary-Tabelle: {e}")
        
        logger.info(f"Report-Generierung abgeschlossen. Dateien in: {output_dir}")
    
    def _extract_metric_scores(self, df: pd.DataFrame, metric: str) -> Optional[pd.DataFrame]:
        scores_list = []
        for idx, row in df.iterrows():
            score_value = None

            # Direkt auf flache Spalten prüfen
            if metric in row:
                score_value = row[metric]

            # Fallback: verschachtelte Strukturen (alte Logs)
            elif metric.startswith('rouge') and 'rouge_scores' in row:
                score_value = row['rouge_scores'].get(metric, {}).get('fmeasure')
            elif metric == 'bertscore' and 'bertscore' in row:
                score_value = row['bertscore'].get('f1')

            if score_value is not None:
                score_dict = {'score': score_value}
                for col in ['model', 'temperature', 'top_p', 'document_id']:
                    if col in row:
                        score_dict[col] = row[col]
                scores_list.append(score_dict)

        return pd.DataFrame(scores_list) if scores_list else None

    
    def _get_model_colors(self, models: List[str]) -> List[str]:
        """
        Gibt Farben für Modelle basierend auf Konfiguration zurück.
        
        Args:
            models: Liste von Modellnamen

        """
        colors = []
        
        for model in models:
            model_lower = str(model).lower()
            
            if 'gpt' in model_lower:
                colors.append(self.colors.get('gpt_models', '#10a37f'))
            elif 'llama' in model_lower:
                colors.append(self.colors.get('llama_models', '#5436da'))
            elif 'deepseek' in model_lower:
                colors.append(self.colors.get('deepseek_models', '#ff6b6b'))
            else:
                colors.append('#999999')  # Default grau
        
        return colors
    
    def _save_plot(self, fig, output_path: Union[str, Path]) -> None:
        """
        Speichert einen Plot mit konfigurierten Einstellungen.
        
        Args:
            fig: Matplotlib Figure
            output_path: Pfad zur Output-Datei
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            fig.savefig(
                output_path,
                dpi=self.dpi,
                bbox_inches='tight',
                format=self.output_format
            )
            logger.info(f"Plot gespeichert: {output_path}")
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Plots: {e}")


# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def compare_models_statistically(
    results: List[Dict],
    metric: str = 'rouge1',
    alpha: float = 0.05
) -> Dict:
    """
    Führt statistische Tests zum Modellvergleich durch.
    
    Args:
        results: Liste von Evaluationsergebnissen
        metric: Zu vergleichende Metrik
        alpha: Signifikanzniveau

    """
    from scipy import stats
    
    logger.info(f"Führe statistische Tests für {metric} durch")
    
    df = pd.DataFrame(results)
    
    if 'model' not in df.columns:
        logger.error("Spalte 'model' nicht gefunden")
        return {}
    
    models = df['model'].unique()
    
    if len(models) < 2:
        logger.warning("Mindestens 2 Modelle für Vergleich erforderlich")
        return {}
    
    # Scores extrahieren
    model_scores = {}
    for model in models:
        model_df = df[df['model'] == model]
        
        scores = []
        for _, row in model_df.iterrows():
            if metric.startswith('rouge') and 'rouge_scores' in row:
                if metric in row['rouge_scores']:
                    scores.append(row['rouge_scores'][metric]['fmeasure'])
            elif metric == 'bertscore' and 'bertscore' in row:
                scores.append(row['bertscore']['f1'])
            elif metric == 'overall_score' and 'overall_score' in row:
                scores.append(row['overall_score'])
        
        model_scores[model] = scores
    
    # Paarweise t-Tests
    test_results = {}
    
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            scores1 = model_scores[model1]
            scores2 = model_scores[model2]
            
            if len(scores1) < 2 or len(scores2) < 2:
                continue
            
            # t-Test
            t_stat, p_value = stats.ttest_ind(scores1, scores2)
            
            # Effektstärke (Cohen's d)
            cohens_d = (np.mean(scores1) - np.mean(scores2)) / np.sqrt(
                (np.std(scores1)**2 + np.std(scores2)**2) / 2
            )
            
            test_results[f'{model1}_vs_{model2}'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'cohens_d': float(cohens_d),
                'mean_diff': float(np.mean(scores1) - np.mean(scores2))
            }
    
    # ANOVA (wenn >2 Modelle)
    if len(models) > 2:
        valid_scores = [s for s in model_scores.values() if len(s) >= 2]
        if len(valid_scores) > 2:
            f_stat, p_value = stats.f_oneway(*valid_scores)
            test_results['anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < alpha
            }
    
    logger.info(f"Statistische Tests abgeschlossen: {len(test_results)} Vergleiche")
    return test_results