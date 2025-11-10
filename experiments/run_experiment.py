"""
Command-Line Interface für Experiment Runner

"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_runner import ExperimentRunner, ExperimentConfig
from experiments.experiment_analyzer import ExperimentAnalyzer

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def load_config_template(config_name: str) -> dict:
    """Lädt vordefinierte Konfiguration"""
    config_file = Path("experiments/experiment_config_templates.json")
    
    if not config_file.exists():
        raise FileNotFoundError(f"Konfigurations-Template nicht gefunden: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        templates = json.load(f)
    
    if config_name not in templates:
        available = [k for k in templates.keys() if k != "description"]
        raise ValueError(
            f"Konfiguration '{config_name}' nicht gefunden. "
            f"Verfügbar: {', '.join(available)}"
        )
    
    return templates[config_name]

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def list_available_configs():
    """Zeigt alle verfügbaren Konfigurationen"""
    config_file = Path("experiments/experiment_config_templates.json")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        templates = json.load(f)
    
    print("\n" + "="*70)
    print("VERFÃœGBARE EXPERIMENT-KONFIGURATIONEN")
    print("="*70 + "\n")
    
    for name, config in templates.items():
        if name == "description":
            continue
        
        print(f"ðŸ“Š {name}")
        print(f"   {config.get('description', 'Keine Beschreibung')}")
        
        if 'estimated_combinations' in config:
            print(f"   Kombinationen: {config['estimated_combinations']}")
        if 'estimated_cost_usd' in config:
            print(f"   Geschätzte Kosten: ${config['estimated_cost_usd']}")
        if 'estimated_duration_hours' in config:
            print(f"   Geschätzte Dauer: {config['estimated_duration_hours']}h")
        
        print()

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def confirm_experiment(config: dict) -> bool:
    """Fordert Bestätigung für Experiment an"""
    print("\n" + "="*70)
    print("EXPERIMENT-KONFIGURATION")
    print("="*70)
    print(f"Name: {config.get('name', 'Unbenannt')}")
    print(f"Beschreibung: {config.get('description', 'N/A')}")
    print(f"\nModelle: {', '.join(config['models'])}")
    print(f"Temperatures: {config['temperatures']}")
    print(f"Top-p Werte: {config['top_p_values']}")
    
    if 'estimated_combinations' in config:
        print(f"Geschätzte Kombinationen: {config['estimated_combinations']}")
    if 'estimated_api_calls' in config:
        print(f"Geschätzte API-Calls: {config['estimated_api_calls']}")
    if 'estimated_cost_usd' in config:
        print(f"Geschätzte Kosten: ${config['estimated_cost_usd']}")
    if 'estimated_duration_hours' in config:
        print(f"Geschätzte Dauer: {config['estimated_duration_hours']} Stunden")
    
    print("="*70 + "\n")
    
    response = input("Experiment starten? [j/N]: ").strip().lower()
    return response in ['j', 'ja', 'y', 'yes']

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def main():
    parser = argparse.ArgumentParser(
        description="Legal Summarization Pipeline - Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,

    )
    
    parser.add_argument(
        '--no-preload',
        action='store_true',
        help='Dokumente NICHT vorladen (langsamer, weniger RAM)'
    )


    parser.add_argument(
        '--config',
        type=str,
        help='Name der Experiment-Konfiguration'
    )
    
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='Zeige alle verfügbaren Konfigurationen'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/input',
        help='Verzeichnis mit Eingabedokumenten (default: data/input)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Unterbrochenes Experiment fortsetzen'
    )
    
    parser.add_argument(
        '--yes',
        '-y',
        action='store_true',
        help='Automatisch bestätigen (keine Nachfrage)'
    )
    
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Nur Analyse durchführen, kein neues Experiment'
    )
    
    parser.add_argument(
        '--experiment-ids',
        nargs='+',
        help='Spezifische Experiment-IDs für Analyse'
    )
    
    args = parser.parse_args()
    
    # Liste Konfigurationen
    if args.list_configs:
        list_available_configs()
        return 0
    
    # Nur Analyse
    if args.analyze_only:
        print("Starte Analyse bestehender Experimente...")
        analyzer = ExperimentAnalyzer()
        
        if args.experiment_ids:
            analyzer.load_experiment_results(experiment_ids=args.experiment_ids)
            analyzer.load_evaluation_results(experiment_ids=args.experiment_ids)
        else:
            analyzer.load_experiment_results()
            analyzer.load_evaluation_results()
        
        analyzer.create_comprehensive_report()
        return 0
    
    # Experiment durchführen
    if not args.config:
        parser.print_help()
        print("Fehler: --config Parameter erforderlich")
        print("Verwenden Sie --list-configs um verfügbare Konfigurationen zu sehen")
        return 1
    
    try:
        # Konfiguration laden
        config = load_config_template(args.config)
        
        # Bestätigung (auÃŸer wenn --yes)
        if not args.yes:
            if not confirm_experiment(config):
                print("Experiment abgebrochen.")
                return 0
        
        # Runner initialisieren
        runner = ExperimentRunner()
        
        # Resume-Modus: Prüfe ob Checkpoint existiert
        resume_from = None
        experiments = None
        
        if args.resume:
            checkpoint_file = Path("results/checkpoint.json")
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                resume_from = checkpoint.get('last_experiment_id')
                
                # Lade Experiment-Configs aus Checkpoint
                if 'experiment_configs' in checkpoint:
                    experiments = [
                        ExperimentConfig.from_dict(cfg) 
                        for cfg in checkpoint['experiment_configs']
                    ]
                    print(f"Lade {len(experiments)} Experimente aus Checkpoint")
                    print(f"Setze fort ab: {resume_from}")
                else:
                    print("Checkpoint enthält keine Experiment-Configs. Erstelle neu...")
        
        # Experimente erstellen (nur wenn nicht aus Checkpoint geladen)
        if experiments is None:
            print(f"\n🔧 Erstelle Experimente für '{args.config}'...")
            experiments = runner.create_grid_search_experiments(
                models=config['models'],
                temperatures=config['temperatures'],
                top_k=config['top_k'],
                top_p_values=config['top_p_values'],
                experiment_name=args.config,
                document_type=config.get('document_type')
            )
        
        print(f"✅ {len(experiments)} Experimente erstellt\n")
        
        # Experimente durchführen
        print("ðŸš€ Starte Batch-Experimente...\n")
        results = runner.run_batch_experiments(
            experiments=experiments,
            input_dir=Path(args.input_dir),
            preload_documents=not args.no_preload,
            evaluate=config.get('evaluate', True),
            use_llm_judge=config.get('use_llm_judge', False),
            resume_from=resume_from,
            include_manual_evaluation=config.get('include_manual_evaluation', False)
        )
        
        print(f"\nâœ… Experimente abgeschlossen: {len(results)}/{len(experiments)}")
                
        print("\n" + "="*70)
        print("ALLE AUFGABEN ABGESCHLOSSEN!")
        print("="*70)
        print(f"Ergebnisse: results/summaries/")
        print(f"Analysen: results/visualizations/")
        print(f"Excel-Export: results/visualizations/experiment_results.xlsx")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\nâŒ Fehler: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())