"""

Verwendung:
    python run_preprocessor.py <input_dir> <output_dir> [--profile {minimal|standard|aggressive}]

"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

from preprocessor import create_preprocessing_pipeline
from readers import read_documents_from_directory


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def preprocess_directory(
    input_dir: Path,
    output_dir: Path,
    profile: str = 'standard'
) -> None:
    """
    Verarbeitet alle Dokumente in einem Verzeichnis.
    
    Args:
        input_dir: Eingabeverzeichnis
        output_dir: Ausgabeverzeichnis
        profile: Preprocessing-Profil ('minimal', 'standard', 'aggressive')
    """
    logger.info(f"Starte Preprocessing mit Profil: {profile}")
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Preprocessor erstellen
    preprocessor = create_preprocessing_pipeline(profile=profile)
    logger.info(f"Aktive Schritte: {', '.join(preprocessor.steps)}")
    
    # Dokumente einlesen
    documents = read_documents_from_directory(
        directory=input_dir,
        extensions=['.pdf', '.docx', '.txt'],
        recursive=False
    )
    
    if not documents:
        logger.warning(f"Keine Dokumente gefunden in {input_dir}")
        return
    
    logger.info(f"Gefunden: {len(documents)} Dokumente")
    
    # Output-Verzeichnis erstellen
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dokumente verarbeiten
    success_count = 0
    error_count = 0
    total_reduction = 0
    
    for i, doc in enumerate(documents, 1):
        filename = doc['metadata']['filename']
        
        try:
            # Preprocessing
            original_text = doc['text']
            processed_text = preprocessor.process(original_text, doc['metadata'])
            
            # Ausgabedatei
            output_filename = Path(filename).stem + '_preprocessed.txt'
            output_path = output_dir / output_filename
            
            # Speichern
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            # Statistik
            reduction = len(original_text) - len(processed_text)
            reduction_pct = (reduction / len(original_text) * 100) if len(original_text) > 0 else 0
            total_reduction += reduction
            
            logger.info(f"[{i}/{len(documents)}] ✓ {filename}: "
                       f"{len(original_text):,} → {len(processed_text):,} Zeichen "
                       f"({reduction_pct:.1f}% Reduktion)")
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"[{i}/{len(documents)}] ✗ {filename}: {e}")
            error_count += 1
    
    # Zusammenfassung
    logger.info(f"\n{'='*70}")
    logger.info(f"PREPROCESSING ABGESCHLOSSEN")
    logger.info(f"{'='*70}")
    logger.info(f"Erfolgreich:  {success_count}")
    logger.info(f"Fehler:       {error_count}")
    logger.info(f"Reduktion:    {total_reduction:,} Zeichen")
    logger.info(f"Ergebnisse:   {output_dir}")

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def main():
    """Hauptfunktion mit CLI."""
    parser = argparse.ArgumentParser(
        description='Preprocessing für juristische Dokumente',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profile:
  minimal    - Nur Unicode & Whitespace (1-3% Reduktion)
  standard   - Empfohlen für die meisten Fälle (4-8% Reduktion)
  aggressive - Maximale Bereinigung (8-12% Reduktion)

        """
    )
    
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Eingabeverzeichnis mit Dokumenten'
    )
    
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Ausgabeverzeichnis für bereinigte Texte'
    )
    
    parser.add_argument(
        '--profile',
        type=str,
        choices=['minimal', 'standard', 'aggressive'],
        default='standard',
        help='Preprocessing-Profil (default: standard)'
    )
    
    args = parser.parse_args()
    
    # Validierung
    if not args.input_dir.exists():
        print(f"Fehler: Eingabeverzeichnis nicht gefunden: {args.input_dir}")
        return 1
    
    if not args.input_dir.is_dir():
        print(f"Fehler: {args.input_dir} ist kein Verzeichnis")
        return 1
    
    # Preprocessing ausführen
    try:
        start_time = datetime.now()
        preprocess_directory(args.input_dir, args.output_dir, args.profile)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Gesamtzeit: {duration:.1f}s")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Abgebrochen durch Benutzer")
        return 1
    except Exception as e:
        logger.error(f"Fehler: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
