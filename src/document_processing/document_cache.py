"""
Document Cache für effiziente Wiederverwendung eingelesener Dokumente.

Diese Klasse vermeidet redundantes Einlesen von Dokumenten bei mehreren
Experiment-Durchläufen durch Zwischenspeicherung im RAM.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
from datetime import datetime
import sys

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class DocumentCache:
    """
    Cache für eingelesene Dokumente zur Vermeidung redundanter Lesevorgänge.
    
    Diese Klasse lädt Dokumente einmal und stellt sie für mehrere Experimente
    zur Verfügung.
    
    Attributes:
        _documents: Liste von (text, metadata) Tupeln
        _metadata: Metadaten über den Cache (Ladezeit, Quellverzeichnis, etc.)
        logger: Logger-Instanz
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialisiert einen leeren Document Cache.
        
        Args:
            logger: Optional Logger-Instanz für Logging
        """
        self._documents: List[Tuple[str, Dict]] = []
        self._metadata: Dict = {}
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.debug("DocumentCache initialisiert")
    
    def load_from_directory(
        self,
        directory: Path,
        extensions: List[str],
        max_documents: Optional[int] = None,
        recursive: bool = False
    ) -> int:
        """
        Lädt alle Dokumente aus einem Verzeichnis in den Cache.
        
        Args:
            directory: Eingabeverzeichnis mit Dokumenten
            extensions: Dateierweiterungen (Standard: ['.pdf', '.docx', '.txt'])
            max_documents: Maximale Anzahl zu lesender Dokumente (optional)
            recursive: Rekursiv Unterverzeichnisse durchsuchen (Standard: False)
        """
        from .readers import read_documents_from_directory
        
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Verzeichnis nicht gefunden: {directory}")
        
        if extensions is None:
            extensions = ['.pdf', '.docx', '.txt']
        
        self.logger.info("=" * 70)
        self.logger.info("📦 DOCUMENT CACHE: Lade Dokumente")
        self.logger.info("=" * 70)
        self.logger.info(f"📂 Verzeichnis: {directory}")
        self.logger.info(f"📄 Erweiterungen: {', '.join(extensions)}")
        
        start_time = datetime.now()
        
        # Lese Dokumente mit existierendem Reader
        documents_raw = read_documents_from_directory(
            directory=directory,
            extensions=extensions,
            recursive=recursive
        )
        
        if not documents_raw:
            self.logger.warning(f"⚠️ Keine Dokumente in {directory} gefunden!")
            return 0
        
        # Begrenze Anzahl falls gewünscht
        if max_documents and max_documents < len(documents_raw):
            self.logger.info(f"🔢 Begrenze auf {max_documents} von {len(documents_raw)} Dokumenten")
            documents_raw = documents_raw[:max_documents]
        
        # Konvertiere zu Tupel-Format (text, metadata)
        self._documents = [
            (doc['text'], doc['metadata']) 
            for doc in documents_raw
        ]
        
        # Berechne Statistiken
        duration = (datetime.now() - start_time).total_seconds()
        total_chars = sum(len(text) for text, _ in self._documents)
        avg_chars = total_chars / len(self._documents) if self._documents else 0
        
        # Speichere Metadaten
        self._metadata = {
            'source_directory': str(directory),
            'num_documents': len(self._documents),
            'extensions': extensions,
            'recursive': recursive,
            'max_documents': max_documents,
            'load_timestamp': datetime.now().isoformat(),
            'load_duration_seconds': duration,
            'total_characters': total_chars,
            'average_characters_per_document': avg_chars
        }
        
        self.logger.info(f"✅ {len(self._documents)} Dokumente geladen")
        self.logger.info(f"⏱️ Ladezeit: {duration:.2f} Sekunden")
        self.logger.info(f"📊 Durchschnitt: {avg_chars:,.0f} Zeichen pro Dokument")
        self.logger.info("=" * 70)
        
        return len(self._documents)
    
    def load_from_files(
        self,
        file_paths: List[Path],
        show_progress: bool = True
    ) -> int:
        """
        Lädt spezifische Dateien in den Cache.
        
        Args:
            file_paths: Liste von Dateipfaden
            show_progress: Fortschritt anzeigen (Standard: True)
        """
        from .readers import DocumentReaderFactory
        
        self.logger.info("=" * 70)
        self.logger.info(f"📄 Lade {len(file_paths)} spezifische Dateien")
        self.logger.info("=" * 70)
        
        start_time = datetime.now()
        self._documents = []
        errors = []
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                if show_progress:
                    self.logger.info(f"[{i}/{len(file_paths)}] Lese {file_path.name}")
                
                reader = DocumentReaderFactory.get_reader(file_path)
                text, metadata = reader.read(file_path)
                self._documents.append((text, metadata))
                
            except Exception as e:
                self.logger.error(f"❌ Fehler bei {file_path}: {e}")
                errors.append({
                    'file': str(file_path),
                    'error': str(e)
                })
                continue
        
        duration = (datetime.now() - start_time).total_seconds()
        total_chars = sum(len(text) for text, _ in self._documents)
        
        self._metadata = {
            'source_files': [str(p) for p in file_paths],
            'num_documents': len(self._documents),
            'num_errors': len(errors),
            'errors': errors,
            'load_timestamp': datetime.now().isoformat(),
            'load_duration_seconds': duration,
            'total_characters': total_chars
        }
        
        success_rate = (len(self._documents) / len(file_paths) * 100) if file_paths else 0
        
        self.logger.info(f"✅ {len(self._documents)} Dokumente erfolgreich geladen")
        self.logger.info(f"❌ {len(errors)} Fehler")
        self.logger.info(f"📊 Erfolgsrate: {success_rate:.1f}%")
        self.logger.info(f"⏱️ Ladezeit: {duration:.2f} Sekunden")
        self.logger.info("=" * 70)
        
        return len(self._documents)
    
    def get_documents(self) -> List[Tuple[str, Dict]]:
        """
        Gibt alle gecachten Dokumente zurück.
        """
        if not self._documents:
            self.logger.warning("⚠️ Cache ist leer! Dokumente zuerst mit load_* laden.")
        
        return self._documents
    
    def get_document(self, index: int) -> Tuple[str, Dict]:
        """
        Gibt ein einzelnes Dokument aus dem Cache zurück.
        
        Args:
            index: Index des Dokuments (0-basiert)
        """
        if index < 0 or index >= len(self._documents):
            raise IndexError(
                f"Index {index} außerhalb des Bereichs [0, {len(self._documents)-1}]"
            )
        
        return self._documents[index]
    
    def filter_documents(
        self,
        condition: Callable[[str, Dict], bool]
    ) -> List[Tuple[str, Dict]]:
        """
        Filtert Dokumente basierend auf einer benutzerdefinierten Bedingung.
        
        Args:
            condition: Funktion die (text, metadata) erhält und bool zurückgibt
        """
        filtered = [
            (text, meta) 
            for text, meta in self._documents 
            if condition(text, meta)
        ]
        
        self.logger.debug(
            f"Gefiltert: {len(filtered)}/{len(self._documents)} Dokumente erfüllen Bedingung"
        )
        
        return filtered
    
    def get_metadata(self) -> Dict:
        """
        Gibt Cache-Metadaten zurück.

        """
        
        return {
            **self._metadata,
            'current_size': len(self._documents)
        }
    
    def get_statistics(self) -> Dict:
        """
        Gibt detaillierte Statistiken über die gecachten Dokumente zurück.
        """
        if not self._documents:
            return {
                'num_documents': 0,
                'total_characters': 0,
                'avg_characters': 0,
                'min_characters': 0,
                'max_characters': 0,
                'file_types': {}
            }
        
        # Zeichenstatistiken
        char_counts = [len(text) for text, _ in self._documents]
        
        # Dateityp-Verteilung
        file_types = {}
        for _, meta in self._documents:
            filetype = meta.get('filetype', 'unknown')
            file_types[filetype] = file_types.get(filetype, 0) + 1
        
        return {
            'num_documents': len(self._documents),
            'total_characters': sum(char_counts),
            'avg_characters': sum(char_counts) / len(char_counts),
            'min_characters': min(char_counts),
            'max_characters': max(char_counts),
            'file_types': file_types
        }
    
    def clear(self):
        """
        Leert den Cache und gibt den Speicher frei.
        """
        num_docs = len(self._documents)
        
        self._documents = []
        self._metadata = {}
        
        self.logger.info(f"🗑️ Cache geleert ({num_docs} Dokumente entfernt)")
    
    def is_empty(self) -> bool:
        """
        Prüft ob der Cache leer ist.
        """
        return len(self._documents) == 0
    
    def __len__(self) -> int:
        """
        Gibt die Anzahl gecachter Dokumente zurück.
        """
        return len(self._documents)