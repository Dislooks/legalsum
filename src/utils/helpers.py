
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import re

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def generate_document_id(file_path: Union[str, Path]) -> str:
    """
    Generiert eindeutige ID für Dokument basierend auf Dateinamen.
    
    Args:
        file_path: Pfad zur Datei

    """
    path = Path(file_path)
    # Entferne Extension und sanitize
    doc_id = path.stem
    # Entferne Sonderzeichen
    doc_id = re.sub(r'[^\w\-]', '_', doc_id)
    return doc_id.lower()

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def generate_experiment_id(
    model: str,
    temperature: float,
    top_p: float
    timestamp = None) -> str:
        
        """
        Generiert eindeutige Experiment-ID.
        
        Args:
            model: Modellname
            temperature: Temperatur-Parameter
            top_p: Top-P Parameter
            timestamp: Optional Zeitstempel

        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Bereinige Modellname
        model_clean = re.sub(r'[^\w]', '', model.lower())
        
        # Formatiere Parameter
        temp_str = f"t{temperature}".replace('.', '')
        top_p_str = f"k{top_p}"
        
        # Zeitstempel
        time_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        return f"exp_{model_clean}_{temp_str}_{top_p_str}_{time_str}"

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def calculate_file_hash(file_path: Union[str, Path]) -> str:
    """
    Berechnet SHA256-Hash einer Datei.
    
    Args:
        file_path: Pfad zur Datei

    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Bereinigt Dateinamen für sicheres Speichern.
    
    Args:
        filename: Original-Dateiname
        max_length: Maximale Länge

    """
    # Entferne gefährliche Zeichen
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Entferne Sonderzeichen (behalte nur Alphanumerisch, Punkt, Unterstrich, Bindestrich)
    safe_name = re.sub(r'[^\w\.\-]', '_', safe_name)
    
    # Entferne mehrfache Unterstriche
    safe_name = re.sub(r'_+', '_', safe_name)
    
    # Kürze falls nötig
    if len(safe_name) > max_length:
        name, ext = safe_name.rsplit('.', 1) if '.' in safe_name else (safe_name, '')
        name = name[:max_length - len(ext) - 1]
        safe_name = f"{name}.{ext}" if ext else name
    
    return safe_name

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def count_tokens_estimate(text: str, method: str = 'words') -> int:
    """
    Schätzt Token-Anzahl (für Kosten-Kalkulation).
    
    Args:
        text: Zu zählender Text
        method: 'words' (schnell) oder 'chars' (genauer)

    """
    if method == 'words':
        word_count = len(text.split())
        # Deutsche Texte haben tendenziell längere Wörter
        return int(word_count * 1.5)
    
    elif method == 'chars':
        # Grobe Schätzung: 4 Zeichen = 1 Token
        return len(text) // 4
    
    else:
        raise ValueError(f"Unbekannte Methode: {method}")

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def format_duration(seconds: float) -> str:
    """
    Formatiert Dauer human-readable.
    
    Args:
        seconds: Dauer in Sekunden

    """
    if seconds < 1:
        return f"{seconds:.2f}s"
    
    if seconds < 60:
        return f"{int(seconds)}s"
    
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Kürzt Text intelligent (an Wortgrenzen).
    
    Args:
        text: Zu kürzender Text
        max_length: Maximale Länge
        suffix: Anhang bei Kürzung

    """
    if len(text) <= max_length:
        return text
    
    # Kürze an Wortgrenze
    truncated = text[:max_length - len(suffix)]
    
    # Finde letztes Leerzeichen
    last_space = truncated.rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated + suffix

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extrahiert JSON aus Text (nützlich für LLM-Responses).
    
    Args:
        text: Text der JSON enthalten könnte

    """
    # Suche nach JSON-Block
    json_patterns = [
        r'\{[^{}]*\}',  # Einfaches JSON-Objekt
        r'\{.*\}',      # Komplexes JSON (greedy)
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    return None

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def save_json(data: Any, file_path: Union[str, Path], pretty: bool = True) -> None:
    """
    Speichert Daten als JSON-Datei.
    
    Args:
        data: Zu speichernde Daten
        file_path: Ziel-Dateipfad
        pretty: Pretty-Print mit Einrückung
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False)

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def load_json(file_path: Union[str, Path]) -> Any:
    """
    Lädt JSON-Datei.
    
    Args:
        file_path: Pfad zur JSON-Datei

    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def create_metadata(
    document_id: str,
    model: str,
    temperature: float,
    top_p: float,
    input_tokens: int,
    output_tokens: int,
    duration: float,
    additional_info: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Erstellt standardisiertes Metadaten-Dictionary.
    
    Args:
        document_id: Dokument-ID
        model: Verwendetes Modell
        temperature: Temperatur-Parameter
        top_p: Top-P Parameter
        input_tokens: Input-Token-Count
        output_tokens: Output-Token-Count
        duration: Verarbeitungszeit in Sekunden
        additional_info: Zusätzliche Metadaten

    """
    metadata = {
        'document_id': document_id,
        'timestamp': datetime.now().isoformat(),
        'model': model,
        'parameters': {
            'temperature': temperature,
            'top_p': top_p
        },
        'tokens': {
            'input': input_tokens,
            'output': output_tokens,
            'total': input_tokens + output_tokens
        },
        'duration_seconds': round(duration, 3),
        'cost_usd': estimate_api_cost(input_tokens, output_tokens, model)
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    return metadata

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Teilt Liste in Batches auf.
    
    Args:
        items: Liste von Items
        batch_size: Größe jedes Batches

    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Merged zwei Config-Dictionaries (override hat Vorrang).
    
    Args:
        base: Basis-Konfiguration
        override: Override-Werte

    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class ProgressTracker:
    """Einfacher Progress-Tracker für lange Operationen."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialisiert Tracker.
        
        Args:
            total: Gesamtanzahl Items
            description: Beschreibung der Operation
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
        
    def update(self, increment: int = 1) -> None:
        """Aktualisiert Progress."""
        self.current += increment
        self._print_progress()
        
    def _print_progress(self) -> None:
        """Gibt Progress-Bar aus."""
        percentage = (self.current / self.total) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Schätze verbleibende Zeit
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = format_duration(eta)
        else:
            eta_str = "?"
        
        bar_length = 30
        filled = int(bar_length * self.current / self.total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(
            f"\r{self.description}: {bar} {self.current}/{self.total} "
            f"({percentage:.1f}%) | ETA: {eta_str}",
            end='',
            flush=True
        )
        
        if self.current >= self.total:
            print()  # Neue Zeile am Ende

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def ensure_directory_structure(base_path: Union[str, Path]) -> Dict[str, Path]:
    """
    Erstellt vollständige Verzeichnisstruktur für Pipeline.
    
    Args:
        base_path: Basis-Projektverzeichnis

    """
    base = Path(base_path)
    
    directories = {
        'data_input': base / 'data' / 'input',
        'data_output': base / 'data' / 'output',
        'gold_standards': base / 'data' / 'gold_standards',
        'results_summaries': base / 'results' / 'summaries',
        'results_evaluations': base / 'results' / 'evaluations',
        'results_visualizations': base / 'results' / 'visualizations',
        'logs': base / 'logs',
        'experiments': base / 'experiments' / 'experiment_configs',
    }
    
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    
    return directories

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def clean_text(text: str) -> str:
    """
    Grundlegende Text-Bereinigung.
    
    Args:
        text: Zu bereinigender Text
        
    Returns:
        Bereinigter Text
    """
    # Mehrfache Leerzeichen
    text = re.sub(r'\s+', ' ', text)
    
    # Mehrfache Zeilenumbrüche
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Trim
    text = text.strip()
    
    return text

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def word_count(text: str) -> int:
    """Zählt Wörter in Text."""
    return len(text.split())

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def char_count(text: str, include_spaces: bool = True) -> int:
    """Zählt Zeichen in Text."""
    if include_spaces:
        return len(text)
    return len(text.replace(' ', '').replace('\n', '').replace('\t', ''))

