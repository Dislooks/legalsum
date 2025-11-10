"""
Textvorverarbeitung und -bereinigung für juristische Dokumente.

Dieses Modul bietet Funktionen zur Bereinigung und Normalisierung
von Texten vor der Zusammenfassung.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class TextPreprocessor:
    """
    Hauptklasse für Textvorverarbeitung.
    
    Bietet flexible Pipeline für verschiedene Bereinigungsschritte.
    """
    
    def __init__(self, steps: Optional[List[str]] = None):
        """
        Args:
            steps: Liste der anzuwendenden Bereinigungsschritte
                   Mögliche Werte: 'normalize_whitespace', 'remove_headers_footers',
                   'normalize_unicode', 'remove_page_numbers', 'fix_hyphenation',
                   'remove_special_chars', 'normalize_legal_citations', 'remove_beck_online_artifacts'
        """
        if steps is None:
            # Standard-Pipeline für juristische Texte
            steps = [
                'normalize_unicode',
                'remove_beck_online_artifacts',
                'remove_headers_footers',
                'fix_hyphenation',
                'normalize_whitespace',
                'remove_page_numbers',
                'normalize_legal_citations'
            ]
        
        self.steps = steps
        self._step_functions = {
            'normalize_whitespace': self._normalize_whitespace,
            'remove_headers_footers': self._remove_headers_footers,
            'normalize_unicode': self._normalize_unicode,
            'remove_page_numbers': self._remove_page_numbers,
            'fix_hyphenation': self._fix_hyphenation,
            'remove_special_chars': self._remove_special_chars,
            'normalize_legal_citations': self._normalize_legal_citations,
            'remove_toc': self._remove_table_of_contents,
            'remove_beck_online_artifacts': self._remove_beck_online_artifacts
        }
    
    def process(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Führt alle konfigurierten Bereinigungsschritte aus.
        
        Args:
            text: Zu bereinigender Text
            metadata: Optional: Metadaten für kontextbezogene Bereinigung
        """
        if not text or not text.strip():
            logger.warning("Leerer Text übergeben")
            return ""
        
        processed_text = text
        
        for step in self.steps:
            if step in self._step_functions:
                try:
                    processed_text = self._step_functions[step](processed_text)
                    logger.debug(f"Schritt '{step}' erfolgreich angewendet")
                except Exception as e:
                    logger.error(f"Fehler bei Schritt '{step}': {e}")
                    continue
            else:
                logger.warning(f"Unbekannter Bereinigungsschritt: {step}")
        
        return processed_text
    
    def _remove_beck_online_artifacts(self, text: str) -> str:
        """
        Entfernt Beck-Online spezifische Artefakte.
        
        Entfernt:
        - Seitenheader (Kopie von..., abgerufen am...)
        - Seitenfooter (URLs, Copyright)
        - Seitentrenner (wiederkehrende Titel mit Fundstellen)
        """
        # Beck-Online spezifische Patterns kompilieren
        beck_patterns = [
            # Seitenheader: "Kopie von..., abgerufen am... - Quelle: beck-online DIE DATENBANK"
            re.compile(
                r'Kopie.*?abgerufen.*?beck-online.*?DATENBANK',
                re.IGNORECASE | re.DOTALL
            ),
            
            # Seitenfooter URLs - alle beck-online URLs
            re.compile(
                r'https?://beck-online[^\s]*',
                re.IGNORECASE
            ),
            
            # Seitenfooter Copyright mit Seitenzahl
            re.compile(
                r'\d+\s*von\s*\d+\s*©.*?Beck.*?\d{4}',
                re.IGNORECASE
            ),
            
            # Copyright-Zeile am Ende (ohne "von")
            re.compile(
                r'©.*?Verlag.*?Beck.*?(?:GmbH|Co\.?KG).*?\d{4}',
                re.IGNORECASE
            ),
            
            # Seitentrenner: Gericht mit Doppelpunkt und Fundstelle in Klammern
            re.compile(
                r'^[A-ZÄÖÜ]{2,}:.*?\([A-Z]{2,}\s+\d{4},\s*\d+\)',
                re.MULTILINE | re.IGNORECASE
            ),
            
            # Seitentrenner: Lange Zeile mit Fundstelle am Ende
            re.compile(
                r'^.{40,150}\s*\([A-Z]{2,}\s+\d{4},\s*\d+\)\s*$',
                re.MULTILINE | re.IGNORECASE
            ),
            
            # Zeilen die mittels/durch/nach etc. enthalten und mit Klammer enden
            re.compile(
                r'^.*?(?:mittels|durch|nach|bei|von|zu|über|unter).*?\([A-Z]+\s+\d{4},\s*\d+\)',
                re.MULTILINE | re.IGNORECASE
            ),
        ]
        cleaned = text
        removed_count = 0
        
        # Wende alle Beck-Online Patterns an
        for i, pattern in enumerate(beck_patterns):
            matches = pattern.findall(cleaned)
            if matches:
                logger.debug(f"Pattern {i} gefunden: {len(matches)} Treffer")
                removed_count += len(matches)
            cleaned = pattern.sub('', cleaned)
        
        logger.debug(f"Beck-Online Artifacts: {removed_count} Patterns entfernt")
        
        # Zusätzliche zeilenbasierte Bereinigung
        lines = cleaned.split('\n')
        cleaned_lines = []
        removed_lines = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            if re.match(r'^[\s\-_=.•]+$', line_stripped):
                removed_lines += 1
                continue
            
            if re.search(r'beck-online', line_stripped, re.IGNORECASE):
                removed_lines += 1
                continue
            
            if '©' in line_stripped and 'beck' in line_stripped.lower():
                removed_lines += 1
                continue
      
            if re.search(r'https?://', line_stripped):
                removed_lines += 1
                continue
            
            if line_stripped == 'Pauschalwerts':
                removed_lines += 1
                continue
            
            if (re.match(r'^[A-ZÄÖÜ]{2,}:', line_stripped) and 
                'Prognose' in line_stripped and 
                len(line_stripped) < 150):
                removed_lines += 1
                continue

            if re.match(r'^\d{4}$', line_stripped):
                removed_lines += 1
                continue

            if (len(line_stripped) < 100 and 
                re.search(r'(?:NJW|BeckRS)\s+\d{4}', line_stripped)):
                removed_lines += 1
                continue

            if (len(line_stripped) < 150 and 
                re.search(r'(?:mittels|Prognose).*?\d{4}', line_stripped)):
                removed_lines += 1
                continue
            
            if re.search(r'\d+\s*von\s*\d+', line_stripped):
                removed_lines += 1
                continue
            
            if re.search(r'\d{2}/\d{2}/\d{4}', line_stripped) and len(line_stripped) < 50:
                removed_lines += 1
                continue

            if (re.match(r'^[A-ZÄÖÜ]{2,}:\s+\w+', line_stripped) and 
                len(line_stripped) < 100 and
                not re.search(r'\.$', line_stripped)):
                removed_lines += 1
                continue
            
            cleaned_lines.append(line)
        
        logger.debug(f"Beck-Online Artifacts: {removed_lines} Zeilen entfernt")
        cleaned = '\n'.join(cleaned_lines)
        
        return cleaned
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalisiert Unicode-Zeichen (NFC-Form).
        
        Konvertiert verschiedene Darstellungen desselben Zeichens
        in eine kanonische Form.
        """
        return unicodedata.normalize('NFC', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalisiert Leerzeichen und Zeilenumbrüche.

        """
        # Mehrfache Leerzeichen durch einzelnes ersetzen
        text = re.sub(r' +', ' ', text)
        
        # Tabs durch Leerzeichen ersetzen
        text = text.replace('\t', ' ')
        
        # Mehrfache Zeilenumbrüche reduzieren (max. 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Leerzeichen am Anfang/Ende jeder Zeile entfernen
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _remove_headers_footers(self, text: str) -> str:
        """
        Entfernt typische Kopf- und Fußzeilen.

        """
        lines = text.split('\n')
        
        # Entferne sehr kurze Zeilen am Anfang (oft Header)
        while lines and len(lines[0].strip()) < 50:
            lines.pop(0)
        
        # Entferne sehr kurze Zeilen am Ende (oft Footer)
        while lines and len(lines[-1].strip()) < 50:
            lines.pop()
        
        return '\n'.join(lines)
    
    def _remove_page_numbers(self, text: str) -> str:
        """
        Entfernt Seitenzahlen.

        """
        # "Seite X" oder "Page X"
        text = re.sub(r'\b(?:Seite|Page|S\.)\s*\d+\b', '', text, flags=re.IGNORECASE)
        
        # Alleinstehende Zahlen in eckigen Klammern
        text = re.sub(r'^\s*\[\d+\]\s*$', '', text, flags=re.MULTILINE)
        
        # Zahlen mit Bindestrichen: - 1 -
        text = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)
        
        # Alleinstehende Zahlen am Zeilenanfang (nur wenn < 1000)
        text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
        
        return text
    
    def _fix_hyphenation(self, text: str) -> str:
        """
        Behebt Silbentrennung am Zeilenende.
        
        """
        # Worttrennungen am Zeilenende (Bindestrich + Zeilenumbruch + Kleinbuchstabe)
        text = re.sub(r'-\s*\n\s*([a-zäöüß])', r'\1', text)
        
        return text
    
    def _remove_special_chars(self, text: str) -> str:
        """
        Entfernt oder ersetzt spezielle Steuerzeichen.

        """
        # Entferne Steuerzeichen außer Zeilenumbruch und Tab
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        # Ersetzt nicht-standardmäßige Anführungszeichen
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Ersetzt Gedankenstriche durch normale Bindestriche
        text = text.replace('–', '-').replace('—', '-')
        
        return text
    
    def _normalize_legal_citations(self, text: str) -> str:
        """
        Normalisiert juristische Zitierweisen.
        
        """
        # § mit Leerzeichen normalisieren: §5 -> § 5
        text = re.sub(r'§\s*(\d+)', r'§ \1', text)
        
        # "Abs." normalisieren
        text = re.sub(r'\bAbs\.\s*(\d+)', r'Abs. \1', text)
        
        # "Art." normalisieren
        text = re.sub(r'\bArt\.\s*(\d+)', r'Art. \1', text)
        
        return text
    
    def _remove_table_of_contents(self, text: str) -> str:
        """
        Versucht Inhaltsverzeichnisse zu erkennen und zu entfernen.
        
        """
        lines = text.split('\n')
        cleaned_lines = []
        in_toc = False
        toc_pattern = re.compile(r'^.{3,60}\.{3,}\s*\d{1,3}$')
        
        for line in lines:
            # Erkenne TOC-Start
            if 'Inhaltsverzeichnis' in line or 'Table of Contents' in line.lower():
                in_toc = True
                continue
            
            # Erkenne TOC-Zeilen
            if in_toc and toc_pattern.match(line.strip()):
                continue
            
            # TOC-Ende (erste substantielle Zeile nach TOC)
            if in_toc and len(line.strip()) > 100:
                in_toc = False
            
            if not in_toc:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
def create_preprocessing_pipeline(
    profile: str = 'standard',
    custom_steps: Optional[List[str]] = None
) -> TextPreprocessor:
    """
    Factory-Funktion für vordefinierte Preprocessing-Profile.
    
    Args:
        profile: Vordefiniertes Profil ('minimal', 'standard', 'aggressive')
        custom_steps: Eigene Schritte (überschreibt Profil)

    """
    profiles = {
        'minimal': [
            'normalize_unicode',
            'normalize_whitespace'
        ],
        'standard': [
            'normalize_unicode',
            'remove_headers_footers',
            'remove_beck_online_artifacts',
            'fix_hyphenation',
            'normalize_whitespace',
            'remove_page_numbers',
            'normalize_legal_citations'
        ],
        'aggressive': [
            'normalize_unicode',
            'remove_beck_online_artifacts',
            'remove_toc',
            'remove_headers_footers',
            'fix_hyphenation',
            'remove_special_chars',
            'normalize_whitespace',
            'remove_page_numbers',
            'normalize_legal_citations'
        ]
    }
    
    if custom_steps:
        steps = custom_steps
    elif profile in profiles:
        steps = profiles[profile]
    else:
        logger.warning(f"Unbekanntes Profil '{profile}'. Verwende 'standard'.")
        steps = profiles['standard']
    
    return TextPreprocessor(steps=steps)
