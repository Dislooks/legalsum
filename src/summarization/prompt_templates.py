"""
Verwaltung und Generierung von Prompts für verschiedene Zusammenfassungsszenarien.
"""

from typing import Dict, Optional
import logging

# Bei Entwicklung, Testing, Debugging und Erweiterung wurde Claude Sonnet 4.5 genutzt
class PromptTemplate:
    """
    Verwaltung von Prompt-Templates für Dokumentenzusammenfassungen.

    """
    
    TEMPLATES: Dict[str, str] = {
       
        
        "court_ruling": 
"""Du bist ein spezialisierter Assistent für die Zusammenfassung von Urteilen des Bundesgerichtshofs (BGH) im Zivilrecht. Erstelle eine präzise, strukturierte Zusammenfassung, die alle wesentlichen Elemente des Urteils vollständig erfasst.

#Zielsetzung und Umfang
Erstelle eine Zusammenfassung mit einer Zielumfang von 400-800 Wörtern. Bleibe zwingend innerhalb dieser Spanne! 

Passe den Umfang an die Länge und Komplexität des Urteils an. Einfachere Urteile können kürzer ausfallen, besonders komplexe Urteile mit mehreren Rechtsfragen dürfen länger sein.
Wichtig: Vollständigkeit geht zwar vor Kürze, der vorgegebene Zielumfang ist dennoch zwingend einzuhalten.

Nutze keine Markdown-Formatierungen in Deiner Zusammenfassung.

#Struktureller Aufbau der Zusammenfassung
Die Zusammenfassung folgt einem dreistufigen Aufbau im Fließtextformat (die Zwischenüberschriften müssen nicht zwingend enthalten sein):

1. Sachverhalt (ca. 15-20% der Zusammenfassung)
Fasse den Tatbestand knapp aber vollständig zusammen:

Beteiligte Parteien und ihre Positionen
Rechtserhebliche Tatsachen in chronologischer Reihenfolge
Streitgegenstand und Klageanträge
Entscheidungen der Vorinstanzen (wenn relevant für das Verständnis)

Stil: Sachlich, präzise, ohne überflüssige Details, aber mit allen entscheidungserheblichen Fakten.

2. Entscheidung (1-2 Sätze)
Fasse das Ergebnis der BGH-Entscheidung prägnant zusammen:

Tenor (z.B. "Revision zurückgewiesen", "Berufungsurteil aufgehoben und zurückverwiesen")
Kurze Kernaussage zur Begründung (z.B. "da die Voraussetzungen des § 823 Abs. 1 BGB vorliegen")

3. Entscheidungsgründe (ca. 75-80% der Zusammenfassung)
Dies ist der Hauptteil - hier liegt der Schwerpunkt der Zusammenfassung. Erläutere die rechtliche Argumentation des BGH vollständig und systematisch:

#Für jede behandelte Rechtsfrage:

- Formuliere die Rechtsfrage präzise
- Nenne die einschlägigen Normen mit vollständigen Paragraphenverweisen
- Erläutere die dogmatische Herleitung und Argumentation des BGH
- Zeige auf, wie der BGH die Normen auf den konkreten Sachverhalt subsumiert
- Erwähne abweichende Ansichten (Vorinstanzen, Literatur), wenn der BGH darauf eingeht
- Zitiere relevante Präjudizien mit Fundstelle (z.B. "BGH, Urteil vom 12.03.2020 - III ZR 42/19")

#Beachtung der argumentativen Struktur:

- Folge der hierarchischen Gliederung des Urteils (Randnummern, Argumentationsebenen)
- Unterscheide zwischen Haupterwägungen (höhere Strukturebene) und unterstützenden Argumenten (tiefere Ebenen)
- Führe Haupterwägungen ausführlicher aus, unterstützende Argumente nur knapp

BGH-Urteile folgen einer hochstrukturierten Argumentationsarchitektur mit baumartigem Aufbau:

- Randnummern (Rn.): Jeder Absatz ist durchnummeriert und bildet eine logische Texteinheit
- Argumentationsebenen: Die Entscheidungsgründe folgen einer hierarchischen Struktur, in der Rechtsfragen auf verschiedenen Ebenen erörtert werden

#Standardgliederung:

- Rubrum: Gericht, Aktenzeichen, Datum, Parteien
- Leitsätze: Prägnante Formulierung der zentralen normativen Aussagen (nicht bei allen Urteilen vorhanden)
- Tatbestand: Sachverhalt und Prozessgeschichte
- Entscheidungsgründe: Rechtliche Würdigung in baumartig strukturierter Argumentation

#Kritische Anforderungen: Begriffliche und normative Präzision
#Juristische Fachbegriffe

- Verwende ausschließlich die juristischen Termini technici aus dem Originaltext
- Niemals umgangssprachlich umschreiben (falsch: "normaler Gebrauch" | richtig: "vertragsgemäße Verwendung")
- Bei Rechtsinstituten: Exakte Bezeichnung (z.B. "culpa in contrahendo", nicht "Verschulden bei Vertragsanbahnung")
- Behalte die präzise Rechtssprache bei: "mangelhaft" ≠ "fehlerhaft", "Rücktritt" ≠ "Kündigung"

#Paragraphenverweise - NIEMALS ohne korrekte Zitation
Dies ist wichtig für die Qualität der Zusammenfassung.
Zitiere alle relevanten Normen im exakt korrekten Format:

#Standardformat:
- Einfache Norm: § 123 Abs. 1 BGB
- Mit Satz: § 433 Abs. 1 S. 1 BGB
- Mit Nummer: § 6 Abs. 1 S. 1 Nr. 1 UWG
- Mit Buchstabe: § 21 Abs. 2 S. 1 Nr. 3 Buchst. a GmbHG

#Mehrere Normen:
- Aufzählung: § 280 Abs. 1, § 241 Abs. 2 BGB
- Bei identischen Absätzen: §§ 145, 147 BGB
- Verweisketten: § 346 Abs. 1 i.V.m. § 357 Abs. 1 BGB
- Entsprechende Anwendung: § 280 Abs. 1 BGB analog

#Niemals:
- Paragraphen paraphrasieren ("nach den gesetzlichen Vorschriften")
- Unvollständige Zitate ("§ 280 BGB" statt "§ 280 Abs. 1 BGB")
- Falsche Abkürzungen ("§§ 280ff. BGB" statt präziser Nennung)
- Weglassen von Paragraphen

Jede Rechtsaussage muss mit der zugrunde liegenden Norm verknüpft sein.

Text:
{text}

Zusammenfassung:""",
        
        "contract": """
Du bist ein spezialisierter Assistent für die vollständige abstraktive Zusammenfassung von rechtlichen Verträgen (Kaufverträge, Arbeitsverträge, Franchiseverträge, Mietverträge, Dienstleistungsverträge, etc.). Deine Aufgabe ist es, ein umfassendes Nachschlagewerk zu erstellen, das alle rechtlich erheblichen Klauseln des Vertrags neutral und vollständig erfasst.

Der Zielumfang ist etwa 300-800 Worte. Bleibe zwingend innerhalb dieser Spanne! 

Passe den Umfang an die Länge und Komplexität des Vertrags an. Einfachere Verträge können kürzer ausfallen, besonders komplexe Verträge mit mehr rechtlich erheblichem Inhalt dürfen länger sein.
Wichtig: Vollständigkeit geht zwar vor Kürze, der vorgegebene Zielumfang ist dennoch zwingend einzuhalten.

Nutze keine Markdown-Formatierungen in Deiner Zusammenfassung.

#Zielsetzung
Diese Zusammenfassung dient zwei kritischen Anwendungsfällen:

Nachschlagewerk: Schnelles Auffinden spezifischer Vertragsinhalte ohne Durchsuchen des gesamten Dokuments
Due Diligence-Grundlage: Ermöglicht menschlichen Bearbeitern die zügige Identifikation von Risiken und ungewöhnlichen Klauseln

#Kritisches Prinzip: VOLLSTÄNDIGKEIT VOR KÜRZE

- Keine Klausel mit rechtlicher Relevanz darf fehlen
- Jede Regelung, die Rechte oder Pflichten begründet, muss erfasst werden
- Im Zweifelsfall: Aufnehmen, nicht weglassen

#Neutralitätsprinzip: BESCHREIBEN, NICHT BEWERTEN

- Gib den Vertragsinhalt objektiv wieder
- Keine Risikobewertungen oder Einschätzungen wie "problematisch", "ungewöhnlich", "riskant"
- Keine Empfehlungen zur Vertragsgestaltung
- Stelle Einseitigkeiten oder Besonderheiten faktisch dar, ohne zu werten

#Besonderheiten der Dokumentenklasse "Vertrag"
Strukturelle Merkmale:

- Modularer Aufbau: Verträge bestehen aus nummerierten Paragraphen (§ 1, § 2, etc.)
- Hierarchie: Paragraphen mit Absätzen, Sätzen, Nummern und Buchstaben
- Querbezüge: Klauseln verweisen aufeinander ("gemäß § 5 Abs. 2")
- Präambeln und Anlagen: Oft Einleitungen und angehängte Dokumente

#Struktur der Zusammenfassung
WICHTIG: Folge der Paragraphen-/Artikelstruktur des Originalvertrags.
Die Zusammenfassung gliedert sich wie folgt:

Vor der eigentlichen Paragraphen-Zusammenfassung:
- Vertragstyp bzw. Vertragsbezeichnung
- Vertragsparteien mit vollständigen Bezeichnungen und Rollen (Hinweis, falls nicht bekannt)
- Präambel-Inhalt (falls vorhanden): Zusammenfassung von Hintergrund, Zweck, Zielen

Paragraphen-weise Zusammenfassung:
Für jeden Paragraphen (oder Artikel) des Vertrags:
Format: § [Nummer] - [Überschrift aus Original]

[Vollständige inhaltliche Zusammenfassung aller Absätze, Sätze und Unterpunkte dieses Paragraphen]

<Absatznummerierung nicht anzeigen> [Inhalt]
<Absatznummerierung nicht anzeigen> [Inhalt]

#Beispiel:
"§ 3 - Vergütung

Bruttomonatsgehalt € 5.500, Zahlung bis 1. Werktag des Folgemonats
Zusätzlich leistungsabhängige Bonuszahlung bis € 10.000 jährlich nach Ermessen des Arbeitgebers, Auszahlung im ersten Quartal des Folgejahres
Gesetzliche Lohn- und Gehaltsfortzahlung bei Krankheit gemäß § 3 EFZG
Arbeitgeber übernimmt 50% der Beiträge für betriebliche Altersvorsorge nach § 1a BetrAVG, maximal € 150 monatlich"

#Detaillierungsgrad je Paragraph:
- Einfache Paragraphen (1-2 Absätze): Kompakte Zusammenfassung in 2-5 Sätzen
- Komplexe Paragraphen (3+ Absätze, Untergliederungen): Strukturierte Darstellung mit Abs. 1, Abs. 2, etc.
- Sehr umfangreiche Paragraphen: Zusätzliche Untergliederung nach Sätzen oder Nummern bei Bedarf

#Keine Formulierungen wie:
- "Diese Klausel ist problematisch/bedenklich/riskant"
- "Dies könnte zu Nachteilen führen"
- "Unüblich/unangemessen/unfair"
- "Empfehlung: Diese Klausel sollte geändert werden"

#Stattdessen neutral:
- "Die Regelung betrifft nur Partei X"
- "Abweichung von gesetzlicher Frist (§ Y BGB: Z Tage)"
- "Keine entsprechende Gegenregelung für Partei X"
- "Umfang nicht durch [Standard/Kriterium] begrenzt"

Text:
{text}

Zusammenfassung:""",

        "legal_detailed": """Du bist ein juristischer Fachanwalt. Erstelle eine strukturierte Zusammenfassung des folgenden juristischen Dokuments.

Struktur der Zusammenfassung:
1. Sachverhalt (kurz)
2. Rechtliche Fragestellung
3. Entscheidungsgründe
4. Ergebnis
5. Relevante Rechtsnormen

Text:
{text}

Zusammenfassung:""",
        
        "legal_abstract": """Erstelle ein prägnantes Abstract des folgenden juristischen Textes. Konzentriere dich auf:
- Die Kernaussage in 1-2 Sätzen
- Die wichtigste rechtliche Einordnung
- Das praktische Ergebnis

Maximale Länge: ca. {max_length} Zeichen.

Text:
{text}

Abstract:""",
        
        "extractive": """Identifiziere und extrahiere die wichtigsten Sätze aus dem folgenden Text. Gib diese Sätze in ihrer ursprünglichen Form wieder, aber in logischer Reihenfolge als Zusammenfassung.

Maximale Länge der Zusammenfassung: ca. {max_length} Zeichen.

Text:
{text}

Wichtigste Sätze:""",
        
        "bullet_points": """Erstelle eine Zusammenfassung des folgenden Textes in Form von prägnanten Bullet Points. Jeder Punkt soll einen Kernaspekt erfassen.

Text:
{text}

Zusammenfassung (Bullet Points):""",
    }
    
    def __init__(self, custom_templates: Optional[Dict[str, str]] = None):
        """
        Initialisiert den PromptTemplate-Manager.
        
        Args:
            custom_templates: Optional zusätzliche oder überschreibende Templates
        """
        self.logger = logging.getLogger(__name__)
        self.templates = self.TEMPLATES.copy()
        
        if custom_templates:
            self.templates.update(custom_templates)
            self.logger.info(f"{len(custom_templates)} benutzerdefinierte Templates hinzugefügt")
    
    def create_prompt(
        self,
        template_name: str,
        text: str,
        max_length: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Erstellt einen Prompt basierend auf einem Template.
        
        Args:
            template_name: Name des zu verwendenden Templates
            text: Der zu zusammenfassende Text
            max_length: Optional maximale Länge der Zusammenfassung
            **kwargs: Zusätzliche Variablen für Template-Substitution

        """
        if template_name not in self.templates:
            available = ", ".join(self.templates.keys())
            raise ValueError(
                f"Template '{template_name}' nicht gefunden. "
                f"Verfügbare Templates: {available}"
            )
        
        template = self.templates[template_name]
        
        # Bereite Variablen für die Substitution vor
        variables = {
            "text": text,
            "max_length": max_length or "nicht spezifiziert",
            **kwargs
        }
        
        try:
            prompt = template.format(**variables)
        except KeyError as e:
            raise ValueError(
                f"Template-Variable {e} nicht bereitgestellt. "
                f"Benötigte Variablen: {self._get_template_variables(template)}"
            )
        
        return prompt
    
    def add_template(self, name: str, template: str) -> None:
        """
        Fügt ein neues Template hinzu oder überschreibt ein existierendes.
        
        Args:
            name: Name des Templates
            template: Template-String mit {variables}
        """
        if name in self.templates:
            self.logger.warning(f"Template '{name}' wird überschrieben")
        
        self.templates[name] = template
        self.logger.info(f"Template '{name}' hinzugefügt")
    
    def list_templates(self) -> Dict[str, str]:
        """
        Gibt alle verfügbaren Templates zurück.

        """
        return {
            name: template[:100] + "..." if len(template) > 100 else template
            for name, template in self.templates.items()
        }
    
    def get_template(self, name: str) -> str:
        """
        Ruft ein spezifisches Template ab.
        
        Args:
            name: Template-Name

        """
        if name not in self.templates:
            raise ValueError(f"Template '{name}' nicht gefunden")
        return self.templates[name]
    
    @staticmethod
    def _get_template_variables(template: str) -> list:
        """
        Extrahiert alle Variablen aus einem Template-String.
        
        Args:
            template: Template-String

        """
        import re
        return re.findall(r'\{(\w+)\}', template)
    
    def create_custom_prompt(
        self,
        instruction: str,
        text: str,
        constraints: Optional[str] = None,
        format_spec: Optional[str] = None
    ) -> str:
        """
        Erstellt einen benutzerdefinierten Prompt ohne vordefiniertes Template.
        
        Args:
            instruction: Die Hauptanweisung für das LLM
            text: Der zu verarbeitende Text
            constraints: Optional Einschränkungen (z.B. Länge, Stil)
            format_spec: Optional Formatvorgaben

        """
        parts = [instruction]
        
        if constraints:
            parts.append(f"\nEinschränkungen:\n{constraints}")
        
        if format_spec:
            parts.append(f"\nFormat:\n{format_spec}")
        
        parts.append(f"\nText:\n{text}")
        parts.append("\nZusammenfassung:")
        
        return "\n".join(parts)
    
    def optimize_for_token_limit(
        self,
        template_name: str,
        text: str,
        max_tokens: int,
        chars_per_token: float = 4.0
    ) -> str:
        """
        Erstellt einen Prompt und kürzt den Text bei Bedarf, um Token-Limits einzuhalten.
        
        Args:
            template_name: Name des Templates
            text: Originaltext
            max_tokens: Maximale Anzahl Tokens für den gesamten Prompt
            chars_per_token: Geschätzte Zeichen pro Token (Durchschnitt)

        """
        # Erstelle zunächst Prompt ohne Text, um Overhead zu messen
        template = self.templates[template_name]
        prompt_overhead = template.replace("{text}", "").replace("{max_length}", "500")
        overhead_tokens = len(prompt_overhead) / chars_per_token
        
        # Berechne verfügbare Tokens für Text
        available_tokens = max_tokens - overhead_tokens - 100  # 100 Token Puffer
        max_text_chars = int(available_tokens * chars_per_token)
        
        # Kürze Text wenn nötig
        if len(text) > max_text_chars:
            self.logger.warning(
                f"Text wird von {len(text)} auf {max_text_chars} Zeichen gekürzt, "
                f"um Token-Limit einzuhalten"
            )
            text = text[:max_text_chars] + "\n[...Text gekürzt...]"
        
        return self.create_prompt(template_name, text)
