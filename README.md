# Legal Summarization Pipeline

## Überblick
Diese Pipeline verarbeitet juristische Dokumente und generiert automatisierte Zusammenfassungen unter Verwendung verschiedener Large Language Models. Das System unterstützt Azure OpenAI und Hugging Face Modelle und bietet umfassende Evaluationsmöglichkeiten.

## Projektstruktur
Das Projekt folgt einer modularen Architektur mit klarer Trennung zwischen Datenverarbeitung, LLM-Integration, Zusammenfassungsgenerierung und Evaluation. Die Hauptkomponenten befinden sich im src-Verzeichnis und sind nach Funktionsbereichen organisiert.

## Voraussetzungen
Python 3.8 oder höher
API-Zugang zu Azure OpenAI oder Hugging Face
Ausreichend Speicherplatz für Dokumentverarbeitung

## Installation
Repository klonen
Virtuelle Umgebung erstellen und aktivieren
Abhängigkeiten installieren: pip install -r requirements.txt
Konfigurationsdateien im config-Verzeichnis anpassen
API-Credentials in credentials.env eintragen

## Konfiguration
Die Datei models_config.yaml enthält Einstellungen für verfügbare Modelle und deren Parameter. In evaluation_config.yaml werden Evaluationskriterien und Metriken definiert.

## Verwendung
Ein Experiment wird über run_experiment.py gestartet. Eingabedokumente werden im data/input-Verzeichnis abgelegt. Das System verarbeitet unterstützte Formate (PDF, DOCX, TXT), generiert Zusammenfassungen und speichert diese in results/summaries.

## Experimentverwaltung
Experimentkonfigurationen werden als JSON-Dateien im experiments-Verzeichnis verwaltet. Dies ermöglicht reproduzierbare Testläufe mit verschiedenen Modellen und Parametern.

## Daten
Die im Projekt verwendeten Daten liegen unter data/contracts und data/court_rulings. Dort sind auch Gold-Standard-Zusammenfassungen zu finden.

## Evaluation
Das System bietet automatisierte Metriken wie ROUGE und BERTScore sowie LLM-basierte Bewertungen. Referenzzusammenfassungen können im data/gold_standards-Verzeichnis hinterlegt werden. Evaluationsergebnisse werden in results/evaluations gespeichert und können als Visualisierungen exportiert werden.

## Erweiterung
Neue Dokumentformate können durch Erweiterung der Reader-Klassen integriert werden. Zusätzliche LLM-Provider erfordern eine neue Client-Implementierung basierend auf base_client.py. Prompt-Templates können in prompt_templates.py angepasst werden.



