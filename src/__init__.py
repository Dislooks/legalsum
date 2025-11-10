"""
Hauptpackage für die Dokumentenzusammenfassung
"""
from .summarization.pipeline import SummarizationPipeline, SummarizationConfig
from .summarization.output_handler import OutputHandler
from .summarization.prompt_templates import PromptTemplate

__all__ = [
    'SummarizationPipeline',
    'SummarizationConfig',
    'OutputHandler',
    'PromptTemplate'

]
