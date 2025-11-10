"""
Summarization Module
Enthält Pipeline, Templates und Output-Handler
"""

from .pipeline import SummarizationPipeline, SummarizationConfig, SummarizationResult
from .prompt_templates import PromptTemplate
from .output_handler import OutputHandler

__all__ = [
    'SummarizationPipeline',
    'SummarizationConfig',
    'SummarizationResult',
    'PromptTemplate',
    'OutputHandler'
]