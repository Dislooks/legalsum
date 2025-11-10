"""
Utils Package für Legal Summarization Pipeline

Enthält Logging, Validierung und Helper-Funktionen.
"""

from .logger import (
    PipelineLogger,
    get_logger,
    init_logger
)

from .validators import (
    ValidationError,
    ConfigValidator,
    ParameterValidator,
    FileValidator,
    APIResponseValidator,
    TextValidator,
    validate_config_files,
    quick_validate
)

from .helpers import (
    generate_document_id,
    generate_experiment_id,
    calculate_file_hash,
    sanitize_filename,
    count_tokens_estimate,
    estimate_api_cost,
    format_duration,
    truncate_text,
    extract_json_from_text,
    save_json,
    load_json,
    create_metadata,
    batch_items,
    merge_configs,
    ProgressTracker,
    ensure_directory_structure,
    clean_text,
    word_count,
    char_count
)

__version__ = '1.0.0'

__all__ = [
    # Logger
    'PipelineLogger',
    'get_logger',
    'init_logger',
    
    # Validators
    'ValidationError',
    'ConfigValidator',
    'ParameterValidator',
    'FileValidator',
    'APIResponseValidator',
    'TextValidator',
    'validate_config_files',
    'quick_validate',
    
    # Helpers
    'generate_document_id',
    'generate_experiment_id',
    'calculate_file_hash',
    'sanitize_filename',
    'count_tokens_estimate',
    'estimate_api_cost',
    'format_duration',
    'truncate_text',
    'extract_json_from_text',
    'save_json',
    'load_json',
    'create_metadata',
    'batch_items',
    'merge_configs',
    'ProgressTracker',
    'ensure_directory_structure',
    'clean_text',
    'word_count',
    'char_count',
]