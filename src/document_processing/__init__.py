
from .readers import (
    DocumentReader,
    PDFReader,
    DOCXReader,
    TXTReader,
    DocumentReaderFactory,
    read_documents_from_directory
)

from .preprocessor import (
    TextPreprocessor,
    create_preprocessing_pipeline
)

__version__ = '1.0.0'

__all__ = [
    # Readers
    'DocumentReader',
    'PDFReader',
    'DOCXReader',
    'TXTReader',
    'DocumentReaderFactory',
    'read_documents_from_directory',
    
    # Preprocessor
    'TextPreprocessor',
    'create_preprocessing_pipeline',

]