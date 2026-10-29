"""
IndoRad-VQA Research Package.
Contains configuration, data loading, and analysis scripts for the IndoRad-VQA dataset.
"""

from src import config
from src.data import data_loader, answer_parser
from src import schema_audit
from src import data_quality
from src import language_analysis
from src import bilingual_pairing
from src import image_audit
from src import text_analysis
from src import leakage_audit
from src import statistical_analysis
from src import visualization
from src import feasibility

__all__ = [
    'config',
    'data_loader',
    'answer_parser',
    'schema_audit',
    'data_quality',
    'language_analysis',
    'bilingual_pairing',
    'image_audit',
    'text_analysis',
    'leakage_audit',
    'statistical_analysis',
    'visualization',
    'feasibility'
]
