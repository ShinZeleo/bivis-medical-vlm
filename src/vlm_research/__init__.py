"""
VLM Research Package: Medical Vision-Language Models & Selective Prediction.
Provides modular components for data validation, bilingual pairing, causal scoring,
selective prediction, and statistical analysis under cross-lingual distribution shift.
"""

from . import config
from .data import data_loader, answer_parser, data_quality, bilingual_pairing, leakage_audit
from . import schema_audit
from . import language_analysis
from . import image_audit
from . import text_analysis
from .evaluation import statistical_analysis
from .visualization import risk_coverage as visualization
from . import feasibility

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
