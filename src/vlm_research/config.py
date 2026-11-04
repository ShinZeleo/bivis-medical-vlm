import os
import sys
from pathlib import Path

# Random Seed for reproducibility
RANDOM_SEED = 42

# Paths
# PROJECT_ROOT is the root directory of the repository
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw' / 'indorad_vqa'
PROCESSED_DIR = DATA_DIR / 'processed'
MANIFESTS_DIR = DATA_DIR / 'manifests'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
FIGURES_DIR = ARTIFACTS_DIR / 'experiments' / 'decisive_validation' / 'rev05' / 'figures'
CORRECTIVE_FIGURES_DIR = FIGURES_DIR / 'corrective_audit'
IMAGES_DIR = RAW_DIR / 'images'
REPORTS_DIR = PROJECT_ROOT / 'docs' / 'audit'
OUTPUTS_DIR = ARTIFACTS_DIR / 'outputs'
CONFIGS_DIR = PROJECT_ROOT / 'configs'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# Dataset configuration
DATASET_NAME = 'Lab-IS/IndoRad-VQA'
PARQUET_SPLITS = {
    'train': 'data/train-00000-of-00001.parquet',
    'test': 'data/test-00000-of-00001.parquet'
}

# Column Names Constants
COL_QID = 'qid'
COL_IMAGE = 'image'
COL_IMAGE_NAME = 'image_name'
COL_QUESTION = 'question'
COL_QUESTION_ID = 'question_indonesian'
COL_ANSWER = 'answer'
COL_ANSWER_ID = 'answer_indonesian'
COL_ANSWER_TYPE = 'answer_type'
COL_QUESTION_TYPE = 'question_type'
COL_IMAGE_ORGAN = 'image_organ'
COL_PHRASE_TYPE = 'phrase_type'
COL_SPLIT = '_split'
COL_HF_INDEX = '_hf_index'
COL_META_Q_SCORE = 'meta_question_score'
COL_META_A_SCORE = 'meta_answer_score'
COL_META_Q_ATTEMPTS = 'meta_question_attempts'
COL_META_A_ATTEMPTS = 'meta_answer_attempts'

# Column Groups
TEXT_COLS_EN = [COL_QUESTION, COL_ANSWER]
TEXT_COLS_ID = [COL_QUESTION_ID, COL_ANSWER_ID]
CATEGORICAL_COLS = [COL_ANSWER_TYPE, COL_QUESTION_TYPE, COL_IMAGE_ORGAN, COL_PHRASE_TYPE]
META_COLS = [COL_META_Q_SCORE, COL_META_A_SCORE, COL_META_Q_ATTEMPTS, COL_META_A_ATTEMPTS]

# Visualization Settings
FIGURE_DPI = 150
FIGURE_FORMAT = 'png'
MAX_DISPLAY_ROWS = 20

def ensure_dirs():
    """Ensure all required project directories exist."""
    dirs_to_create = [
        DATA_DIR, RAW_DIR, PROCESSED_DIR, MANIFESTS_DIR, IMAGES_DIR,
        FIGURES_DIR, CORRECTIVE_FIGURES_DIR, REPORTS_DIR, OUTPUTS_DIR
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

def get_library_versions() -> dict:
    """Return dictionary of key library versions."""
    import pandas as pd
    import numpy as np
    import matplotlib as mpl
    import seaborn as sns
    import sklearn
    
    return {
        'python': sys.version,
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'matplotlib': mpl.__version__,
        'seaborn': sns.__version__,
        'scikit-learn': sklearn.__version__,
    }

# Ensure directories exist upon importing the config
ensure_dirs()
