# Source Code Architecture (`src/vlm_research`)

The `src/vlm_research` directory contains the modular, unit-tested Python package powering the research pipeline.

---

## Package Structure

```text
src/vlm_research/
├── __init__.py                  # Package root initializer
├── config.py                   # Central path resolver and constant definitions
│
├── data/                       # Data ingestion, audit, and bilingual cohort pairing
│   ├── __init__.py
│   ├── answer_parser.py        # Normalizes model outputs into canonical YES/NO space
│   ├── bilingual_pairing.py    # Formulates matched EN-ID QA pairs
│   ├── data_loader.py          # HuggingFace Parquet and local CSV loader
│   ├── data_quality.py         # Cohort validation, missing values, duplicate checks
│   ├── image_utils.py          # Pixel verification and black canvas generator
│   └── leakage_audit.py        # Image cluster leakage verification
│
├── inference/                  # Multimodal model execution and causal logit extraction
│   ├── __init__.py
│   └── decisive_validation.py  # Causal sequence log-likelihood and blank perturbation engine
│
├── evaluation/                 # Scoring mechanisms, selective prediction, and statistical tests
│   ├── __init__.py
│   ├── scoring.py              # Length-normalized log-likelihood and directional Delta P_dir
│   ├── selective_prediction.py # Empirical discrete AURC and risk estimators (M1, M_norm, S_LR)
│   └── statistical_analysis.py # Paired cluster permutation test and Holm-Bonferroni correction
│
├── models/                     # Foundation VLM loaders and architecture wrappers
│   ├── __init__.py
│   └── qwen_loader.py          # Qwen2.5-VL and HuatuoGPT-Vision loader with device placement
│
└── visualization/              # Publication-grade diagnostic charting
    ├── __init__.py
    └── risk_coverage.py        # Generates Risk-Coverage trade-off curves
```

---

## Installation & Usage

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

Import cleanly in any script or notebook:

```python
import vlm_research.config as cfg
from vlm_research.evaluation.scoring import compute_sequence_log_likelihood
from vlm_research.evaluation.selective_prediction import compute_discrete_aurc
```
