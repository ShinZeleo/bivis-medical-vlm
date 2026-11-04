# Experiment Configurations (`configs/`)

This directory houses authoritative configuration files controlling model hyperparameters, inference decoding, dataset paths, and evaluation metrics for the Medical VLM research pipeline.

---

## Files

| File | Purpose | Key Parameters |
|---|---|---|
| [`evaluation_config.yaml`](evaluation_config.yaml) | Primary configuration for model screening, evaluation, and selective prediction. | `models`, `dataset`, `decoding` (temperature=0.0, max_new_tokens=16), `perturbation` (black_blank). |

---

## Usage in Code

Configurations are loaded deterministically via `vlm_research.config` or PyYAML:

```python
import yaml
from pathlib import Path

config_path = Path("configs/evaluation_config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
```
