# Research Artifacts & Experiment Outputs (`artifacts/`)

This directory houses official evaluation metrics, per-sample prediction checkpoints, generated figures, audit logs, and model checkpoint state.

---

## Directory Organization

```text
artifacts/
├── README.md
├── experiments/
│   ├── README.md
│   ├── decisive_validation/
│   │   ├── rev05/                # OFFICIAL AUTHORITATIVE EXPERIMENTAL OUTPUTS
│   │   │   ├── metrics/          # Empirical metric summary CSVs (RQ1, RQ2, RQ3, Risk Ablation)
│   │   │   ├── predictions/      # Sample-level prediction CSV checkpoints (Qwen & Huatuo)
│   │   │   ├── figures/          # Official publication-grade Risk-Coverage PNG curves
│   │   │   ├── logs/             # Manifest audits, cohort audits, and zip packages
│   │   │   └── perturbed_images/ # Synthesized black blank images (P1 controls)
│   │   ├── rev06/                # Enhanced rev06 prediction checkpoints & metrics
│   │   └── historical/           # Pre-rev05 exploratory runs (rev02, rev03, rev04)
│   ├── legacy_screening/        # Initial model feasibility screening outputs
│   └── pilot_outputs/           # Pilot model screening metrics
├── figures/                      # Dataset & exploratory audit figures
└── checkpoints/
    └── README.md                 # Checkpoint governance and model weights storage guide
```

---

## Metric Preservation Guarantee

No experimental result or metric CSV in `rev05/` may be altered or overwritten. All numbers are cryptographically linked to the execution timestamp `2026-10-25T02:48:53 UTC` on Kaggle Dual Tesla T4 GPUs.
