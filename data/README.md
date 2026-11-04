# Dataset Governance & Manifests (`data/`)

This directory houses raw medical images, cohort manifests, and preprocessed tabular summaries for the `IndoRad-VQA` dataset.

---

## Directory Organization

```text
data/
├── README.md
├── raw/
│   └── indorad_vqa/
│       └── images/               # 314 raw medical radiograph scans (.png)
├── manifests/
│   ├── indorad_vqa_final_manifest.csv  # Authoritative primary clean cohort (1,122 clean QA pairs)
│   ├── manifest_rev05.csv              # Frozen manifest for rev05 experiments (SHA-256: fb55fd75...)
│   ├── full_bilingual_manifest.csv     # Complete bilingual dataset manifest
│   └── recommended_image_split.csv     # Patient-aware group partition mappings
├── processed/
│   ├── schema_data_dictionary.csv      # Detailed schema dictionary
│   └── question_answer_distribution.csv# Class balance and question type distributions
└── archive/
    └── indorad-vqa-research.zip        # Cold storage backup of original dataset
```

---

## Data Governance & Zero-Leakage Policy

1. **Immutable Raw Data:** Files in `data/raw/` must NEVER be overwritten or manually modified.
2. **Cryptographic Validation:** Images are validated against pixel SHA-256 hashes during every preflight test.
3. **Partition Isolation:** The 88 development image clusters and 204 test image clusters have zero overlap ($\mathcal{C}_{\text{dev}} \cap \mathcal{C}_{\text{test}} = \emptyset$), preventing data leakage during risk model training and evaluation.
