# DATA PROCESSING & COHORT SPLITTING PROTOCOL
## Medical Vision-Language Models | IndoRad-VQA Dataset Governance

**Protocol Identifier:** `DPROT-INDORAD-REV05-V10`  
**Dataset Source:** `Lab-IS/IndoRad-VQA` (Hugging Face Datasets)  
**Primary Manifest:** `data/manifests/indorad_vqa_final_manifest.csv` (SHA-256: `fb55fd75378b1c03abcbeb9772fa27d6ab81dd9abdfca4735c03438507c19ee9`)

---

## 1. Raw Data Extraction & Cryptographic Auditing
1. The raw dataset contains **2,244 question-answer instances** paired across 314 physical radiology images.
2. Every image was extracted to `data/raw/indorad_vqa/images/` and verified against a pre-computed RGB pixel SHA-256 hash.
3. Tamper rejection policy: Any corrupted or missing image initiates self-healing re-download from authoritative Parquet shards.

---

## 2. Cohort Filtering & Hygiene Invariants
To ensure scientific validity and eliminate translation artifacts, raw instances are systematically partitioned:

```
[ Raw IndoRad-VQA: 2,244 rows ]
       │
       ├──► [ Open-Ended Questions: 1,053 rows ] ──► EXCLUDED (Grammar / formatting non-comparability)
       │
       └──► [ Yes/No Binary Questions: 1,191 rows ]
                 │
                 ├──► [ Translation / Logic Corruptions: 69 rows ] ──► EXCLUDED
                 │      (e.g., QID 144, 219, 444 where Indonesian inverted English logic)
                 │
                 └──► [ PRIMARY CLEAN COHORT: 1,122 rows ] ──► INCLUDED
```

---

## 3. Group-Aware Cluster Splitting & Zero-Leakage Guarantee
To prevent optimistic bias from patient/image correlation across partitions:
1. **Splitting Criterion:** Partitioning is performed strictly by image cluster prefix (`image_id`). All QA pairs associated with a given radiograph reside in the same split.
2. **Cluster Distribution:**
   - **Development Split (Dev):** 323 QA pairs from **88 unique image clusters**.
   - **Test Split (Test):** 799 QA pairs from **204 unique image clusters**.
   - **Total:** 1,122 QA pairs from **292 unique image clusters** (22 images exclusively associated with open-ended questions).
3. **Data Leakage Proof:**
   $$\mathcal{C}_{\text{dev}} \cap \mathcal{C}_{\text{test}} = \emptyset$$
   Deterministic assertion verifies zero cluster overlap ($0.00\%$ leakage).
