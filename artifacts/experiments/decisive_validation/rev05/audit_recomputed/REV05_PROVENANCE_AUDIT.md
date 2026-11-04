# Rev05 Provenance & Data Integrity Audit Report
**Generated:** 2026-10-31 14:08:47 UTC  
**Authoritative Manifest:** `data/manifests/manifest_rev05.csv`  
**Expected SHA-256:** `fb55fd75378b1c03abcbeb9772fa27d6ab81dd9abdfca4735c03438507c19ee9`

## 1. Cryptographic Manifest Verification
- **Observed Manifest SHA-256:** `fb55fd75378b1c03abcbeb9772fa27d6ab81dd9abdfca4735c03438507c19ee9`
- **Match Status:** `VERIFIED (PASS)`
- **Total Manifest Rows:** 2244
- **Unique QIDs:** 2244
- **Unique Physical Radiographs:** 314

## 2. Cohort Accounting & Sample Completeness
| Cohort Name | Target Rows | Observed in Qwen | Observed in Huatuo | Clusters | Overlap / Leakage | Status |
|---|---:|---:|---:|---:|---:|:---:|
| **Primary Cohort (Total)** | 1,122 | 1,122 | 1,122 | 292 | 0.00% | `PASS` |
| **Primary Test Split** | 799 | 799 | 799 | 204 | 0.00% | `PASS` |
| **Primary Dev Split** | 323 | 323 | 323 | 88 | 0.00% | `PASS` |
| **Historical Benchmark** | 150 | 150 | 150 | 58 | 0.00% | `PASS` |
| **Active Union Evaluated** | 1,133 | 1,133 | 1,133 | 292 | 0.00% | `PASS` |

## 3. Bilingual Semantic Alignment & Leakage Audit
- **QID Alignment Between Models:** Identical (True)
- **Extraneous QIDs Outside Manifest:** 0
- **Duplicate QIDs in Checkpoints:** 0
- **Pairing Mismatches against Manifest (Canonical Image & Answer):** 0 (0.00%)
- **Test-Dev Image Cluster Overlap:** 0 (Strict 0.00% patient leakage)

## 4. Execution Metadata
- **Scoring Method:** `exact_sequence_candidate_normalized_v5`
- **Revision Identifier:** `rev05`
- **Qwen Checkpoint Timestamp:** `2026-10-21T03:21:45.435934+00:00`
- **Huatuo Checkpoint Timestamp:** `2026-10-21T10:10:18.645977+00:00`
