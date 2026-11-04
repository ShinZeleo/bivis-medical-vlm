# Verification & Unit Test Suite (`tests/`)

This directory provides automated unit tests ensuring mathematical correctness, dataset hygiene, causal logit alignment, and zero patient data leakage.

---

## Test Inventory

| Test Module | Coverage & Verification Scope | Execution Time |
|---|---|:---:|
| [`test_decisive_validation.py`](test_decisive_validation.py) | **Master Preflight Suite (9/9 Tests)**: Manifest SHA-256, cohort invariants, pixel hash, logit indexing, discrete AURC, permutation testing, and atomic writing. | ~7.4s |
| [`test_scoring.py`](test_scoring.py) | Sequence log-likelihood calculation, token length normalization, and directional deletion sensitivity. | <0.1s |
| [`test_data_quality.py`](test_data_quality.py) | Invariants on 1,122 clean cohort, 2,244 raw rows, and canonical binary label space `{'YES', 'NO'}`. | <0.1s |
| [`test_bilingual_pairing.py`](test_bilingual_pairing.py) | Matched English-Indonesian question pairing and ground-truth label consistency. | <0.1s |
| [`test_leakage_audit.py`](test_leakage_audit.py) | Mathematical proof of $0.00\%$ cluster leakage between Dev (88 clusters) and Test (204 clusters). | <0.1s |
| [`test_statistical_analysis.py`](test_statistical_analysis.py) | Discrete Riemann sum AURC hand-calculation cases and boundary properties. | <0.1s |

---

## Running the Tests

Run all modular tests:
```bash
python -m unittest tests/test_scoring.py tests/test_data_quality.py tests/test_bilingual_pairing.py tests/test_leakage_audit.py tests/test_statistical_analysis.py
```

Run master preflight verification:
```bash
python tests/test_decisive_validation.py
```
*Expected Result: `ALL 9 PREFLIGHT TESTS PASSED (PASS_UNIT_TEST)`.*
