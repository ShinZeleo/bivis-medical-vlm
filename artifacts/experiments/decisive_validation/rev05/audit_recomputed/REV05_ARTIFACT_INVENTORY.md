# Rev05 Experiment Artifact Inventory Report
**Generated:** 2026-10-31 14:08:47 UTC  
**Target Directory:** `artifacts/experiments/decisive_validation/rev05/`  
**Auditor:** Senior ML Research Scientist & Statistical Auditor

## 1. Directory Structure Overview
```text
rev05/
├── figures/              # Publication Risk-Coverage curve plots
├── logs/                 # Execution metadata, cohort manifests & complete zip archives
├── metrics/              # Canonical aggregated CSV tables (RQ1, RQ2, RQ3, Ablation)
├── perturbed_images/     # P1 visual deletion control images (black canvas)
└── predictions/          # Per-sample live validation checkpoints for Qwen and Huatuo
```

## 2. Complete Inventory Table
| File Name | Size (Bytes) | Rows | Cols | Unique QIDs | Duplicates | Model | Split | Function |
|---|---:|---:|---:|---:|---:|---|---|---|
| `risk_coverage_primary_test_HuatuoGPT_Vision_7B.png` | 170,469 | -1 | -1 | -1 | -1 | HuatuoGPT-Vision-7B-Qwen2.5VL | N/A | Publication Risk-Coverage curve figure |
| `risk_coverage_primary_test_Qwen2.5_VL_7B.png` | 178,899 | -1 | -1 | -1 | -1 | Qwen2.5-VL-7B-Instruct | N/A | Publication Risk-Coverage curve figure |
| `cohort_audit.json` | 1,314 | 6 | -1 | -1 | -1 | N/A | N/A | Experimental provenance log or zip archive |
| `decisive_validation_rev05_complete.zip` | 1,211,279 | -1 | -1 | -1 | -1 | N/A | N/A | Experimental provenance log or zip archive |
| `manifest_audit.json` | 1,257 | 13 | -1 | -1 | -1 | N/A | N/A | Experimental provenance log or zip archive |
| `run_metadata.json` | 384 | 9 | -1 | -1 | -1 | N/A | N/A | Experimental provenance log or zip archive |
| `scoring_validation_report.json` | 738 | 8 | -1 | -1 | -1 | N/A | N/A | Experimental provenance log or zip archive |
| `adaptive_abstention_coverage_targeted.csv` | 6,990 | 40 | 12 | -1 | -1 | N/A | N/A | Adaptive abstention risk-coverage metrics |
| `adaptive_abstention_risk_targeted.csv` | 4,514 | 24 | 11 | -1 | -1 | N/A | N/A | Adaptive abstention risk-coverage metrics |
| `artifact_sha256_manifest.csv` | 42,042 | 306 | 3 | -1 | -1 | N/A | N/A | Authoritative manifest CSV copy |
| `manifest_rev05.csv` | 830,986 | 2244 | 25 | 2244 | 0 | N/A | test: 1600, dev: 644 | Authoritative manifest CSV copy |
| `risk_ablation_dev_test.csv` | 800 | 6 | 7 | -1 | -1 | N/A | N/A | Risk model ablation (LR-conf, LR-visual, LR-full) dev/test metrics |
| `rq1_bilingual_metrics.csv` | 1,568 | 8 | 11 | -1 | -1 | N/A | primary_test, primary_dev, historical_test, historical_dev | RQ1 Bilingual accuracy and consistency metrics |
| `rq2_visual_intervention_metrics.csv` | 557 | 4 | 7 | -1 | -1 | N/A | primary_test, historical_test | RQ2 Visual intervention sensitivity metrics |
| `rq3_selective_prediction_primary_test.csv` | 3,320 | 10 | 19 | -1 | -1 | N/A | primary_test | RQ3 Selective prediction discrete AURC metrics |
| `threshold_calibration_oof_vs_insample.csv` | 6,688 | 40 | 10 | -1 | -1 | N/A | N/A | Threshold calibration out-of-fold vs in-sample metrics |
| `blank_01cab8417235629a.png` | 3,106 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_034724ce4122a47f.png` | 2,586 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_035ff09f03f886b6.png` | 1,113 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_0498dd002989665b.png` | 369 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_0514608f9e6e7000.png` | 1,991 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_0582b4370828cb25.png` | 9,432 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_0593d2ae97d9560f.png` | 2,477 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_061c20fde1826f38.png` | 3,785 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_092b83c85b05653a.png` | 2,190 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_0a0c8bc50b777f08.png` | 653 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_0a3940b8bb94c6d9.png` | 663 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_0a8a6fb00ebfc170.png` | 2,263 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_0c261f82969c18af.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_0e1fb39d30ee436a.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_117904b339c8f6d8.png` | 1,305 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_118e62a88fa61563.png` | 1,113 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_1192356cb996ced2.png` | 2,104 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_11ccc1a1c4e9c8e1.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_12721e287328992a.png` | 333 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_156e70cdf62dad3f.png` | 3,129 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_15a8e1b5ba7a12a7.png` | 2,078 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_173c2d5e7bc14105.png` | 2,173 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_182bf4e839897571.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_18a298f3ebc38978.png` | 1,739 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_193b3f13df9569c5.png` | 320 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_196c0e408dd759ed.png` | 3,533 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_19b41b715a9eb8d2.png` | 2,581 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_1a709231e743c140.png` | 753 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_1b372d082fa2bf05.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_1c6ca2358a3c9b8f.png` | 452 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_1de8239dee07c864.png` | 442 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_1de91a8afc488dfd.png` | 678 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_1e17148bde399c47.png` | 1,969 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_20e826c6d265f2f3.png` | 3,981 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_2120bc1bee31e0ee.png` | 1,061 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_216499bc573ff0f8.png` | 2,670 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_21f2c282d1c6c3d2.png` | 3,936 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_22a0905ba66bd3df.png` | 3,010 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_22b255ea1e3082b8.png` | 1,201 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_22bc1f61ac0299bc.png` | 3,347 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_23fd619fc0ab0d5f.png` | 2,003 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_2778908a018d11af.png` | 522 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_2a3411316757eabd.png` | 2,586 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_2a75a4c01a0a151a.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_2a7f4b068d830fdf.png` | 2,089 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_2b3b0a462e7f7846.png` | 3,122 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_2bb923ed81f118bc.png` | 4,546 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_2d174e34e8c6e2cb.png` | 1,175 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_2d253335a4a09d1e.png` | 3,370 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_2e4a3e79f3c9c492.png` | 1,170 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_31054e4f1069a083.png` | 3,091 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_327b2eb68696afba.png` | 1,428 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_33302c5cb6079b97.png` | 537 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_3355a3a6ba047095.png` | 1,664 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_35b87488ba222c67.png` | 666 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_36a57b72fb71e155.png` | 3,981 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_37042939fc6d2c42.png` | 2,089 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_3707f2cb32a52b4a.png` | 903 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_376a2117295e209a.png` | 3,129 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_3851ff33c808311f.png` | 1,333 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_386078f57d667d2a.png` | 3,629 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_38906bd7e0f21205.png` | 2,030 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_3a18a99e0453cf1f.png` | 408 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_3a659b101a868b6c.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_3a9ec85ea41bb7a9.png` | 3,091 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_3ab0d408e3b97f07.png` | 2,643 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_3afbd94bdafd87ad.png` | 270 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_3b580bfc2de2cee3.png` | 587 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_3d985f2e34b07927.png` | 2,582 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_3f95525dc82c3d9f.png` | 1,111 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_405fb9ae19b4f433.png` | 3,091 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_41475ed2f4214638.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_426b94bb9fac4514.png` | 3,923 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_427fb08d4de13e36.png` | 2,507 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_43033f928744d7be.png` | 2,199 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_430cc6825227e77e.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_4312b741c63c4ae5.png` | 2,586 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_4369fde227b088fc.png` | 735 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_441fce090bd1a461.png` | 578 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_45026f888cf64c28.png` | 1,507 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_45706dd79a6850da.png` | 749 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_46469d88645efd27.png` | 483 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_4661ca5c93547b39.png` | 2,171 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_4846b9987b918068.png` | 1,138 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_49ad7f6a3ad10fe0.png` | 1,678 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_4a8dc4a914454dc4.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_4afdc25a2b573808.png` | 2,009 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_4ba83d5dfd9e11c8.png` | 846 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_4c57e1114bafee13.png` | 839 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_4cee3edfece35117.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_4ed1056e8a067ab6.png` | 2,358 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_4f59c29c417f6775.png` | 2,108 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_51f2d3a02b9252fa.png` | 3,909 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_5277876890c6e739.png` | 577 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_5319c65c85ef47b4.png` | 3,325 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_54c13dcd1c40df38.png` | 3,091 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_563e2995a5395c4a.png` | 3,322 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_565c19d53d91ec55.png` | 3,126 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_56f96a6732f48dd3.png` | 780 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_58e9cbee072d6211.png` | 821 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_596608b93e1897f8.png` | 3,579 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_59d1e27c08cb0c4c.png` | 3,923 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_5a147c7e2caac0d4.png` | 2,366 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_5a2a23e80fea9c69.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_5a8cd8ead48c927e.png` | 3,981 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_5c43818666220941.png` | 2,581 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_5cab8556571f9cec.png` | 3,480 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_5cb1a92c2e23ff51.png` | 2,357 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_5cd5c0aa652e9e6d.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_5d2123e3352d7473.png` | 2,586 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_5e4529cf63780347.png` | 1,198 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_5e7613408c89dcc8.png` | 686 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_5f88e5d493e86c90.png` | 613 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_603c42aed8cb15ea.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_6185dbde43bcbc0c.png` | 1,106 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_624d70cb6c041476.png` | 2,000 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_62d0581e84a18ff8.png` | 3,963 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_645b146b89ec7c36.png` | 2,789 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_665aedcbb0078cea.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_688c88304e262e83.png` | 2,209 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_6bb258db240d21f4.png` | 383 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_6c274b01b297538f.png` | 3,963 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_6c33577f8c7c9405.png` | 810 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_705a36d427070834.png` | 3,507 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_70eac2ce8a8d4827.png` | 3,972 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_716fb64d422d5a44.png` | 2,582 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_71c9710228ef7945.png` | 767 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_72f34f31d279634f.png` | 3,972 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_74d9dd480e24a4f9.png` | 3,296 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_751289a09c00add4.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_75492b9c3aebd522.png` | 2,337 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_75a1bda6498dea76.png` | 3,909 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_7921c99a04532c3c.png` | 3,981 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_79b3831e8502fa50.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_79da4748a46a8361.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_79df1c3988550669.png` | 747 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_79f0f61566fcc9d0.png` | 457 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_7a82c452b9814c08.png` | 568 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_7aa329812ae2d419.png` | 3,524 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_7b0ea63b06b346e8.png` | 3,531 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_7cbb62b5b83aefdb.png` | 2,586 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_7f2cff7f96ee7eb2.png` | 1,786 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_80652d54386512c2.png` | 2,548 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_80f6253a3601d41b.png` | 604 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_8131f15dcd5fcc1a.png` | 3,992 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_82183b94c629cf0b.png` | 993 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_83a233637ae80dfd.png` | 3,224 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_843a5901ec66f18a.png` | 2,882 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_85af7d54e15ec769.png` | 1,209 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_8811d450645c8661.png` | 2,581 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_88d855043eb37cfd.png` | 495 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_89bb931e6b96435e.png` | 3,981 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_8ae34cd40ea684e0.png` | 494 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_8c083cf36cbe6b6c.png` | 811 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_8d80b4ee1f10898f.png` | 2,462 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_8f30e0f45766a570.png` | 3,785 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_8fefb9f43946c1dd.png` | 3,972 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_91aeab7f47f961bd.png` | 1,516 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_95c0a2588c2c1720.png` | 3,429 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_972fb2d384932a95.png` | 3,909 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_9786c5daf749ffa4.png` | 3,215 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_983899416845dc49.png` | 692 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_98d2dfd13d77fb69.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_9aacbe88842b6e91.png` | 704 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_9c4b2a92a3722177.png` | 3,370 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_9c4e2bade8cf39ed.png` | 3,520 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_9c8a4e1d9eab4e12.png` | 1,991 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_a008ffb14e255135.png` | 705 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_a0bcd21fd48d5a67.png` | 630 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_a14d5556b81f098f.png` | 3,081 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_a14f4274867e3a1f.png` | 1,765 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_a22067e7d274b9d5.png` | 464 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_a23e575c0767232d.png` | 358 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_a3580fb18a26cafc.png` | 2,599 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_a41e50707238f077.png` | 795 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_a42a8c6711eb659e.png` | 845 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_a4e5284229be0a8a.png` | 2,586 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_a731e56521f29970.png` | 2,340 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_aad3cf5049c8f196.png` | 2,679 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ab987c216de9f349.png` | 3,093 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ac795f6b8263336f.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_adfc7662bd7f90d5.png` | 505 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_afa786fd709360b9.png` | 3,525 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_b04cfe70c6e92d64.png` | 1,991 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_b0bbd8a3c003b02d.png` | 2,622 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_b50018632edb873e.png` | 3,409 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_b5384d142c0389e5.png` | 3,984 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_b6306d5111af5517.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_b72eddc30fd610e0.png` | 1,286 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_b73e860d1a7ac32b.png` | 2,120 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_b85d24575c6f4850.png` | 1,781 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_b8724267bf34f369.png` | 2,950 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_b92531bd0765dcdb.png` | 3,387 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_b95429055a2d801b.png` | 1,738 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_b96871e48bbbf9aa.png` | 1,175 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_b973d542fc5ecfd3.png` | 3,601 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_babc55f82833e715.png` | 2,586 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_bb452f9735b46484.png` | 2,089 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_bb86ec661078102b.png` | 1,486 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_bb938b295669e4f8.png` | 3,972 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_be6bc466225b947f.png` | 3,126 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_c04e33f8d4d9d149.png` | 3,507 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_c0909bf983e2006f.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_c1739ef249efe239.png` | 1,991 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_c211b2ecf3ca596d.png` | 3,963 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_c24ba5d8d1e131fe.png` | 2,586 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_c30cee74d85f9a0b.png` | 1,738 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_c3ed818440183e54.png` | 3,608 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_c5653fe3731160e5.png` | 2,611 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_c7b050874fa72577.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ca6fd888155f1564.png` | 2,742 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_cb86d4e4f7b9970c.png` | 3,521 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_cbed24e5374f203f.png` | 1,810 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_cd63d9cf5611c456.png` | 2,586 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_cd7f1199d93c414e.png` | 2,498 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_cda2d6fcf610e1b3.png` | 566 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_cddfef1d3e2bac0c.png` | 1,177 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_cee723c403e2eddf.png` | 1,698 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_cf03f46527f9bf8a.png` | 1,195 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_cf08178fa875bcde.png` | 1,050 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_d07a33ad418727cb.png` | 3,981 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_d09a7181111eeb87.png` | 2,263 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_d1bbfdd364ab073f.png` | 648 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_d1e58dcf8ee2808a.png` | 3,564 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_d2485cc9cf40a7d8.png` | 3,978 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_d32eaf48b9c01d09.png` | 3,923 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_d3b38c459ad07c99.png` | 2,027 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_d42e381f40647f8b.png` | 1,197 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_d800890884faa61a.png` | 2,258 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_d803ef582158b46e.png` | 1,239 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_d901e1db019020a8.png` | 3,796 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_d932b1e1088ae5b8.png` | 541 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_d93dc938281bf8c5.png` | 328 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_d96070b1887d82b3.png` | 3,524 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_da88d274978bbe25.png` | 3,909 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_dadbc7af5d9a2282.png` | 2,033 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_db5705fd8ae7dc5c.png` | 2,021 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_dc2770cd71db5e63.png` | 2,770 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_dd34ffa2bcb93e02.png` | 3,429 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_dd3bee5772c32966.png` | 3,122 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ddc2d27738497992.png` | 3,806 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ddfccb6a55b87470.png` | 599 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_de0a2f5c3e567e9f.png` | 1,646 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_df1ad5b92a4d0ffd.png` | 989 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_dfe6cfdce70f335a.png` | 3,126 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_e0062eb9000ce880.png` | 651 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_e045dcb25d05f5b0.png` | 1,459 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_e0792a4f2379590e.png` | 493 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_e0f3540c5bde89b1.png` | 1,046 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_e3cc005d6c19e23b.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_e4449c803cacd361.png` | 1,821 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_e44ad4bc94ed9c5e.png` | 2,107 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_e4c0a068e28b90e1.png` | 3,614 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_e4f9728345e0fadf.png` | 1,476 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_e5d839dc899857eb.png` | 1,048 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_e5e98ba57fd08f97.png` | 1,989 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_e687a105d2b71deb.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_e6c4e28463b52b6d.png` | 2,322 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_e824dd7fbe445f53.png` | 1,821 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ea70fc729d028c78.png` | 2,089 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_eabe10c4fbdb41b7.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_eb555118c5361ec9.png` | 2,071 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_eb6ff30acd1144e6.png` | 3,785 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ebab42ecbd92c183.png` | 865 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ebbcced9f6713610.png` | 1,676 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ec74b4c03fee08ec.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ed17c1158dba4913.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ed32cc8ecad538c3.png` | 1,116 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ed719fb6c977025f.png` | 2,000 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ee9d1fc3a1fcf7b8.png` | 3,909 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ef43da43bce737ed.png` | 3,890 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_f16cd18f893395ea.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_f20f5c04df12ff17.png` | 2,027 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_f291323136d71a90.png` | 1,147 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_f29fb2a31f4920ba.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_f374db39e2383a23.png` | 1,795 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_f3a40338c64a12f3.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_f4325a972302f8d7.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_f4eb47c01c4397c1.png` | 1,630 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_f59665a5f5a70dd9.png` | 3,632 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_f5d8016cff64a995.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_f6dc88fb2261ad08.png` | 1,963 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_f731c8d56baedfdd.png` | 1,224 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_f75fa2c19fecc1db.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_f8c39630ab19b80a.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_fc51213655cd7776.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_fc5eb4fc76acd7d4.png` | 588 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_fc67c911edceea22.png` | 842 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_fd3aebb2147d537c.png` | 589 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_fe5e296954436462.png` | 782 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_fec3d1c2c7ca7420.png` | 1,268 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `blank_ff7d6d76ae06e428.png` | 3,147 | -1 | -1 | -1 | -1 | N/A | N/A | Black blank perturbation canvas (P1 control image) |
| `live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv` | 1,104,185 | 1133 | 62 | 1133 | 0 | HuatuoGPT-Vision-7B-Qwen2.5VL | test: 806, dev: 327 | Prediction checkpoint: per-sample sequence log-likelihood & intervention scores |
| `live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv` | 1,115,745 | 1133 | 62 | 1133 | 0 | Qwen2.5-VL-7B-Instruct | test: 806, dev: 327 | Prediction checkpoint: per-sample sequence log-likelihood & intervention scores |

## 3. Inventory Integrity Summary
- **Total Files Inventoried:** 310
- **Prediction Checkpoints Present:** 2 (Qwen 7B & Huatuo 7B)
- **Perturbed Control Images Present:** 292 black blank PNG images
- **Metric Tables Present:** 8 CSV tables (including all RQ1, RQ2, RQ3, and ablation outputs)
- **Missing Core Artifacts:** 0 (Status: `COMPLETE`)
