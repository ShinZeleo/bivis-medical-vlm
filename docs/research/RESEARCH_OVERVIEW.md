# RESEARCH OVERVIEW & THEORETICAL FOUNDATIONS
## Visual Sensitivity–Aware Selective Prediction for Bilingual Medical VLMs

### 1. Research Background & Clinical Motivation
In developing countries such as Indonesia, automated clinical decision support tools powered by Medical Vision-Language Models (Med-VLMs) must reliably answer radiological visual queries in the national language (Bahasa Indonesia). However, state-of-the-art vision-language foundation models are predominantly pretrained on English-centric biomedical literature (e.g., PubMed, MIMIC-CXR, PMC-OA).

When evaluated on non-English queries, models exhibit two critical failure modes:
1. **Cross-Lingual Representation Drift (Flip Errors):** The model produces contradictory diagnostic classifications ($\hat{y}_{\text{ID}} \neq \hat{y}_{\text{EN}}$) for the exact same patient and radiographic scan merely due to query language translation.
2. **Visual Ungroundedness & Shortcut Learning (*Dangerous Quadrant*):** The model produces highly confident diagnostic answers based solely on memorized language priors without grounding its predictions in the visual pixel evidence (Sadanandan et al., CVPR 2026).

Fine-tuning foundation models on small local clinical datasets introduces a high risk of multimodal representation degradation and catastrophic mode collapse (Sadanandan & Behzadan, CHIL 2026, PMLR 333:703–720). Consequently, this research investigates an **inference-time selective prediction framework** that enables the model to abstain and refer uncertain cases to human expert radiologists.

---

### 2. Research Questions (RQs)
- **RQ1 (Error Correlation):** Does cross-lingual prediction disagreement ($D_{\text{lang}} = \mathbb{I}(\hat{y}_{\text{EN}} \neq \hat{y}_{\text{ID}})$) provide a statistically significant signal for predicting diagnostic errors in Indonesian radiology queries?
- **RQ2 (Heuristic Formulation):** Can unparameterized multiplicative fusion of raw confidence and directional visual deletion sensitivity ($M_{\text{norm}}$) consistently outperform baseline confidence ($M_1$)?
- **RQ3 (Parametric Risk Modeling):** Does out-of-sample regularized risk modeling ($LR_{\text{full}}$) combining confidence, directional sensitivity, and cross-lingual disagreement achieve significant reductions in Area Under the Risk-Coverage curve (AURC)?
- **RQ4 (Model Architectures):** How does domain-specialized pretraining (HuatuoGPT-Vision-7B) compare to generalist foundation models (Qwen2.5-VL-7B) in cross-lingual stability and selective risk ranking?
- **RQ5 (Zero-Leakage Generalization):** Can selective risk models trained on one set of hospital clusters generalize to completely unseen image clusters without patient data leakage?

---

### 3. Empirical Verdict Summary
1. **RQ1:** Strongly Verified ($OR > 2.7\times$, Fisher's $p < 10^{-6}$). Cross-lingual disagreement is an effective error surrogate.
2. **RQ2:** Negative Finding Verified ($M_{\text{norm}}$ degrades on Qwen and is statistically non-significant on Huatuo with $p_{\text{adj}} = 0.7812$). Heuristic weighting fails without parameter learning.
3. **RQ3:** Strongly Verified ($LR_{\text{full}}$ significantly reduces AURC on both models, $p = 0.0120$ for Qwen and $p = 0.0005$ for Huatuo).
4. **RQ4:** Strongly Verified (Huatuo specialist maintains $+2.63\%$ Indonesian accuracy and $+7.38\%$ English accuracy over Qwen generalist).
5. **RQ5:** Strongly Verified ($0.00\%$ image cluster leakage across 88 dev clusters and 204 test clusters).
