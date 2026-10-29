# Strong-Model Validation Report
**Reference:** `AGENT_REVISED_COLLAB.md`
**Model Under Evaluation:** `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL`
**Dataset:** `Lab-IS/IndoRad-VQA`
**Sample Size:** 150 validated bilingual pairs

## 1. Summary of Empirical Findings
- **English Accuracy (Acc_EN):** 76.00%
- **Indonesian Accuracy (Acc_ID):** 70.00%
- **Language Gap:** 6.00 percentage points
- **Cross-Lingual Flip Rate:** 21.33%
- **Joint Correctness:** 62.67%

## 2. Confidence & Calibration Metrics
- **Expected Calibration Error (ECE):** EN = 0.0325, ID = 0.2350
- **Brier Score:** EN = 0.1534, ID = 0.2713
- **Confidently Wrong Rate (>=0.90):** EN = 0.67%, ID = 24.67%

## 3. Counterfactual Visual Sensitivity
- **Visual Flip Rate (P0 Original vs P1 Blank):** 29.33%
- **Mean Delta P (|P_orig - P_blank|):** 0.1594

## 4. Mandatory Feasibility Answer (RQ0)
> **"Does the proposed problem remain observable in a strong-capability VLM?"**
> **YES.** The empirical evidence demonstrates that language shift and reliability misalignments persist even in a high-capacity VLM.

## 5. Formal Experimental Decision
### **DECISION: GO**
**Rationale:**
- Non-trivial cross-lingual flip rate observed (21.3%).
- Poor calibration / substantial ECE observed (0.235).
- High rate of confidently wrong answers in Indonesian (24.7%).

**Next Step:** Proceed to the Main Experiment (Selective Prediction & Factorial Intervention).