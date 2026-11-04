# Statistical Validity & Resampling Audit Report
**Auditor:** Statistical Reproducibility Engineer  
**Core Unit of Analysis:** Physical Radiograph Cluster (`canonical_image_id`, N=204 test clusters)

## 1. Audit of Resampling & Testing Procedures
1. **Cluster Bootstrap (B=1,000 replicates):**
   - Resamples unique image clusters with replacement.
   - Evaluates discrete AURC on concatenated cluster blocks, preserving within-cluster correlation.
   - Yields valid 95% Percentile Confidence Intervals.
2. **Paired Cluster Permutation Test (B=1,000 replicates):**
   - Applies independent Rademacher sign flips $\epsilon_c \in \{-1, +1\}$ at the cluster level.
   - Exact two-sided Monte Carlo p-value: $p = \frac{1 + \sum I(|T_{perm}| \ge |T_{obs}|)}{B + 1}$.
   - **Distinction Noted:** With 204 clusters, exact enumeration requires $2^{204} \approx 2.57 \times 10^{61}$ evaluations. The protocol correctly uses a Monte Carlo approximation (B=1,000), which is computationally tractable and standard in high-dimensional bioinformatics.
3. **Multiple Testing Correction:**
   - Step-down Holm-Bonferroni correction strictly controls Family-Wise Error Rate (FWER) at $\alpha = 0.05$.

## 2. Key Statistical Test Outcomes
- **Heuristic Estimator ($M_{norm}$ vs $M_1$):**
  - Qwen: $p_{perm} = 0.2208$, $p_{holm} = 0.4056$ (Not significant)
  - Huatuo: $p_{perm} = 0.3906$, $p_{holm} = 0.7812$ (Not significant)
- **Learned Multivariate Risk ($LR_{full}$ vs $M_1$):**
  - Qwen: $\Delta \text{AURC} = -0.0369$, $p_{perm} = 0.0310$ (Statistically significant)
  - Huatuo: $\Delta \text{AURC} = -0.0705$, $p_{perm} = 0.0010$ (Highly statistically significant)
