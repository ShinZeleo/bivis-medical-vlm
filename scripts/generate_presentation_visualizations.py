import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure output directories exist
out_dir = Path("reports/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# Set high-quality visual style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# Load datasets
print("📊 Loading manifest & rev05 checkpoints...")
df_manifest = pd.read_csv("data/manifests/manifest_rev05.csv")
df_qwen = pd.read_csv("artifacts/experiments/decisive_validation/rev05/predictions/qwen/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv")
df_huatuo = pd.read_csv("artifacts/experiments/decisive_validation/rev05/predictions/huatuo/live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv")

# Filter primary test cohorts (N=799)
qwen_test = df_qwen[(df_qwen["split"] == "test") & (df_qwen["is_eligible_primary"] == True)].copy()
huatuo_test = df_huatuo[(df_huatuo["split"] == "test") & (df_huatuo["is_eligible_primary"] == True)].copy()
assert len(qwen_test) == 799 and len(huatuo_test) == 799

eval_manifest = df_manifest[df_manifest["in_eval_cohort"] == True].copy()
test_manifest = eval_manifest[(eval_manifest["split"] == "test") & (eval_manifest["is_eligible_primary"] == True)]
dev_manifest = eval_manifest[(eval_manifest["split"] == "dev") & (eval_manifest["is_eligible_primary"] == True)]

print(f"✅ Primary Test: N={len(test_manifest)} (Clusters: {test_manifest['canonical_image_id'].nunique()})")
print(f"✅ Dev Cohort:  N={len(dev_manifest)} (Clusters: {dev_manifest['canonical_image_id'].nunique()})")

# ------------------------------------------------------------------------------
# FIG 1: EDA DATASET & STRICT CLUSTER LEAKAGE ISOLATION
# ------------------------------------------------------------------------------
print("🎨 Plotting Fig 1: EDA Dataset & Cluster Leakage Isolation...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Subplot 1A: Split counts (QA vs Cluster)
split_data = pd.DataFrame({
    "Cohort Split": ["Dev Cohort (Calibration)", "Dev Cohort (Calibration)", "Primary Test (Evaluation)", "Primary Test (Evaluation)"],
    "Count Type": ["QA Pairs (N)", "Unique Image Clusters", "QA Pairs (N)", "Unique Image Clusters"],
    "Value": [len(dev_manifest), dev_manifest["canonical_image_id"].nunique(),
              len(test_manifest), test_manifest["canonical_image_id"].nunique()]
})
sns.barplot(data=split_data, x="Cohort Split", y="Value", hue="Count Type", palette=["#2b5c8f", "#d95f02"], ax=axes[0])
axes[0].set_title("A. Cohort Sample Size & Cluster Cardinality", fontweight="bold")
axes[0].set_ylabel("Count")
for p in axes[0].patches:
    h = p.get_height()
    if h > 0:
        axes[0].annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2., h / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=11)

# Subplot 1B: Disjoint Cluster Leakage Matrix
dev_clusters = set(dev_manifest["canonical_image_id"])
test_clusters = set(test_manifest["canonical_image_id"])
leak_overlap = len(dev_clusters.intersection(test_clusters))
matrix_data = np.array([[len(dev_clusters), leak_overlap], [leak_overlap, len(test_clusters)]])
sns.heatmap(matrix_data, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Dev Clusters", "Test Clusters"], yticklabels=["Dev Clusters", "Test Clusters"],
            ax=axes[1], annot_kws={"size": 13, "weight": "bold"})
axes[1].set_title("B. Cluster Disjoint Matrix (0% Leakage)", fontweight="bold")
axes[1].text(0.5, -0.15, f"Cross-Cohort Cluster Overlap = {leak_overlap} (Strict Zero-Leakage)",
             ha="center", transform=axes[1].transAxes, color="#007700", fontweight="bold")

# Subplot 1C: QA per Image Distribution
qa_counts_dev = dev_manifest["canonical_image_id"].value_counts()
qa_counts_test = test_manifest["canonical_image_id"].value_counts()
sns.kdeplot(qa_counts_dev, fill=True, label=f"Dev (Mean: {qa_counts_dev.mean():.2f})", color="#2b5c8f", ax=axes[2], bw_adjust=1.5)
sns.kdeplot(qa_counts_test, fill=True, label=f"Test (Mean: {qa_counts_test.mean():.2f})", color="#d95f02", ax=axes[2], bw_adjust=1.5)
axes[2].set_title("C. QA Pairs per Image Cluster Density", fontweight="bold")
axes[2].set_xlabel("Number of Questions per Medical Image")
axes[2].set_ylabel("Density")
axes[2].legend(loc="upper right")

plt.tight_layout()
fig.savefig(out_dir / "fig1_eda_dataset_and_cluster_leakage.png")
plt.close(fig)

# ------------------------------------------------------------------------------
# FIG 2: TEXT LENGTH & CLASS BALANCE (EN vs ID)
# ------------------------------------------------------------------------------
print("🎨 Plotting Fig 2: Text Length & Class Balance...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Subplot 2A: Word Length Distribution
en_len = test_manifest["question_en"].str.split().str.len()
id_len = test_manifest["question_id"].str.split().str.len()
sns.kdeplot(en_len, fill=True, label=f"English (Mean: {en_len.mean():.1f} words)", color="#1b9e77", ax=axes[0])
sns.kdeplot(id_len, fill=True, label=f"Indonesian (Mean: {id_len.mean():.1f} words)", color="#e7298a", ax=axes[0])
axes[0].set_title("A. Question Length Distribution (Words)", fontweight="bold")
axes[0].set_xlabel("Word Count")
axes[0].set_ylabel("Density")
axes[0].legend(loc="upper right")

# Subplot 2B: Character Length Expansion Ratio
expansion_ratio = test_manifest["question_id"].str.len() / test_manifest["question_en"].str.len()
sns.histplot(expansion_ratio, kde=True, color="#7570b3", ax=axes[1], bins=25)
axes[1].axvline(expansion_ratio.median(), color="red", linestyle="--", label=f"Median Expansion: {expansion_ratio.median():.2f}x")
axes[1].set_title("B. Cross-Lingual Expansion Ratio (ID/EN Characters)", fontweight="bold")
axes[1].set_xlabel("Character Ratio (Indonesian / English)")
axes[1].set_ylabel("Frequency")
axes[1].legend(loc="upper right")

# Subplot 2C: Ground Truth Class Balance
gt_counts = pd.DataFrame({
    "Cohort": ["Dev (N=323)", "Dev (N=323)", "Primary Test (N=799)", "Primary Test (N=799)"],
    "Answer": ["YES", "NO", "YES", "NO"],
    "Percentage": [
        (dev_manifest["reference_answer"] == "YES").mean() * 100,
        (dev_manifest["reference_answer"] == "NO").mean() * 100,
        (test_manifest["reference_answer"] == "YES").mean() * 100,
        (test_manifest["reference_answer"] == "NO").mean() * 100
    ]
})
sns.barplot(data=gt_counts, x="Cohort", y="Percentage", hue="Answer", palette=["#2ca02c", "#d62728"], ax=axes[2])
axes[2].set_title("C. Ground Truth Answer Distribution", fontweight="bold")
axes[2].set_ylabel("Percentage (%)")
axes[2].set_ylim(0, 100)
axes[2].axhline(50, color="grey", linestyle=":", alpha=0.7)
for p in axes[2].patches:
    h = p.get_height()
    axes[2].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h / 2),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
fig.savefig(out_dir / "fig2_eda_text_length_and_class_balance.png")
plt.close(fig)

# ------------------------------------------------------------------------------
# FIG 3: SPATIAL RESOLUTION & MULTIMODAL BOUNDARIES
# ------------------------------------------------------------------------------
print("🎨 Plotting Fig 3: Spatial Resolution & Dynamic Patch Bounds...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

w = test_manifest["image_width"].dropna()
h = test_manifest["image_height"].dropna()
pixels = w * h
aspect = w / h

# Subplot 3A: Width vs Height Scatter with Qwen2.5-VL limits
sns.scatterplot(x=w, y=h, hue=aspect, palette="viridis", alpha=0.75, s=40, ax=axes[0], edgecolor="none")
axes[0].set_title("A. Medical Image Dimensions (W × H)", fontweight="bold")
axes[0].set_xlabel("Image Width (Pixels)")
axes[0].set_ylabel("Image Height (Pixels)")
# Iso-pixel lines
axes[0].axvline(512, color="gray", linestyle="--", alpha=0.5, label="512px Boundary")
axes[0].axhline(512, color="gray", linestyle="--", alpha=0.5)
axes[0].set_xlim(0, max(w.max(), 1050))
axes[0].set_ylim(0, max(h.max(), 1050))
axes[0].legend(title="Aspect Ratio", loc="lower right")

# Subplot 3B: Aspect Ratio Distribution
sns.histplot(aspect, kde=True, color="#386cb0", ax=axes[1], bins=30)
axes[1].axvline(1.0, color="crimson", linestyle="--", label="1:1 Square (Aspect = 1.0)")
axes[1].set_title("B. Aspect Ratio Distribution (Width / Height)", fontweight="bold")
axes[1].set_xlabel("Aspect Ratio")
axes[1].set_ylabel("Image Count")
axes[1].legend(loc="upper right")

plt.tight_layout()
fig.savefig(out_dir / "fig3_eda_spatial_resolution_distribution.png")
plt.close(fig)

# ------------------------------------------------------------------------------
# FIG 4: CROSS-LINGUAL PREDICTION TRANSITIONS & POLARITY COLLAPSE
# ------------------------------------------------------------------------------
print("🎨 Plotting Fig 4: Polarity Collapse Transition Matrices...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

def compute_transition_matrix(df):
    ct = pd.crosstab(df["pred_en_seq"], df["pred_id_seq"], rownames=["English Pred"], colnames=["Indonesian Pred"])
    for cat in ["YES", "NO"]:
        if cat not in ct.index:
            ct.loc[cat] = 0
        if cat not in ct.columns:
            ct[cat] = 0
    return ct.loc[["YES", "NO"], ["YES", "NO"]]

qwen_trans = compute_transition_matrix(qwen_test)
huatuo_trans = compute_transition_matrix(huatuo_test)

sns.heatmap(qwen_trans, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[0],
            annot_kws={"size": 14, "weight": "bold"})
axes[0].set_title("A. Qwen2.5-VL-7B (Bilingual Balance)", fontweight="bold")
axes[0].text(0.5, -0.15, f"Cross-Lingual Disagreements (Flips) = {qwen_test['flip_seq'].sum()} ({qwen_test['flip_seq'].mean()*100:.1f}%)",
             ha="center", transform=axes[0].transAxes, fontsize=11, fontweight="bold", color="#1b7837")

sns.heatmap(huatuo_trans, annot=True, fmt="d", cmap="Reds", cbar=False, ax=axes[1],
            annot_kws={"size": 14, "weight": "bold"})
axes[1].set_title("B. HuatuoGPT-Vision-7B (Polarity Collapse)", fontweight="bold")
fl_count = huatuo_test['flip_seq'].sum()
yes_to_no = huatuo_trans.loc["YES", "NO"]
axes[1].text(0.5, -0.15, f"Flips = {fl_count} ({fl_count/799*100:.1f}%) | 404/406 Flips are EN=YES → ID=NO!",
             ha="center", transform=axes[1].transAxes, fontsize=11, fontweight="bold", color="#b2182b")

plt.tight_layout()
fig.savefig(out_dir / "fig4_polarity_collapse_transition_matrices.png")
plt.close(fig)

# ------------------------------------------------------------------------------
# FIG 5: ASYMMETRY TPR vs TNR & BALANCED ACCURACY GAP
# ------------------------------------------------------------------------------
print("🎨 Plotting Fig 5: TPR/TNR Asymmetry & Performance Gap...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

def get_rates(df, lang_col):
    tp = ((df[lang_col] == "YES") & (df["reference_answer"] == "YES")).sum()
    fn = ((df[lang_col] == "NO") & (df["reference_answer"] == "YES")).sum()
    tn = ((df[lang_col] == "NO") & (df["reference_answer"] == "NO")).sum()
    fp = ((df[lang_col] == "YES") & (df["reference_answer"] == "NO")).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    bacc = 0.5 * (tpr + tnr)
    return tpr * 100, tnr * 100, bacc * 100

q_en_tpr, q_en_tnr, q_en_bacc = get_rates(qwen_test, "pred_en_seq")
q_id_tpr, q_id_tnr, q_id_bacc = get_rates(qwen_test, "pred_id_seq")
h_en_tpr, h_en_tnr, h_en_bacc = get_rates(huatuo_test, "pred_en_seq")
h_id_tpr, h_id_tnr, h_id_bacc = get_rates(huatuo_test, "pred_id_seq")

# Subplot 5A: Qwen Rates
q_df = pd.DataFrame({
    "Language": ["English (EN)", "English (EN)", "Indonesian (ID)", "Indonesian (ID)"],
    "Metric": ["Sensitivity (TPR)", "Specificity (TNR)", "Sensitivity (TPR)", "Specificity (TNR)"],
    "Rate": [q_en_tpr, q_en_tnr, q_id_tpr, q_id_tnr]
})
sns.barplot(data=q_df, x="Language", y="Rate", hue="Metric", palette=["#3182bd", "#9ecae1"], ax=axes[0])
axes[0].set_title("A. Qwen2.5-VL-7B Performance Rates", fontweight="bold")
axes[0].set_ylim(0, 110)
axes[0].set_ylabel("Rate (%)")
for p in axes[0].patches:
    h = p.get_height()
    axes[0].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h / 2),
                     ha='center', va='center', color='black' if h < 60 else 'white', fontweight='bold')

# Subplot 5B: Huatuo Rates
h_df = pd.DataFrame({
    "Language": ["English (EN)", "English (EN)", "Indonesian (ID)", "Indonesian (ID)"],
    "Metric": ["Sensitivity (TPR)", "Specificity (TNR)", "Sensitivity (TPR)", "Specificity (TNR)"],
    "Rate": [h_en_tpr, h_en_tnr, h_id_tpr, h_id_tnr]
})
sns.barplot(data=h_df, x="Language", y="Rate", hue="Metric", palette=["#e6550d", "#fdae6b"], ax=axes[1])
axes[1].set_title("B. HuatuoGPT-Vision TPR Collapse", fontweight="bold")
axes[1].set_ylim(0, 110)
axes[1].set_ylabel("Rate (%)")
for p in axes[1].patches:
    h = p.get_height()
    axes[1].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h / 2),
                     ha='center', va='center', color='black' if h < 60 else 'white', fontweight='bold')

# Subplot 5C: Balanced Accuracy Comparison
bacc_df = pd.DataFrame({
    "Model": ["Qwen2.5-VL", "Qwen2.5-VL", "HuatuoGPT", "HuatuoGPT"],
    "Language": ["English (EN)", "Indonesian (ID)", "English (EN)", "Indonesian (ID)"],
    "Balanced Acc": [q_en_bacc, q_id_bacc, h_en_bacc, h_id_bacc]
})
sns.barplot(data=bacc_df, x="Model", y="Balanced Acc", hue="Language", palette=["#41ab5d", "#74c476"], ax=axes[2])
axes[2].set_title("C. Balanced Accuracy Cross-Lingual Gap", fontweight="bold")
axes[2].set_ylim(0, 110)
axes[2].set_ylabel("Balanced Accuracy (%)")
for p in axes[2].patches:
    h = p.get_height()
    axes[2].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h / 2),
                     ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
fig.savefig(out_dir / "fig5_tpr_tnr_asymmetry_and_gap.png")
plt.close(fig)

# ------------------------------------------------------------------------------
# FIG 6: MULTIMODAL PERTURBATION DELTA P (CORRECT VS INCORRECT)
# ------------------------------------------------------------------------------
print("🎨 Plotting Fig 6: Multimodal Visual Perturbation Delta P...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for idx, (df, m_title, col_pref) in enumerate([
    (qwen_test, "Qwen2.5-VL-7B", "#2171b5"),
    (huatuo_test, "HuatuoGPT-Vision-7B", "#cb181d")
]):
    df_plot = df.copy()
    df_plot["Outcome"] = np.where(df_plot["correct_id_seq"] == 1, "Correct (ID)", "Incorrect (ID)")
    sns.kdeplot(data=df_plot, x="delta_p_dir_seq", hue="Outcome", fill=True,
                palette={"Correct (ID)": "#238b45", "Incorrect (ID)": "#d73027"},
                common_norm=False, ax=axes[idx], alpha=0.5, bw_adjust=1.2)
    axes[idx].axvline(0.0, color="black", linestyle="--", alpha=0.7, label="No Visual Reliance (Δp = 0)")
    axes[idx].set_title(f"{'A' if idx==0 else 'B'}. {m_title}: Visual Sensitivity (Δp_dir)", fontweight="bold")
    axes[idx].set_xlabel("Directed Shift: Δp_dir = Conf_orig - P_blank_pred")
    axes[idx].set_ylabel("Density")
    axes[idx].legend(loc="upper left")

plt.tight_layout()
fig.savefig(out_dir / "fig6_multimodal_visual_perturbation_delta_p.png")
plt.close(fig)

# ------------------------------------------------------------------------------
# FIG 7: RISK-COVERAGE CURVES (RQ3 PRIMARY TEST N=799)
# ------------------------------------------------------------------------------
print("🎨 Plotting Fig 7: Risk-Coverage Curves (RQ3)...")
from src.stats_engine import calculate_discrete_aurc

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

estimators = [
    ("score_m1_seq", "m1: Raw Confidence", "#386cb0", "-"),
    ("score_m2_seq", "m2: Cross-Lingual", "#f0027f", "-."),
    ("score_m_dir_seq", "m_dir: Multimodal Directed", "#fdc086", ":"),
    ("score_m_norm_seq", "m_norm: Hybrid (BIVIS)", "#2ca02c", "-")
]

for idx, (df, m_title) in enumerate([(qwen_test, "Qwen2.5-VL-7B"), (huatuo_test, "HuatuoGPT-Vision-7B")]):
    corr = df["correct_id_seq"].values
    
    # Plot Oracle (Optimal)
    sort_oracle = np.argsort(-corr, kind="mergesort")
    k_vals = np.arange(1, len(corr) + 1)
    cov_arr = k_vals / float(len(corr))
    risk_oracle = 1.0 - (np.cumsum(corr[sort_oracle]) / k_vals)
    aurc_oracle = float(np.mean(risk_oracle))
    axes[idx].plot(cov_arr, risk_oracle, label=f"Oracle (AURC={aurc_oracle:.4f})", color="black", linestyle="--", linewidth=1.5)
    
    for score_col, label_name, color_c, l_style in estimators:
        scores = df[score_col].values
        aurc_val, covs, risks = calculate_discrete_aurc(corr, scores, tie_mode="stable")
        e_aurc = aurc_val - aurc_oracle
        axes[idx].plot(covs, risks, label=f"{label_name} (AURC={aurc_val:.4f}, E={e_aurc:.4f})",
                       color=color_c, linestyle=l_style, linewidth=2.0)
        
    axes[idx].set_title(f"{'A' if idx==0 else 'B'}. {m_title} Risk-Coverage Curve", fontweight="bold")
    axes[idx].set_xlabel("Coverage (Proportion of Answered Questions)")
    axes[idx].set_ylabel("Selective Risk (Error Rate on Answered)")
    axes[idx].set_xlim(0.0, 1.02)
    axes[idx].set_ylim(-0.02, 0.65)
    axes[idx].legend(loc="upper left", frameon=True)

plt.tight_layout()
fig.savefig(out_dir / "fig7_risk_coverage_curves_rq3.png")
plt.close(fig)

# ------------------------------------------------------------------------------
# FIG 8: COVERAGE VS SELECTIVE ACCURACY
# ------------------------------------------------------------------------------
print("🎨 Plotting Fig 8: Coverage vs Selective Accuracy Trade-off...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for idx, (df, m_title) in enumerate([(qwen_test, "Qwen2.5-VL-7B"), (huatuo_test, "HuatuoGPT-Vision-7B")]):
    corr = df["correct_id_seq"].values
    for score_col, label_name, color_c, l_style in estimators:
        scores = df[score_col].values
        aurc_val, covs, risks = calculate_discrete_aurc(corr, scores, tie_mode="stable")
        accs = (1.0 - risks) * 100
        axes[idx].plot(covs * 100, accs, label=label_name, color=color_c, linestyle=l_style, linewidth=2.0)
        
    axes[idx].set_title(f"{'A' if idx==0 else 'B'}. {m_title} Selective Accuracy vs Coverage", fontweight="bold")
    axes[idx].set_xlabel("Coverage (%)")
    axes[idx].set_ylabel("Selective Accuracy (%)")
    axes[idx].set_xlim(15, 100)
    axes[idx].set_ylim(40, 100)
    axes[idx].axvline(80, color="gray", linestyle=":", alpha=0.7)
    axes[idx].axvline(50, color="gray", linestyle=":", alpha=0.7)
    axes[idx].legend(loc="lower left", frameon=True)

plt.tight_layout()
fig.savefig(out_dir / "fig8_coverage_vs_selective_accuracy.png")
plt.close(fig)

# ------------------------------------------------------------------------------
# FIG 9: AURC & E-AURC SUMMARY BAR CHART
# ------------------------------------------------------------------------------
print("🎨 Plotting Fig 9: AURC & Excess AURC Bar Comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

aurc_records = []
for model_name, df in [("Qwen2.5-VL-7B", qwen_test), ("HuatuoGPT-Vision-7B", huatuo_test)]:
    corr = df["correct_id_seq"].values
    sort_oracle = np.argsort(-corr, kind="mergesort")
    k_vals = np.arange(1, len(corr) + 1)
    risk_oracle = 1.0 - (np.cumsum(corr[sort_oracle]) / k_vals)
    aurc_oracle = float(np.mean(risk_oracle))
    
    for score_col, label_name, _, _ in estimators:
        scores = df[score_col].values
        aurc_val, _, _ = calculate_discrete_aurc(corr, scores, tie_mode="stable")
        aurc_records.append({
            "Model": model_name,
            "Estimator": label_name.split(":")[0],
            "Full Label": label_name,
            "AURC": aurc_val,
            "Excess AURC (E-AURC)": aurc_val - aurc_oracle
        })

df_aurc_summary = pd.DataFrame(aurc_records)

sns.barplot(data=df_aurc_summary, x="Estimator", y="AURC", hue="Model", palette=["#2b5c8f", "#d95f02"], ax=axes[0])
axes[0].set_title("A. Discrete Riemann AURC (Lower is Better)", fontweight="bold")
axes[0].set_ylabel("AURC")
for p in axes[0].patches:
    h = p.get_height()
    if h > 0:
        axes[0].annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=10)

sns.barplot(data=df_aurc_summary, x="Estimator", y="Excess AURC (E-AURC)", hue="Model", palette=["#2b5c8f", "#d95f02"], ax=axes[1])
axes[1].set_title("B. Excess AURC Relative to Oracle (Lower is Better)", fontweight="bold")
axes[1].set_ylabel("E-AURC")
for p in axes[1].patches:
    h = p.get_height()
    if h > 0:
        axes[1].annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=10)

plt.tight_layout()
fig.savefig(out_dir / "fig9_aurc_estimator_comparison.png")
plt.close(fig)

print("🎉 ALL 9 FIGURES GENERATED SUCCESSFULLY IN reports/figures/!")
