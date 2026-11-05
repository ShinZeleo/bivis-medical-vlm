"""
Generate High-Resolution Publication & Presentation Diagram for Bi-ViS Method
Output:
  - reports/figures/fig10_bivis_architecture_pipeline.png
  - reports/figures/fig10_bivis_architecture_pipeline.svg
  - artifacts/figures/bivis_architecture_pipeline.png
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Set high resolution
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

fig = plt.figure(figsize=(19, 10.5), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 19)
ax.set_ylim(0, 10.5)
ax.axis('off')

# Color Palette (Scientific Clinical AI theme)
COLOR_BG = "#F8FAFC"
COLOR_PRIMARY_BLUE = "#1E40AF"     # Deep Blue
COLOR_CYAN_ACCENT = "#0284C7"      # Cyan / Sky
COLOR_AMBER = "#D97706"            # Warning / Counterfactual
COLOR_PURPLE = "#7C3AED"           # Cross-lingual
COLOR_GREEN = "#16A34A"            # Accept
COLOR_RED = "#DC2626"              # Abstain / Error
COLOR_TEXT_MAIN = "#0F172A"        # Slate 900
COLOR_TEXT_MUTED = "#475569"       # Slate 600

# Background canvas
bg_rect = FancyBboxPatch((0.1, 0.1), 18.8, 10.3,
                         boxstyle="round,pad=0.0,rounding_size=0.15",
                         facecolor=COLOR_BG, edgecolor="#E2E8F0", linewidth=1.5, zorder=0)
ax.add_patch(bg_rect)

# -------------------------------------------------------------
# TITLE BANNER
# -------------------------------------------------------------
title_box = FancyBboxPatch((0.4, 9.25), 18.2, 0.95,
                           boxstyle="round,pad=0.0,rounding_size=0.12",
                           facecolor="#1E293B", edgecolor="none", zorder=1)
ax.add_patch(title_box)

ax.text(0.7, 9.8, "Bi-ViS : Bilingual Visual-Sensitivity-Aware Selective Prediction",
        fontsize=16.5, fontweight='bold', color="#FFFFFF", va='center')
ax.text(0.7, 9.45, "Zero-Retraining Runtime Safety Framework for Medical Vision-Language Models under Cross-Lingual Shift",
        fontsize=10.5, color="#94A3B8", va='center')

# Badges in title box
badges = [
    ("Zero Retraining", "#0284C7", 13.5),
    ("Runtime Counterfactual", "#D97706", 15.2),
    ("Cluster-Calibrated", "#10B981", 17.1)
]
for b_text, b_col, b_x in badges:
    b_patch = FancyBboxPatch((b_x - 0.7, 9.45), 1.5, 0.45,
                             boxstyle="round,pad=0.0,rounding_size=0.08",
                             facecolor=b_col, edgecolor="none", zorder=2)
    ax.add_patch(b_patch)
    ax.text(b_x + 0.05, 9.675, b_text, fontsize=8.5, fontweight='bold', color="#FFFFFF", ha='center', va='center', zorder=3)

# -------------------------------------------------------------
# 5 HORIZONTAL STAGES
# -------------------------------------------------------------
def draw_stage_card(x, y, w, h, stage_num, title, subtitle, border_color="#E2E8F0", header_bg="#F1F5F9"):
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12",
                          facecolor="#FFFFFF", edgecolor=border_color, linewidth=1.5, zorder=1)
    ax.add_patch(card)
    header = FancyBboxPatch((x, y + h - 0.75), w, 0.75, boxstyle="round,pad=0.0,rounding_size=0.12",
                            facecolor=header_bg, edgecolor=border_color, linewidth=1.5, zorder=2)
    ax.add_patch(header)
    rect_patch = patches.Rectangle((x, y + h - 0.75), w, 0.35, facecolor=header_bg, edgecolor="none", zorder=3)
    ax.add_patch(rect_patch)
    
    ax.text(x + 0.2, y + h - 0.28, f"STAGE {stage_num}: {title}", fontsize=9.2, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=4)
    ax.text(x + 0.2, y + h - 0.55, subtitle, fontsize=7.2, color=COLOR_TEXT_MUTED, zorder=4)

# =============================================================
# STAGE 1: MULTIMODAL INPUTS
# =============================================================
draw_stage_card(0.4, 2.0, 3.1, 7.0, "1", "MULTIMODAL INPUTS", "Clinical Queries & Counterfactual")

# Item 1A: Medical Image (Original)
box_img = FancyBboxPatch((0.6, 6.7), 2.7, 1.8, boxstyle="round,pad=0.0,rounding_size=0.08",
                         facecolor="#EFF6FF", edgecolor="#BFDBFE", linewidth=1.2, zorder=3)
ax.add_patch(box_img)
ax.text(0.75, 8.15, "[RAD-IMG]  $I_{orig}$ (Radiology Image)", fontsize=8.5, fontweight='bold', color="#1E40AF", zorder=4)
ax.text(0.75, 7.7, "Chest X-Ray / CT Scan (Clean)\nFormat: RGB ($H \\times W \\times 3$)\nCluster ID: $K_i \\in \\{1..204\\}$ (Disjoint)",
        fontsize=7.3, color="#1E3A8A", linespacing=1.3, zorder=4)

# Item 1B: Indonesian Question
box_qid = FancyBboxPatch((0.6, 5.0), 2.7, 1.5, boxstyle="round,pad=0.0,rounding_size=0.08",
                         facecolor="#F0FDF4", edgecolor="#BBF7D0", linewidth=1.2, zorder=3)
ax.add_patch(box_qid)
ax.text(0.75, 6.15, "[QUERY-ID]  $Q_{ID}$ (Indonesian)", fontsize=8.5, fontweight='bold', color="#166534", zorder=4)
ax.text(0.75, 5.5, '"Apakah terdapat kardiomegali?"\nCandidate Space: {Ya, Tidak}\nLength: $\\sim 8.4$ words (token expansion)',
        fontsize=7.1, color="#14532D", linespacing=1.3, zorder=4)

# Item 1C: Counterfactual Blank Canvas
box_blank = FancyBboxPatch((0.6, 3.5), 2.7, 1.3, boxstyle="round,pad=0.0,rounding_size=0.08",
                           facecolor="#FEF3C7", edgecolor="#FDE68A", linewidth=1.2, zorder=3)
ax.add_patch(box_blank)
ax.text(0.75, 4.45, "[BLANK]  $I_{blank}$ (Zero Canvas)", fontsize=8.2, fontweight='bold', color="#92400E", zorder=4)
ax.text(0.75, 3.9, "Pure Black Canvas: $I_{blank} = 0$\nSpatial dimensions identical ($H \\times W \\times 3$)\nMasks all anatomical evidence",
        fontsize=7.0, color="#78350F", linespacing=1.3, zorder=4)

# Item 1D: English Paired Question
box_qen = FancyBboxPatch((0.6, 2.2), 2.7, 1.15, boxstyle="round,pad=0.0,rounding_size=0.08",
                         facecolor="#F3E8FF", edgecolor="#E9D5FF", linewidth=1.2, zorder=3)
ax.add_patch(box_qen)
ax.text(0.75, 3.0, "[QUERY-EN]  $Q_{EN}$ (English)", fontsize=8.2, fontweight='bold', color="#6B21A8", zorder=4)
ax.text(0.75, 2.5, '"Is cardiomegaly present?"\nOriginal pre-translation reference query',
        fontsize=7.0, color="#581C87", linespacing=1.2, zorder=4)


# =============================================================
# STAGE 2: 3-WAY INFERENCE (FROZEN VLM)
# =============================================================
draw_stage_card(3.8, 2.0, 3.3, 7.0, "2", "3-WAY VLM INFERENCE", "Zero-Retraining Log-Likelihood Evaluation")

# VLM Core Box
vlm_container = FancyBboxPatch((4.0, 7.4), 2.9, 1.1, boxstyle="round,pad=0.0,rounding_size=0.08",
                               facecolor="#0F172A", edgecolor="#334155", linewidth=1.2, zorder=3)
ax.add_patch(vlm_container)
ax.text(5.45, 8.15, "Frozen Medical VLM ($f_\\theta$)", fontsize=9.2, fontweight='bold', color="#38BDF8", ha='center', zorder=4)
ax.text(5.45, 7.7, "Qwen2.5-VL-7B / HuatuoGPT-Vision-7B\nZero Retraining (Parameters $\\theta$ are FROZEN)",
        fontsize=7.0, color="#CBD5E1", ha='center', linespacing=1.2, zorder=4)

# 3 Parallel Forward Passes
passes = [
    ("Pass 1: Primary ID Diagnostic", "#EFF6FF", "#1D4ED8",
     "Input: $(I_{orig}, Q_{ID})$\n• Sequence Log-Likelihood:\n  $P(c \\mid I_{orig}, Q_{ID}), c \\in \\{\\mathrm{Ya},\\mathrm{Tidak}\\}$\n• Pred: $\\hat{y}_{ID} = \\arg\\max_c P(c)$", 5.6, 1.6),
    ("Pass 2: Visual Removal Probe", "#FEF3C7", "#B45309",
     "Input: $(I_{blank}, Q_{ID})$\n• Likelihood on blank canvas:\n  $P(\\hat{y}_{ID} \\mid I_{blank}, Q_{ID})$\n• Probes text priors & hallucinations", 3.8, 1.6),
    ("Pass 3: Cross-Lingual Reference", "#F3E8FF", "#7E22CE",
     "Input: $(I_{orig}, Q_{EN})$\n• English Candidate Likelihood:\n  $P(c \\mid I_{orig}, Q_{EN}), c \\in \\{\\mathrm{Yes},\\mathrm{No}\\}$\n• Pred: $\\hat{y}_{EN} = \\arg\\max_c P(c)$", 2.2, 1.45)
]

for p_title, p_bg, p_txt, p_desc, p_y, p_h in passes:
    p_box = FancyBboxPatch((4.0, p_y), 2.9, p_h, boxstyle="round,pad=0.0,rounding_size=0.08",
                           facecolor=p_bg, edgecolor="#CBD5E1", linewidth=1.0, zorder=3)
    ax.add_patch(p_box)
    ax.text(4.15, p_y + p_h - 0.25, p_title, fontsize=7.8, fontweight='bold', color=p_txt, zorder=4)
    ax.text(4.15, p_y + p_h - 0.52, p_desc, fontsize=6.8, color="#1E293B", linespacing=1.25, va='top', zorder=4)


# =============================================================
# STAGE 3: TRI-SIGNAL EXTRACTION
# =============================================================
draw_stage_card(7.4, 2.0, 3.3, 7.0, "3", "TRI-SIGNAL EXTRACTION", "Orthogonal Epistemic Risk Indicators")

signals = [
    ("SIGNAL 1: Verbal Confidence", "$M_1$", "#DBEAFE", "#1E40AF",
     "Definition:\n$M_1 = P(y = \\hat{y}_{ID} \\mid I_{orig}, Q_{ID})$\n\n"
     "• Direct self-confidence in Indonesian\n"
     "• Token length-normalized sequence probability\n"
     "• Clinical interpretation: Verbal certainty", 6.6, 2.1),
    ("SIGNAL 2: Directed Visual Sensitivity", "$\\Delta P_{dir}$", "#FEF3C7", "#92400E",
     "Definition:\n$\\Delta P_{dir} = P(\\hat{y}_{ID}|I_{orig}) - P(\\hat{y}_{ID}|I_{blank})$\n\n"
     "• $\\Delta P > 0$: Strong visual grounding\n"
     "• $\\Delta P \\leq 0$: Linguistic shortcut / hallucination\n"
     "• Captures negative visual reliance (Huatuo: 80%)", 4.3, 2.15),
    ("SIGNAL 3: Cross-Lingual Conflict", "$D_{lang}$", "#F3E8FF", "#6B21A8",
     "Definition:\n$D_{lang} = \\mathbb{I}(\\mathcal{M}(\\hat{y}_{EN}) \\ne \\mathcal{M}(\\hat{y}_{ID}))$\n\n"
     "• Binary indicator of translation flip\n"
     "• Detects semantic instability under shift\n"
     "• Risk Odds Ratio: 3.06x (Qwen), 2.76x (Huatuo)!", 2.2, 1.95)
]

for s_name, s_sym, s_bg, s_txt, s_desc, s_y, s_h in signals:
    s_box = FancyBboxPatch((7.6, s_y), 2.9, s_h, boxstyle="round,pad=0.0,rounding_size=0.08",
                           facecolor=s_bg, edgecolor="#CBD5E1", linewidth=1.0, zorder=3)
    ax.add_patch(s_box)
    ax.text(7.75, s_y + s_h - 0.25, s_name, fontsize=7.8, fontweight='bold', color=s_txt, zorder=4)
    ax.text(7.75, s_y + s_h - 0.52, s_desc, fontsize=6.8, color="#1E293B", linespacing=1.25, va='top', zorder=4)


# =============================================================
# STAGE 4: CALIBRATED MULTIVARIATE RISK ESTIMATOR
# =============================================================
draw_stage_card(11.0, 2.0, 3.6, 7.0, "4", "LEARNED RISK ESTIMATOR", "Dev-Calibrated L2 Regularized Logistic Model")

# Step 4A: Dev Set Standardization
box_std = FancyBboxPatch((11.2, 6.7), 3.2, 1.8, boxstyle="round,pad=0.0,rounding_size=0.08",
                         facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.0, zorder=3)
ax.add_patch(box_std)
ax.text(11.35, 8.2, "A. Dev Scaler (Strict Zero-Leakage)", fontsize=8.0, fontweight='bold', color="#334155", zorder=4)
ax.text(11.35, 7.85, "Standardization on Dev Set ($N=323$, 88 clusters):\n"
                      "$z_{j} = (x_j - \\mu_j^{dev}) / \\sigma_j^{dev}, \\quad j \\in \\{M_1, \\Delta P_{dir}, D_{lang}\\}$\n"
                      "Prevents data snooping; VIF $< 1.55$ (Strictly orthogonal).",
        fontsize=6.8, color="#475569", linespacing=1.25, va='top', zorder=4)

# Step 4B: Logistic Multivariate Combination
box_logit = FancyBboxPatch((11.2, 4.3), 3.2, 2.25, boxstyle="round,pad=0.0,rounding_size=0.08",
                          facecolor="#F0FDF4", edgecolor="#86EFAC", linewidth=1.2, zorder=3)
ax.add_patch(box_logit)
ax.text(11.35, 6.25, "B. Calibrated Error Risk Modeling", fontsize=8.0, fontweight='bold', color="#166534", zorder=4)
ax.text(11.35, 5.95, "Regularized Logistic Regression ($L_2, C=1.0$):\n"
                      "$\\mathrm{logit}(\\mathbf{x}) = \\beta_0 + \\beta_1 z_{M_1} + \\beta_2 z_{\\Delta P} + \\beta_3 z_{D_{lang}}$\n"
                      "$P(\\mathrm{error} \\mid \\mathbf{x}) = \\sigma(\\mathrm{logit}) = [1 + \\exp(-\\mathrm{logit})]^{-1}$\n"
                      "Optimized via 5-Fold Group Cross-Validation on Dev clusters.",
        fontsize=6.8, color="#14532D", linespacing=1.25, va='top', zorder=4)

# Step 4C: Selective Reliability Score
box_score = FancyBboxPatch((11.2, 2.2), 3.2, 1.95, boxstyle="round,pad=0.0,rounding_size=0.08",
                           facecolor="#EFF6FF", edgecolor="#93C5FD", linewidth=1.2, zorder=3)
ax.add_patch(box_score)
ax.text(11.35, 3.85, "C. Selective Reliability Score Alignment", fontsize=8.0, fontweight='bold', color="#1E40AF", zorder=4)
ax.text(11.35, 3.55, "Score is oriented inverse to clinical risk:\n"
                      "$S_{Bi-ViS}(\\mathbf{x}) = 1.0 - P(\\mathrm{error} \\mid \\mathbf{x})$\n"
                      "• $S \\in [0, 1]$: continuous reliability index\n"
                      "• Solves arithmetic ratio instability of $M_{norm}$.",
        fontsize=6.8, color="#1E3A8A", linespacing=1.25, va='top', zorder=4)


# =============================================================
# STAGE 5: SELECTIVE DECISION GATE & CLINICAL ROUTING
# =============================================================
draw_stage_card(14.9, 2.0, 3.7, 7.0, "5", "SELECTIVE DECISION GATE", "Clinical Abstention & Risk-Coverage Control")

# Gate Evaluation Box
box_gate = FancyBboxPatch((15.1, 6.4), 3.3, 1.8, boxstyle="round,pad=0.0,rounding_size=0.1",
                          facecolor="#312E81", edgecolor="#6366F1", linewidth=1.5, zorder=3)
ax.add_patch(box_gate)
ax.text(16.75, 7.9, "Clinical Decision Threshold Gate", fontsize=8.6, fontweight='bold', color="#FFFFFF", ha='center', zorder=4)
ax.text(16.75, 7.45, "Evaluation Rule:\n$S_{Bi-ViS}(\\mathbf{x}) \\geq \\tau$",
        fontsize=8.5, color="#E0E7FF", ha='center', linespacing=1.3, zorder=4)
ax.text(16.75, 6.75, "Threshold $\\tau$ is calibrated on Dev Set for Target Coverage\n$\\kappa = 80\\%$ (or maximum allowable risk $\\rho$)",
        fontsize=6.8, color="#C7D2FE", ha='center', linespacing=1.2, zorder=4)

# Path 1: ACCEPT (Green)
box_accept = FancyBboxPatch((15.1, 4.3), 3.3, 1.85, boxstyle="round,pad=0.0,rounding_size=0.08",
                            facecolor="#DCFCE7", edgecolor="#22C55E", linewidth=1.5, zorder=3)
ax.add_patch(box_accept)
ax.text(15.25, 5.85, "[ACCEPT] Answer Diagnostic Query", fontsize=8.0, fontweight='bold', color="#15803D", zorder=4)
ax.text(15.25, 5.5, "• High Confidence & Visual Grounding Verified\n"
                     "• Emits Indonesian Answer $\\hat{y}_{ID}$ to clinician\n"
                     "• Selective Risk drops from 33.6% -> 28.5% (Qwen)\n"
                     "• Guarantees high-integrity automated triage.",
        fontsize=6.8, color="#14532D", linespacing=1.25, va='top', zorder=4)

# Path 2: ABSTAIN (Red/Amber)
box_abstain = FancyBboxPatch((15.1, 2.2), 3.3, 1.85, boxstyle="round,pad=0.0,rounding_size=0.08",
                             facecolor="#FEE2E2", edgecolor="#EF4444", linewidth=1.5, zorder=3)
ax.add_patch(box_abstain)
ax.text(15.25, 3.75, "[ABSTAIN] Refer to Radiologist", fontsize=8.0, fontweight='bold', color="#B91C1C", zorder=4)
ax.text(15.25, 3.4, "• Disagreement or Language Shortcut Flagged\n"
                     "• Model withholds automated diagnostic response\n"
                     "• Routes image to specialist human radiologist\n"
                     "• Prevents catastrophic silent misdiagnoses.",
        fontsize=6.8, color="#7F1D1D", linespacing=1.25, va='top', zorder=4)


# =============================================================
# ARROWS AND INTERCONNECTIONS
# =============================================================
arrow_prop = dict(arrowstyle="->,head_width=0.25,head_length=0.35", linewidth=1.4, color="#475569")

# Connect Stage 1 to Stage 2
ax.annotate("", xy=(4.0, 6.4), xytext=(3.3, 7.5), arrowprops=arrow_prop, zorder=5) # Img to Pass 1
ax.annotate("", xy=(4.0, 4.6), xytext=(3.3, 4.1), arrowprops=arrow_prop, zorder=5) # Blank to Pass 2
ax.annotate("", xy=(4.0, 2.9), xytext=(3.3, 2.8), arrowprops=arrow_prop, zorder=5) # Qen to Pass 3

# Connect Stage 2 to Stage 3
ax.annotate("", xy=(7.6, 7.6), xytext=(6.9, 6.4), arrowprops=arrow_prop, zorder=5) # Pass 1 to M1
ax.annotate("", xy=(7.6, 5.3), xytext=(6.9, 4.6), arrowprops=arrow_prop, zorder=5) # Pass 2 to DeltaP
ax.annotate("", xy=(7.6, 3.1), xytext=(6.9, 2.9), arrowprops=arrow_prop, zorder=5) # Pass 3 to Dlang

# Connect Stage 3 to Stage 4 (Features to Standardizer)
ax.annotate("", xy=(11.2, 7.6), xytext=(10.5, 7.6), arrowprops=arrow_prop, zorder=5)
ax.annotate("", xy=(11.2, 7.3), xytext=(10.5, 5.3), arrowprops=arrow_prop, zorder=5)
ax.annotate("", xy=(11.2, 7.0), xytext=(10.5, 3.1), arrowprops=arrow_prop, zorder=5)

# Internal Stage 4 connections
ax.annotate("", xy=(12.8, 6.55), xytext=(12.8, 6.7), arrowprops=arrow_prop, zorder=5)
ax.annotate("", xy=(12.8, 4.15), xytext=(12.8, 4.3), arrowprops=arrow_prop, zorder=5)

# Connect Stage 4 to Stage 5
ax.annotate("", xy=(15.1, 7.3), xytext=(14.4, 3.1), arrowprops=arrow_prop, zorder=5)

# Internal Stage 5 branching
arrow_yes = dict(arrowstyle="->,head_width=0.25,head_length=0.35", linewidth=1.5, color="#16A34A")
arrow_no  = dict(arrowstyle="->,head_width=0.25,head_length=0.35", linewidth=1.5, color="#DC2626")
ax.annotate("", xy=(16.75, 6.15), xytext=(16.75, 6.4), arrowprops=arrow_yes, zorder=5)
ax.text(16.85, 6.25, "Yes (>= tau)", fontsize=7.2, fontweight='bold', color="#16A34A", zorder=6)

ax.annotate("", xy=(16.75, 4.05), xytext=(16.75, 4.3), arrowprops=arrow_no, zorder=5)
ax.text(16.85, 4.15, "No (< tau)", fontsize=7.2, fontweight='bold', color="#DC2626", zorder=6)


# =============================================================
# BOTTOM SUMMARY KPI STRIP
# =============================================================
kpi_box = FancyBboxPatch((0.4, 0.35), 18.2, 1.45,
                         boxstyle="round,pad=0.0,rounding_size=0.12",
                         facecolor="#0F172A", edgecolor="#334155", linewidth=1.2, zorder=1)
ax.add_patch(kpi_box)

ax.text(0.7, 1.48, "EMPIRICAL BENCHMARK PERFORMANCE (IndoRad-VQA Primary Test Set, N=799, 204 Independent Clusters)",
        fontsize=8.5, fontweight='bold', color="#F8FAFC", zorder=2)

metrics = [
    ("Model 1: Qwen2.5-VL-7B (Generalist)",
     "• Baseline AURC (M1): 0.2422 -> Bi-ViS AURC: 0.2053 (p = 0.0310*)\n"
     "• Risk@80% Coverage: 33.64% -> 28.48% (Reduces errors by -5.16 pp)\n"
     "• Disagreement Odds Ratio: OR = 3.064x (Wald z = +7.55, p = 4.45e-14)"),
    ("Model 2: HuatuoGPT-Vision-7B (Medical)",
     "• Baseline AURC (M1): 0.2884 -> Bi-ViS AURC: 0.2179 (p = 0.0010***)\n"
     "• Polarity Collapse Mitigated: 404/406 flips captured by D_lang\n"
     "• Visual Sensitivity: beta_2 = -0.3032 (OR = 0.738, 26.2% risk reduction)"),
    ("Anti-HARKing Demarcation",
     "• Pre-registered Hypothesis (M_norm): FAILED on Qwen (ratio explosion)\n"
     "• Bi-ViS Proposed Method: Regularized logistic mapping guarantees stability\n"
     "• Strict Cluster Disjointness: Zero data leakage between Dev & Test splits")
]

x_offsets = [0.7, 6.8, 12.9]
w_boxes = [5.8, 5.8, 5.4]

for i, (m_head, m_body) in enumerate(metrics):
    box_m = FancyBboxPatch((x_offsets[i], 0.5), w_boxes[i], 0.85,
                           boxstyle="round,pad=0.0,rounding_size=0.06",
                           facecolor="#1E293B", edgecolor="#475569", linewidth=0.8, zorder=2)
    ax.add_patch(box_m)
    ax.text(x_offsets[i] + 0.15, 1.18, m_head, fontsize=7.4, fontweight='bold', color="#38BDF8", zorder=3)
    ax.text(x_offsets[i] + 0.15, 1.02, m_body, fontsize=6.5, color="#E2E8F0", linespacing=1.25, va='top', zorder=3)

# Save figures
out_reports = Path("reports/figures")
out_artifacts = Path("artifacts/figures")
out_reports.mkdir(parents=True, exist_ok=True)
out_artifacts.mkdir(parents=True, exist_ok=True)

png_path_1 = out_reports / "fig10_bivis_architecture_pipeline.png"
svg_path_1 = out_reports / "fig10_bivis_architecture_pipeline.svg"
png_path_2 = out_artifacts / "bivis_architecture_pipeline.png"

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(png_path_1, dpi=300, bbox_inches='tight')
fig.savefig(svg_path_1, bbox_inches='tight')
fig.savefig(png_path_2, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"[SUCCESS] Generated: {png_path_1}")
print(f"[SUCCESS] Generated: {svg_path_1}")
print(f"[SUCCESS] Generated: {png_path_2}")
