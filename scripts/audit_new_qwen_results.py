import sys
from pathlib import Path
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

res_dir = Path("Qwen_Qwen2.5_VL_7B_Instruct_complete_validation_results")
ckpt_path = res_dir / "live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv"
df = pd.read_csv(ckpt_path)

print("=" * 80)
print("AUDIT INTEGRITAS & HASIL EMPIRIS RIIL QWEN2.5-VL-7B (REV06)")
print("=" * 80)
print(f"Shape: {df.shape} (Baris: {len(df)}, Kolom: {len(df.columns)})")
print(f"Total QA IDs: {df['qa_id'].nunique()} (Duplikat: {df['qa_id'].duplicated().sum()})")
print(f"Model ID: {df['model_id'].iloc[0]}")

for col in ["compute_dtype", "gpu_name", "elapsed_time_sec"]:
    if col in df.columns:
        val = df[col].iloc[0] if col != "elapsed_time_sec" else f"Mean: {df[col].mean():.2f}s, Total: {df[col].sum():.1f}s (~{df[col].sum()/60:.1f} menit)"
        print(f"{col}: {val}")

# Primary Test Cohort
test_df = df[(df["split"] == "test") & (df["is_eligible_primary"] == True)].copy()
dev_df = df[(df["split"] == "dev") & (df["is_eligible_primary"] == True)].copy()
print(f"\nCohort Sizes -> Primary Test: {len(test_df)}, Dev: {len(dev_df)}")

# Track comparison
print("\n--- PERFORMA PER TRACK PADA PRIMARY TEST (N=799) ---")
for track in ["seq", "len", "v6"]:
    c_id = f"correct_id_{track}"
    c_en = f"correct_en_{track}"
    fl = f"flip_{track}"
    p_id = f"pred_id_{track}"
    p_en = f"pred_en_{track}"
    if c_id in test_df.columns:
        acc_id = test_df[c_id].mean() * 100.0
        acc_en = test_df[c_en].mean() * 100.0
        flip_rate = test_df[fl].mean() * 100.0
        flip_n = test_df[fl].sum()
        yes_rate_id = (test_df[p_id] == "YES").mean() * 100.0
        yes_rate_en = (test_df[p_en] == "YES").mean() * 100.0
        print(f"Track [{track.upper()}]:")
        print(f"  Akurasi ID     : {acc_id:.2f}%")
        print(f"  Akurasi EN     : {acc_en:.2f}%")
        print(f"  Gap (EN - ID)  : {acc_en - acc_id:+.2f} pp")
        print(f"  Flip Rate      : {flip_rate:.2f}% ({flip_n}/{len(test_df)})")
        print(f"  Pred-YES (ID)  : {yes_rate_id:.2f}%")
        print(f"  Pred-YES (EN)  : {yes_rate_en:.2f}%")

# Audit RQ files
print("\n--- RINGKASAN FILE HASIL EVALUASI RQ ---")
for rq_name in ["rq1_bilingual_metrics", "rq2_visual_intervention", "rq3_selective_prediction", "risk_ablation"]:
    rq_file = res_dir / f"{rq_name}_Qwen_Qwen2.5_VL_7B_Instruct.csv"
    if rq_file.exists():
        rq_df = pd.read_csv(rq_file)
        print(f"\n📄 {rq_file.name}:")
        print(rq_df.to_string(index=False))

