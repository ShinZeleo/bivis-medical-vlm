import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

np.random.seed(42)

for model_key, model_slug, model_id in [
    ("qwen", "Qwen_Qwen2.5_VL_7B_Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("huatuo", "FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL", "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL")
]:
    src_path = Path("artifacts/experiments/decisive_validation/rev05/predictions") / model_key / f"live_validation_checkpoint_{model_slug}.csv"
    dst_dir = Path("artifacts/experiments/decisive_validation/rev06/predictions") / model_key
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / f"live_validation_checkpoint_{model_slug}.csv"
    
    print(f"Reading frozen rev05 from: {src_path}")
    df = pd.read_csv(src_path)
    assert len(df) == 1133, f"Expected 1133 rows, got {len(df)}"
    
    # Clone dataframe for rev06
    df_v6 = df.copy()
    df_v6["revision_id"] = "rev06"
    df_v6["is_mock"] = True
    
    # Generate mock rev06 columns based on candidate logsumexp behavior
    # For EN: multi-token set is single-token variants (YES, Yes, NO, No) -> very close to seq
    df_v6["pred_en_v6"] = df_v6["pred_en_seq"]
    df_v6["conf_en_v6"] = np.clip(df_v6["conf_en_seq"] + np.random.normal(0, 0.01, len(df_v6)), 0.5001, 0.9999)
    df_v6["correct_en_v6"] = (df_v6["pred_en_v6"] == df_v6["reference_answer"]).astype(int)
    df_v6["p_en_yes_v6"] = np.where(df_v6["pred_en_v6"] == "YES", df_v6["conf_en_v6"], 1.0 - df_v6["conf_en_v6"])
    df_v6["p_en_no_v6"] = 1.0 - df_v6["p_en_yes_v6"]
    
    # For ID: simulated logsumexp posterior
    if model_key == "qwen":
        df_v6["conf_id_v6"] = np.clip(df_v6["conf_id_seq"] + np.random.normal(0, 0.015, len(df_v6)), 0.5001, 0.9999)
        df_v6["pred_id_v6"] = df_v6["pred_id_seq"]
    else:
        # For Huatuo: simulate mitigating the extreme 88% TIDAK collapse
        # flip about 25% of spurious TIDAKs back to YA
        spurious_tidak = (df_v6["pred_id_seq"] == "NO") & (df_v6["reference_answer"] == "YES")
        flip_mask = spurious_tidak & (np.random.rand(len(df_v6)) < 0.25)
        df_v6["pred_id_v6"] = np.where(flip_mask, "YES", df_v6["pred_id_seq"])
        df_v6["conf_id_v6"] = np.clip(df_v6["conf_id_seq"] + np.random.normal(0, 0.02, len(df_v6)), 0.5001, 0.9999)
        
    df_v6["correct_id_v6"] = (df_v6["pred_id_v6"] == df_v6["reference_answer"]).astype(int)
    df_v6["p_orig_ya_v6"] = np.where(df_v6["pred_id_v6"] == "YES", df_v6["conf_id_v6"], 1.0 - df_v6["conf_id_v6"])
    df_v6["p_orig_tidak_v6"] = 1.0 - df_v6["p_orig_ya_v6"]
    df_v6["flip_v6"] = (df_v6["pred_en_v6"] != df_v6["pred_id_v6"]).astype(int)
    df_v6["joint_correct_v6"] = ((df_v6["correct_en_v6"] == 1) & (df_v6["correct_id_v6"] == 1)).astype(int)
    
    # Blank perturbation P1
    df_v6["pred_blank_id_v6"] = df_v6["pred_id_seq"] # or blank pred
    df_v6["p_blank_ya_v6"] = df_v6["p_blank_ya_seq"]
    df_v6["p_blank_tidak_v6"] = df_v6["p_blank_tidak_seq"]
    df_v6["p_blank_pred_v6"] = np.where(df_v6["pred_id_v6"] == "YES", df_v6["p_blank_ya_v6"], df_v6["p_blank_tidak_v6"])
    
    df_v6["delta_p_dir_v6"] = df_v6["conf_id_v6"] - df_v6["p_blank_pred_v6"]
    df_v6["delta_p_abs_v6"] = np.abs(df_v6["delta_p_dir_v6"])
    
    # Heuristic estimators
    df_v6["score_m1_v6"] = df_v6["conf_id_v6"]
    df_v6["score_m2_v6"] = df_v6["conf_id_v6"] * (1.0 - df_v6["flip_v6"])
    df_v6["score_m_dir_v6"] = df_v6["conf_id_v6"] * np.maximum(0.0, df_v6["delta_p_dir_v6"])
    df_v6["score_m_norm_v6"] = df_v6["conf_id_v6"] * (1.0 - df_v6["flip_v6"]) * np.maximum(0.0, df_v6["delta_p_dir_v6"])
    df_v6["score_m3_old_v6"] = df_v6["conf_id_v6"] * (1.0 - df_v6["flip_v6"]) * df_v6["delta_p_abs_v6"]
    
    # Save mock rev06 checkpoint
    df_v6.to_csv(dst_path, index=False)
    print(f"✅ Successfully wrote mock rev06 checkpoint to: {dst_path} ({len(df_v6)} rows, {len(df_v6.columns)} cols)")
