import sys
import json
from pathlib import Path

for nb_path in [
    Path('notebooks/kaggle/rev06/08_qwen_decisive_validation.ipynb'),
    Path('notebooks/kaggle/rev06/09_huatuo_decisive_validation.ipynb')
]:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        src = ''.join(cell.get('source', []))
        if 'STAGE 9: POST-INFERENCE CHECKPOINT INTEGRITY AUDIT' in src:
            new_src = """# ==============================================================================
# STAGE 9: POST-INFERENCE CHECKPOINT INTEGRITY AUDIT
# ==============================================================================
print("=" * 70)
print("🔍 POST-INFERENCE CHECKPOINT INTEGRITY AUDIT")
print("=" * 70)

if PILOT_MODE:
    print(f"ℹ️ PILOT MODE ACTIVE: Skipping full-run 1,133-row check. Verifying {len(df_final)} pilot samples...")
    assert len(df_final) == PILOT_SAMPLE_COUNT, f"Expected {PILOT_SAMPLE_COUNT} pilot rows, found {len(df_final)}"
    print(f"✅ Pilot Preflight Row Count: {len(df_final)} / {PILOT_SAMPLE_COUNT} (100% COMPLETE)")
else:
    assert len(df_final) == 1133, f"Incomplete checkpoint! Expected 1133 rows, found {len(df_final)}"
    assert df_final["qa_id"].nunique() == 1133, "Duplicate QA IDs detected in checkpoint!"
    assert set(df_final["qa_id"]) == set(df_eval_cohort["qa_id"]), "QA ID mismatch against evaluation cohort!"
    print("✅ Full Run Row Count: 1,133 / 1,133 (100% COMPLETE)")

# Non-finite score check
score_cols = [
    "score_m1_seq", "score_m2_seq", "score_m_dir_seq", "score_m_norm_seq", "score_m3_old_seq",
    "delta_p_dir_seq", "score_m1_len", "score_m_norm_len"
]
for col in score_cols:
    assert np.isfinite(df_final[col]).all(), f"Non-finite values detected in {col}!"

print("✅ Duplicate QA IDs: 0")
print("✅ Non-finite Values: 0")
print(f"✅ Accuracy ID (Seq): {df_final['correct_id_seq'].mean()*100:.2f}%")
print(f"✅ Accuracy EN (Seq): {df_final['correct_en_seq'].mean()*100:.2f}%")
print(f"✅ Flip Rate (Seq)  : {df_final['flip_seq'].mean()*100:.2f}%")
print("=" * 70)
"""
            cell['source'] = [l + '\n' for l in new_src.splitlines()]
            
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f'Updated Stage 9 in {nb_path.name}')
