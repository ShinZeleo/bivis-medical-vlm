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
        if 'STAGE 8B: RESILIENT INFERENCE LOOP' in src:
            # Fix df_existing typo
            old_str = "df_final = atomic_save(new_records, checkpoint_dest, df_base=None if PILOT_MODE else df_existing)"
            new_str = "base_df = None if PILOT_MODE else (df_existing_ckpt if 'df_existing_ckpt' in globals() else None)\n            df_final = atomic_save(new_records, checkpoint_dest, df_base=base_df)"
            if old_str in src:
                src = src.replace(old_str, new_str)
            else:
                src = src.replace("df_existing", "(df_existing_ckpt if 'df_existing_ckpt' in globals() else None)")
            
            # Also ensure df_final is defined if loop has zero iterations
            if "t_elapsed = time.time() - t_start" in src and "if 'df_final' not in locals():" not in src:
                src = src.replace(
                    "t_elapsed = time.time() - t_start",
                    "if 'df_final' not in locals():\n        df_final = df_existing_ckpt if ('df_existing_ckpt' in globals() and df_existing_ckpt is not None) else pd.DataFrame(new_records)\n    t_elapsed = time.time() - t_start"
                )
            cell['source'] = [l + '\n' for l in src.splitlines()]
            
        if 'STAGE 16: ARTIFACT PACKAGING' in src:
            # Make zip packaging ultra resilient
            old_zip_block = """with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(checkpoint_dest, arcname=checkpoint_dest.name)
    if rq1_csv.exists():
        zf.write(rq1_csv, arcname=rq1_csv.name)
    if rq2_csv.exists():
        zf.write(rq2_csv, arcname=rq2_csv.name)
    if rq3_csv.exists():
        zf.write(rq3_csv, arcname=rq3_csv.name)
    if ablation_csv.exists():
        zf.write(ablation_csv, arcname=ablation_csv.name)
    for plot_p in RESULTS_ROOT.glob("*.png"):
        zf.write(plot_p, arcname=plot_p.name)"""
            new_zip_block = """with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    if checkpoint_dest.exists():
        zf.write(checkpoint_dest, arcname=checkpoint_dest.name)
    for cand_var in ["rq1_csv", "rq2_csv", "rq3_csv", "ablation_csv"]:
        if cand_var in globals():
            p_val = globals()[cand_var]
            if isinstance(p_val, Path) and p_val.exists():
                zf.write(p_val, arcname=p_val.name)
    for plot_p in RESULTS_ROOT.glob("*.png"):
        zf.write(plot_p, arcname=plot_p.name)"""
            if old_zip_block in src:
                src = src.replace(old_zip_block, new_zip_block)
            cell['source'] = [l + '\n' for l in src.splitlines()]

    # Clear cell outputs
    for c in nb['cells']:
        if c.get('cell_type') == 'code':
            c['outputs'] = []
            c['execution_count'] = None

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Fixed {nb_path.name}")
