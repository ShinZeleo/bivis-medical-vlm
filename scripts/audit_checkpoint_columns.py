import subprocess
import json
import pandas as pd
import ast
import re

# 1. Load actual checkpoint CSV
qwen_csv_path = "artifacts/experiments/decisive_validation/rev05/predictions/qwen/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv"
huatuo_csv_path = "artifacts/experiments/decisive_validation/rev05/predictions/huatuo/live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv"

df_qwen = pd.read_csv(qwen_csv_path)
df_huatuo = pd.read_csv(huatuo_csv_path)

csv_cols_qwen = list(df_qwen.columns)
csv_cols_huatuo = list(df_huatuo.columns)

print(f"Qwen CSV shape: {df_qwen.shape}, num cols: {len(csv_cols_qwen)}")
print(f"Huatuo CSV shape: {df_huatuo.shape}, num cols: {len(csv_cols_huatuo)}")
print(f"Columns identical between Qwen and Huatuo CSV: {csv_cols_qwen == csv_cols_huatuo}")

print("\nMetadata values in CSV:")
print("Qwen revision_id:", df_qwen["revision_id"].unique().tolist())
print("Qwen scoring_method:", df_qwen["scoring_method"].unique().tolist())
print("Huatuo revision_id:", df_huatuo["revision_id"].unique().tolist())
print("Huatuo scoring_method:", df_huatuo["scoring_method"].unique().tolist())

# 2. Extract keys from 08_decisive_validation_rev05.ipynb (monolithic STAGE 21 & 22)
d08 = subprocess.check_output(['git', 'show', 'v11.0-rev05-frozen:main/notebooks/kaggle/08_decisive_validation_rev05.ipynb'])
nb08 = json.loads(d08)
c7_src = ''.join(nb08['cells'][7]['source'])
idx1 = c7_src.find('results.append({')
idx2 = c7_src.find('})', idx1)
dict_str_08 = c7_src[idx1 + len('results.append('): idx2 + 1]
# extract keys using regex
keys_08 = re.findall(r'"([^"]+)":', dict_str_08)
print(f"\nKeys in 08_decisive_validation_rev05.ipynb (monolithic): {len(keys_08)}")

# 3. Extract keys from 09_infer_huatuo_resume.ipynb
d09 = subprocess.check_output(['git', 'show', 'v11.0-rev05-frozen:main/notebooks/kaggle/09_infer_huatuo_resume.ipynb'])
nb09 = json.loads(d09)
c8_src_09 = ''.join(nb09['cells'][8]['source'])
idx1 = c8_src_09.find('new_records.append({')
idx2 = c8_src_09.find('})', idx1)
dict_str_09 = c8_src_09[idx1 + len('new_records.append('): idx2 + 1]
keys_09 = re.findall(r'"([^"]+)":', dict_str_09)
print(f"Keys in 09_infer_huatuo_resume.ipynb: {len(keys_09)}")

# Compare keys
print("\nComparison with Qwen CSV columns:")
print(f"CSV cols count: {len(csv_cols_qwen)}")
print(f"Monolithic 08 keys count: {len(keys_08)}")
diff_08_missing_in_csv = set(keys_08) - set(csv_cols_qwen)
diff_csv_missing_in_08 = set(csv_cols_qwen) - set(keys_08)
print(f"Keys in 08 but missing in CSV: {diff_08_missing_in_csv}")
print(f"Cols in CSV but missing in 08: {diff_csv_missing_in_08}")
print(f"Keys order matches CSV exactly: {keys_08 == csv_cols_qwen}")

print("\nComparison 09_infer_huatuo with Huatuo CSV columns:")
print(f"Keys in 09 but missing in CSV: {set(keys_09) - set(csv_cols_huatuo)}")
print(f"Cols in CSV but missing in 09: {set(csv_cols_huatuo) - set(keys_09)}")
print(f"Keys order matches CSV exactly: {keys_09 == csv_cols_huatuo}")
