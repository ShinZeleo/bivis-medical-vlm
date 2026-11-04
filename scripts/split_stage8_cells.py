import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def split_stage_8_in_notebook(nb_path):
    print(f"🔧 Splitting Stage 8 in: {nb_path}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    c8_idx = 8
    c8 = nb["cells"][c8_idx]
    src = c8["source"]
    
    # Verify split locations
    split1 = None
    split2 = None
    for i, line in enumerate(src):
        if "if len(df_queue) > 0" in line:
            split1 = i
        if "# STAGE 8B: PILOT VERIFICATION" in line:
            split2 = i
            
    assert split1 is not None and split2 is not None, f"Could not find split points in {nb_path}"
    
    # 1. Part 8A: Queue Setup & Pilot Stratification
    part_8a_lines = [
        "# ==============================================================================\n",
        "# STAGE 8A: INFERENCE QUEUE SETUP & STRATIFIED PILOT SAMPLING\n",
        "# ==============================================================================\n",
        "import os\n",
        "import shutil\n",
        "import time\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "def atomic_save(new_records, dest_path, df_base=None):\n",
        "    df_new = pd.DataFrame(new_records)\n",
        "    if df_base is not None and len(df_base) > 0:\n",
        "        df_all = pd.concat([df_base, df_new], ignore_index=True)\n",
        "    else:\n",
        "        df_all = df_new\n",
        "    df_all = df_all.drop_duplicates(subset=[\"qa_id\"], keep=\"last\").reset_index(drop=True)\n",
        "    tmp_path = dest_path.parent / f\"{dest_path.stem}.tmp_{os.getpid()}.csv\"\n",
        "    df_all.to_csv(tmp_path, index=False)\n",
        "    shutil.move(tmp_path, dest_path)\n",
        "    return df_all\n",
        "\n",
        "# Enforce active evaluation cohort dataframe source\n",
        "cohort_source = df_eval_cohort if \"df_eval_cohort\" in globals() else (df_manifest if \"df_manifest\" in globals() else None)\n",
        "assert cohort_source is not None, \"❌ Dataframe kohort evaluasi tidak ditemukan! Jalankan Sel 3 terlebih dahulu.\"\n",
        "\n",
        "if PILOT_MODE:\n",
        "    checkpoint_dest = MODEL_OUT_DIR / f\"pilot_checkpoint_{MODEL_SLUG}.csv\"\n",
        f"    print(f\"🔬 PILOT PREFLIGHT MODE ACTIVE: Stratifying {{PILOT_SAMPLE_COUNT}} samples (12 Dev + 28 Test, Seed: {{RANDOM_SEED}})...\")\n",
        "    dev_pool = cohort_source[(cohort_source[\"split\"] == \"dev\") & (cohort_source[\"is_eligible_primary\"] == True)]\n",
        "    test_pool = cohort_source[(cohort_source[\"split\"] == \"test\") & (cohort_source[\"is_eligible_primary\"] == True)]\n",
        "    n_dev_sample = min(12, len(dev_pool))\n",
        "    n_test_sample = min(28, len(test_pool))\n",
        "    pilot_dev = dev_pool.sample(n=n_dev_sample, random_state=RANDOM_SEED)\n",
        "    pilot_test = test_pool.sample(n=n_test_sample, random_state=RANDOM_SEED)\n",
        "    df_queue = pd.concat([pilot_dev, pilot_test]).sort_values(\"qa_id\").reset_index(drop=True)\n",
        "else:\n",
        "    checkpoint_dest = MODEL_OUT_DIR / f\"live_validation_checkpoint_{MODEL_SLUG}.csv\"\n",
        "    df_queue = df_remaining\n",
        "\n",
        "print(f\"📁 Target Checkpoint Path : {checkpoint_dest}\")\n",
        "print(f\"🎯 Antrean Sampel Diproses: {len(df_queue)} sampel\")\n",
        "print(f\"   Breakdown Split        : {dict(df_queue['split'].value_counts())}\\n\")\n"
    ]
    
    # 2. Part 8B: Resilient Inference Loop
    part_8b_lines = [
        "# ==============================================================================\n",
        "# STAGE 8B: RESILIENT INFERENCE LOOP (KV-CACHE FORWARD & ATOMIC FLUSH)\n",
        "# ==============================================================================\n",
        "from tqdm.auto import tqdm\n",
        "import time\n",
        "import torch\n",
        "\n",
        "new_records = []\n",
        "\n"
    ]
    # append lines from split1 to split2
    part_8b_lines.extend(src[split1:split2])
    
    # 3. Part 8C: Pilot Verification Suite
    part_8c_lines = [
        "# ==============================================================================\n",
        "# STAGE 8C: PILOT PREFLIGHT EVALUATION & HARD GATE DECISION\n",
        "# ==============================================================================\n"
    ]
    # append remaining lines
    # Ensure df_pilot loads from checkpoint if new_records empty
    rest_lines = src[split2+3:] # skip the old header lines
    # replace first line if PILOT_MODE:
    part_8c_lines.append("if PILOT_MODE:\n")
    part_8c_lines.append("    if len(new_records) == 0 and checkpoint_dest.exists():\n")
    part_8c_lines.append("        df_pilot = pd.read_csv(checkpoint_dest)\n")
    part_8c_lines.append("    else:\n")
    part_8c_lines.append("        df_pilot = pd.DataFrame(new_records)\n")
    
    # append after df_pilot assignment
    for l in rest_lines:
        if "df_pilot = pd.DataFrame(new_records)" in l:
            continue
        part_8c_lines.append(l)
        
    cell_8a = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": part_8a_lines
    }
    cell_8b = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": part_8b_lines
    }
    cell_8c = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": part_8c_lines
    }
    
    # Replace cell 8 with the three cells
    new_cells = nb["cells"][:c8_idx] + [cell_8a, cell_8b, cell_8c] + nb["cells"][c8_idx+1:]
    nb["cells"] = new_cells
    
    # Strip any outputs
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
            
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
    print(f"✅ Successfully split into 3 cells in {nb_path}. Total cells now: {len(nb['cells'])}")

split_stage_8_in_notebook(Path("notebooks/kaggle/rev06/08_qwen_decisive_validation.ipynb"))
split_stage_8_in_notebook(Path("notebooks/kaggle/rev06/09_huatuo_decisive_validation.ipynb"))
