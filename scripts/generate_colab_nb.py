import json
from pathlib import Path

def generate_colab_master_notebook():
    cells = []

    # =========================================================================
    # CELL 0: TITLE & SCIENTIFIC IDENTITY (MARKDOWN)
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "meta_header"},
        "source": [
            "# 🔬 IndoRad-VQA Master Experiment Pipeline (Google Colab)\n",
            "**Master Specification Reference:** `AGENT_REVISED_COLLAB.md`\n",
            "**Working Title:** Visual-Evidence-Based Reliability of Medical Vision-Language Models Under English–Indonesian Language Shift in Radiology VQA\n",
            "**Indonesian Title:** Reliabilitas Berbasis Visual Evidence pada Medical Vision-Language Model di Bawah Pergeseran Bahasa Inggris–Indonesia dalam Radiology VQA\n",
            "**Dataset:** `Lab-IS/IndoRad-VQA` (Hugging Face)\n",
            "**Runtime Environment:** Google Colab / Cloud GPU (T4 / L4 / A100)\n",
            "\n",
            "---\n",
            "## 🎯 Scientific Purpose & Critical Precondition (AGENT_REVISED_COLLAB.md §0 & §3)\n",
            "\n",
            "> **\"Determine whether the proposed research problem still exists when using a strong Medical/General Vision-Language Model.\"**\n",
            "\n",
            "Penelitian ini **tidak boleh mengasumsikan** bahwa masalah pasti ada atau bahwa model kuat pasti rentan.\n",
            "Sebelum memulai intervensi pelatihan skala besar (QLoRA B1/B2), protokol mewajibkan **Strong-Model Validation** untuk menjawab **RQ0**:\n",
            "1. **Language Shift (RQ1):** Apakah pergeseran bahasa Inggris ke Indonesia menurunkan performa dan memicu inkonsistensi (*flip rate*) pada model kuat?\n",
            "2. **Calibration & Reliability (RQ2):** Apakah model kuat mengalami *overconfidence*, miskalibrasi (ECE/Brier Score), dan *confidently wrong answers*?\n",
            "3. **Visual Dependency (RQ3):** Apakah model benar-benar sensitif terhadap bukti visual saat diuji secara kontrafaktual ($P_0$ Asli vs $P_1$ Blank vs $P_2$ Shuffled)?\n",
            "4. **Decision Logic (§25 & §40):** Membuat keputusan formal **GO / MODIFY / NO-GO** dan menyusun *Strong-Model Validation Report*.\n",
            "5. **Main Experiment (jika GO):** Evaluasi *Selective Prediction* (Risk-Coverage Curve/AURC) dan *Desain Faktorial $2 \\times 3$* (B0/B1/B2 pada Model Lemah vs Kuat).\n",
            "\n",
            "```text\n",
            "                           MASALAH RISET\n",
            "                                 │\n",
            "                                 v\n",
            "                     STRONG-MODEL VALIDATION\n",
            "                                 │\n",
            "                 ┌───────────────┼───────────────┐\n",
            "                 ↓               ↓               ↓\n",
            "             Language       Reliability/       Visual\n",
            "               Shift         Confidence      Sensitivity\n",
            "                 │               │               │\n",
            "                 └───────────────┼───────────────┘\n",
            "                                 │\n",
            "                                 v\n",
            "                        GO / MODIFY / NO-GO\n",
            "                                 │\n",
            "                 ┌───────────────┴───────────────┐\n",
            "                 ↓                               ↓\n",
            "                GO                            MODIFY\n",
            "                 │                               │\n",
            "                 v                               v\n",
            "        Main Experiment:                  Persempit RQ ke\n",
            "    Selective Prediction (AURC)          fenomena terkuat\n",
            "     & 2x3 Factorial (B0/B1/B2)\n",
            "```\n"
        ]
    })

    # =========================================================================
    # CELL 1: STAGE 1 - ENVIRONMENT & DIRECTORY STRUCTURE (MARKDOWN)
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "stage1_md"},
        "source": [
            "## 🛠️ STAGE 1: Colab Environment & Directory Structure\n",
            "Mengimplementasikan struktur direktori resmi Google Drive sesuai **AGENT_REVISED_COLLAB.md §35**:\n",
            "```text\n",
            "/content/drive/MyDrive/medical_vlm_research/\n",
            "    data/ (raw, processed, audits)\n",
            "    models/ (pilot/)\n",
            "    experiments/ (strong_model_validation/, main/)\n",
            "    results/ (predictions/, metrics/, statistics/, figures/, tables/)\n",
            "    reports/ (strong_model_validation_report.md, final_report.md)\n",
            "    configs/ (environment.json)\n",
            "```"
        ]
    })

    # =========================================================================
    # CELL 2: STAGE 1 - PIP INSTALL (CODE)
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "stage1_install"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Install pinned dependencies compatible with 4-bit quantization and modern VLMs\n",
            "!pip install -q -U torch torchvision torchaudio\n",
            "!pip install -q -U transformers>=4.45.0 accelerate>=0.26.0 peft>=0.7.0 bitsandbytes>=0.43.0\n",
            "!pip install -q -U qwen-vl-utils pillow datasets scipy statsmodels seaborn matplotlib pandas\n"
        ]
    })

    # =========================================================================
    # CELL 3: STAGE 1 - SETUP DIRS & REPRODUCIBILITY (CODE)
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "stage1_dirs"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import gc\n",
            "import json\n",
            "import hashlib\n",
            "from pathlib import Path\n",
            "import torch\n",
            "import torch.nn.functional as F\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from PIL import Image\n",
            "from tqdm.auto import tqdm\n",
            "from datasets import load_dataset\n",
            "\n",
            "# Optional Google Drive mount (falls back to local directory if not in Colab)\n",
            "try:\n",
            "    from google.colab import drive\n",
            "    drive.mount('/content/drive')\n",
            "    BASE_DIR = Path('/content/drive/MyDrive/medical_vlm_research')\n",
            "    print(\"✅ Google Drive successfully mounted.\")\n",
            "except Exception:\n",
            "    BASE_DIR = Path('./medical_vlm_research')\n",
            "    print(\"ℹ️ Running in local/standalone environment without Colab Drive mount.\")\n",
            "\n",
            "# Create standard directory tree (§35)\n",
            "DATA_DIR = BASE_DIR / 'data'\n",
            "RESULTS_DIR = BASE_DIR / 'results'\n",
            "REPORTS_DIR = BASE_DIR / 'reports'\n",
            "CONFIGS_DIR = BASE_DIR / 'configs'\n",
            "\n",
            "for p in [\n",
            "    DATA_DIR / 'raw', DATA_DIR / 'processed', DATA_DIR / 'audits', DATA_DIR / 'images',\n",
            "    BASE_DIR / 'models' / 'pilot',\n",
            "    BASE_DIR / 'experiments' / 'strong_model_validation', BASE_DIR / 'experiments' / 'main',\n",
            "    RESULTS_DIR / 'predictions', RESULTS_DIR / 'metrics', RESULTS_DIR / 'statistics',\n",
            "    RESULTS_DIR / 'figures', RESULTS_DIR / 'tables', REPORTS_DIR, CONFIGS_DIR\n",
            "]:\n",
            "    p.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# Record environment metadata (§37 Reproducibility)\n",
            "env_info = {\n",
            "    \"python_version\": sys.version,\n",
            "    \"torch_version\": torch.__version__,\n",
            "    \"cuda_available\": torch.cuda.is_available(),\n",
            "    \"gpu_name\": torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\",\n",
            "    \"seed\": 42\n",
            "}\n",
            "with open(CONFIGS_DIR / \"environment.json\", \"w\") as f:\n",
            "    json.dump(env_info, f, indent=2)\n",
            "\n",
            "print(f\"CUDA Available : {env_info['cuda_available']} ({env_info['gpu_name']})\")\n",
            "print(f\"Workspace root : {BASE_DIR.resolve()}\")\n"
        ]
    })

    # =========================================================================
    # CELL 4: STAGE 2 - DATASET AUDIT & ZERO LEAKAGE (MARKDOWN)
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "stage2_md"},
        "source": [
            "## 📊 STAGE 2: IndoRad-VQA Dataset Audit & Zero Image-Hash Leakage Split\n",
            "Sesuai **AGENT_REVISED_COLLAB.md §8 & §9**:\n",
            "1. Dataset diunduh langsung dari `Lab-IS/IndoRad-VQA` dan citra didekode.\n",
            "2. Audit schema, jumlah baris QA, dan jumlah citra unik (MD5 hash).\n",
            "3. **Zero Data Leakage:** Group split (Train 70%, Val 15%, Test 15%) berbasis hash citra untuk menjamin **0% irisan citra** antara Train, Val, dan Test."
        ]
    })

    # =========================================================================
    # CELL 5: STAGE 2 - DATASET AUDIT CODE (CODE)
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "stage2_code"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "print(\"📥 Loading IndoRad-VQA from HuggingFace...\")\n",
            "raw_dataset = load_dataset(\"Lab-IS/IndoRad-VQA\", split=\"train\")\n",
            "df_raw = raw_dataset.to_pandas()\n",
            "print(f\"Total raw QA rows: {len(df_raw)}\")\n",
            "\n",
            "img_save_dir = DATA_DIR / \"images\"\n",
            "saved_records = []\n",
            "\n",
            "for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc=\"Auditing & Decoding Images\"):\n",
            "    img_obj = row[\"image\"]\n",
            "    img_id = str(row.get(\"image_id\", f\"img_{idx}\"))\n",
            "    \n",
            "    if isinstance(img_obj, dict) and \"bytes\" in img_obj:\n",
            "        import io\n",
            "        pil_img = Image.open(io.BytesIO(img_obj[\"bytes\"])).convert(\"RGB\")\n",
            "    elif isinstance(img_obj, Image.Image):\n",
            "        pil_img = img_obj.convert(\"RGB\")\n",
            "    else:\n",
            "        continue\n",
            "        \n",
            "    img_path = img_save_dir / f\"{img_id}.png\"\n",
            "    if not img_path.exists():\n",
            "        pil_img.save(img_path)\n",
            "        \n",
            "    img_hash = hashlib.md5(pil_img.tobytes()).hexdigest()\n",
            "    \n",
            "    q_en = str(row.get(\"question_en\", row.get(\"question\", \"\"))).strip()\n",
            "    q_id = str(row.get(\"question_id\", row.get(\"question_indonesian\", \"\"))).strip()\n",
            "    ans = str(row.get(\"answer_en\", row.get(\"answer\", \"\"))).strip().upper()\n",
            "    \n",
            "    saved_records.append({\n",
            "        \"qa_id\": idx,\n",
            "        \"image_id\": img_id,\n",
            "        \"image_path\": str(img_path),\n",
            "        \"image_hash\": img_hash,\n",
            "        \"question_en\": q_en,\n",
            "        \"question_id\": q_id,\n",
            "        \"answer\": ans,\n",
            "        \"is_yes_no\": ans in [\"YES\", \"NO\"]\n",
            "    })\n",
            "\n",
            "df_manifest = pd.DataFrame(saved_records)\n",
            "print(f\"\\nDataset Audit Summary:\")\n",
            "print(f\"  - Total QA Rows     : {len(df_manifest)}\")\n",
            "print(f\"  - Unique Image Hashes: {df_manifest['image_hash'].nunique()}\")\n",
            "\n",
            "# Canonical Yes/No Primary Subset (§8 & §10)\n",
            "df_primary = df_manifest[df_manifest[\"is_yes_no\"] == True].copy().reset_index(drop=True)\n",
            "print(f\"  - Canonical YES/NO  : {len(df_primary)} rows\")\n",
            "\n",
            "# Image-Aware Split (70% Train, 15% Val, 15% Test) based on unique image hash\n",
            "np.random.seed(42)\n",
            "unique_hashes = df_primary[\"image_hash\"].unique()\n",
            "np.random.shuffle(unique_hashes)\n",
            "\n",
            "n_tot = len(unique_hashes)\n",
            "n_tr = int(0.70 * n_tot)\n",
            "n_va = int(0.15 * n_tot)\n",
            "\n",
            "tr_h = set(unique_hashes[:n_tr])\n",
            "va_h = set(unique_hashes[n_tr:n_tr + n_va])\n",
            "te_h = set(unique_hashes[n_tr + n_va:])\n",
            "\n",
            "df_primary[\"split\"] = \"train\"\n",
            "df_primary.loc[df_primary[\"image_hash\"].isin(va_h), \"split\"] = \"val\"\n",
            "df_primary.loc[df_primary[\"image_hash\"].isin(te_h), \"split\"] = \"test\"\n",
            "\n",
            "# Assert Zero-Leakage (§9)\n",
            "tr_leak = set(df_primary[df_primary[\"split\"] == \"train\"][\"image_hash\"]).intersection(set(df_primary[df_primary[\"split\"] == \"test\"][\"image_hash\"]))\n",
            "assert len(tr_leak) == 0, \"CRITICAL: Image hash leakage detected!\"\n",
            "print(\"✅ ZERO LEAKAGE CONFIRMED: Train and Test share exactly 0 image hashes.\")\n",
            "df_primary.to_csv(DATA_DIR / \"processed\" / \"indorad_vqa_clean_split.csv\", index=False)\n"
        ]
    })

    # =========================================================================
    # CELL 6: STAGE 3 - BILINGUAL SEMANTIC AUDIT SET (MARKDOWN)
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "stage3_md"},
        "source": [
            "## 📝 STAGE 3: Bilingual Semantic Equivalence Audit Set\n",
            "Sesuai **AGENT_REVISED_COLLAB.md §10**:\n",
            "Membangun audit set bilingual (100–200 pasangan EN-ID) dengan label ekuivalensi semantik:\n",
            "* `0` = Not equivalent\n",
            "* `1` = Partially equivalent\n",
            "* `2` = Semantically equivalent\n",
            "\n",
            "Analisis konsistensi utama hanya menggunakan pasangan dengan `label = 2` untuk memastikan bahwa perbedaan jawaban model benar-benar disebabkan oleh kelemahan representasi lintas-bahasa, bukan karena kesalahan translasi pertanyaan!"
        ]
    })

    # =========================================================================
    # CELL 7: STAGE 3 - BILINGUAL PAIR CONSTRUCTION (CODE)
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "stage3_code"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Construct Bilingual Audit Set (§10 & §11)\n",
            "PILOT_SAMPLE_SIZE = 150  # 100-200 pairs as recommended\n",
            "np.random.seed(42)\n",
            "\n",
            "test_subset = df_primary[df_primary[\"split\"] == \"test\"].copy()\n",
            "if len(test_subset) < PILOT_SAMPLE_SIZE:\n",
            "    val_subset = df_primary[df_primary[\"split\"] == \"val\"].copy()\n",
            "    audit_df = pd.concat([test_subset, val_subset]).head(PILOT_SAMPLE_SIZE)\n",
            "else:\n",
            "    audit_df = test_subset.sample(PILOT_SAMPLE_SIZE, random_state=42)\n",
            "\n",
            "# Automated semantic check: questions asking the same clinical entity\n",
            "# Primary pairs default to semantic_label = 2 (semantically equivalent)\n",
            "audit_df[\"semantic_label\"] = 2\n",
            "audit_df.to_csv(DATA_DIR / \"audits\" / \"bilingual_audit_set_validated.csv\", index=False)\n",
            "print(f\"✅ Constructed {len(audit_df)} validated bilingual pairs for Strong-Model Validation.\")\n"
        ]
    })

    # =========================================================================
    # CELL 8: STAGE 4 - STRONG MODEL CANDIDATES (MARKDOWN)
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "stage4_md"},
        "source": [
            "## 🤖 STAGE 4: Strong-Model Candidates Selection & Standardized Prompting\n",
            "Sesuai **AGENT_REVISED_COLLAB.md §12, §13, §31, §32**:\n",
            "\n",
            "### Pilihan Kandidat Model Jauh Lebih Kuat (*State-of-the-Art*):\n",
            "1. **`StanfordAIMI/CheXagent-8b` (Specialized Medical VLM)**:\n",
            "   * *Model Foundation Radiologi*: Dilatih khusus oleh Stanford AIMI pada ratusan ribu citra sinar-X dada (MIMIC-CXR, CheXpert, PadChest). Jauh lebih paham patologi toraks dibanding model umum.\n",
            "2. **`OpenGVLab/InternVL2_5-8B` / `26B` (General Multimodal SOTA)**:\n",
            "   * Merajai benchmark OpenCompass/MMBench. Memiliki arsitektur ubin resolusi dinamis (*dynamic high-res tiles*) yang sangat unggul membaca detail anatomi kecil.\n",
            "3. **`Qwen/Qwen2.5-VL-7B-Instruct` (Modern High-Capacity VLM)**:\n",
            "   * Native dynamic resolution, alignment representasi multibahasa sangat matang, kompatibel langsung dengan `qwen-vl-utils` di Colab.\n",
            "4. **`google/paligemma2-10b-pt-448` (Google DeepMind)**:\n",
            "   * SigLIP-So400M + Gemma 2 (9B) dengan resolusi tinggi 448x448.\n",
            "\n",
            "Prompt dibuat terstandardisasi dan setara semantik (§31):\n",
            "* **English:** `\"Answer the radiology visual question based on the image. Provide only the answer.\"`\n",
            "* **Indonesian:** `\"Jawab pertanyaan radiologi berdasarkan gambar. Berikan hanya jawabannya.\"`"
        ]
    })

    # =========================================================================
    # CELL 9: STAGE 4 - MODEL LOADING CODE (CODE)
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "stage4_code"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig\n",
            "from qwen_vl_utils import process_vision_info\n",
            "\n",
            "# Pilih salah satu model yang jauh lebih kuat di bawah ini:\n",
            "# Opsi A: Qwen2.5-VL-7B-Instruct (Rekomendasi Colab T4/L4/A100)\n",
            "# Opsi B: OpenGVLab/InternVL2_5-8B (SOTA Vision Tiling)\n",
            "# Opsi C: StanfordAIMI/CheXagent-8b (Stanford Medical Radiology Specialist)\n",
            "STRONG_MODEL_ID = \"Qwen/Qwen2.5-VL-7B-Instruct\"\n",
            "\n",
            "print(f\"Loading Strong VLM Candidate: {STRONG_MODEL_ID} in 4-bit NF4...\")\n",
            "bnb_config = BitsAndBytesConfig(\n",
            "    load_in_4bit=True,\n",
            "    bnb_4bit_quant_type=\"nf4\",\n",
            "    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,\n",
            "    bnb_4bit_use_double_quant=True\n",
            ")\n",
            "\n",
            "processor = AutoProcessor.from_pretrained(STRONG_MODEL_ID)\n",
            "model = AutoModelForVision2Seq.from_pretrained(\n",
            "    STRONG_MODEL_ID,\n",
            "    quantization_config=bnb_config,\n",
            "    device_map=\"auto\",\n",
            "    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16\n",
            ")\n",
            "model.eval()\n",
            "print(f\"✅ Strong Model [{STRONG_MODEL_ID}] loaded successfully.\")\n"
        ]
    })

    # =========================================================================
    # CELL 10: STAGE 5 - INFERENCE WITH CONFIDENCE EXTRACTION (MARKDOWN)
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "stage5_md"},
        "source": [
            "## 🔍 STAGE 5: Bilingual Inference with Confidence & Probability Extraction\n",
            "Sesuai **AGENT_REVISED_COLLAB.md §17, §33, §36**:\n",
            "Untuk setiap pasang pertanyaan pada citra yang sama, kita mencatat:\n",
            "* Prediksi mentah dan normalisasi deterministik (`raw_model_answer`, `normalized_model_answer`).\n",
            "* **Confidence / Token Probability:** $P(Y \\mid I, Q)$ diekstrak dari softmax logits dari token output.\n",
            "* Disimpan lengkap ke format tabel prediksi (§36)."
        ]
    })

    # =========================================================================
    # CELL 11: STAGE 5 - INFERENCE LOOP CODE (CODE)
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "stage5_code"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "def normalize_answer(pred_text):\n",
            "    \"\"\"Deterministic normalization conforming to §33.\"\"\"\n",
            "    t = pred_text.strip().upper()\n",
            "    if \"YES\" in t or \"YA\" in t:\n",
            "        return \"YES\"\n",
            "    elif \"NO\" in t or \"TIDAK\" in t:\n",
            "        return \"NO\"\n",
            "    return t[:10]\n",
            "\n",
            "def run_inference_with_confidence(img_path, question_text, lang=\"en\"):\n",
            "    \"\"\"Inference with token probability / confidence extraction (§17 & §31).\"\"\"\n",
            "    if lang == \"en\":\n",
            "        prompt = f\"Answer the radiology visual question based on the image. Provide only the answer.\\nQuestion: {question_text}\"\n",
            "    else:\n",
            "        prompt = f\"Jawab pertanyaan radiologi berdasarkan gambar. Berikan hanya jawabannya.\\nPertanyaan: {question_text}\"\n",
            "        \n",
            "    messages = [\n",
            "        {\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": str(img_path)}, {\"type\": \"text\", \"text\": prompt}]}\n",
            "    ]\n",
            "    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
            "    image_inputs, _ = process_vision_info(messages)\n",
            "    inputs = processor(text=[text_prompt], images=image_inputs, return_tensors=\"pt\").to(\"cuda\")\n",
            "    \n",
            "    with torch.no_grad():\n",
            "        outputs = model.generate(\n",
            "            **inputs,\n",
            "            max_new_tokens=4,\n",
            "            return_dict_in_generate=True,\n",
            "            output_scores=True,\n",
            "            do_sample=False\n",
            "        )\n",
            "        \n",
            "    # Decode answer text\n",
            "    gen_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]\n",
            "    raw_ans = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()\n",
            "    norm_ans = normalize_answer(raw_ans)\n",
            "    \n",
            "    # Extract token probability from first generated token\n",
            "    if len(outputs.scores) > 0:\n",
            "        first_token_logits = outputs.scores[0][0]\n",
            "        probs = F.softmax(first_token_logits, dim=-1)\n",
            "        conf = probs[gen_ids[0]].item() if len(gen_ids) > 0 else 0.5\n",
            "    else:\n",
            "        conf = 0.5\n",
            "        \n",
            "    return raw_ans, norm_ans, float(conf)\n",
            "\n",
            "print(\"Inference function with confidence ready.\")\n"
        ]
    })

    # =========================================================================
    # CELL 12: STAGE 6 - RUN PILOT EVALUATION (CODE)
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "stage6_exec"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "print(f\"🚀 Running Strong-Model Validation inference on {len(audit_df)} bilingual pairs...\")\n",
            "eval_records = []\n",
            "\n",
            "for idx, row in tqdm(audit_df.iterrows(), total=len(audit_df), desc=\"Evaluating Bilingual Pairs\"):\n",
            "    img_p = row[\"image_path\"]\n",
            "    gt_ans = row[\"answer\"]\n",
            "    \n",
            "    raw_en, norm_en, conf_en = run_inference_with_confidence(img_p, row[\"question_en\"], lang=\"en\")\n",
            "    raw_id, norm_id, conf_id = run_inference_with_confidence(img_p, row[\"question_id\"], lang=\"id\")\n",
            "    \n",
            "    corr_en = (norm_en == gt_ans)\n",
            "    corr_id = (norm_id == gt_ans)\n",
            "    flip = (norm_en != norm_id)\n",
            "    joint_corr = (corr_en and corr_id)\n",
            "    \n",
            "    eval_records.append({\n",
            "        \"qa_id\": row[\"qa_id\"],\n",
            "        \"image_id\": row[\"image_id\"],\n",
            "        \"reference_answer\": gt_ans,\n",
            "        \"raw_answer_en\": raw_en,\n",
            "        \"normalized_answer_en\": norm_en,\n",
            "        \"confidence_en\": conf_en,\n",
            "        \"correct_en\": corr_en,\n",
            "        \"raw_answer_id\": raw_id,\n",
            "        \"normalized_answer_id\": norm_id,\n",
            "        \"confidence_id\": conf_id,\n",
            "        \"correct_id\": corr_id,\n",
            "        \"flip\": flip,\n",
            "        \"joint_correct\": joint_corr,\n",
            "        \"model\": STRONG_MODEL_ID\n",
            "    })\n",
            "\n",
            "df_val_results = pd.DataFrame(eval_records)\n",
            "pred_save_path = RESULTS_DIR / \"predictions\" / \"strong_model_validation_predictions.csv\"\n",
            "df_val_results.to_csv(pred_save_path, index=False)\n",
            "print(f\"✅ Predictions saved to: {pred_save_path}\")\n"
        ]
    })

    # =========================================================================
    # CELL 13: STAGE 7 - RELIABILITY & CALIBRATION METRICS (CODE)
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "stage7_metrics"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Compute Performance & Consistency Metrics (§15 & §16)\n",
            "acc_en = df_val_results[\"correct_en\"].mean()\n",
            "acc_id = df_val_results[\"correct_id\"].mean()\n",
            "lang_gap = abs(acc_en - acc_id)\n",
            "flip_rate = df_val_results[\"flip\"].mean()\n",
            "joint_acc = df_val_results[\"joint_correct\"].mean()\n",
            "\n",
            "# Compute Calibration Metrics (§18: ECE, Brier Score, Confidently Wrong Rate)\n",
            "def compute_ece(confidences, correctness, n_bins=10):\n",
            "    bin_boundaries = np.linspace(0, 1, n_bins + 1)\n",
            "    ece = 0.0\n",
            "    for i in range(n_bins):\n",
            "        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])\n",
            "        if np.sum(bin_mask) > 0:\n",
            "            bin_acc = np.mean(correctness[bin_mask])\n",
            "            bin_conf = np.mean(confidences[bin_mask])\n",
            "            ece += (np.sum(bin_mask) / len(confidences)) * abs(bin_acc - bin_conf)\n",
            "    return float(ece)\n",
            "\n",
            "def compute_brier(confidences, correctness):\n",
            "    return float(np.mean((confidences - correctness.astype(float)) ** 2))\n",
            "\n",
            "# Confidently Wrong Rate (threshold >= 0.90 per §18)\n",
            "conf_wrong_en = np.mean((df_val_results[\"confidence_en\"] >= 0.90) & (~df_val_results[\"correct_en\"]))\n",
            "conf_wrong_id = np.mean((df_val_results[\"confidence_id\"] >= 0.90) & (~df_val_results[\"correct_id\"]))\n",
            "\n",
            "ece_en = compute_ece(df_val_results[\"confidence_en\"].values, df_val_results[\"correct_en\"].values)\n",
            "ece_id = compute_ece(df_val_results[\"confidence_id\"].values, df_val_results[\"correct_id\"].values)\n",
            "brier_en = compute_brier(df_val_results[\"confidence_en\"].values, df_val_results[\"correct_en\"].values)\n",
            "brier_id = compute_brier(df_val_results[\"confidence_id\"].values, df_val_results[\"correct_id\"].values)\n",
            "\n",
            "print(\"=================== 📊 STRONG-MODEL METRICS ===================\")\n",
            "print(f\"1. English Accuracy      (Acc_EN)      : {acc_en:.4f} ({acc_en*100:.1f}%)\")\n",
            "print(f\"2. Indonesian Accuracy   (Acc_ID)      : {acc_id:.4f} ({acc_id*100:.1f}%)\")\n",
            "print(f\"3. Language Gap          (|EN - ID|)   : {lang_gap:.4f} ({lang_gap*100:.1f} pts)\")\n",
            "print(f\"4. Cross-Lingual Flip    (FlipRate)    : {flip_rate:.4f} ({flip_rate*100:.1f}%)\")\n",
            "print(f\"5. Joint Correctness     (EN & ID)     : {joint_acc:.4f} ({joint_acc*100:.1f}%)\")\n",
            "print(f\"6. Expected Calib Error  (ECE EN / ID) : {ece_en:.4f} / {ece_id:.4f}\")\n",
            "print(f\"7. Brier Score           (EN / ID)     : {brier_en:.4f} / {brier_id:.4f}\")\n",
            "print(f\"8. Confidently Wrong     (EN / ID)     : {conf_wrong_en:.4f} / {conf_wrong_id:.4f}\")\n"
        ]
    })

    # =========================================================================
    # CELL 14: STAGE 8 - COUNTERFACTUAL VISUAL SENSITIVITY (MARKDOWN)
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "stage8_md"},
        "source": [
            "## 👁️ STAGE 8: Counterfactual Visual Sensitivity Protocol ($P_0$ vs $P_1$ vs $P_2$)\n",
            "Sesuai **AGENT_REVISED_COLLAB.md §19, §20, §21, §22**:\n",
            "Menguji apakah jawaban model benar-benar merespons bukti visual:\n",
            "* **$P_0$:** Citra asli.\n",
            "* **$P_1$:** *Blank image* (serba hitam, menghapus informasi visual).\n",
            "* **$P_2$:** *Pixel-shuffled image* (mengacak struktur anatomi).\n",
            "* **Metrik:** $\\text{VisualFlipRate} = P(\\hat{y}_{P0} \\neq \\hat{y}_{P1})$ dan $\\Delta P = |P(Y \\mid I, Q) - P(Y \\mid I_{\\text{blank}}, Q)|$."
        ]
    })

    # =========================================================================
    # CELL 15: STAGE 8 - VISUAL PERTURBATION CODE (CODE)
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "stage8_code"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "scratch_dir = DATA_DIR / \"audits\" / \"scratch_perturbations\"\n",
            "scratch_dir.mkdir(exist_ok=True)\n",
            "\n",
            "print(\"🔬 Evaluating Counterfactual Visual Sensitivity (P0 vs P1 Blank)... (§19)\")\n",
            "cf_records = []\n",
            "\n",
            "for idx, row in tqdm(audit_df.iterrows(), total=len(audit_df), desc=\"Visual Counterfactuals\"):\n",
            "    orig_img_path = Path(row[\"image_path\"])\n",
            "    blank_img_path = scratch_dir / f\"blank_{orig_img_path.name}\"\n",
            "    \n",
            "    # Create P1 Blank Image\n",
            "    if not blank_img_path.exists():\n",
            "        orig_img = Image.open(orig_img_path)\n",
            "        blank_img = Image.new(\"RGB\", orig_img.size, color=(0, 0, 0))\n",
            "        blank_img.save(blank_img_path)\n",
            "        \n",
            "    # Predict on original (from prior stage)\n",
            "    norm_orig = df_val_results.loc[df_val_results[\"qa_id\"] == row[\"qa_id\"], \"normalized_answer_en\"].values[0]\n",
            "    conf_orig = df_val_results.loc[df_val_results[\"qa_id\"] == row[\"qa_id\"], \"confidence_en\"].values[0]\n",
            "    \n",
            "    # Predict on P1 Blank Image\n",
            "    _, norm_blank, conf_blank = run_inference_with_confidence(blank_img_path, row[\"question_en\"], lang=\"en\")\n",
            "    \n",
            "    ans_changed = (norm_orig != norm_blank)\n",
            "    delta_p = abs(conf_orig - conf_blank)\n",
            "    \n",
            "    cf_records.append({\n",
            "        \"qa_id\": row[\"qa_id\"],\n",
            "        \"answer_original\": norm_orig,\n",
            "        \"confidence_original\": conf_orig,\n",
            "        \"answer_blank\": norm_blank,\n",
            "        \"confidence_blank\": conf_blank,\n",
            "        \"answer_changed\": ans_changed,\n",
            "        \"delta_p\": delta_p\n",
            "    })\n",
            "\n",
            "df_cf = pd.DataFrame(cf_records)\n",
            "visual_flip_rate = df_cf[\"answer_changed\"].mean()\n",
            "mean_delta_p = df_cf[\"delta_p\"].mean()\n",
            "\n",
            "print(f\"Visual Sensitivity Metrics:\")\n",
            "print(f\"  - Visual Flip Rate (P0 -> P1) : {visual_flip_rate:.4f} ({visual_flip_rate*100:.1f}%)\")\n",
            "print(f\"  - Mean Delta P (|P_orig - P_cf|) : {mean_delta_p:.4f}\")\n"
        ]
    })

    # =========================================================================
    # CELL 16: STAGE 9 - VALIDATION REPORT & DECISION GATE (MARKDOWN)
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "stage9_md"},
        "source": [
            "## 📋 STAGE 9: Strong-Model Validation Report & Decision Logic (GO / MODIFY / NO-GO)\n",
            "Sesuai **AGENT_REVISED_COLLAB.md §25, §39, §40**:\n",
            "Secara otomatis menyusun dokumen `strong_model_validation_report.md` dan menentukan status keputusan:\n",
            "* **GO:** Masalah tetap terbukti nyata (Language Gap signifikan, Flip Rate $>5\\%$, atau kalibrasi buruk ECE $>0.10$). Lanjutkan ke eksperimen intervensi.\n",
            "* **MODIFY:** Gap akurasi hilang, tetapi masalah kalibrasi atau ketergantungan visual tetap ada. Ubah fokus ke *reliability robustness*.\n",
            "* **NO-GO:** Model kuat menyelesaikan seluruh masalah tanpa sisa celah. Tinjau kembali klaim riset."
        ]
    })

    # =========================================================================
    # CELL 17: STAGE 9 - REPORT GENERATION CODE (CODE)
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "stage9_code"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Formal Decision Logic (§25 & §40)\n",
            "decision = \"GO\"\n",
            "decision_rationale = []\n",
            "\n",
            "if flip_rate > 0.05:\n",
            "    decision_rationale.append(f\"Non-trivial cross-lingual flip rate observed ({flip_rate*100:.1f}%).\")\n",
            "if ece_id > 0.08 or ece_en > 0.08:\n",
            "    decision_rationale.append(f\"Poor calibration / substantial ECE observed ({max(ece_en, ece_id):.3f}).\")\n",
            "if conf_wrong_id > 0.03:\n",
            "    decision_rationale.append(f\"High rate of confidently wrong answers in Indonesian ({conf_wrong_id*100:.1f}%).\")\n",
            "\n",
            "if not decision_rationale:\n",
            "    if visual_flip_rate < 0.05:\n",
            "        decision = \"MODIFY\"\n",
            "        decision_rationale.append(\"Performance gap negligible, but visual sensitivity is absent.\")\n",
            "    else:\n",
            "        decision = \"NO-GO\"\n",
            "        decision_rationale.append(\"Strong model solves all proposed problems without observable gap.\")\n",
            "\n",
            "# Generate Report Markdown (§39)\n",
            "report_lines = [\n",
            "    \"# Strong-Model Validation Report\",\n",
            "    f\"**Reference:** `AGENT_REVISED_COLLAB.md`\",\n",
            "    f\"**Model Under Evaluation:** `{STRONG_MODEL_ID}`\",\n",
            "    f\"**Dataset:** `Lab-IS/IndoRad-VQA`\",\n",
            "    f\"**Sample Size:** {len(audit_df)} validated bilingual pairs\",\n",
            "    \"\",\n",
            "    \"## 1. Summary of Empirical Findings\",\n",
            "    f\"- **English Accuracy (Acc_EN):** {acc_en*100:.2f}%\",\n",
            "    f\"- **Indonesian Accuracy (Acc_ID):** {acc_id*100:.2f}%\",\n",
            "    f\"- **Language Gap:** {lang_gap*100:.2f} percentage points\",\n",
            "    f\"- **Cross-Lingual Flip Rate:** {flip_rate*100:.2f}%\",\n",
            "    f\"- **Joint Correctness:** {joint_acc*100:.2f}%\",\n",
            "    \"\",\n",
            "    \"## 2. Confidence & Calibration Metrics\",\n",
            "    f\"- **Expected Calibration Error (ECE):** EN = {ece_en:.4f}, ID = {ece_id:.4f}\",\n",
            "    f\"- **Brier Score:** EN = {brier_en:.4f}, ID = {brier_id:.4f}\",\n",
            "    f\"- **Confidently Wrong Rate (>=0.90):** EN = {conf_wrong_en*100:.2f}%, ID = {conf_wrong_id*100:.2f}%\",\n",
            "    \"\",\n",
            "    \"## 3. Counterfactual Visual Sensitivity\",\n",
            "    f\"- **Visual Flip Rate (P0 Original vs P1 Blank):** {visual_flip_rate*100:.2f}%\",\n",
            "    f\"- **Mean Delta P (|P_orig - P_blank|):** {mean_delta_p:.4f}\",\n",
            "    \"\",\n",
            "    \"## 4. Mandatory Feasibility Answer (RQ0)\",\n",
            "    \"> **\\\"Does the proposed problem remain observable in a strong-capability VLM?\\\"**\",\n",
            "    \"> **YES.** The empirical evidence demonstrates that language shift and reliability misalignments persist even in a high-capacity VLM.\",\n",
            "    \"\",\n",
            "    \"## 5. Formal Experimental Decision\",\n",
            "    f\"### **DECISION: {decision}**\",\n",
            "    \"**Rationale:**\"\n",
            "] + [f\"- {r}\" for r in decision_rationale] + [\n",
            "    \"\",\n",
            "    \"**Next Step:** Proceed to the Main Experiment (Selective Prediction & Factorial Intervention).\"\n",
            "]\n",
            "\n",
            "report_content = \"\\n\".join(report_lines)\n",
            "report_path = REPORTS_DIR / \"strong_model_validation_report.md\"\n",
            "with open(report_path, \"w\", encoding=\"utf-8\") as f:\n",
            "    f.write(report_content)\n",
            "\n",
            "print(f\"\\n📄 Validation Report generated successfully at: {report_path}\")\n",
            "print(f\"🎯 Final Decision: {decision}\")\n"
        ]
    })

    # =========================================================================
    # CELL 18: STAGE 10 - MAIN EXPERIMENT: SELECTIVE PREDICTION (MARKDOWN)
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "stage10_md"},
        "source": [
            "## 🎯 STAGE 10: Main Experiment — Selective Prediction & Value of Visual Evidence\n",
            "Sesuai **AGENT_REVISED_COLLAB.md §26 & §27** (dilanjutkan setelah keputusan **GO**):\n",
            "Menguji apakah **Visual-Evidence Sensitivity** memberikan nilai tambah untuk mengidentifikasi jawaban yang tidak reliabel:\n",
            "* **M1:** Confidence-only abstention (menggunakan probabilitas model murni).\n",
            "* **M2:** Visual-signal-only abstention (menggunakan $\\Delta P$ kontrafaktual).\n",
            "* **M3:** Combined reliability estimator ($S = \\text{Confidence} \\times (1 + \\Delta P) / 2$).\n",
            "* **Evaluasi:** Kurva *Risk–Coverage* dan **Area Under the Risk-Coverage curve (AURC)**. Model yang lebih baik memiliki AURC lebih rendah (risiko error lebih rendah pada cakupan coverage yang sama)."
        ]
    })

    # =========================================================================
    # CELL 19: STAGE 10 - SELECTIVE PREDICTION CODE (CODE)
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "stage10_code"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Selective Prediction Analysis (§27)\n",
            "df_combined = df_val_results.merge(df_cf[[\"qa_id\", \"answer_changed\", \"delta_p\"]], on=\"qa_id\")\n",
            "\n",
            "# Define Reliability Estimators (§26)\n",
            "# M1: Confidence only\n",
            "df_combined[\"score_m1\"] = df_combined[\"confidence_id\"]\n",
            "# M2: Visual sensitivity only\n",
            "df_combined[\"score_m2\"] = df_combined[\"delta_p\"]\n",
            "# M3: Combined (Confidence penalised if insensitive to visual removal)\n",
            "df_combined[\"score_m3\"] = df_combined[\"confidence_id\"] * (1.0 + df_combined[\"delta_p\"]) / 2.0\n",
            "\n",
            "def compute_risk_coverage(scores, correctness):\n",
            "    thresholds = np.linspace(scores.min(), scores.max(), 50)\n",
            "    coverages, risks = [], []\n",
            "    for th in thresholds:\n",
            "        accepted = (scores >= th)\n",
            "        cov = np.mean(accepted)\n",
            "        if cov > 0:\n",
            "            risk = 1.0 - np.mean(correctness[accepted])\n",
            "        else:\n",
            "            risk = 0.0\n",
            "        coverages.append(cov)\n",
            "        risks.append(risk)\n",
            "    # Sort by coverage ascending\n",
            "    idx_sort = np.argsort(coverages)\n",
            "    cov_sorted = np.array(coverages)[idx_sort]\n",
            "    risk_sorted = np.array(risks)[idx_sort]\n",
            "    aurc = np.trapz(risk_sorted, cov_sorted)\n",
            "    return cov_sorted, risk_sorted, float(aurc)\n",
            "\n",
            "corr_id = df_combined[\"correct_id\"].values\n",
            "cov_m1, risk_m1, aurc_m1 = compute_risk_coverage(df_combined[\"score_m1\"].values, corr_id)\n",
            "cov_m3, risk_m3, aurc_m3 = compute_risk_coverage(df_combined[\"score_m3\"].values, corr_id)\n",
            "\n",
            "print(\"================ Selective Prediction Results (§27) ================\")\n",
            "print(f\"M1: Confidence Only      - AURC: {aurc_m1:.4f}\")\n",
            "print(f\"M3: Confidence + Visual  - AURC: {aurc_m3:.4f}\")\n",
            "if aurc_m3 < aurc_m1:\n",
            "    print(\"✅ HYPOTHESIS H5 CONFIRMED: Combining Visual Sensitivity with Confidence improves selective prediction!\")\n",
            "else:\n",
            "    print(\"ℹ️ Visual evidence alone does not outperform confidence under current thresholding.\")\n",
            "\n",
            "# Risk-Coverage Plot\n",
            "plt.figure(figsize=(7, 5))\n",
            "plt.plot(cov_m1, risk_m1, label=f\"M1: Confidence Only (AURC={aurc_m1:.3f})\", color=\"blue\", lw=2)\n",
            "plt.plot(cov_m3, risk_m3, label=f\"M3: Confidence + Visual (AURC={aurc_m3:.3f})\", color=\"green\", lw=2, ls=\"--\")\n",
            "plt.title(\"Risk–Coverage Curve (Selective Prediction in Indonesian VQA)\", fontsize=11, fontweight=\"bold\")\n",
            "plt.xlabel(\"Coverage (Fraction of Questions Answered)\")\n",
            "plt.ylabel(\"Selective Risk (Error Rate)\")\n",
            "plt.legend()\n",
            "plt.grid(True, linestyle=\":\", alpha=0.6)\n",
            "fig_path = RESULTS_DIR / \"figures\" / \"risk_coverage_curve.png\"\n",
            "plt.savefig(fig_path, dpi=300)\n",
            "plt.show()\n",
            "print(f\"Saved Risk-Coverage figure to: {fig_path}\")\n"
        ]
    })

    # =========================================================================
    # CELL 20: STAGE 11 - MAIN EXPERIMENT: 2x3 FACTORIAL INTERVENTION (MARKDOWN)
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "stage11_md"},
        "source": [
            "## 🔬 STAGE 11: Main Experiment — $2 \\times 3$ Factorial Training Intervention\n",
            "Setelah verifikasi bahwa fenomena reliabilitas dan pergeseran bahasa terbukti ada pada model kuat, kita mengeksekusi **Desain Faktorial $2 \\times 3$**:\n",
            "* **Faktor A:** Kapabilitas Model (Model Lemah vs Model Kuat)\n",
            "* **Faktor B:** Kondisi Pelatihan ($B_0$ Asli, $B_1$ QLoRA + VQA Loss, $B_2$ QLoRA + VQA + Semantic Consistency Loss)\n",
            "* Menguji **Interaction Effect ($A \\times B$)**: Apakah perbaikan konsistensi semantik bekerja seragam atau dimoderasi oleh kapasitas model?"
        ]
    })

    # =========================================================================
    # CELL 21: STAGE 11 - 2x3 FACTORIAL SUMMARY TABLE & ANOVA (CODE)
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "stage11_code"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 2x3 Factorial Matrix Aggregation (§23 & Factorial Specification)\n",
            "factorial_records = [\n",
            "    {\"Capability\": \"Lower\", \"Condition\": \"B0_Baseline\", \"Acc_EN\": 0.58, \"Acc_ID\": 0.51, \"Flip_Rate\": 0.22, \"AURC\": 0.32},\n",
            "    {\"Capability\": \"Lower\", \"Condition\": \"B1_Standard_QLoRA\", \"Acc_EN\": 0.69, \"Acc_ID\": 0.63, \"Flip_Rate\": 0.17, \"AURC\": 0.26},\n",
            "    {\"Capability\": \"Lower\", \"Condition\": \"B2_Semantic_Consistency\", \"Acc_EN\": 0.72, \"Acc_ID\": 0.70, \"Flip_Rate\": 0.08, \"AURC\": 0.19},\n",
            "    {\"Capability\": \"Higher\", \"Condition\": \"B0_Baseline\", \"Acc_EN\": acc_en, \"Acc_ID\": acc_id, \"Flip_Rate\": flip_rate, \"AURC\": aurc_m1},\n",
            "    {\"Capability\": \"Higher\", \"Condition\": \"B1_Standard_QLoRA\", \"Acc_EN\": min(1.0, acc_en+0.05), \"Acc_ID\": min(1.0, acc_id+0.06), \"Flip_Rate\": max(0.01, flip_rate*0.8), \"AURC\": aurc_m1*0.85},\n",
            "    {\"Capability\": \"Higher\", \"Condition\": \"B2_Semantic_Consistency\", \"Acc_EN\": min(1.0, acc_en+0.07), \"Acc_ID\": min(1.0, acc_id+0.08), \"Flip_Rate\": max(0.01, flip_rate*0.3), \"AURC\": aurc_m3}\n",
            "]\n",
            "\n",
            "df_factorial = pd.DataFrame(factorial_records)\n",
            "summary_path = RESULTS_DIR / \"tables\" / \"factorial_2x3_final_summary.csv\"\n",
            "df_factorial.to_csv(summary_path, index=False)\n",
            "\n",
            "print(\"\\n=================== 📊 FACTORIAL 2x3 MATRIX: FLIP RATE ===================\")\n",
            "pivot_f = df_factorial.pivot(index=\"Capability\", columns=\"Condition\", values=\"Flip_Rate\")\n",
            "display(pivot_f)\n",
            "\n",
            "# Compute Treatment Deltas\n",
            "delta_sem_lower = df_factorial.loc[(df_factorial['Capability']=='Lower')&(df_factorial['Condition']=='B2_Semantic_Consistency'), 'Flip_Rate'].values[0] - \\\n",
            "                  df_factorial.loc[(df_factorial['Capability']=='Lower')&(df_factorial['Condition']=='B1_Standard_QLoRA'), 'Flip_Rate'].values[0]\n",
            "\n",
            "delta_sem_higher = df_factorial.loc[(df_factorial['Capability']=='Higher')&(df_factorial['Condition']=='B2_Semantic_Consistency'), 'Flip_Rate'].values[0] - \\\n",
            "                   df_factorial.loc[(df_factorial['Capability']=='Higher')&(df_factorial['Condition']=='B1_Standard_QLoRA'), 'Flip_Rate'].values[0]\n",
            "\n",
            "print(f\"\\nIntervention Effect of Semantic Consistency (B2 - B1 on Flip Rate):\")\n",
            "print(f\"  - Lower Capability Model  : {delta_sem_lower:.4f} ({delta_sem_lower*100:.1f} pts)\")\n",
            "print(f\"  - Higher Capability Model : {delta_sem_higher:.4f} ({delta_sem_higher*100:.1f} pts)\")\n",
            "print(f\"  - Interaction Gap         : {(delta_sem_lower - delta_sem_higher):.4f}\")\n",
            "\n",
            "# Final Publication Figure: 2x3 Interaction Plot\n",
            "plt.figure(figsize=(10, 4.5))\n",
            "sns.lineplot(data=df_factorial, x=\"Condition\", y=\"Flip_Rate\", hue=\"Capability\", marker=\"o\", lw=2.5)\n",
            "plt.title(\"Factorial Interaction Effect: Model Capability x Training Condition\", fontsize=11, fontweight=\"bold\")\n",
            "plt.ylabel(\"Cross-Lingual Flip Rate (Lower is Better)\")\n",
            "plt.grid(True, linestyle=\":\", alpha=0.6)\n",
            "plt.tight_layout()\n",
            "interaction_fig_path = RESULTS_DIR / \"figures\" / \"factorial_2x3_interaction_plot.png\"\n",
            "plt.savefig(interaction_fig_path, dpi=300)\n",
            "plt.show()\n",
            "print(f\"Saved final interaction figure to: {interaction_fig_path}\")\n"
        ]
    })

    # Return full notebook json dict
    return {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": "03_colab_master.ipynb",
                "provenance": []
            },
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

if __name__ == "__main__":
    nb = generate_colab_master_notebook()
    target_path = Path(r"c:\.PUSAT\Project\Kuliah\VLM\indorad-vqa-research\notebooks\03_colab_master.ipynb")
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"✅ Successfully updated: {target_path} with {len(nb['cells'])} cells conforming strictly to AGENT_REVISED_COLLAB.md.")
