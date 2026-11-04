import os
import sys
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data.image_utils import get_image_path, ensure_images_extracted
from src.models.qwen_loader import load_qwen_vl
from qwen_vl_utils import process_vision_info

def create_blank_image(img_path, save_path):
    img = Image.open(img_path)
    blank = Image.new(img.mode, img.size, color=0)
    blank.save(save_path)
    return save_path

def create_shuffled_image(img_path, save_path):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    shape = arr.shape
    arr_flat = arr.reshape(-1, 3)
    np.random.shuffle(arr_flat)
    arr_shuffled = arr_flat.reshape(shape)
    shuffled_img = Image.fromarray(arr_shuffled)
    shuffled_img.save(save_path)
    return save_path

def evaluate_pair(model, processor, target_device, img_path, q_en, q_id):
    prompts = {
        "en": f"Answer only YES or NO.\n{q_en}",
        "id": f"Answer only YES or NO. Question:\n{q_id}"
    }
    preds = {}
    for lang, prompt_text in prompts.items():
        messages = [
            {"role": "user", "content": [{"type": "image", "image": str(img_path)}, {"type": "text", "text": prompt_text}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(target_device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1)
        pred = processor.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
        # Clean prediction to canonical YES/NO if possible
        if "YES" in pred or "YA" in pred: pred = "YES"
        elif "NO" in pred or "TIDAK" in pred: pred = "NO"
        preds[lang] = pred
    return preds["en"], preds["id"]

def run_pilot(sample_size=100, seed=config.RANDOM_SEED):
    ensure_images_extracted()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = config.DATA_DIR / "indorad_vqa_final_manifest.csv"
    df = pd.read_csv(manifest_path)
    eligible_df = df[df["is_eligible_primary"] == True]
    if len(eligible_df) < sample_size:
        eligible_df = df
    
    sample_df = eligible_df.sample(min(sample_size, len(eligible_df)), random_state=seed)
    print(f"Starting pilot evaluation on {len(sample_df)} pairs...")
    
    processor, model, target_device = load_qwen_vl()
    
    scratch_dir = PROJECT_ROOT / "data" / "scratch"
    scratch_dir.mkdir(exist_ok=True)
    
    results = []
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        img_id = row.get("image_id", "")
        try:
            img_path = get_image_path(img_id)
        except Exception:
            continue
            
        q_en = str(row.get('question_en', row.get('question', '')))
        q_id = str(row.get('question_id', row.get('question_indonesian', '')))
        gt = str(row.get('answer_en', row.get('answer', ''))).strip().upper()
        
        # Original Image
        pred_en, pred_id = evaluate_pair(model, processor, target_device, img_path, q_en, q_id)
        
        # Counterfactual: Blank Image
        blank_path = scratch_dir / f"blank_{img_path.name}"
        create_blank_image(img_path, blank_path)
        pred_en_blank, _ = evaluate_pair(model, processor, target_device, blank_path, q_en, q_id)
        
        # Counterfactual: Shuffled Image
        shuffle_path = scratch_dir / f"shuffle_{img_path.name}"
        create_shuffled_image(img_path, shuffle_path)
        pred_en_shuffle, _ = evaluate_pair(model, processor, target_device, shuffle_path, q_en, q_id)
        
        results.append({
            "qid": row.get("qid", idx),
            "ground_truth": gt,
            "pred_en_orig": pred_en,
            "pred_id_orig": pred_id,
            "pred_en_blank": pred_en_blank,
            "pred_en_shuffle": pred_en_shuffle
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(results_dir / "pilot_predictions_Qwen2.5-VL-3B.csv", index=False)
    
    # Calculate Metrics
    # 1. Accuracy
    acc_en = (res_df["pred_en_orig"] == res_df["ground_truth"]).mean()
    acc_id = (res_df["pred_id_orig"] == res_df["ground_truth"]).mean()
    lang_gap = acc_en - acc_id
    
    # 2. Flip Rate (Inconsistency)
    flip_rate = (res_df["pred_en_orig"] != res_df["pred_id_orig"]).mean()
    
    # 3. Visual Sensitivity (Change rate when image is blanked/shuffled)
    blank_change_rate = (res_df["pred_en_orig"] != res_df["pred_en_blank"]).mean()
    shuffle_change_rate = (res_df["pred_en_orig"] != res_df["pred_en_shuffle"]).mean()
    
    metrics = {
        "Model": "Qwen2.5-VL-3B-Instruct",
        "Condition": "B0 (Original)",
        "EN_Acc": acc_en,
        "ID_Acc": acc_id,
        "Lang_Gap": lang_gap,
        "Flip_Rate": flip_rate,
        "VS_Blank_Change": blank_change_rate,
        "VS_Shuffle_Change": shuffle_change_rate
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / "model_screening.csv", index=False)
    
    print("\n--- PILOT METRICS ---")
    print(metrics_df.to_string(index=False))
    print("Saved to results/model_screening.csv")

if __name__ == "__main__":
    # We run on 100 images for a robust pilot
    run_pilot(sample_size=100)
