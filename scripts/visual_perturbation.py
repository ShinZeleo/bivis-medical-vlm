import os
import sys
from typing import Union, Optional
import pandas as pd
import torch
from pathlib import Path
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data.image_utils import get_image_path, ensure_images_extracted
from src.models.qwen_loader import load_qwen_vl
from qwen_vl_utils import process_vision_info

def create_blank_image(image_path: Union[str, Path]) -> Image.Image:
    try:
        orig = Image.open(image_path)
        return Image.new('RGB', orig.size, (0, 0, 0))
    except Exception:
        return Image.new('RGB', (224, 224), (0, 0, 0))

def run_visual_perturbation(
    sample_size: int = 10,
    model: Optional[object] = None,
    processor: Optional[object] = None,
    target_device: Optional[torch.device] = None,
    random_seed: int = config.RANDOM_SEED,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Run visual perturbation pilot (Original vs. Blank image) to evaluate visual sensitivity.
    Can be called directly from Jupyter Notebook or CLI.
    """
    ensure_images_extracted()
    
    manifest_path = config.DATA_DIR / "indorad_vqa_final_manifest.csv"
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / "data" / "indorad_vqa_final_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        
    df = pd.read_csv(manifest_path)
    eligible_df = df[df["is_eligible_primary"] == True]
    if len(eligible_df) == 0:
        eligible_df = df
        
    sample_df = eligible_df.sample(min(sample_size, len(eligible_df)), random_state=random_seed)
    print(f"[run_visual_perturbation] Evaluating visual perturbation on {len(sample_df)} pairs...")
    
    if model is None or processor is None:
        processor, model, target_device = load_qwen_vl()
    elif target_device is None:
        target_device = next(model.parameters()).device
        
    token_yes_upper = processor.tokenizer.encode("YES", add_special_tokens=False)[0]
    token_no_upper = processor.tokenizer.encode("NO", add_special_tokens=False)[0]
    token_yes_title = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    token_no_title = processor.tokenizer.encode("No", add_special_tokens=False)[0]
    
    results = []
    out_dir = config.OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_blank_path = out_dir / "temp_blank.jpg"
    
    for idx, row in sample_df.iterrows():
        img_id = row.get("image_id", "")
        try:
            img_path = get_image_path(img_id)
        except Exception as e:
            print(f"Skipping {img_id}: {e}")
            continue
            
        q_en = str(row.get('question_en', row.get('question', 'Is this normal?')))
        q_id = str(row.get('question_id', row.get('question_indonesian', 'Apakah ini normal?')))
        
        pairs = [("en", f"Answer only YES or NO.\n{q_en}"), ("id", f"Jawab hanya dengan YES atau NO.\n{q_id}")]
        
        for lang, prompt_text in pairs:
            # 1. Condition A: Original Image
            msgs_orig = [{"role": "user", "content": [{"type": "image", "image": str(img_path.resolve())}, {"type": "text", "text": prompt_text}]}]
            text_orig = processor.apply_chat_template(msgs_orig, tokenize=False, add_generation_prompt=True)
            image_inputs_orig, _ = process_vision_info(msgs_orig)
            inputs_orig = processor(text=[text_orig], images=image_inputs_orig, padding=True, return_tensors="pt").to(target_device)
            
            with torch.no_grad():
                out_orig = model.generate(**inputs_orig, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
            probs_orig = torch.softmax(out_orig.scores[0][0], dim=-1)
            p_yes_orig = probs_orig[token_yes_upper].item() + probs_orig[token_yes_title].item()
            p_no_orig = probs_orig[token_no_upper].item() + probs_orig[token_no_title].item()
            tot_orig = p_yes_orig + p_no_orig
            norm_yes_orig = p_yes_orig / tot_orig if tot_orig > 0 else 0.5
            norm_no_orig = p_no_orig / tot_orig if tot_orig > 0 else 0.5
            conf_orig = max(norm_yes_orig, norm_no_orig)
            pred_orig = "YES" if norm_yes_orig >= norm_no_orig else "NO"
            
            # 2. Condition B: Blank Image
            blank_img = create_blank_image(img_path)
            blank_img.save(temp_blank_path)
            
            msgs_pert = [{"role": "user", "content": [{"type": "image", "image": str(temp_blank_path.resolve())}, {"type": "text", "text": prompt_text}]}]
            text_pert = processor.apply_chat_template(msgs_pert, tokenize=False, add_generation_prompt=True)
            image_inputs_pert, _ = process_vision_info(msgs_pert)
            inputs_pert = processor(text=[text_pert], images=image_inputs_pert, padding=True, return_tensors="pt").to(target_device)
            
            with torch.no_grad():
                out_pert = model.generate(**inputs_pert, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
            probs_pert = torch.softmax(out_pert.scores[0][0], dim=-1)
            p_yes_pert = probs_pert[token_yes_upper].item() + probs_pert[token_yes_title].item()
            p_no_pert = probs_pert[token_no_upper].item() + probs_pert[token_no_title].item()
            tot_pert = p_yes_pert + p_no_pert
            norm_yes_pert = p_yes_pert / tot_pert if tot_pert > 0 else 0.5
            norm_no_pert = p_no_pert / tot_pert if tot_pert > 0 else 0.5
            conf_pert = max(norm_yes_pert, norm_no_pert)
            pred_pert = "YES" if norm_yes_pert >= norm_no_pert else "NO"
            
            answer_changed = (pred_orig != pred_pert)
            delta_conf = abs(conf_orig - conf_pert)
            
            results.append({
                "qid": row.get("qid", idx),
                "image_id": img_id,
                "language": lang,
                "pred_orig": pred_orig,
                "conf_orig": round(conf_orig, 4),
                "pred_pert": pred_pert,
                "conf_pert": round(conf_pert, 4),
                "answer_changed": answer_changed,
                "delta_conf": round(delta_conf, 4)
            })
            print(f"[{lang.upper()}] Orig: {pred_orig} ({conf_orig:.2f}) -> Blank: {pred_pert} ({conf_pert:.2f}) | Changed: {answer_changed}")
            
    if temp_blank_path.exists():
        temp_blank_path.unlink()
        
    res_df = pd.DataFrame(results)
    
    if save_output:
        out_file = out_dir / "visual_perturbation_results.csv"
        res_df.to_csv(out_file, index=False)
        print(f"[run_visual_perturbation] Saved visual perturbation results to {out_file}")
        
    return res_df

def main():
    run_visual_perturbation(sample_size=10, save_output=True)

if __name__ == "__main__":
    main()
