import io
from pathlib import Path
from typing import Union, Optional
import pandas as pd
from PIL import Image
try:
    from vlm_research import config
except ImportError:
    from .. import config

def ensure_images_extracted() -> int:
    """
    Ensure all unique images from IndoRad-VQA parquet files are extracted to config.IMAGES_DIR.
    Returns the number of images in IMAGES_DIR.
    """
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(config.IMAGES_DIR.glob("*.png")) + list(config.IMAGES_DIR.glob("*.jpg"))
    if len(existing) >= 314:
        return len(existing)

    manifest_path = config.MANIFESTS_DIR / "image_identity_manifest.csv"
    if not manifest_path.exists():
        manifest_path = config.DATA_DIR / "indorad_vqa_final_manifest.csv"
        
    qid_to_img_id = {}
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        img_col = "canonical_image_id" if "canonical_image_id" in manifest.columns else "image_id"
        if "qid" in manifest.columns and img_col in manifest.columns:
            qid_to_img_id = dict(zip(manifest["qid"], manifest[img_col]))

    # Locate parquet cache or raw files
    parquet_files = []
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    for p in hf_cache_dir.glob("**/datasets--Lab-IS--IndoRad-VQA/**/*.parquet"):
        if p.is_file():
            parquet_files.append(p)
            
    if not parquet_files:
        for split in ["train", "test"]:
            raw_p = config.RAW_DIR / f"{split}.parquet"
            if raw_p.exists():
                parquet_files.append(raw_p)

    saved_count = 0
    saved_ids = {p.stem for p in existing}
    
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            if "image" not in df.columns or "qid" not in df.columns:
                continue
            for _, row in df.iterrows():
                qid = row["qid"]
                img_id = qid_to_img_id.get(qid)
                if not img_id:
                    img_id = str(row.get("image_name", f"img_{qid}"))
                    
                if img_id not in saved_ids:
                    img_data = row["image"]
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        img_bytes = img_data["bytes"]
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        img.save(config.IMAGES_DIR / f"{img_id}.png")
                        saved_ids.add(img_id)
                        saved_count += 1
        except Exception as e:
            print(f"Warning extracting from {pf}: {e}")

    return len(list(config.IMAGES_DIR.glob("*.png")))

def get_image_path(image_id: Union[str, int]) -> Path:
    """
    Get the absolute Path to an image file given its image_id or canonical_image_id.
    """
    str_id = str(image_id).strip()
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    candidates = [
        config.IMAGES_DIR / f"{str_id}.png",
        config.IMAGES_DIR / f"{str_id}.jpg",
        config.IMAGES_DIR / str_id,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
            
    # Try extraction if not found
    ensure_images_extracted()
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
            
    # Fallback search by prefix or stem
    for f in config.IMAGES_DIR.iterdir():
        if f.is_file() and (f.stem == str_id or f.stem.startswith(str_id[:8])):
            return f.resolve()
            
    raise FileNotFoundError(f"Image '{image_id}' not found in {config.IMAGES_DIR}")

def get_image(image_id: Union[str, int]) -> Image.Image:
    """Load and return image as a PIL Image in RGB format."""
    p = get_image_path(image_id)
    return Image.open(p).convert("RGB")
