import pandas as pd
from pathlib import Path
from typing import Dict, Union, Optional

def audit_images(df: pd.DataFrame, split_name: str, max_check: Optional[int] = None) -> pd.DataFrame:
    """Audit images in dataset: check image reference integrity and metadata."""
    audits = []
    subset = df if max_check is None else df.head(max_check)
    
    for idx, row in subset.iterrows():
        img_id = str(row.get('image_name', row.get('image', idx)))
        has_img = row.get('image') is not None
        
        audits.append({
            'split': split_name,
            'image_id': img_id,
            'has_image_object': has_img,
            'status': 'OK' if has_img else 'MISSING'
        })
        
    return pd.DataFrame(audits)

def compute_image_stats(audit_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """Compute summary statistics for image audit."""
    total = len(audit_df)
    unique_imgs = audit_df['image_id'].nunique() if 'image_id' in audit_df.columns else total
    ok_count = (audit_df['status'] == 'OK').sum() if 'status' in audit_df.columns else total
    
    return {
        'total_records': total,
        'unique_images': unique_imgs,
        'valid_images': ok_count,
        'valid_pct': round(ok_count / total * 100, 2) if total > 0 else 100.0
    }

def save_image_audit(audit_df: pd.DataFrame, stats: Dict, output_dir: Union[str, Path]):
    """Save image audit report to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audit_df.to_csv(output_dir / "image_audit.csv", index=False)
