import pandas as pd
from pathlib import Path
from typing import Dict
from huggingface_hub import hf_hub_download
try:
    from vlm_research import config
except ImportError:
    from .. import config

def load_parquet_from_hf(split_name: str) -> pd.DataFrame:
    """
    Load a parquet split from HuggingFace Hub.
    
    Args:
        split_name: Either 'train' or 'test'.
        
    Returns:
        pd.DataFrame containing the dataset split.
    """
    if split_name not in config.PARQUET_SPLITS:
        raise ValueError(f"Unknown split name: {split_name}. Expected one of {list(config.PARQUET_SPLITS.keys())}")
        
    filename = config.PARQUET_SPLITS[split_name]
    
    try:
        # First attempt: Try direct download using pandas if possible
        # This works if pyarrow is installed and handles HF datasets smoothly
        url = f"hf://datasets/{config.DATASET_NAME}/{filename}"
        df = pd.read_parquet(url)
        print(f"Successfully loaded '{split_name}' split directly via hf:// URL.")
        return df
    except Exception as e:
        print(f"Direct hf:// URL failed for '{split_name}' split ({e}). Falling back to hf_hub_download...")
        
        # Fallback: Download to local cache and load
        local_path = hf_hub_download(
            repo_id=config.DATASET_NAME,
            filename=filename,
            repo_type="dataset"
        )
        df = pd.read_parquet(local_path)
        print(f"Successfully loaded '{split_name}' split via hf_hub_download.")
        return df

def load_all_splits() -> Dict[str, pd.DataFrame]:
    """
    Load both train and test splits from HuggingFace.
    
    Returns:
        Dictionary mapping split names to DataFrames.
    """
    dfs = {}
    for split in config.PARQUET_SPLITS.keys():
        print(f"Loading '{split}' split...")
        dfs[split] = load_parquet_from_hf(split)
    return dfs

def get_split_summary(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate a summary of the loaded splits.
    
    Args:
        dfs: Dictionary mapping split names to DataFrames.
        
    Returns:
        DataFrame with split summary (rows, columns).
    """
    summary_data = []
    for split_name, df in dfs.items():
        summary_data.append({
            'Split': split_name,
            'Rows': len(df),
            'Columns': len(df.columns)
        })
    return pd.DataFrame(summary_data)

def save_raw_data(dfs: Dict[str, pd.DataFrame], output_dir: Path):
    """
    Save the loaded dataframes to the raw directory as parquet.
    Drops the 'image' column to save space if it exists.
    
    Args:
        dfs: Dictionary mapping split names to DataFrames.
        output_dir: Path to directory where files should be saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, df in dfs.items():
        save_path = output_dir / f"{split_name}.parquet"
        
        # Drop image column if present to save space
        if config.COL_IMAGE in df.columns:
            df_to_save = df.drop(columns=[config.COL_IMAGE])
        else:
            df_to_save = df
            
        df_to_save.to_parquet(save_path, index=False)
        print(f"Saved '{split_name}' (without images) to {save_path}")
