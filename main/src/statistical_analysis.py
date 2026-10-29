import pandas as pd
from typing import Dict, Union

def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for numerical metadata fields."""
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        return df[num_cols].describe()
    return pd.DataFrame()
