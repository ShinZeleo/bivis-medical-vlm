import pandas as pd
from pathlib import Path
from typing import Dict, Union

def audit_schema(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Audit dataframe schema: data types, non-null counts, null percentages, unique counts.
    Handles unhashable column values (e.g. image dicts/objects) safely.
    """
    audit_data = []
    total_rows = len(df)
    
    for col in df.columns:
        non_null = df[col].notnull().sum()
        null_count = total_rows - non_null
        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
        
        try:
            unique_count = df[col].nunique(dropna=True)
        except TypeError:
            try:
                unique_count = len(df[col].dropna().astype(str).unique())
            except Exception:
                unique_count = -1
                
        dtype = str(df[col].dtype)
        
        audit_data.append({
            'column': col,
            'dtype': dtype,
            'non_null_count': non_null,
            'null_count': null_count,
            'null_pct': round(null_pct, 2),
            'unique_count': unique_count
        })
        
    return pd.DataFrame(audit_data)

def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a comprehensive data dictionary for dataset columns."""
    field_descriptions = {
        'qid': 'Unique question identifier',
        'image': 'Image binary or path object',
        'image_name': 'Filename or identifier of the radiology image',
        'question': 'English radiology question',
        'question_indonesian': 'Indonesian translated question',
        'answer': 'English ground truth answer',
        'answer_indonesian': 'Indonesian translated ground truth answer',
        'answer_type': 'Question answer type category (CLOSED / OPEN)',
        'question_type': 'Medical domain topic of question (organ, abnormality, etc.)',
        'image_organ': 'Target anatomical organ system',
        'phrase_type': 'Syntactic structure type of the question',
        '_split': 'Original HF dataset split label',
        '_hf_index': 'Original HuggingFace dataset row index',
        'meta_question_score': 'Quality translation score for question',
        'meta_answer_score': 'Quality translation score for answer',
        'meta_question_attempts': 'Number of translation attempts for question',
        'meta_answer_attempts': 'Number of translation attempts for answer',
    }
    
    dict_data = []
    for col in df.columns:
        dict_data.append({
            'Field Name': col,
            'Data Type': str(df[col].dtype),
            'Description': field_descriptions.get(col, 'Dataset metadata field'),
            'Example Value': str(df[col].iloc[0]) if len(df) > 0 else ''
        })
    return pd.DataFrame(dict_data)

def save_schema_audit(data: Dict[str, pd.DataFrame], output_dir: Union[str, Path]):
    """Save schema audit tables to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for key, df in data.items():
        save_path = output_dir / f"schema_{key}.csv"
        df.to_csv(save_path, index=False)
