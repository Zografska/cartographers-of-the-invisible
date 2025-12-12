"""
Data utilities for loading and processing data.
"""

import pandas as pd
from typing import Optional, Dict, Any


def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        filepath: Path to the data file
        **kwargs: Additional arguments to pass to pandas read functions
        
    Returns:
        DataFrame containing the loaded data
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath, **kwargs)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath, **kwargs)
    elif filepath.endswith(('.xls', '.xlsx')):
        return pd.read_excel(filepath, **kwargs)
    elif filepath.endswith('.parquet'):
        return pd.read_parquet(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def process_data(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    dropna: bool = False,
    normalize: bool = False
) -> pd.DataFrame:
    """
    Process and clean data.
    
    Args:
        df: Input DataFrame
        columns: List of columns to keep (None keeps all)
        dropna: Whether to drop rows with missing values
        normalize: Whether to normalize numeric columns
        
    Returns:
        Processed DataFrame
    """
    result = df.copy()
    
    if columns:
        result = result[columns]
    
    if dropna:
        result = result.dropna()
    
    if normalize:
        numeric_cols = result.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            min_val = result[col].min()
            max_val = result[col].max()
            if max_val > min_val:
                result[col] = (result[col] - min_val) / (max_val - min_val)
    
    return result


def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing summary statistics
    """
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {}
    }
