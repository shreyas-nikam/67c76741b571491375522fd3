
import pandas as pd
import os
from typing import Optional

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a dataset from the specified CSV file path.

    Args:
        file_path (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty or cannot be parsed.
        pd.errors.ParserError: For malformed CSV files.
        ValueError: If the file path is invalid.
    """
    if not isinstance(file_path, str):
        raise ValueError(f"Invalid file path: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
    except pd.errors.EmptyDataError:
        raise
    except pd.errors.ParserError as e:
        raise
    except Exception as e:
        raise
    return df
