# scripts/utils/data_utils.py

import os
import pandas as pd
import hashlib

def hash_text(text: str) -> str:
    """
    Generate a SHA-256 hash string for the given text.

    Args:
        text (str): Input text to hash.

    Returns:
        str: Hex digest of the text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def deduplicate_emails(new_df: pd.DataFrame, existing_path: str, key_col: str = "Cleaned Body") -> pd.DataFrame:
    """
    Deduplicate emails based on a specified column (default: 'Cleaned Body').

    This function compares the SHA-256 hashes of the target column in the new data
    against hashes from an existing cleaned TSV file. Only truly new records are returned.

    Args:
        new_df (pd.DataFrame): Newly fetched email DataFrame.
        existing_path (str): Path to existing cleaned email file (TSV).
        key_col (str): Column name to use for comparison and hashing (default: 'Cleaned Body').

    Returns:
        pd.DataFrame: Filtered DataFrame containing only deduplicated rows.

    Raises:
        ValueError: If the key column is not found in the input DataFrame.
    """
    if new_df.empty:
        return new_df

    if key_col not in new_df.columns:
        raise ValueError(f"Key column '{key_col}' not found in new_df")

    new_df["__hash__"] = new_df[key_col].apply(hash_text)

    if not existing_path or not os.path.exists(existing_path):
        return new_df.drop(columns="__hash__")

    existing_df = pd.read_csv(existing_path, sep="\t")
    if key_col not in existing_df.columns:
        return new_df.drop(columns="__hash__")

    existing_hashes = set(existing_df[key_col].dropna().apply(hash_text))
    deduped_df = new_df[~new_df["__hash__"].isin(existing_hashes)]
    return deduped_df.drop(columns="__hash__")

def deduplicate_chunks(new_df: pd.DataFrame, existing_path: str, text_col: str = "Chunk") -> pd.DataFrame:
    """
    Deduplicate chunks based on content using SHA-256 hash of the specified text column.

    Args:
        new_df (pd.DataFrame): New chunked data.
        existing_path (str): Path to existing metadata TSV.
        text_col (str): Column name containing chunk text (default: 'Chunk').

    Returns:
        pd.DataFrame: Deduplicated chunk DataFrame.
    """
    if new_df.empty:
        return new_df

    if text_col not in new_df.columns:
        raise ValueError(f"Text column '{text_col}' not found in new_df")

    new_df["__hash__"] = new_df[text_col].apply(hash_text)

    if not existing_path or not os.path.exists(existing_path):
        return new_df.drop(columns="__hash__")

    existing_df = pd.read_csv(existing_path, sep="\t")
    if text_col not in existing_df.columns:
        return new_df.drop(columns="__hash__")

    existing_hashes = set(existing_df[text_col].dropna().apply(hash_text))
    deduped_df = new_df[~new_df["__hash__"].isin(existing_hashes)]
    return deduped_df.drop(columns="__hash__")
