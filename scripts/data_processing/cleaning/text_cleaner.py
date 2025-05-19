import pandas as pd
from typing import Callable, List

class TextCleaner:
    """Handles text cleaning operations on a DataFrame."""

    @staticmethod
    def clean_dataframe(df: pd.DataFrame, cleaning_steps: List[Callable[[str], str]]) -> pd.DataFrame:
        """
        Clean the 'content' column of the DataFrame using specified steps.

        Args:
            df (pd.DataFrame): The input DataFrame with 'content' and 'content_type' columns.
            cleaning_steps (List[Callable[[str], str]]): A list of cleaning functions to apply.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        def clean_content(row):
            text = row["content"]
            for step in cleaning_steps:
                text = step(text, row["content_type"])  # Pass content and type to each step
            return text

        df["content"] = df.apply(clean_content, axis=1)
        return df

    @staticmethod
    def remove_urls(text: str, content_type: str) -> str:
        """Removes URLs from the text."""
        import re
        if content_type == "plain_text":  # Apply only to plain text
            return re.sub(r"http\S+|www\S+", "", text)
        return text

    @staticmethod
    def normalize_whitespace(text: str, content_type: str) -> str:
        """Normalize whitespace in the text."""
        return " ".join(text.split())  # Remove excess whitespace in all types

    @staticmethod
    def lowercase(text: str, content_type: str) -> str:
        """Convert text to lowercase for specific types."""
        if content_type == "plain_text":  # Apply only to plain text
            return text.lower()
        return text

    @staticmethod
    def standardize_bullets(text: str, content_type: str) -> str:
        """Standardize bullet markers."""
        if content_type == "bullet_list":
            return text.replace("-", "•")
        return text
