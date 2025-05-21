"""
This module offers the `TextCleaner` class, a collection of static methods
designed for cleaning text data within Pandas DataFrames. It supports various
operations such as URL removal, whitespace normalization, case conversion,
and bullet point standardization, which can be applied based on content type.
"""
import pandas as pd
from typing import Callable, List

class TextCleaner:
    """
    Provides a collection of static methods for cleaning textual data,
    typically within Pandas DataFrames.

    This class serves as a namespace for text cleaning utilities. It is not
    intended to be instantiated; its methods are static and can be called
    directly (e.g., `TextCleaner.remove_urls(...)`).

    The primary method, `clean_dataframe`, iterates through a list of provided
    cleaning functions, applying them to a specified column (usually 'content')
    in a DataFrame. Individual static methods address specific cleaning tasks:
    - `remove_urls`: Strips URLs from text.
    - `normalize_whitespace`: Consolidates multiple whitespace characters into single spaces.
    - `lowercase`: Converts text to lowercase.
    - `standardize_bullets`: Converts common bullet characters (like '-') to a
      standard symbol (e.g., '•').

    A key feature is that many cleaning functions operate conditionally based on a
    `content_type` string (passed alongside the text to each cleaning function),
    allowing for targeted cleaning rules (e.g., only lowercasing 'plain_text'
    content or standardizing bullets only for 'bullet_list' content type).
    """

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
        """
        Removes URLs (http, https, www) from the text if `content_type` is 'plain_text'.
        Otherwise, returns the text unchanged.

        Args:
            text (str): The input string.
            content_type (str): The type of content; URLs are only removed if this is 'plain_text'.

        Returns:
            str: The text with URLs removed (if applicable), or the original text.
        """
        import re
        if content_type == "plain_text":  # Apply only to plain text
            return re.sub(r"http\S+|www\S+", "", text)
        return text

    @staticmethod
    def normalize_whitespace(text: str, content_type: str) -> str:
        """
        Normalizes whitespace in the text by replacing multiple consecutive
        whitespace characters (spaces, tabs, newlines) with a single space and
        stripping leading/trailing whitespace. The `content_type` parameter is
        accepted for consistency but not used in the current logic.

        Args:
            text (str): The input string.
            content_type (str): The type of content (currently not used by this function).

        Returns:
            str: The text with normalized whitespace.
        """
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
