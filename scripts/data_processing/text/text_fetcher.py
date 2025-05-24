import os
import pandas as pd
from datetime import datetime
from pathlib import Path # Added for robust path handling

# Assuming LoggerManager is in a utility module, e.g., scripts.utils.logger_manager
# from scripts.utils.logger_manager import LoggerManager # Placeholder

# Placeholder for LoggerManager until its actual location is known
class LoggerManager:
    @staticmethod
    def get_logger(name):
        # This is a mock logger. In a real scenario, this would configure and return a logging.Logger instance.
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers: # Avoid adding multiple handlers if get_logger is called multiple times
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

logger = LoggerManager.get_logger(__name__)

from scripts.utils.constants import COL_FILE_PATH, COL_RAW_TEXT, COL_CLEANED_TEXT

class TextFileFetcher:
    """
    Fetches and processes text files from a specified directory.
    """
    def __init__(self, config: dict):
        """
        Initializes the TextFileFetcher with a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing necessary paths.
                           Expected keys:
                           - config["text_files"]["input_dir"] (str): Path to the directory containing text files.
                           - config["paths"]["text_output_dir"] (str): Path to the directory to save processed files.
        """
        self.config = config
        self.input_dir = Path(config["text_files"]["input_dir"])
        self.output_dir = Path(config["paths"]["text_output_dir"])

        if not self.input_dir.is_dir():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TextFileFetcher initialized. Input: {self.input_dir}, Output: {self.output_dir}")

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text.
        (Placeholder implementation - returns text as is)

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        # In a real implementation, this would involve more sophisticated cleaning
        # e.g., removing special characters, normalizing whitespace, etc.
        logger.debug(f"Cleaning text (length: {len(text)})...")
        return text.strip()

    def fetch_text_files(self, return_dataframe: bool = False, save: bool = True):
        """
        Reads all .txt files from the input directory, processes them,
        and optionally saves them to a TSV file.

        Args:
            return_dataframe (bool): If True, returns the DataFrame.
                                     Otherwise, returns the path to the saved file if 'save' is True.
            save (bool): If True, saves the DataFrame to a TSV file.

        Returns:
            pd.DataFrame or str or None:
                - Pandas DataFrame if return_dataframe is True.
                - Path to the saved TSV file if return_dataframe is False and save is True.
                - None if return_dataframe is False and save is False.
        """
        logger.info(f"Starting to fetch text files from: {self.input_dir}")
        file_data = []
        
        # Iterate over all .txt files in the input directory
        for file_path in self.input_dir.glob("*.txt"):
            if file_path.is_file():
                try:
                    logger.debug(f"Processing file: {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        raw_text = f.read()
                    
                    cleaned_text = self.clean_text(raw_text)
                    
                    file_data.append({
                        COL_FILE_PATH: str(file_path.resolve()),
                        COL_RAW_TEXT: raw_text,
                        COL_CLEANED_TEXT: cleaned_text
                    })
                    logger.debug(f"Successfully processed and added: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            else:
                logger.warning(f"Skipping non-file item: {file_path}")

        if not file_data:
            logger.warning("No .txt files found in the input directory.")
            if return_dataframe:
                return pd.DataFrame(columns=[COL_FILE_PATH, COL_RAW_TEXT, COL_CLEANED_TEXT])
            if save: # If save is true but no data, maybe return a message or handle as per requirement
                logger.info("No data to save.")
                return None # Or an empty file path if that's preferred
            return None

        df = pd.DataFrame(file_data)
        logger.info(f"Successfully created DataFrame with {len(df)} text files.")

        if save:
            # Ensure output directory exists (it should from __init__, but double check)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"text_files_{timestamp}.tsv"
            output_path = self.output_dir / output_filename
            
            try:
                df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
                logger.info(f"DataFrame saved to: {output_path}")
                if return_dataframe:
                    return df
                return str(output_path)
            except Exception as e:
                logger.error(f"Error saving DataFrame to {output_path}: {e}", exc_info=True)
                # If saving fails, but dataframe is requested, still return it.
                if return_dataframe:
                    return df
                return None # Or raise error, depending on desired behavior for save failure

        # If not saving, but dataframe is requested
        if return_dataframe:
            return df
        
        return None # If not saving and not returning dataframe
