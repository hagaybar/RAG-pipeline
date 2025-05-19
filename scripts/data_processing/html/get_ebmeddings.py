
#!/usr/bin/env python3
"""
HTML Embedding Processor

A class-based solution for processing HTML files, extracting data,
generating embeddings, and exporting results.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd

from api_clients.openai.gptApiClient import APIClient
from data_processing.html.data_processor import DataProcessor


class HTMLEmbeddingProcessor:
    """
    A class that handles the entire workflow of processing HTML files,
    generating embeddings, and exporting results.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_key_env_var: str = "OPEN_AI_KEY",
        budget_limit: float = 1.0,
        output_file: str = "embeddings_output.xlsx",
        log_level: int = logging.INFO
    ):
        """
        Initialize the HTML Embedding Processor.
        
        Args:
            api_key: API key for the embedding service (optional)
            api_key_env_var: Environment variable name for the API key
            budget_limit: Budget limit for API calls
            output_file: Default output file path
            log_level: Logging level
        """
        # Set up logging
        self.logger = self._setup_logger(log_level)
        
        # Store configuration
        self.config = {
            "api_key": api_key,
            "api_key_env_var": api_key_env_var,
            "budget_limit": budget_limit,
            "output_file": output_file
        }
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.api_client = self._initialize_api_client()
        
        # Initialize state
        self.dataframe = None
        self.processed = False
        self.embeddings_generated = False

    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Set up and configure logger."""
        logger = logging.getLogger(__name__)
        
        # Only add handler if not already added to avoid duplicate logs
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.setLevel(log_level)
        return logger

    def _initialize_api_client(self) -> APIClient:
        """
        Initialize the API client, leveraging its built-in API key handling.
        
        Returns:
            Initialized API client
        """
        # We can pass the API key directly (which may be None)
        # and let the APIClient handle the environment variable fallback
        return APIClient(
            api_key=self.config.get("api_key"),
            budget_limit=self.config["budget_limit"]
        )

    def process_html_file(self, file_path: Union[str, Path]) -> bool:
        """
        Process an HTML file and extract data into a DataFrame.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            True if processing was successful, False otherwise
        """
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return False
            
        try:
            self.dataframe = self.data_processor.process_html_file(file_path)
            self.processed = True
            self.logger.info(f"Successfully processed {file_path}")
            self.logger.info(f"Extracted {len(self.dataframe)} rows of data")
            return True
        except Exception as e:
            self.logger.error(f"Error processing HTML file: {e}")
            return False


    def _create_embedding(self, row):
        """
        Create an embedding for a DataFrame row with enhanced context from all relevant columns.
        
        Args:
            row: DataFrame row containing data
            
        Returns:
            List of embedding values or None if no text was found
        """
        # Handle DataProcessor format with multiple columns
        if 'text' in row.index:
            # Start with the main text content
            text_content = str(row['text']) if pd.notnull(row['text']) else ""
            
            # Skip if main content is empty
            if not text_content.strip():
                self.logger.debug("Empty text found for row, skipping embedding generation")
                return None
            
            # Add tag name as context
            tag_info = ""
            if 'tag' in row.index and pd.notnull(row['tag']):
                tag_info = f"Element: {row['tag']} | "
            
            # Add nesting level context (helps understand document structure)
            nesting_info = ""
            if 'nesting_level' in row.index and pd.notnull(row['nesting_level']):
                nesting_level = row['nesting_level']
                if nesting_level <= 3:
                    importance = "High"
                elif nesting_level <= 6:
                    importance = "Medium"
                else:
                    importance = "Low"
                nesting_info = f"Level: {nesting_level} ({importance} importance) | "
            
            # Add relevant attributes (class, id, href, etc.)
            attr_info = ""
            if 'attributes' in row.index and pd.notnull(row['attributes']):
                attrs = row['attributes']
                if isinstance(attrs, dict):
                    # Extract useful attributes that provide context
                    useful_attrs = []
                    
                    # Extract id attribute (often indicates purpose)
                    if 'id' in attrs:
                        useful_attrs.append(f"id={attrs['id']}")
                    
                    # Extract class attribute (indicates type/purpose)
                    if 'class' in attrs:
                        if isinstance(attrs['class'], list):
                            class_str = ' '.join(attrs['class'])
                        else:
                            class_str = attrs['class']
                        useful_attrs.append(f"class={class_str}")
                    
                    # For links, extract href
                    if 'href' in attrs:
                        useful_attrs.append(f"href={attrs['href']}")
                    
                    # For images, extract alt text
                    if 'alt' in attrs:
                        useful_attrs.append(f"alt={attrs['alt']}")
                    
                    if useful_attrs:
                        attr_info = f"Attributes: {', '.join(useful_attrs)} | "
            
            # Combine all information with appropriate context
            combined_text = f"{tag_info}{nesting_info}{attr_info}Content: {text_content}"
        
        else:
            # Fallback for cell_X format
            cell_columns = [col for col in row.index if col.startswith("cell_")]
            combined_text = " ".join(str(row[col]) for col in cell_columns if pd.notnull(row[col]))
            
            if not combined_text.strip():
                self.logger.debug("Empty text found for row, skipping embedding generation")
                return None
        
        try:
            # Truncate if necessary
            max_text_length = 8000
            if len(combined_text) > max_text_length:
                self.logger.debug(f"Truncating text from {len(combined_text)} to {max_text_length} chars")
                combined_text = combined_text[:max_text_length]
            
            # Get the embedding
            embedding = self.api_client.get_embedding(combined_text)
            return embedding
        except Exception as e:
            self.logger.error(f"Error creating embedding: {e}")
            return None


    def generate_embeddings(self) -> bool:
        """
        Generate embeddings for each row in the DataFrame.
        
        Returns:
            True if embeddings were generated, False otherwise
        """
        if not self.processed or self.dataframe is None:
            self.logger.error("No data to generate embeddings for. Process an HTML file first.")
            return False
        
        self.logger.info("Generating embeddings for dataset")
        
        try:
            # Track progress for large DataFrames
            total_rows = len(self.dataframe)
            processed_count = 0
            skipped_count = 0
            last_progress = 0
            
            # Function to update progress
            def update_progress():
                nonlocal last_progress
                progress = int((processed_count / total_rows) * 100)
                if progress >= last_progress + 10:  # Log every 10% progress
                    self.logger.info(f"Embedding progress: {progress}% ({processed_count}/{total_rows} rows)")
                    last_progress = progress
            
            # Create embeddings with progress tracking
            embeddings = []
            for idx, row in self.dataframe.iterrows():
                embedding = self._create_embedding(row)
                embeddings.append(embedding)
                
                processed_count += 1
                if embedding is None:
                    skipped_count += 1
                    
                update_progress()
            
            # Add the embeddings to the DataFrame
            self.dataframe['embedding'] = embeddings
            
            # Log statistics
            successful_count = processed_count - skipped_count
            self.logger.info(f"Generated {successful_count} embeddings out of {total_rows} rows")
            if skipped_count > 0:
                self.logger.info(f"Skipped {skipped_count} rows with empty or invalid text")
            
            self.embeddings_generated = True
            return True
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return False



    def generate_embeddings(self) -> bool:
        """
        Generate embeddings for each row in the DataFrame.
        
        Returns:
            True if embeddings were generated, False otherwise
        """
        if not self.processed or self.dataframe is None:
            self.logger.error("No data to generate embeddings for. Process an HTML file first.")
            return False
        
        self.logger.info("Generating embeddings for dataset")
        
        try:
            # Apply the embedding function to each row
            self.dataframe['embedding'] = self.dataframe.apply(
                self._create_embedding, axis=1
            )
            
            # Count embeddings generated
            embedding_count = self.dataframe['embedding'].notna().sum()
            self.logger.info(f"Generated {embedding_count} embeddings out of {len(self.dataframe)} rows")
            
            self.embeddings_generated = True
            return True
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return False

    def export_results(self, output_file: Optional[str] = None) -> bool:
        """
        Export the DataFrame with embeddings to an Excel file.
        
        Args:
            output_file: Path to the output Excel file (optional)
            
        Returns:
            True if export was successful, False otherwise
        """
        if not self.embeddings_generated:
            self.logger.error("No embeddings to export. Generate embeddings first.")
            return False
            
        # Use provided output file or fall back to config
        output_path = output_file or self.config["output_file"]
        
        try:
            self.data_processor.export_df_to_excel(self.dataframe, output_path)
            self.logger.info(f"DataFrame with embeddings saved to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving DataFrame to Excel: {e}")
            return False
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get the current DataFrame with embeddings.
        
        Returns:
            DataFrame with embeddings or None if not processed
        """
        return self.dataframe.copy() if self.dataframe is not None else None
    
    def process_pipeline(self, file_path: Union[str, Path], output_file: Optional[str] = None) -> bool:
        """
        Run the complete processing pipeline on a single file.
        
        Args:
            file_path: Path to the HTML file
            output_file: Path to the output Excel file (optional)
            
        Returns:
            True if all steps were successful, False otherwise
        """
        if not self.process_html_file(file_path):
            return False
            
        if not self.generate_embeddings():
            return False
            
        return self.export_results(output_file)


def main():
    """Example usage of the HTMLEmbeddingProcessor class."""
    # Initialize the processor
    processor = HTMLEmbeddingProcessor(
        api_key_env_var="OPEN_AI_KEY",
        budget_limit=1.0,
        output_file="embeddings_output.xlsx"
    )
    
    # Run the complete pipeline
    html_file_path = "path/to/your/Opening.html"  # Update with actual path
    success = processor.process_pipeline(html_file_path)
    
    if success:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline failed. Check logs for details.")
    
    # Alternative: Run steps individually
    # if processor.process_html_file(html_file_path):
    #     if processor.generate_embeddings():
    #         processor.export_results("custom_output.xlsx")


if __name__ == "__main__":
    main()