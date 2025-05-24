"""
This module offers path management utilities for RAG pipeline tasks.
It includes `generate_run_id()` for creating unique, timestamp-based run
identifiers, and the `TaskPaths` class, which defines and manages a
standardized directory structure and provides easy access to file paths
for a given task (e.g., for chunks, embeddings, logs).
"""
# scripts/utils/paths.py

import os
from pathlib import Path
from typing import Optional

def generate_run_id() -> str:
    """
    Generates a string identifier based on the current datetime.

    Args:
        None.

    Returns:
        str: A string formatted as "YYYYMMDD_HHMMSS", suitable for use as a
             unique run ID.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

class TaskPaths:
    """
    Utility class to compute and manage directory and file paths
    for a given task within the RAG pipeline.
    """

    def __init__(self, task_name: str, base_dir: str = "runs"):
        """
        Initializes the TaskPaths object for a given task.

        Sets up attributes for various standard subdirectories (emails, chunks,
        embeddings, logs, runs, updates) under `base_dir/task_name/` and calls
        `_create_dirs()` to ensure these directories exist.

        Args:
            task_name (str): The name of the task, used to create a root
                             directory for its files.
            base_dir (str, optional): The base directory under which
                                      task-specific directories will be created.
                                      Defaults to "runs".
        """
        self.task_name = task_name
        self.base_dir = base_dir
        self.task_root = os.path.join(base_dir, task_name)

        # Standard subfolders
        self.emails_dir = os.path.join(self.task_root, "emails")
        self.chunks_dir = os.path.join(self.task_root, "chunks") 
        self.embeddings_dir = os.path.join(self.task_root, "embeddings")
        self.logs_dir = os.path.join(self.task_root, "logs")
        self.runs_dir = os.path.join(self.task_root, "runs")
        self.updates_dir = os.path.join(self.task_root, "updates")
        self.raw_text_output_dir = os.path.join(self.task_root, "raw_text_output") # Added for TextFileFetcher raw output

        # Ensure directory structure exists
        self._create_dirs()

    def _create_dirs(self):
        """
        (Private method) Creates all standard subdirectories if they do not exist.

        The directories created are those defined as instance attributes in
        `__init__` (e.g., `self.emails_dir`, `self.chunks_dir`, etc.).

        Args:
            None.

        Returns:
            None.
        """
        os.makedirs(self.emails_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.updates_dir, exist_ok=True)
        os.makedirs(self.raw_text_output_dir, exist_ok=True) # Ensure creation

    def get_chunk_file(self, data_type: str = "email") -> str:
        """
        Returns the full path to the standard TSV file for storing chunked data.

        The specific filename depends on the data_type.
        - "email": "chunked_emails.tsv"
        - "text_file": "chunked_text_files.tsv"

        Args:
            data_type (str, optional): The type of data. Defaults to "email".
                                       Supported values: "email", "text_file".

        Returns:
            str: The absolute path to the chunked data file within the task's
                 chunks directory (`self.chunks_dir`).
        
        Raises:
            ValueError: If the data_type is unsupported.
        """
        if data_type == "email":
            return os.path.join(self.chunks_dir, "chunked_emails.tsv")
        elif data_type == "text_file":
            return os.path.join(self.chunks_dir, "chunked_text_files.tsv")
        else:
            # Consider adding logging here if a logger is available in this class
            raise ValueError(f"Unsupported data_type for chunk file: {data_type}. Supported types are 'email' and 'text_file'.")

    def get_raw_text_output_dir(self) -> str:
        """
        Returns the full path to the directory for storing raw text outputs from TextFileFetcher.

        Returns:
            str: The absolute path to the raw text output directory.
        """
        return self.raw_text_output_dir

    def get_cleaned_email_file(self) -> str:
        """
        Returns the full path to the standard TSV file for storing cleaned email data.

        Args:
            None.

        Returns:
            str: The absolute path to "cleaned_emails.tsv" within the task's
                 emails directory (`self.emails_dir`).
        """
        return os.path.join(self.emails_dir, "cleaned_emails.tsv")

    def get_index_file(self) -> str:
        """
        Returns the full path to the standard FAISS index file.

        Args:
            None.

        Returns:
            str: The absolute path to "chunks.index" within the task's
                 embeddings directory (`self.embeddings_dir`).
        """
        return os.path.join(self.embeddings_dir, "chunks.index")

    def get_metadata_file(self) -> str:
        """
        Returns the full path to the standard TSV file for storing chunk metadata.

        Args:
            None.

        Returns:
            str: The absolute path to "chunks_metadata.tsv" within the task's
                 embeddings directory (`self.embeddings_dir`).
        """
        return os.path.join(self.embeddings_dir, "chunks_metadata.tsv")

    def get_run_dir(self, run_id: str) -> str:
        """
        Returns the path to a directory for a specific run, creating it if it doesn't exist.

        Args:
            run_id (str): The unique identifier for the run.

        Returns:
            str: The absolute path to the directory `self.runs_dir/run_id/`.
        """
        path = os.path.join(self.runs_dir, run_id)
        os.makedirs(path, exist_ok=True)
        return path

    def get_update_dir(self, update_id: str) -> str:
        """
        Returns the path to a directory for a specific update operation, creating it if it doesn't exist.

        Args:
            update_id (str): The unique identifier for the update.

        Returns:
            str: The absolute path to the directory `self.updates_dir/update_id/`.
        """
        path = os.path.join(self.updates_dir, update_id)
        os.makedirs(path, exist_ok=True)
        return path

    def get_config_path(self) -> str:
        """
        Returns the full path to the standard configuration file for the task.

        Args:
            None.

        Returns:
            str: The absolute path to "config.yaml" within the task's root
                 directory (`self.task_root`).
        """
        return os.path.join(self.task_root, "config.yaml")

    def get_log_path(self, run_id: Optional[str] = None) -> str:
        """
        Returns the full path to a log file for the task.

        If `run_id` is provided, it returns a path to a run-specific log file
        (e.g., `logs_dir/run_id.log`). Otherwise, it returns the path to a
        general task log file (e.g., `logs_dir/task.log`).

        Args:
            run_id (Optional[str]): The unique identifier for a specific run.
                                    Defaults to None.

        Returns:
            str: The absolute path to the log file.
        """
        if run_id:
            return os.path.join(self.logs_dir, f"{run_id}.log")
        return os.path.join(self.logs_dir, "task.log")