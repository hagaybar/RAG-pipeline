# scripts/utils/paths.py

import os
from pathlib import Path
from typing import Optional

def generate_run_id() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

class TaskPaths:
    """
    Utility class to compute and manage directory and file paths
    for a given task within the RAG pipeline.
    """

    def __init__(self, task_name: str, base_dir: str = "runs"):
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

        # Ensure directory structure exists
        self._create_dirs()

    def _create_dirs(self):
        os.makedirs(self.emails_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.updates_dir, exist_ok=True)

    def get_chunk_file(self) -> str:
        return os.path.join(self.chunks_dir, "chunked_emails.tsv")

    def get_cleaned_email_file(self) -> str:
        return os.path.join(self.emails_dir, "cleaned_emails.tsv")

    def get_index_file(self) -> str:
        return os.path.join(self.embeddings_dir, "chunks.index")

    def get_metadata_file(self) -> str:
        return os.path.join(self.embeddings_dir, "chunks_metadata.tsv")

    def get_run_dir(self, run_id: str) -> str:
        path = os.path.join(self.runs_dir, run_id)
        os.makedirs(path, exist_ok=True)
        return path

    def get_update_dir(self, update_id: str) -> str:
        path = os.path.join(self.updates_dir, update_id)
        os.makedirs(path, exist_ok=True)
        return path

    def get_config_path(self) -> str:
        return os.path.join(self.task_root, "config.yaml")

    def get_log_path(self, run_id: Optional[str] = None) -> str:
        if run_id:
            return os.path.join(self.logs_dir, f"{run_id}.log")
        return os.path.join(self.logs_dir, "task.log")