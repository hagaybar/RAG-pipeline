"""
This module supplies the `get_default_config` function, which generates a
standardized default configuration template (Python dictionary) for RAG
pipeline tasks. This template provides baseline settings for embedding,
retrieval, generation, file paths, and Outlook email fetching, serving as an
initial setup for new task configurations.
"""
from pathlib import Path

MODEL_MODE_COMPATIBILITY = {
    "api": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
    "local": ["sentence-transformers/all-MiniLM-L6-v2"],
    "batch": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
}

MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "sentence-transformers/all-MiniLM-L6-v2": 384
}

def get_default_config(task_name: str) -> dict:
    """
    Returns the default configuration template for a pipeline task.
    """
    return {
        "task_name": task_name,
        "embedding": {
            "mode": "local",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384, # Default to MiniLM
            "output_dir": "embeddings",
            "index_filename": "chunks.index",
            "metadata_filename": "chunks_metadata.tsv"
        },
        "chunking": {
            "max_chunk_size": 450,
            "overlap": 50,
            "min_chunk_size": 150, # Character count
            "similarity_threshold": 0.8,
            "language_model": "en_core_web_sm",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2" # For sentence similarity
        },
        "retrieval": {
            "top_k": 5,
            "strategy": "dense_vector"
        },
        "generation": {
            "model": "openai-gpt-4",
            "prompt_template": "standard_qa"
        },
        "paths": {
            "chunked_emails": Path("embeddings", "chunked_emails.tsv").as_posix()
        },
        "outlook": {
            "account_name": "YOUR_ACCOUNT_NAME",
            "folder_path": "Inbox",
            "days_to_fetch": 3
        }
    }
