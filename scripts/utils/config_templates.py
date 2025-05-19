from pathlib import Path

def get_default_config(task_name: str) -> dict:
    """
    Returns the default configuration template for a pipeline task.
    """
    return {
        "task_name": task_name,
        "embedding": {
            "mode": "local",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "output_dir": "embeddings",
            "index_filename": "chunks.index",
            "metadata_filename": "chunks_metadata.tsv"
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
