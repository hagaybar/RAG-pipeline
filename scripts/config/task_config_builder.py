"""
This module contains the `TaskConfigBuilder` class, which provides an
interactive command-line interface (CLI) for generating task-specific YAML
configuration files for the RAG pipeline. It prompts the user for settings
related to embedding, chunking, Outlook email fetching, retrieval, and
answer generation.
"""
# scripts/config/task_config_builder.py

import yaml
import os
from typing import Optional

class TaskConfigBuilder:
    def __init__(self):
        self.config = {}

    def start(self,  default_task_name: Optional[str] = None):
        print("🔧 Starting interactive configuration...")
        self.ask_task_name(default=default_task_name)
        self.ask_embedding_settings()
        self.ask_chunking_settings()
        self.ask_outlook_settings()
        self.ask_retrieval_settings()
        self.ask_generation_settings()
        print("✅ Configuration building complete.")

    def ask_task_name(self, default: Optional[str] = None):
        task_name = input(f"Enter a unique task name{f' (default: {default})' if default else ''}: ").strip()
        if not task_name and default:
            task_name = default
        self.config["task_name"] = task_name

    def ask_embedding_settings(self):
        print("\n🔹 Embedding Settings")

        mode = input("Choose embedding mode [local/api/batch] (default: local): ").strip() or "local"

        if mode == "api":
            print("➡️ Available API models: text-embedding-3-small, text-embedding-3-large")
        elif mode == "local":
            print("➡️ Available local models: sentence-transformers/all-MiniLM-L6-v2")
        elif mode == "batch":
            print("➡️ Batch uses API-compatible models (e.g., text-embedding-3-small)")

        model = input("Enter embedding model name: ").strip()

        # Validate compatibility
        if mode in ["api", "batch"] and model.startswith("sentence-transformers"):
            print("❌ Incompatible: You selected 'api' or 'batch' mode but chose a local model.")
            print("🔄 Switching embedding mode to 'local' automatically.")
            mode = "local"

        # Default dimensions
        DEFAULT_DIMENSIONS = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "sentence-transformers/all-MiniLM-L6-v2": 384
        }
        embedding_dim = DEFAULT_DIMENSIONS.get(model)

        if embedding_dim is None:
            dim_input = input("Enter embedding dimension (no default available): ").strip()
            embedding_dim = int(dim_input)
        else:
            print(f"ℹ️ Auto-selected embedding dimension: {embedding_dim}")

        self.config["embedding"] = {
            "mode": mode,
            "model_name": model,
            "embedding_dim": embedding_dim,
            "output_dir": "embeddings",
            "index_filename": "chunks.index",
            "metadata_filename": "chunks_metadata.tsv"
        }

    def ask_chunking_settings(self):
        print("\n🔹 Chunking Settings")
        max_size = input("Max chunk size (default: 450): ").strip() or "450"
        overlap = input("Chunk overlap (default: 50): ").strip() or "50"
        min_size = input("Min chunk size (default: 150): ").strip() or "150"
        threshold = input("Similarity threshold (default: 0.8): ").strip() or "0.8"
        language_model = input("SpaCy language model (default: en_core_web_sm): ").strip() or "en_core_web_sm"
        embedding_model = input("Chunking embedding model (default: sentence-transformers/all-MiniLM-L6-v2): ").strip() or "sentence-transformers/all-MiniLM-L6-v2"

        self.config["chunking"] = {
            "max_chunk_size": int(max_size),
            "overlap": int(overlap),
            "min_chunk_size": int(min_size),
            "similarity_threshold": float(threshold),
            "language_model": language_model,
            "embedding_model": embedding_model
        }

    def ask_outlook_settings(self):
        print("\n🔹 Outlook Settings")
        account = input("Outlook account name (e.g., 'Your Name'): ").strip()
        folder = input("Folder path (e.g., Inbox > Subfolder): ").strip()
        days = input("Days to fetch (default: 3): ").strip() or "3"

        self.config["outlook"] = {
            "account_name": account,
            "folder_path": folder,
            "days_to_fetch": int(days)
        }

    def ask_retrieval_settings(self):
        print("\n🔹 Retrieval Settings")
        k = input("Top-K chunks to retrieve (default: 5): ").strip() or "5"
        strategy = input("Retrieval strategy (default: dense_vector): ").strip() or "dense_vector"

        self.config["retrieval"] = {
            "top_k": int(k),
            "strategy": strategy
        }

    def ask_generation_settings(self):
        print("\n🔹 Answer Generation Settings")
        model = input("OpenAI model for generation (default: openai-gpt-4): ").strip() or "openai-gpt-4"
        template = input("Prompt template name (default: standard_qa): ").strip() or "standard_qa"

        self.config["generation"] = {
            "model": model,
            "prompt_template": template
        }

    def write_yaml(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)
        print(f"📁 Configuration saved to: {output_path}")

