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
    """
    Facilitates the interactive creation of task-specific YAML configuration files.

    This class guides users through a command-line interface (CLI) to define
    all necessary parameters for a new RAG pipeline task. It systematically
    prompts for settings across various stages:
    - Task identification (e.g., `task_name`).
    - Embedding parameters (e.g., `embedding.mode`, `embedding.model_name`, `embedding.embedding_dim`).
    - Chunking strategies (e.g., `chunking.max_chunk_size`, `chunking.overlap`, `chunking.language_model`).
    - Outlook email fetching details (e.g., `outlook.account_name`, `outlook.folder_path`, `outlook.days_to_fetch`).
    - Retrieval settings (e.g., `retrieval.top_k`, `retrieval.strategy`).
    - Answer generation model (e.g., `generation.model`).

    The collected configuration is stored internally in the `self.config` dictionary
    and can be written to a YAML file using the `write_yaml` method. This approach
    simplifies the creation of new task configurations by interactively ensuring all
    key aspects and configuration groups are covered.
    """
    def __init__(self):
        """
        Initializes the TaskConfigBuilder.

        Sets up an empty dictionary `self.config` to store configuration settings
        as they are gathered through the interactive process.
        """
        self.config = {}

    def start(self,  default_task_name: Optional[str] = None):
        """
        Initiates and manages the interactive configuration building process.

        This method orchestrates the gathering of all necessary settings by
        calling the various `ask_*` methods in sequence. It starts by asking for
        a task name, then proceeds to embedding, chunking, Outlook, retrieval,
        and generation settings.

        Args:
            default_task_name (Optional[str], optional): An optional default name
                for the task. If provided, this name will be used if the user
                does not enter a specific name. Defaults to None.
        """
        print("🔧 Starting interactive configuration...")
        self.ask_task_name(default=default_task_name)
        self.ask_embedding_settings()
        self.ask_chunking_settings()
        self.ask_outlook_settings()
        self.ask_retrieval_settings()
        self.ask_generation_settings()
        print("✅ Configuration building complete.")

    def ask_task_name(self, default: Optional[str] = None):
        """
        Prompts the user to enter a unique name for the task.

        The entered task name is stored in the `self.config` dictionary under
        the key "task_name".

        Args:
            default (Optional[str], optional): An optional default task name.
                If provided and the user enters no name, this default is used.
                Defaults to None.
        """
        task_name = input(f"Enter a unique task name{f' (default: {default})' if default else ''}: ").strip()
        if not task_name and default:
            task_name = default
        self.config["task_name"] = task_name

    def ask_embedding_settings(self):
        """
        Interactively prompts the user for embedding-related settings.

        This method gathers information about the desired embedding mode (local,
        api, or batch), the specific model name, and the embedding dimension.
        It includes logic to validate model compatibility with the chosen mode
        (e.g., preventing selection of a local sentence-transformer model if
        'api' mode is chosen) and attempts to auto-select the embedding
        dimension based on known models. The collected settings are stored in
        `self.config['embedding']`.
        """
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
        """
        Prompts the user for various chunking parameters.

        This includes the maximum chunk size (in tokens), token overlap between
        chunks, minimum chunk size (in characters), semantic similarity threshold
        for grouping sentences, the SpaCy language model for sentence tokenization,
        and the sentence-transformer model for similarity calculations.
        The settings are stored in `self.config['chunking']`.
        """
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
        """
        Gathers Outlook email fetching settings from the user.

        Prompts for the Outlook account name, the specific folder path from which
        to fetch emails (e.g., "Inbox > Subfolder"), and the number of past days
        of emails to retrieve. These settings are stored in `self.config['outlook']`.
        """
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
        """
        Asks the user for settings related to the chunk retrieval process.

        This includes the number of top-K chunks to retrieve for a given query
        and the retrieval strategy to be used (e.g., "dense_vector").
        The settings are stored in `self.config['retrieval']`.
        """
        print("\n🔹 Retrieval Settings")
        k = input("Top-K chunks to retrieve (default: 5): ").strip() or "5"
        strategy = input("Retrieval strategy (default: dense_vector): ").strip() or "dense_vector"

        self.config["retrieval"] = {
            "top_k": int(k),
            "strategy": strategy
        }

    def ask_generation_settings(self):
        """
        Collects settings for the answer generation phase.

        Prompts the user for the OpenAI model to be used for generating answers
        and the name of the prompt template to apply. These settings are stored
        in `self.config['generation']`.
        """
        print("\n🔹 Answer Generation Settings")
        model = input("OpenAI model for generation (default: openai-gpt-4): ").strip() or "openai-gpt-4"
        template = input("Prompt template name (default: standard_qa): ").strip() or "standard_qa"

        self.config["generation"] = {
            "model": model,
            "prompt_template": template
        }

    def write_yaml(self, output_path: str):
        """
        Saves the gathered configuration to a YAML file.

        The `self.config` dictionary, which contains all settings collected
        through the `ask_*` methods, is dumped to a YAML file at the
        specified `output_path`. The method ensures that the directory for
        the output file exists before writing.

        Args:
            output_path (str): The file path where the YAML configuration
                               will be saved.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)
        print(f"📁 Configuration saved to: {output_path}")

