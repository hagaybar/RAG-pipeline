"""
This module defines the `RAGPipeline` class, the central orchestrator for the
Retrieval Augmented Generation workflow. It manages all key stages, from
configuration loading, data extraction (e.g., emails) and chunking, to
embedding generation, context retrieval, and AI-driven answer synthesis.
The class supports modular execution of pipeline steps, dynamic task
configuration, and methods for updating existing embeddings.
"""
import os
import json
from pathlib import Path
from typing import Optional
import pandas as pd
import yaml
import csv # Added for quoting options in to_csv
from datetime import datetime


from scripts.data_processing.email.config_loader import ConfigLoader
from scripts.data_processing.email.email_fetcher import EmailFetcher

from scripts.chunking.text_chunker_v2 import TextChunker
from scripts.retrieval.chunk_retriever_v3 import ChunkRetriever
from scripts.prompting.prompt_builder import EmailPromptBuilder
from scripts.api_clients.openai.gptApiClient import APIClient


from scripts.utils.logger import LoggerManager
import logging
from scripts.utils.yaml_utils import SmartQuotedStringDumper
from scripts.utils.paths import TaskPaths, generate_run_id
from scripts.utils.yaml_utils import enforce_scalar_types

from scripts.formatting.citation_formatter import CitationFormatter



class RAGPipeline:
    """
    Orchestrates the end-to-end Retrieval Augmented Generation (RAG) workflow.

    This class is the main driver of the RAG pipeline, managing all stages from
    initial data ingestion and processing to the final generation of answers.
    It integrates various components to perform tasks such as data fetching,
    chunking, embedding, retrieval, and answer synthesis.

    Key Responsibilities and Functionalities:
    - **Initialization**: Can be initialized with a path to a YAML configuration
      file (`config_path`). Upon initialization with a path, it loads, validates,
      and applies the configuration, setting up paths and instantiating necessary
      components like the appropriate embedder based on the config.
    - **Core RAG Stages**:
        - `extract_and_chunk()`: Fetches raw data (e.g., emails via `EmailFetcher`),
          cleans it, and divides it into smaller, manageable text chunks using
          `TextChunker`.
        - `embed_chunks()` / `embed_chunks_batch()`: Takes the processed chunks and
          generates vector embeddings using the configured `GeneralPurposeEmbedder`.
          Supports both regular and batch embedding modes.
        - `retrieve()`: Accepts a user query, embeds it using the same embedding
          client, and then uses `ChunkRetriever` to find and return the most
          semantically similar chunks from the FAISS index.
        - `generate_answer()`: Constructs a detailed prompt using `EmailPromptBuilder`
          (incorporating the user query and retrieved context chunks), sends it to
          an LLM via `APIClient`, and formats the response using `CitationFormatter`
          to include source citations.
    - **Component Integration**: Serves as a central hub that instantiates and
      coordinates various specialized components, including `ConfigLoader`,
      `EmailFetcher`, `TextChunker`, `GeneralPurposeEmbedder` (with either
      `LocalModelEmbedder` or an API-based embedder), `ChunkRetriever`,
      `EmailPromptBuilder`, `APIClient`, and `CitationFormatter`.
    - **Configuration Handling**: Manages pipeline configurations through methods
      like `load_config()`, `validate_config()`, and `configure_task()` (for
      creating new task configurations dynamically).
    - **Flexible Pipeline Execution**: Allows for defining a custom sequence of
      pipeline steps via `add_step()` and executing them using `run_steps()`.
      The `run_full_pipeline()` method provides a shortcut for a standard end-to-end run.
    - **State Management**: Maintains critical state information, including the
      loaded configuration, paths to data files (e.g., `chunked_file`,
      `index_path`, `metadata_path`), the user's query, and the last retrieved chunks.
    - **Utility and introspection**: Offers `pipe_review()` to display the current
      pipeline setup and `describe_steps()` to list available operations.
    - **Logging**: Employs `LoggerManager` for comprehensive logging of its operations.

    The class defines `STEP_DEPENDENCIES` to manage the valid order of operations
    when steps are added manually.
    """
    STEP_DEPENDENCIES = {
    "extract_and_chunk": [],
    "embed_chunks": [],
    "embed_chunks_batch": [],
    "get_user_query": [],
    "retrieve": ["embed_chunks", "get_user_query"],
    "generate_answer": ["retrieve"]

}

    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the RAGPipeline instance.

        Sets up a logger and initializes various internal attributes to their
        default states (e.g., `config`, `embedder`, `steps`, `query`). If a
        `config_path` (path to a YAML configuration file) is provided, this
        method will also call `self.load_config()` to load and apply that
        configuration immediately.

        Args:
            config_path (Optional[str], optional): Path to a YAML configuration
                                                   file. Defaults to None.
        """
        self.logger = LoggerManager.get_logger("RAGPipeline")
        self.logger.info("Initializing RAGPipeline...")
        self.config_loader = None
        self.config = None
        self.config_path = None
        self.mode = None
        self.embedder = None
        self.retriever = None
        self.chunked_file = None
        self.index_path = None
        self.metadata_path = None
        self.steps = []  # list of (step_name, kwargs)
        self.query = None
        self.last_chunks = None  #

        if config_path:
            self.config_path = config_path
            self.load_config(config_path)

    def load_config(self, path: str = None) -> None:
        """
        Loads, validates, and applies a YAML configuration from the given path.

        This method uses `ConfigLoader` to read the YAML file. After loading,
        it validates the configuration using `self.validate_config()`.
        It then sets up critical instance attributes based on the loaded config,
        such as `self.mode` (embedding mode), `self.index_path`, and
        `self.metadata_path`. Finally, it calls `self._create_embedder()`
        to instantiate the appropriate embedding client.

        Args:
            path (str, optional): The file path to the YAML configuration file.
                                  If None, it uses `self.config_path` (if set
                                  during initialization). Defaults to None.
        """
        self.logger.info("Loading configuration...")
        if path is None:
            path = self.config_path
        self.config_loader = ConfigLoader(path)
        self.config = self.config_loader._load_config()
        self.validate_config()
        self.logger.info(f"Configuration loaded successfully from: {path}")
        self.mode = self.config["embedding"]["mode"]
        output_dir = self.config["embedding"]["output_dir"]
        self.index_path = os.path.join(output_dir, self.config["embedding"]["index_filename"])
        self.metadata_path = os.path.join(output_dir, self.config["embedding"]["metadata_filename"])


        self.embedder = self._create_embedder()

    def ensure_config_loaded(self):
        """
        Checks if a configuration has been successfully loaded into `self.config`.

        This is a helper method to guard other methods that depend on a loaded
        configuration.

        Args:
            None.

        Returns:
            None.

        Raises:
            RuntimeError: If `self.config` is None, indicating that no
                          configuration has been loaded.
        """
        if not self.config:
            raise RuntimeError("No configuration loaded. Please call load_config(path) first.")

    def get_user_query(self, query: str):
        """
        Set the user query for downstream use in retrieval and generation steps.
        """
        self.logger.info("Setting user query...")
        self.query = query
        print(f"🔍 Query set: {query}")
        self.logger.info(f"User query set: {query}")

    def _create_embedder(self):
        """
        (Private) Creates and returns a `GeneralPurposeEmbedder` instance.

        The type of embedder (and its underlying client, e.g., local model or API)
        is determined by the 'embedding' section of the loaded `self.config`.
        It supports 'local', 'api', and 'batch' modes, instantiating the
        appropriate clients like `LocalModelEmbedder` or `APIClient`.

        Args:
            None (relies on `self.config`).

        Returns:
            GeneralPurposeEmbedder: The configured `GeneralPurposeEmbedder` instance.

        Raises:
            RuntimeError: If `self.config` has not been loaded.
            ValueError: If the `embedding.mode` in the config is unsupported.
        """
        self.logger.info("Creating embedder...")
        if not self.config:
            self.logger.error("Config not loaded. Cannot create embedder.")
            raise RuntimeError("Config not loaded. Cannot create embedder.")

        mode = self.config["embedding"]["mode"]
        model_name = self.config["embedding"]["model_name"]
        embedding_dim = self.config["embedding"]["embedding_dim"]
        output_dir = self.config["embedding"]["output_dir"]
        index_filename = self.config["embedding"]["index_filename"]
        metadata_filename = self.config["embedding"]["metadata_filename"]

        if mode == "local":
            from scripts.embedding.local_model_embedder import LocalModelEmbedder
            from scripts.embedding.general_purpose_embedder import GeneralPurposeEmbedder

            embedder_client = LocalModelEmbedder(model_name)
            self.logger.info("Local model embedder created.")
            return GeneralPurposeEmbedder(
                embedder_client=embedder_client,
                embedding_dim=embedding_dim,
                output_dir=output_dir,
                index_filename=index_filename,
                metadata_filename=metadata_filename
            )

        elif mode == "api":
            from scripts.api_clients.openai.gptApiClient import APIClient
            from scripts.embedding.general_purpose_embedder import GeneralPurposeEmbedder

            api_client = APIClient(config=self.config)
            self.logger.info("API client created.")
            return GeneralPurposeEmbedder(
                embedder_client=api_client,
                embedding_dim=embedding_dim,
                output_dir=output_dir,
                index_filename=index_filename,
                metadata_filename=metadata_filename
            )
        elif mode == "batch":
            from scripts.api_clients.openai.gptApiClient import APIClient
            from scripts.embedding.general_purpose_embedder import GeneralPurposeEmbedder

            api_client = APIClient(config=self.config)
            self.logger.info("Batch API client created for embedding.")
            return GeneralPurposeEmbedder(
                embedder_client=api_client,
                embedding_dim=embedding_dim,
                output_dir=output_dir,
                index_filename=index_filename,
                metadata_filename=metadata_filename
            )


        else:
            raise ValueError(f"Unsupported embedding mode: {mode}")

    def extract_and_chunk(self) -> str:
        """
        Orchestrates the fetching of emails and their subsequent chunking.

        This method uses `EmailFetcher` to retrieve emails based on the 'outlook'
        section of the loaded configuration. The fetched email bodies are then
        processed by `TextChunker`, configured via the 'chunking' section of
        the config, to produce smaller text segments. The resulting chunked
        data is saved to a TSV file, the path to which is also determined by
        the configuration.

        Args:
            None (relies on `self.config`).

        Returns:
            str: The file path to the output TSV file containing the chunked text.

        Raises:
            ValueError: If `EmailFetcher` returns no emails (e.g., empty DataFrame).
        """
        self.logger.info("Starting email extraction and chunking...")
        self.ensure_config_loaded()

        # ✅ Step 1: Fetch emails using the EmailFetcher class
        fetcher = EmailFetcher(self.config)
        # tsv_path = fetcher.fetch_emails_from_folder(return_dataframe=False)
        raw_emails_df = fetcher.fetch_emails_from_folder(return_dataframe=True)


        # ✅ Step 2: check if the DataFrame is empty
        if raw_emails_df.empty or raw_emails_df.columns.empty:
            raise ValueError("❌ No emails fetched — DataFrame is empty. Check Outlook folder path or email filtering.")


        # ✅ Step 3: Chunk based on "Cleaned Body" (assumes it's already cleaned)
        output_file = self.config["paths"]["chunked_emails"]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print("📦 raw_emails preview:")
        print(raw_emails_df.head(3).to_string())


        chunk_cfg = self.config["chunking"]
        chunker = TextChunker(
            max_chunk_size=chunk_cfg.get("max_chunk_size", 500),
            overlap=chunk_cfg.get("overlap", 50),
            min_chunk_size=chunk_cfg.get("min_chunk_size", 150),
            similarity_threshold=chunk_cfg.get("similarity_threshold", 0.8),
            language_model=chunk_cfg.get("language_model", "en_core_web_sm"),
            embedding_model=chunk_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )

        df = pd.DataFrame(raw_emails_df)
        df["Chunks"] = df["Cleaned Body"].apply(lambda x: chunker.chunk_text(str(x)))
        df_chunks = df.explode("Chunks").reset_index(drop=True).rename(columns={"Chunks": "Chunk"})

        # Sanitize the 'Chunk' column as a safeguard before saving to TSV
        def sanitize_text_for_tsv(text: str) -> str:
            if not isinstance(text, str):
                return ""
            # Replace tab, newline, and carriage return with a single space
            sanitized_text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            # Replace multiple consecutive spaces with a single space
            return ' '.join(sanitized_text.split())

        # Ensure 'Chunk' column is string type before applying sanitization
        if "Chunk" in df_chunks.columns:
            df_chunks["Chunk"] = df_chunks["Chunk"].astype(str).apply(sanitize_text_for_tsv)
        
        df_chunks.to_csv(output_file, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)

        self.chunked_file = output_file
        print(f"✅ Chunked email data saved to: {output_file}")
        self.logger.info(f"Chunked email data saved to: {output_file}")
        return output_file

    def embed_chunks(self) -> None:
        """
        Embeds text chunks from the file generated by `extract_and_chunk`.

        This method calls the `run` method of the configured `self.embedder`
        (a `GeneralPurposeEmbedder` instance), passing the path to the
        `self.chunked_file` (output from `extract_and_chunk`) and the name
        of the text column to be embedded.

        Args:
            None (relies on `self.chunked_file` and `self.embedder`).

        Returns:
            None.

        Raises:
            RuntimeError: If `self.chunked_file` is not set (i.e., if
                          `extract_and_chunk` was not run successfully before this).
        """
        self.ensure_config_loaded()
        self.logger.info("Starting chunk embedding...")
        if not self.chunked_file:
            self.logger.error("No chunked file available. Run extract_and_chunk() first.")
            raise RuntimeError("No chunked file available. Run extract_and_chunk() first.")
        self.embedder.run(self.chunked_file, text_column="Chunk")

    def embed_chunks_batch(self) -> None:
        """
        Embeds text chunks in batch mode using the configured embedder.

        This method calls the `run_batch` method of `self.embedder`. It requires
        that `self.chunked_file` has been set by a prior call to
        `extract_and_chunk` and that the configured embedder supports batch
        operations.

        Args:
            None (relies on `self.chunked_file` and `self.embedder`).

        Returns:
            None.

        Raises:
            RuntimeError: If `self.chunked_file` is not set, or if the current
                          embedder (`self.embedder`) does not have a `run_batch` method.
        """
        self.ensure_config_loaded()
        self.logger.info("Starting batch chunk embedding...")

        if not self.chunked_file:
            self.logger.error("No chunked file available. Run extract_and_chunk() first.")
            raise RuntimeError("No chunked file available. Run extract_and_chunk() first.")

        if not hasattr(self.embedder, "run_batch"):
            raise RuntimeError("Current embedder does not support batch embedding.")

        self.embedder.run_batch(self.chunked_file, text_column="Chunk")

    def update_embeddings(self) -> None:
        """
        Updates existing embeddings with new content.

        The process involves:
        1. Fetching new emails using `EmailFetcher`.
        2. Deduplicating these new emails against already processed and cleaned emails.
        3. Chunking the truly new emails using `TextChunker`.
        4. Deduplicating these new chunks against existing chunk metadata.
        5. Embedding only the unique new chunks using `self.embedder.run()`.
        6. Saving metadata about this update run, including counts and paths.

        This method is designed to incrementally add new information to the
        embedding store without reprocessing existing data.

        Args:
            None (relies on `self.config` and existing data files).

        Returns:
            None.
        """
        
        from scripts.utils.data_utils import deduplicate_emails, deduplicate_chunks
        from scripts.data_processing.email.email_fetcher import EmailFetcher
        from scripts.chunking.text_chunker_v2 import TextChunker

        self.ensure_config_loaded()
        task_name = self.config["task_name"]
        run_id = generate_run_id()
        task_paths = TaskPaths(task_name)
        logger = LoggerManager.get_logger("UpdatePipeline", task_paths=task_paths, run_id=run_id)

        logger.info("Starting update_embeddings() run...")

        # Step 1: Fetch new emails
        fetcher = EmailFetcher(self.config)
        new_emails = fetcher.fetch_emails_from_folder(return_dataframe=True, save=False)
        logger.info(f"Fetched {len(new_emails)} raw emails.")

        # Step 2: Deduplicate emails against cleaned dataset
        cleaned_email_path = task_paths.get_cleaned_email_file()
        deduped_emails = deduplicate_emails(new_emails, cleaned_email_path)
        logger.info(f"Deduplicated: {len(deduped_emails)} new emails remain.")

        if deduped_emails.empty:
            logger.info("No new emails to process after deduplication. Exiting update.")
            return

        # Step 3: Save updated cleaned emails
        deduped_emails.to_csv(cleaned_email_path, sep="\t", mode="a", index=False, header=not os.path.exists(cleaned_email_path))
        logger.info(f"Appended cleaned emails to: {cleaned_email_path}")

        # Step 4: Chunk new emails
        chunk_cfg = self.config["chunking"]
        chunker = TextChunker(
            max_chunk_size=chunk_cfg.get("max_chunk_size", 500),
            overlap=chunk_cfg.get("overlap", 50),
            min_chunk_size=chunk_cfg.get("min_chunk_size", 150),
            similarity_threshold=chunk_cfg.get("similarity_threshold", 0.8),
            language_model=chunk_cfg.get("language_model", "en_core_web_sm"),
            embedding_model=chunk_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )

        deduped_emails["Chunks"] = deduped_emails["Cleaned Body"].apply(lambda x: chunker.chunk_text(str(x)))
        chunk_df = deduped_emails.explode("Chunks").reset_index(drop=True).rename(columns={"Chunks": "Chunk"})

        # Step 5: Deduplicate chunks against existing metadata
        chunk_file = task_paths.get_chunk_file()
        chunk_df.to_csv(chunk_file, sep="\t", mode="a", index=False, header=not os.path.exists(chunk_file))

        chunk_meta_path = task_paths.get_metadata_file()
        final_chunks = deduplicate_chunks(chunk_df, chunk_meta_path, text_col="Chunk")
        logger.info(f"Deduplicated chunks: {len(final_chunks)} to embed.")

        # Step 6: Embed chunks and append to FAISS + metadata
        if final_chunks.empty:
            logger.info("No new unique chunks to embed. Skipping embedding stage.")
            return

        self.embedder.run(chunk_file, text_column="Chunk")
        logger.info("Embedding complete. FAISS and metadata updated.")

        log_file_path = None
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                log_file_path = getattr(h, "baseFilename", None)
                break

        if not log_file_path:
            log_file_path = "unknown"

        # Optional: Save run metadata
        run_dir = task_paths.get_update_dir(run_id)
        metadata = {
            "task_name": task_name,
            "run_id": run_id,
            "run_type": "update_embeddings",
            "timestamp": datetime.now().isoformat(),
            "input_summary": {
                "emails_fetched": len(new_emails),
                "emails_after_dedup": len(deduped_emails),
                "chunks_created": len(chunk_df),
                "unique_chunks_embedded": len(final_chunks)
            },
            "output_paths": {
                "cleaned_emails_file": cleaned_email_path,
                "chunk_file": chunk_file,
                "index_file": task_paths.get_index_file(),
                "metadata_file": chunk_meta_path,
                "log_file": log_file_path
            },
            "embedding_model": self.config["embedding"]["model_name"],
            "embedding_dim": self.config["embedding"]["embedding_dim"]
        }
        pd.Series(metadata).to_json(os.path.join(run_dir, "run_metadata.json"), indent=2)
        logger.info(f"Run metadata saved to: {run_dir}/run_metadata.json")

    def retrieve(self, query: Optional[str] = None) -> dict:
        """
        Retrieves relevant text chunks for a given query.

        First, the query is embedded using `self.embedder.embed_query()` to get
        its vector representation. Then, a `ChunkRetriever` instance is used
        to search the FAISS index (`self.index_path`) and find the `top_k`
        most similar chunks based on their embeddings and associated metadata
        (from `self.metadata_path`). The result, including formatted context
        and detailed chunk information, is stored in `self.last_chunks`.

        Args:
            query (Optional[str], optional): The user's query string. If None,
                                             `self.query` (set by `get_user_query`)
                                             is used. Defaults to None.

        Returns:
            dict: A dictionary as returned by `ChunkRetriever.retrieve()`,
                  containing the query, formatted context string, a list of
                  top chunk details, and the path to a debug file.

        Raises:
            ValueError: If no query is available (neither passed directly nor
                        set via `self.query`).
        """
        self.logger.info("Starting chunk retrieval...")
        self.ensure_config_loaded()

        top_k = self.config.get("retrieval", {}).get("top_k", 5)  # ← Read from config

        retriever = ChunkRetriever(
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            top_k=top_k,
            config=self.config
        )

        query = query or self.query
        if not query:
            raise ValueError("No query provided. Use get_user_query() or pass query explicitly.")

        query_vector = self.embedder.embed_query(query)
        result = retriever.retrieve(query_vector=query_vector)
        self.last_chunks = result

        self.logger.info(f"Retrieved {len(result['context'])} relevant chunks for query: {query}")
        return result

    def generate_answer(self, query: Optional[str] = None, chunks: Optional[dict] = None, run_id: Optional[str] = None) -> str:
        """
        Generates a natural language answer based on a query and retrieved chunks.

        This method constructs a prompt using `EmailPromptBuilder` with the given
        query and context from `chunks`. It then sends this prompt to a language
        model via `APIClient` to get a raw answer. Finally, `CitationFormatter`
        is used to process this raw answer, renumbering citations and appending
        a formatted list of sources. The final answer, debug information, and
        run metadata are saved to a run-specific directory.

        Args:
            query (Optional[str], optional): The user's query. If None,
                                             `self.query` is used. Defaults to None.
            chunks (Optional[dict], optional): The dictionary of retrieved chunks
                                               (output from `self.retrieve`). If None,
                                               `self.last_chunks` is used. Defaults to None.
            run_id (Optional[str], optional): An optional identifier for this specific
                                              generation run. If None, a new ID is
                                              generated. Defaults to None.

        Returns:
            str: The generated and formatted answer string.

        Raises:
            ValueError: If no query or no chunks are available.
        """
        self.ensure_config_loaded()
        task_name = self.config.get("task_name", "unnamed_task")
        run_id = run_id or generate_run_id()
        task_paths = TaskPaths(task_name)
        logger = LoggerManager.get_logger("GenerateAnswer", task_paths=task_paths, run_id=run_id)
        run_dir = task_paths.get_run_dir(run_id)
        os.makedirs(run_dir, exist_ok=True)

        logger.info("Generating answer...")

        query = query or self.query
        if not query:
            logger.error("No query provided.")
            raise ValueError("No query provided. Use get_user_query() before calling generate_answer().")

        chunks = chunks or getattr(self, "last_chunks", None)
        if not chunks:
            logger.error("No chunks provided.")
            raise ValueError("No chunks provided. Ensure retrieve() ran before generate_answer().")

        logger.info("Building prompt...")
        prompt_builder = EmailPromptBuilder(style="references")
        client = APIClient(config=self.config)
        prompt = prompt_builder.build(query, chunks["context"])

        logger.info("Sending prompt to OpenAI...")
        answer = client.send_completion_request(prompt)

        # Format citations
        formatter = CitationFormatter(top_chunks=chunks["top_chunks"])
        answer = formatter.finalize_answer(answer)

        logger.info("Answer received.")

        # Save outputs
        answer_path = os.path.join(run_dir, "answer.txt")
        with open(answer_path, "w", encoding="utf-8") as f:
            f.write(answer.strip())

        debug_path = os.path.join(run_dir, "query_debug.txt")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write("Query:\n" + query.strip() + "\n\n")
            f.write("Prompt:\n" + prompt.strip() + "\n\n")
            f.write("Chunks Used:\n\n")
            for chunk in chunks.get("top_chunks", []):
                f.write(f"[{chunk['rank']}] {chunk['text']}\n\n")

        metadata_path = os.path.join(run_dir, "run_metadata.json")
        metadata = {
            "task_name": task_name,
            "run_id": run_id,
            "run_type": "generate_answer",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "model": self.config.get("generation", {}).get("model", "unknown"),
            "top_k": self.config.get("retrieval", {}).get("top_k", "unknown"),
            "num_chunks_used": len(chunks.get("top_chunks", [])),
            "output_paths": {
                "answer_file": answer_path,
                "debug_file": debug_path,
                "log_file": logger.handlers[0].baseFilename if logger.handlers else "unknown"
            }
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print("💬 Generated Answer:\n", answer)
        logger.info(f"Answer saved to: {answer_path}")
        logger.info(f"Run metadata saved to: {metadata_path}")
        return answer

    def run_full_pipeline(self, query: str) -> str:
        """
        Executes a predefined sequence of core pipeline steps for a given query.

        The sequence is:
        1. `self.extract_and_chunk()`
        2. `self.embed_chunks()`
        3. `self.retrieve(query)`
        4. `self.generate_answer(query, chunks)`

        This method provides a convenient way to run the most common end-to-end
        RAG workflow.

        Args:
            query (str): The user's query string.

        Returns:
            str: The final generated and formatted answer from the pipeline.
        """
        self.ensure_config_loaded()
        self.extract_and_chunk()
        self.embed_chunks()
        chunks = self.retrieve(query)
        answer = self.generate_answer(query, chunks)
        return answer

    def configure_task(self, task_name: str, output_format: str = "yaml", overrides: Optional[dict] = None, interactive: bool = False) -> str:
        """
        Create and save a configuration file for a new task.

        Args:
            task_name (str): Unique identifier for the task.
            output_format (str): 'yaml' or 'json'.
            overrides (dict, optional): Values to override in the default template.
            interactive (bool): If True, run interactive CLI to gather config.

        Returns:
            str: Path to the saved configuration file.
        """
        if interactive:
            from scripts.config.task_config_builder import TaskConfigBuilder
            builder = TaskConfigBuilder()
            builder.start(default_task_name=task_name)
            config = builder.config
            task_name = config["task_name"]

        else:
            from scripts.utils.config_templates import get_default_config
            from scripts.utils.merge_utils import deep_merge
            config = get_default_config(task_name)
            if overrides:
                config = deep_merge(config, overrides)

        # Compute task folder structure
        from scripts.utils.paths import TaskPaths
        task_paths = TaskPaths(task_name)

        # Inject paths into config
        config["paths"] = {
            "chunked_emails": os.path.normpath(task_paths.get_chunk_file()),
            "email_dir": os.path.normpath(task_paths.emails_dir),
            "log_dir": os.path.normpath(task_paths.logs_dir),
            "output_dir": os.path.normpath(task_paths.embeddings_dir),
        }
        config["embedding"]["output_dir"] = os.path.normpath(task_paths.embeddings_dir)

        # Save to disk
        os.makedirs("configs/tasks", exist_ok=True)
        save_path = os.path.join("configs/tasks", f"{task_name}.{output_format}")
        if output_format == "yaml":
            with open(save_path, "w", encoding="utf-8") as f:
                clean_config = enforce_scalar_types(config)
                yaml.safe_dump(clean_config, f, default_flow_style=False, sort_keys=False)
        elif output_format == "json":
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
        else:
            raise ValueError("Output format must be 'yaml' or 'json'.")

        print(f"✅ Task configuration saved to: {save_path}")
        return save_path
        
    def validate_config(self):
        """
        Validates the loaded `self.config` against a predefined set of required
        keys and their expected data types.

        This method checks for the presence and correct type of essential
        configuration parameters needed for the pipeline to operate correctly.
        It iterates through a list of required key paths (e.g., "embedding.model_name")
        and their expected types (e.g., `str`).

        Args:
            None (operates on `self.config`).

        Returns:
            None.

        Raises:
            KeyError: If a required configuration key is missing from `self.config`.
            TypeError: If a configuration key has a value of an incorrect data type.
        """
        required = [
            ("embedding", dict),
            ("embedding.model_name", str),
            ("embedding.output_dir", str),
            ("embedding.embedding_dim", int),
            ("retrieval.top_k", int),
            ("outlook.account_name", str),
            ("outlook.folder_path", str),
            ("outlook.days_to_fetch", int),
        ]

        for key_path, expected_type in required:
            parts = key_path.split(".")
            value = self.config
            for part in parts:
                if part not in value:
                    raise KeyError(f"Missing required config key: {key_path}")
                value = value[part]
            if not isinstance(value, expected_type):
                self.logger.error(f"Config key {key_path} must be of type {expected_type.__name__}, got {type(value).__name__}")
                raise TypeError(f"Config key {key_path} must be of type {expected_type.__name__}, got {type(value).__name__}")
            else:
                pass
                # print(f"Config key {key_path} is valid.")
                # self.logger.info(f"Config key {key_path} is valid.")

    def add_step(self, step_name: str, force: bool = False, **kwargs):
        """
        Add a step to the pipeline, validating dependencies and required config keys.

        Each step must meet two criteria:
        1. All dependent steps must already be added (unless force=True)
        2. All required configuration keys for the step must be present in the config

        If any config key is missing, the step is not added and an error is raised.

        Args:
            step_name (str): The name of the pipeline step (e.g., 'embed_chunks').
            force (bool): If True, bypasses dependency checks (not recommended).
            **kwargs: Optional arguments passed to the step during execution.

        Raises:
            AttributeError: If step_name is not a method of this pipeline.
            ValueError: If dependencies or required config keys are missing.
        """
        from scripts.config.step_config_requirements import STEP_REQUIRED_CONFIG

        if not hasattr(self, step_name):
            raise AttributeError(f"Step '{step_name}' is not a method of RAGPipeline.")

        if step_name not in self.STEP_DEPENDENCIES:
            raise ValueError(f"Step '{step_name}' is not a recognized pipeline step.")

        added_steps = [s for s, _ in self.steps]

        # Step dependency validation
        missing_dependencies = [
            dep for dep in self.STEP_DEPENDENCIES[step_name]
            if dep not in added_steps
        ]
        if missing_dependencies and not force:
            raise ValueError(
                f"Cannot add step '{step_name}' — missing prerequisite(s): {missing_dependencies}.\n"
                f"You can override this check with force=True if you are sure these steps were already completed."
            )

        # Step config validation
        required_keys = STEP_REQUIRED_CONFIG.get(step_name, [])
        missing_keys = []
        for key in required_keys:
            try:
                self.config_loader.get(key)
            except KeyError:
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(
                f"Step '{step_name}' cannot be added — missing required config keys:\n" +
                "\n".join(f"  - {k}" for k in missing_keys)
            )

        self.steps.append((step_name, kwargs))
        print(f"✅ Step '{step_name}' added{' (force override)' if force else ''}.")

    def clear_steps(self):
        """Remove all configured steps from the pipeline."""
        self.steps.clear()
        print("🧹 Pipeline steps cleared.")

    def run_steps(self):
        """
        Execute all steps added via `add_step()` in order.
        """
        print("🚀 Running configured pipeline steps...\n")
        for step_name, kwargs in self.steps:
            step_fn = getattr(self, step_name)
            print(f"➡️ Running step: {step_name}")
            try:
                result = step_fn(**kwargs)
                print(f"✅ Step '{step_name}' completed.\n")

                # If final step is generate_answer → show + save result
                if step_name == "generate_answer":
                    print("🧠 Final Answer:")
                    print(result)

                    task_name = self.config.get("task_name", "unnamed_task")
                    output_path = os.path.join("outputs", "answers", f"{task_name}.txt")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(result)
                    print(f"📁 Answer saved to: {output_path}")

            except Exception as e:
                print(f"❌ Step '{step_name}' failed with error: {e}")
                raise

    def pipe_review(self):
        """
        Display a review of the current pipeline configuration, including steps, model, and paths.
        """
        self.ensure_config_loaded()

        print("\n🚀 RAG Pipeline Configuration Review")
        print("==========================================")
        print(f"Config Source: {getattr(self.config_loader, 'config_path', '[Not loaded via config_loader]')}\n")

        print("[Steps in Pipeline]")
        if self.steps:
            for i, (step, kwargs) in enumerate(self.steps, start=1):
                print(f"  {i}. {step} {'(with arguments)' if kwargs else ''}")
        else:
            print("  (No steps added to pipeline yet)")
        print()

        emb_cfg = self.config.get("embedding", {})
        print("[Chunking]")
        print(f"Max Chunk Size: {self.config['chunking']['max_chunk_size']}")
        print(f"Overlap: {self.config['chunking']['overlap']}")
        print(f"Min Chunk Size: {self.config['chunking']['min_chunk_size']}")
        print(f"Similarity Threshold: {self.config['chunking']['similarity_threshold']}")
        print(f"Language Model: {self.config['chunking']['language_model']}")
        print(f"Embedding Model for Similarity: {self.config['chunking']['embedding_model']}\n")

        print("[Embedding]")
        print(f"  Mode: {emb_cfg.get('mode', 'N/A')}")
        print(f"  Model: {emb_cfg.get('model_name', 'N/A')}")
        print(f"  Embedding Dimension: {emb_cfg.get('embedding_dim', 'N/A')}")
        print(f"  Output Directory: {emb_cfg.get('output_dir', 'N/A')}\n")

        print("[Retrieval]")
        print(f"  FAISS Index Path: {getattr(self, 'index_path', 'N/A')}")
        print(f"  Metadata Path: {getattr(self, 'metadata_path', 'N/A')}")
        print(f"  Top-K: {self.config.get('retrieval', {}).get('top_k', 'N/A')}\n")

        print("[Prompting]")
        print("  Prompt Builder: EmailPromptBuilder (default)\n")

        print("[Answer Generation]")
        gen_cfg = self.config.get("generation", {})
        print(f"  Model: {gen_cfg.get('model', 'openai-gpt-4')}")
        print("==========================================\n")

    def describe_steps(self):
        """
        Print a list of available pipeline steps, their descriptions, and dependencies.
        """
        STEP_INFO = {
            "extract_and_chunk": {
                "desc": "Fetch emails from Outlook and chunk them into segments.",
                "depends_on": []
            },
            "embed_chunks": {
                "desc": "Embed chunked text using a local model or OpenAI API.",
                "depends_on": []
            },
            "embed_chunks_batch": {
            "desc": "Embed chunked text using OpenAI's batch embedding API.",
            "depends_on": []
            },

            "get_user_query": {
                "desc": "Prompt the user (or system) to input a natural language query.",
                "depends_on": []
            },
            "retrieve": {
                "desc": "Use the embedded query to retrieve top-K relevant chunks.",
                "depends_on": ["embed_chunks", "get_user_query"]
            },
            "generate_answer": {
                "desc": "Generate a natural language answer using retrieved chunks.",
                "depends_on": ["retrieve"]
            },
        }

        print("\n📚 Available RAG Pipeline Steps")
        print("=======================================")
        for step, info in STEP_INFO.items():
            deps = ", ".join(info["depends_on"]) if info["depends_on"] else "None"
            print(f"🔹 {step}")
            print(f"    Description: {info['desc']}")
            print(f"    Depends on: {deps}\n")
