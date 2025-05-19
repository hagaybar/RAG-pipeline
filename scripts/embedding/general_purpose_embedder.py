# scripts/embedding/general_purpose_embedder.py

import os
import pandas as pd
import numpy as np
import faiss
from typing import Optional, Union, List
from scripts.api_clients.openai.batch_embedder import BatchEmbedder

class GeneralPurposeEmbedder:
    """
    GeneralPurposeEmbedder

    Embeds text chunks from a TSV file into vector embeddings and saves them into a FAISS index
    along with metadata.

    Supports:
    - Any embedding backend (local models like sentence-transformers OR API clients like OpenAI)
    - Flexible embedding dimensions
    - Duplicate checking to avoid redundant embeddings
    - Configuration via YAML file for paths, model choices, and modes
    - Batch embedding using OpenAI's batch API via the `run_batch()` method for efficient large-scale processing

    Expected Usage:
    ----------------
    1. Initialize an embedder client that has an `embed(texts: List[str]) -> List[List[float]]` method.
    2. Pass the client and configuration into GeneralPurposeEmbedder.
    3. Call `run(chunked_file_path, text_column)` to perform synchronous embedding.
    4. Call `run_batch(chunked_file_path, text_column)` to perform asynchronous batch embedding using OpenAI.

    Args:
        embedder_client (object): Must expose an `.embed(texts)` method.
        embedding_dim (int): The dimension of the output embeddings.
        output_dir (str): Directory for saving the FAISS index and metadata.
        index_filename (str): Filename for storing the FAISS index.
        metadata_filename (str): Filename for storing chunk metadata.

    Example:
    --------
    >>> embedder = GeneralPurposeEmbedder(LocalModelEmbedder(), 384)
    >>> embedder.run("data/chunks.tsv", text_column="Chunk")
    >>> embedder.run_batch("data/chunks.tsv", text_column="Chunk")
    """

    def __init__(self,
                 embedder_client: object,
                 embedding_dim: int,
                 output_dir: str = "embeddings",
                 index_filename: str = "chunks.index",
                 metadata_filename: str = "chunks_metadata.tsv"):
        self.embedder_client = embedder_client
        self.embedding_dim = embedding_dim
        self.output_dir = output_dir
        self.index_path = os.path.join(output_dir, index_filename)
        self.metadata_path = os.path.join(output_dir, metadata_filename)

        os.makedirs(self.output_dir, exist_ok=True)


    def embed_dataframe(self, df: pd.DataFrame, text_column: str = "Chunk") -> np.ndarray:
        """Embeds the text in the specified column of a DataFrame."""
        texts = df[text_column].tolist()
        embeddings = self.embedder_client.embed(texts)
        return np.array(embeddings, dtype="float32")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query using the same client used for chunk embeddings.
        Ensures consistency in embedding space.
        """
        return self.embedder_client.embed([query])[0]


    def save_index(self, new_embeddings: np.ndarray) -> None:
        """Saves or appends new embeddings to the FAISS index."""
        if os.path.exists(self.index_path):
            index = faiss.read_index(self.index_path)
            index.add(new_embeddings)
        else:
            index = faiss.IndexFlatL2(self.embedding_dim)
            index.add(new_embeddings)
        faiss.write_index(index, self.index_path)

    def save_metadata(self, new_df: pd.DataFrame) -> None:
        """Appends new metadata to the metadata file."""
        if os.path.exists(self.metadata_path):
            new_df.to_csv(self.metadata_path, sep="\t", mode='a', index=False, header=False)
        else:
            new_df.to_csv(self.metadata_path, sep="\t", index=False)

    def run(self, chunked_file_path: str, text_column: str = "Chunk") -> None:
        """Full embedding workflow: load chunks, embed, update index and metadata."""
        df = pd.read_csv(chunked_file_path, sep="\t")
        if text_column not in df.columns:
            raise ValueError(f"'{text_column}' column not found in file: {chunked_file_path}")

        if os.path.exists(self.metadata_path):
            existing_chunks = pd.read_csv(self.metadata_path, sep="\t", usecols=[text_column])
            before_count = len(df)
            df = df[~df[text_column].isin(existing_chunks[text_column])]
            after_count = len(df)
            print(f"🧹 Filtered out {before_count - after_count} duplicate chunks. {after_count} new chunks remain.")

        if df.empty:
            print("⚠️ No new chunks to embed.")
            return

        new_embeddings = self.embed_dataframe(df, text_column=text_column)
        self.save_index(new_embeddings)
        self.save_metadata(df)

    def run_batch(self, chunked_file_path: str, text_column: str = "Chunk") -> None:
            """
            Full batch embedding workflow using OpenAI's batch API.
            Loads text chunks, submits batch job, parses results, and saves to FAISS + metadata.
            """
            df = pd.read_csv(chunked_file_path, sep="\t")
            if text_column not in df.columns:
                raise ValueError(f"'{text_column}' column not found in file: {chunked_file_path}")

            if os.path.exists(self.metadata_path):
                existing_chunks = pd.read_csv(self.metadata_path, sep="\t", usecols=[text_column])
                before_count = len(df)
                df = df[~df[text_column].isin(existing_chunks[text_column])]
                after_count = len(df)
                print(f"\U0001f9f9 Filtered out {before_count - after_count} duplicate chunks. {after_count} new chunks remain.")

            if df.empty:
                print("⚠️ No new chunks to embed.")
                return

            texts = df[text_column].tolist()
            ids = [f"chunk-{i}" for i in range(len(texts))]
            df["custom_id"] = ids

            batch_embedder = BatchEmbedder(model="text-embedding-3-small", output_dir=self.output_dir)
            embeddings_dict = batch_embedder.run(texts)

            # Order embeddings according to the original custom_id list
            ordered_embeddings = [embeddings_dict[cid] for cid in df["custom_id"] if cid in embeddings_dict]
            embedding_matrix = np.array(ordered_embeddings, dtype="float32")

            self.save_index(embedding_matrix)
            self.save_metadata(df.drop(columns=["custom_id"]))
