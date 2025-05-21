"""
This module defines the ChunkRetriever class for searching and retrieving
relevant text chunks from a FAISS index. It supports querying by text (which
it embeds) or by a pre-computed vector, filtering by metadata (e.g., date
ranges), and formats the output for use in Retrieval-Augmented Generation
(RAG) systems.
"""
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import faiss
import os
from scripts.api_clients.openai.gptApiClient import APIClient
import logging


class ChunkRetriever:
    """
    Retrieves relevant text chunks from a FAISS index based on vector similarity to a query.

    This class encapsulates the logic for finding and fetching the `top_k` most
    relevant text chunks that match a given query. It uses a pre-built FAISS
    index for efficient vector search and a corresponding metadata file that stores
    additional information about each chunk.

    Key Features:
    - **Initialization**: Requires paths to the FAISS index (`index_path`) and a
      metadata file (`metadata_path`). It's also configured with `top_k` (the
      number of chunks to retrieve) and can take an optional `config` for its
      internal `APIClient`, which is used if a raw text query needs embedding.
      The `text_column` and `date_column` parameters specify the relevant columns
      in the metadata file.
    - **Querying**: The `retrieve` method can be called with either:
        - A raw query string: The retriever will use its `APIClient` to generate an
          embedding for this query.
        - A pre-computed query vector.
    - **Date Filtering**: If a `date_range` (tuple of start and end date strings)
      is provided to `retrieve` and a valid `date_column` (specified during
      initialization) exists in the metadata, the search space is filtered to include
      only chunks within that date range before performing the similarity search.
      Dates in metadata are converted to datetime objects for comparison.
    - **Similarity Search**: It performs a dot product similarity search between the
      query vector and the chunk embeddings in the FAISS index (or the filtered
      subset if date filtering is applied).
    - **Context Formatting**: The retrieved `top_k` chunks are formatted for use as
      context in a language model. This includes adding ranking labels (e.g., `[1]`),
      and prepending key metadata like Sender and Date (from the metadata file,
      using specified column names) to each chunk's text.
    - **Debugging Output**: Saves a debug file (in the `debug_dir` specified at
      initialization) containing the query and the content of the retrieved, labeled
      chunks for inspection.
    - **Return Value**: The `retrieve` method returns a dictionary containing:
        - `query`: The original query string or a placeholder if a vector was provided.
        - `context`: A single string where all formatted and labeled top chunks
          are concatenated, ready for LLM consumption.
        - `top_chunks`: A list of dictionaries, where each dictionary holds detailed
          information about one retrieved chunk (rank, original index, similarity
          score, text, label, and its full metadata).
        - `debug_file`: The path to the saved debug file.

    It relies on `faiss` for vector search, `pandas` for metadata handling, and an
    `APIClient` (from `scripts.api_clients.openai.gptApiClient`) for on-the-fly
    query embedding.
    """
    def __init__(self,
                 index_path: str,
                 metadata_path: str,
                 text_column: str = "Chunk",
                 date_column: str = "Received",
                 top_k: int = 5,
                 debug_dir: str = "debug",
                 config: Optional[dict] = None):
        """
        Initializes the ChunkRetriever.

        Loads the FAISS index and metadata, initializes an APIClient for query
        embedding, and sets up directories and parameters for retrieval.

        Args:
            index_path (str): Path to the pre-built FAISS index file.
            metadata_path (str): Path to the TSV file containing metadata
                                 associated with the chunks in the FAISS index.
            text_column (str, optional): The name of the column in the metadata
                                         file that holds the actual text of the
                                         chunks. Defaults to "Chunk".
            date_column (str, optional): The name of the column in the metadata
                                         file that holds date information, used for
                                         optional date-based filtering. Defaults to "Received".
            top_k (int, optional): The default number of top relevant chunks to
                                   retrieve. Defaults to 5.
            debug_dir (str, optional): The directory where debug files (containing
                                       query and retrieved chunks) will be saved.
                                       Defaults to "debug".
            config (Optional[dict], optional): A configuration dictionary passed
                                               to the internal `APIClient`, typically
                                               for API key and budget settings if
                                               query embedding is needed. Defaults to None.
        """
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_csv(metadata_path, sep="\t")
        self.text_column = text_column
        self.date_column = date_column
        self.top_k = top_k
        self.api_client = APIClient(config=config)
        os.makedirs(debug_dir, exist_ok=True)
        self.debug_dir = debug_dir

        if self.date_column in self.metadata.columns:
            self.metadata[self.date_column] = pd.to_datetime(self.metadata[self.date_column], errors="coerce")

    def retrieve(
        self,
        query: Optional[str] = None,
        query_vector: Optional[np.ndarray] = None,
        date_range: Optional[Tuple[str, str]] = None
    ) -> Dict:
        """
        Retrieves the top-K most relevant chunks from the FAISS index based on
        similarity to the given query (either text or vector). Supports optional
        date filtering.

        If a text `query` is provided, it's first embedded using the internal
        API client. If a `query_vector` is provided, it's used directly.
        If `date_range` is specified, chunks are filtered by this range before
        the similarity search. The results include formatted context for LLMs
        and detailed information about each retrieved chunk.

        Args:
            query (Optional[str], optional): The raw text query. If provided and
                                             `query_vector` is None, this query will be
                                             embedded using the internal API client.
                                             Defaults to None.
            query_vector (Optional[np.ndarray], optional): A pre-computed embedding
                                                          vector for the query. If both
                                                          `query` and `query_vector` are
                                                          provided, `query_vector` takes
                                                          precedence. Defaults to None.
            date_range (Optional[Tuple[str, str]], optional): A tuple `(start_date, end_date)`
                                                            for filtering chunks by date.
                                                            Dates should be in a format
                                                            parseable by `pd.to_datetime`.
                                                            Defaults to None.

        Returns:
            Dict: A dictionary containing:
                  - "query" (str): The original query string, or "[embedded vector]"
                                   if a vector was input.
                  - "context" (str): A single string concatenating all labeled and
                                     formatted top chunks, suitable for LLM context.
                  - "top_chunks" (List[Dict]): A list of dictionaries, where each
                                               details a retrieved chunk (rank,
                                               original index, score, text, label,
                                               metadata).
                  - "debug_file" (str): Path to the saved debug file for this retrieval.
                  Returns an empty dictionary if no relevant chunks are found after filtering.

        Raises:
            ValueError: If neither `query` nor `query_vector` is provided.
        """
        if query_vector is None:
            if not query:
                raise ValueError("Must provide either `query` or `query_vector`.")
            embedding = self.api_client.get_embedding(query)
            query_vector = np.array([embedding], dtype="float32")
        else:
            if isinstance(query_vector, list):
                query_vector = np.array(query_vector, dtype="float32")
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

        metadata_filtered = self.metadata.copy()

        if date_range and self.date_column in metadata_filtered.columns:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            metadata_filtered = metadata_filtered[
                (metadata_filtered[self.date_column] >= start) &
                (metadata_filtered[self.date_column] <= end)
            ]

        if metadata_filtered.empty:
            return {}

        faiss_ids = metadata_filtered.index.to_list()
        all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
        subset_vectors = np.array([all_vectors[i] for i in faiss_ids], dtype="float32")

        scores = np.dot(subset_vectors, query_vector.T).flatten()
        top_indices = np.argsort(scores)[-self.top_k:][::-1]

        results = []
        labeled_chunks = []
        for rank, idx in enumerate(top_indices):
            metadata_idx = faiss_ids[idx]
            row = self.metadata.iloc[metadata_idx]
            label = f"[{rank + 1}]"
            date_str = row[self.date_column].strftime("%Y-%m-%d") if pd.notna(row[self.date_column]) else "Unknown"
            label_header = f"{label} (Sender: {row.get('Sender', 'Unknown')}, Date: {date_str})"
            chunk_text = f"{label_header}\n{row[self.text_column].strip()}"
            labeled_chunks.append(chunk_text)

            result = {
                "rank": rank + 1,
                "index": int(metadata_idx),
                "score": float(scores[idx]),
                "text": row[self.text_column],
                "label": label,
                "metadata": row.to_dict()
            }
            results.append(result)

        openai_context = "\n\n".join(labeled_chunks)
        debug_file_path = self._save_debug_file(query or "[embedded vector]", results, labeled_chunks)

        return {
            "query": query or "[embedded vector]",
            "context": openai_context,
            "top_chunks": results,
            "debug_file": debug_file_path
        }



    def _save_debug_file(self, query: str, results: List[Dict], labeled_chunks: List[str]) -> str:
        """
        (Private) Saves debugging information for a retrieval operation to a
        timestamped text file.

        The file includes the original query and the list of formatted, labeled
        chunks that were retrieved.

        Args:
            query (str): The query string used for the retrieval.
            results (List[Dict]): The list of raw result dictionaries for the
                                  top retrieved chunks (currently not directly used
                                  in the file content but passed for potential future use).
            labeled_chunks (List[str]): The list of formatted, labeled chunk strings
                                        that are written to the debug file.

        Returns:
            str: The absolute path to the created debug file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.debug_dir, f"query_{timestamp}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n\n")
            for chunk in labeled_chunks:
                f.write(chunk + "\n\n")
        return file_path