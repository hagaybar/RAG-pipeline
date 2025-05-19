from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import faiss
import os
from scripts.api_clients.openai.gptApiClient import APIClient
import logging


class ChunkRetriever:
    def __init__(self,
                 index_path: str,
                 metadata_path: str,
                 text_column: str = "Chunk",
                 date_column: str = "Received",
                 top_k: int = 5,
                 debug_dir: str = "debug",
                 config: Optional[dict] = None):
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
        Retrieve top-K chunks based on query or query vector.
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.debug_dir, f"query_{timestamp}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n\n")
            for chunk in labeled_chunks:
                f.write(chunk + "\n\n")
        return file_path