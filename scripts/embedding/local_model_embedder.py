# scripts/embedding/local_model_embedder.py
"""
This module contains the `LocalModelEmbedder` class, an adapter for utilizing
local sentence-transformer models to generate text embeddings. It ensures
compatibility with the broader embedding infrastructure, such as the
`GeneralPurposeEmbedder`, by providing a standardized `embed` method.
"""
from typing import List
from sentence_transformers import SentenceTransformer

class LocalModelEmbedder:
    """
    Adapter class for local sentence-transformer models
    to be compatible with GeneralPurposeEmbedder.

    Exposes .embed(texts) -> List[List[float]]
    """

    def __init__(self, model_name: str):
        """
        Initializes the LocalModelEmbedder.

        Args:
            model_name (str): The name or path of the sentence-transformer model
                              to load (e.g., "sentence-transformers/all-MiniLM-L6-v2").
                              The model is loaded using the SentenceTransformer library.
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts using the loaded sentence-transformer model.
        Normalizes embeddings and shows a progress bar during encoding.

        Args:
            texts (List[str]): A list of text strings to be embedded.

        Returns:
            List[List[float]]: A list of embedding vectors, where each vector
                               is a list of floats.
        """
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string (returns a 1D vector).
        """
        return self.embed([query])[0]
