# scripts/embedding/local_model_embedder.py

from typing import List
from sentence_transformers import SentenceTransformer

class LocalModelEmbedder:
    """
    Adapter class for local sentence-transformer models
    to be compatible with GeneralPurposeEmbedder.

    Exposes .embed(texts) -> List[List[float]]
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts.
        """
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string (returns a 1D vector).
        """
        return self.embed([query])[0]
