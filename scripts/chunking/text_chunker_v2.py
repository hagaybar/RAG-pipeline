"""
This module provides the TextChunker class, which is responsible for splitting
text into meaningful chunks. It employs a combination of syntactic strategies,
such as sentence boundary detection, and semantic strategies, like analyzing
embedding similarity, to achieve coherent chunking.
"""
import re
import spacy
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import logging

class TextChunker:
    """
    Splits text into semantically meaningful and appropriately sized chunks.

    This class implements a multi-step chunking strategy to create chunks
    that are both semantically coherent and suitable for downstream embedding
    and retrieval tasks. The process involves:

    1.  **Sentence Segmentation**: Utilizes a spaCy language model (specified by
        `language_model`) to initially break down the input text into
        individual sentences. This forms the basic syntactic units.
    2.  **Sentence Embedding**: Generates vector embeddings for each sentence using a
        sentence-transformer model (specified by `embedding_model`). These
        embeddings capture the semantic meaning of the sentences.
    3.  **Semantic Chunking**: Groups consecutive sentences based on the cosine
        similarity of their embeddings. Sentences with similarity scores above
        the `similarity_threshold` are merged into initial semantic chunks.
    4.  **Size Enforcement**:
        -   **Max Size**: Ensures that no chunk exceeds `max_chunk_size` (in tokens).
            If a semantic chunk is too large, it is further divided, maintaining
            an `overlap` between the new sub-chunks to preserve context.
        -   **Min Size**: Merges chunks that are shorter than `min_chunk_size`
            (character length) with adjacent chunks to avoid overly fragmented text,
            enhancing contextual integrity.

    The overall aim is to produce text segments that are rich in semantic meaning
    while adhering to size constraints optimal for language model processing and
    effective information retrieval.

    Key Initialization Parameters:
        language_model (str): The spaCy language model identifier used for sentence
                              segmentation (e.g., "en_core_web_sm").
        embedding_model (str): The sentence-transformers model identifier used for
                               generating sentence embeddings (e.g.,
                               "sentence-transformers/all-MiniLM-L6-v2").
        max_chunk_size (int): The maximum number of tokens allowed in a single chunk.
        overlap (int): The number of tokens to overlap between chunks when a larger
                       chunk is split due to exceeding `max_chunk_size`.
        similarity_threshold (float): The cosine similarity score (0.0 to 1.0) above
                                      which consecutive sentences will be grouped
                                      during semantic chunking.
        min_chunk_size (int): The minimum character length for a chunk. Chunks shorter
                              than this are merged with adjacent ones if possible.
    """

    def __init__(self, language_model: str = "en_core_web_sm", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_chunk_size: int = 500, overlap: int = 50, similarity_threshold: float = 0.8, min_chunk_size: int = 150):
        """
        Initialize the TextChunker with SpaCy and embedding models.

        Args:
            language_model (str): The SpaCy language model to use for sentence segmentation.
            embedding_model (str): The sentence-transformers model to use for semantic embeddings.
            max_chunk_size (int): Maximum size of each chunk (in tokens).
            overlap (int): Overlapping tokens between chunks.
            similarity_threshold (float): Cosine similarity threshold for semantic chunking.
        """
        self.nlp = spacy.load(language_model)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using SpaCy.

        Args:
            text (str): The input text.

        Returns:
            List[str]: A list of sentences.
        """
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of sentences.

        Args:
            sentences (List[str]): List of sentences.

        Returns:
            np.ndarray: Array of sentence embeddings.
        """
        if not sentences:
            logging.debug("No sentences to embed. Skipping embedding.")
            return np.zeros((0, self.embedding_dim))
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def _semantic_chunking(self, sentences: List[str]) -> List[str]:
        """
        Perform semantic chunking based on sentence embeddings and similarity.

        Args:
            sentences (List[str]): List of sentences.

        Returns:
            List[str]: List of semantically coherent chunks.
        """
        embeddings = self._get_sentence_embeddings(sentences)
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            if similarity >= self.similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_text(self, text: str) -> List[str]:
        """
        Split and group text into chunks using both syntactic and semantic strategies.

        Args:
            text (str): The input text.

        Returns:
            List[str]: A list of text chunks.
        """
        if not text.strip():
            return []
        
        sentences = self._split_sentences(text)
        initial_chunks = self._semantic_chunking(sentences)

        final_chunks = []
        for chunk in initial_chunks:
            tokens = self.tokenizer.encode(chunk, add_special_tokens=False)
            if len(tokens) > self.max_chunk_size:
                split_chunks = [
                    tokens[i:i + self.max_chunk_size]
                    for i in range(0, len(tokens), self.max_chunk_size - self.overlap)
                ]
                for split_chunk in split_chunks:
                    text = self.tokenizer.decode(split_chunk, skip_special_tokens=True)
                    final_chunks.append(text)
            else:
                final_chunks.append(chunk)


        # Post-process: merge short chunks
        final_filtered_chunks = []
        buffer = ""

        for chunk in final_chunks:
            if len(chunk) < self.min_chunk_size:
                buffer += " " + chunk
            else:
                if buffer:
                    final_filtered_chunks.append(buffer.strip())
                    buffer = ""
                final_filtered_chunks.append(chunk.strip())

        if buffer:
            final_filtered_chunks.append(buffer.strip())

        return final_filtered_chunks

