# scripts/config/step_config_requirements.py

# This dictionary maps each step to its required configuration keys.
# The keys are the step names, and the values are lists of configuration keys that are required for each step.  

STEP_REQUIRED_CONFIG = {
    "extract_and_chunk": [
        "chunking.max_chunk_size",
        "chunking.overlap",
        "chunking.min_chunk_size",
        "chunking.similarity_threshold",
        "chunking.language_model",
        "chunking.embedding_model"
    ],
    "embed_chunks": [
        "embedding.model_name",
        "embedding.output_dir",
        "embedding.embedding_dim",
        "embedding.index_filename",
        "embedding.metadata_filename"
    ],
    "embed_chunks_batch": [
        "embedding.model_name",
        "embedding.output_dir",
        "embedding.embedding_dim",
        "embedding.index_filename",
        "embedding.metadata_filename"
    ],
    "get_user_query": [],
    "retrieve": [
        "retrieval.top_k",
        "embedding.model_name"
    ],
    "generate_answer": [
        "generation.model"
    ],
    "update_embeddings": [
        "task_name",
        "chunking.max_chunk_size",
        "embedding.model_name"
    ]
}
