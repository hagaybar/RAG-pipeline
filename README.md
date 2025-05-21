# 📚 RAG Email Pipeline Prototype

This repository contains a modular Retrieval-Augmented Generation (RAG) pipeline designed to extract and process email data, generate vector embeddings, and provide AI-assisted answers using the OpenAI API. Currently email data from local outlook client is the only input, the input options should be extended soon.

---

## 🚀 Project Overview

This project demonstrates a complete RAG workflow using email content as input:

1. **Email Extraction** – Fetch emails from Outlook folders
2. **Text Cleaning** – Normalize content, remove signatures and quoted replies
3. **Chunking** – Split emails into semantically coherent text chunks
4. **Embedding** – Convert chunks into vectors using local or API models
5. **Retrieval** – Match queries against embedded chunks
6. **Answer Generation** – Construct prompts and query OpenAI to synthesize answers

---

## 🧬 Core Components

The RAG pipeline is orchestrated by the `RAGPipeline` class (`scripts/pipeline/rag_pipeline.py`). It manages the entire workflow, from initial data ingestion to final answer generation. The pipeline is highly modular and configurable via YAML files, allowing users to customize its behavior and components.

Key classes include:

-   **`RAGPipeline` (`scripts/pipeline/rag_pipeline.py`)**: The central orchestrator. It loads the configuration and manages the sequential execution of different stages:
    -   Data extraction (e.g., fetching emails)
    -   Text processing and chunking
    -   Vector embedding generation
    -   Information retrieval based on user queries
    -   Prompt construction and answer generation using a language model.
-   **`EmailFetcher` (`scripts/data_processing/email/email_fetcher.py`)**: Responsible for fetching emails from various sources, with current support focused on Microsoft Outlook.
-   **`TextChunker` (`scripts/chunking/text_chunker_v2.py`)**: Implements logic to split large texts (like email bodies) into smaller, semantically coherent chunks suitable for embedding and retrieval.
-   **`GeneralPurposeEmbedder` (`scripts/embedding/general_purpose_embedder.py`)**: Manages the creation of vector embeddings from text chunks. It can be configured to use different embedding models, including local ones or those accessed via APIs.
-   **`LocalModelEmbedder` (`scripts/embedding/local_model_embedder.py`)**: A specific embedder that utilizes local sentence transformer models to generate embeddings, allowing for offline processing.
-   **`APIClient` (`scripts/api_clients/openai/gptApiClient.py`)**: Handles communication with external APIs, primarily the OpenAI API. It's used for generating embeddings (if using OpenAI's models) and for the final answer generation step through language models like GPT.
-   **`ChunkRetriever` (`scripts/retrieval/chunk_retriever_v3.py`)**: Performs similarity searches against a vector store (e.g., FAISS) to find and retrieve the most relevant text chunks based on the user's query.
-   **`EmailPromptBuilder` (`scripts/prompting/prompt_builder.py`)**: Constructs carefully engineered prompts for the language model. These prompts typically include the user's query and the context provided by the retrieved text chunks to guide the AI in generating relevant and accurate answers.
-   **`CitationFormatter` (`scripts/formatting/citation_formatter.py`)**: Post-processes the language model's output to include citations, linking the generated answer back to the source email documents.

The pipeline's modular design allows for individual components to be swapped or customized, providing flexibility for different use cases and data sources. Configuration for each step and component is managed through YAML files located in the `configs/` directory.

---

## 🧱 Project Structure

```
configs/            # YAML configuration files (general + task-specific)
data/               # Cleaned emails, chunks, logs, embeddings
debug/              # Query debug output
outputs/            # Final generated answers
scripts/            # Modular pipeline components
    api_clients/    # Manages communication with external APIs (e.g., OpenAI, Google)
    chunking/       # Contains logic for splitting text into smaller, manageable chunks
    config/         # Handles loading, validation, and building of configuration files
    data_processing/ # Includes modules for cleaning and processing various data types (e.g., emails, HTML)
    embedding/      # Responsible for generating vector embeddings from text using different models
    formatting/     # Utilities for formatting text, like citations
    pipeline/       # Core orchestration logic for the RAG pipeline
    prompting/      # Utilities for constructing prompts for language models
    retrieval/      # Handles searching and retrieving relevant chunks based on queries
    ui/             # User interface components (if applicable)
    utils/          # Common utility functions used across the project
tests/              # Test cases and sample inputs
docs/               # Notes, planning, and requirements
```

---

## ⚙️ Configuration

Main settings live in YAML files under `configs/`. Task configs include:
- `config.yaml` – Core settings
- `test_full_api.yaml`, `test_full_local.yaml` – API and local embedding runs

Use `RAGPipeline.configure_task()` to create a new config dynamically.

---

## 🔧 Requirements

Install dependencies with:

```bash
pip install -r docs/requirements.txt
```

Recommended:
- Python 3.10+
- Windows with Outlook installed (for email extraction)

---

## 🧪 Running the Pipeline

Example full pipeline run:

```python
from scripts.pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(config_path="configs/tasks/test_full_api.yaml")
pipeline.run_full_pipeline(query="How can I find deleted POLs in Alma?")
```

Or build it step by step using `add_step()` and `run_steps()`.

---

## 🔐 API Keys

Set your OpenAI key in an environment variable:

```bash
export OPEN_AI=your-api-key
```

Or supply it directly when creating the `APIClient`.

---

## 📄 License

[MIT License](https://opensource.org/license/mit)

---

## 🤝 Acknowledgments

Built with:
- [OpenAI API](https://platform.openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [sentence-transformers](https://www.sbert.net/)
- Microsoft Outlook COM API (via `pywin32`)
