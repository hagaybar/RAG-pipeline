# 📚 RAG Email Pipeline Prototype

This repository contains a modular Retrieval-Augmented Generation (RAG) pipeline designed to extract and process various data sources like emails, generic XML files, and MARCXML bibliographic records. It generates vector embeddings from this data and provides AI-assisted answers using the OpenAI API.

---

## 🚀 Project Overview

This project demonstrates a complete RAG workflow using data sources such as emails, XML files, or MARCXML records as input:

1. **Data Extraction** – Fetch data from Outlook folders (for emails), local directories (for text files, XML files, or MARCXML files).
2. **Text Cleaning/Parsing** – Normalize content (e.g., remove email signatures), parse XML structure, or extract specific bibliographic data from MARCXML records.
3. **Chunking** – Split extracted text content into semantically coherent text chunks.
4. **Embedding** – Convert chunks into vectors using local or API models
5. **Retrieval** – Match queries against embedded chunks
6. **Answer Generation** – Construct prompts and query OpenAI to synthesize answers

---

## 🧬 Core Components

The RAG pipeline is orchestrated by the `RAGPipeline` class (`scripts/pipeline/rag_pipeline.py`). It manages the entire workflow, from initial data ingestion to final answer generation. The pipeline is highly modular and configurable via YAML files, allowing users to customize its behavior and components.

Key classes include:

-   **`RAGPipeline` (`scripts/pipeline/rag_pipeline.py`)**: The central orchestrator. It loads the configuration and manages the sequential execution of different stages:
    -   Data extraction (e.g., fetching emails, text files, generic XML files, MARCXML records).
    -   Text processing, parsing (including targeted MARCXML data extraction), and chunking.
    -   Vector embedding generation.
    -   Information retrieval based on user queries.
    -   Prompt construction and answer generation using a language model.
-   **`EmailFetcher` (`scripts/data_processing/email/email_fetcher.py`)**: Responsible for fetching emails from Microsoft Outlook.
-   **`TextFileFetcher` (`scripts/data_processing/text/text_fetcher.py`)**: Responsible for fetching and processing plain text files from a directory.
-   **`XMLFetcher` (`scripts/data_processing/xml/xml_fetcher.py`)**: Responsible for fetching, parsing, and extracting all text content from generic XML files in a directory.
-   **`MARCXMLFetcher` (`scripts/data_processing/marc_xml/marc_xml_fetcher.py`)**: Specialized for processing MARCXML bibliographic records. It uses the `pymarc` library to parse records and implements a refined text extraction logic. This logic selectively extracts data from common bibliographic fields such as titles (e.g., MARC tag 245), summaries (e.g., 520), author/creator names (e.g., 1XX, 7XX), and subject headings (e.g., 6XX). Additionally, it extracts and formats key information from control field `008`, specifically the publication year and language code, making them human-readable within the "Cleaned Text" output.
-   **`TextChunker` (`scripts/chunking/text_chunker_v2.py`)**: Implements logic to split large texts (like email bodies or extracted text from files) into smaller, semantically coherent chunks suitable for embedding and retrieval.
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
    - `config.yaml` – Core settings template.
    - `test_full_api.yaml`, `test_full_local.yaml` – Example task configurations for API and local embedding runs.

Use `RAGPipeline.configure_task()` to create a new task-specific configuration file.

Key configuration options in your task-specific YAML file include:

-   `task_name`: A unique name for your task.
-   `embedding`: Settings for the embedding model (mode, model_name, dimension, output paths).
    -   `mode`: Can be "local", "api", or "batch".
-   `chunking`: Parameters for text chunking (max_chunk_size, overlap, etc.).
-   `data_sources`: Configuration for input data types. This section should contain one or more of the following keys based on the data you want to process:
    -   `outlook` (for emails): Requires `account_name`, `folder_path`, `days_to_fetch`.
    -   `text_files` (for plain .txt files): Requires `input_dir`.
    -   `xml_files` (for generic .xml files): Requires `input_dir`.
    -   `marcxml_files` (for MARCXML .xml files): Requires `input_dir`.
-   `paths`: Defines various input and output paths. Ensure the relevant chunked file path is configured:
    -   `chunked_emails`: Output path for chunked email data (TSV).
    -   `chunked_text_files`: Output path for chunked text file data (TSV).
    -   `chunked_xml_files`: Output path for chunked generic XML file data (TSV).
    -   `chunked_marcxml_files`: Output path for chunked MARCXML record data (TSV).
    -   Other paths for logs, raw data outputs, etc.
-   `retrieval`: Settings for the retrieval process (e.g., `top_k` chunks to retrieve).
-   `generation`: Configuration for the answer generation model (e.g., OpenAI model name).

When using `RAGPipeline.extract_and_chunk(data_type="...")`, ensure the corresponding configuration sections (e.g., `outlook`, `text_files`, `xml_files`, `marcxml_files`) and paths (e.g., `paths.chunked_emails`, `paths.chunked_text_files`, `paths.chunked_xml_files`, `paths.chunked_marcxml_files`) are correctly set up in your task's YAML file. The `data_type` parameter can be "email", "text_file", "xml", or "marcxml".

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
- [pymarc](https://github.com/edsu/pymarc) (for MARCXML processing)
