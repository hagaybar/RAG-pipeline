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

## ⚙️ Setup and Installation

Follow these steps to set up and run the RAG Email Pipeline:

1.  **Get the Code:**
    *   Clone the repository to your local machine:
        ```bash
        git clone <repository_url_placeholder>
        ```
    *   Alternatively, download the source code as a ZIP file and extract it.

2.  **Navigate to Project Directory:**
    *   Open your terminal or command prompt and change to the project's root directory:
        ```bash
        cd path/to/rag-email-pipeline 
        ```
        (Replace `path/to/rag-email-pipeline` with the actual path to the cloned/extracted directory).

3.  **Set up Python Environment (Recommended):**
    *   This project requires Python 3.10 or newer. Ensure you have a compatible version installed. You can check your Python version by running `python --version` or `python3 --version`.
    *   It's highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python projects:
        ```bash
        python -m venv venv
        ```
        (This command creates a new directory named `venv` in your project folder).
    *   Activate the virtual environment:
        *   On macOS and Linux:
            ```bash
            source venv/bin/activate
            ```
        *   On Windows:
            ```bash
            venv\Scripts\activate
            ```
        (Your terminal prompt might change to indicate the virtual environment is active).

4.  **Install Dependencies:**
    *   With your virtual environment activated, install the required Python packages using the `requirements.txt` file located in the project root:
        ```bash
        pip install -r requirements.txt
        ```

5.  **Set Up API Keys:**
    *   This pipeline uses the OpenAI API for certain functionalities (like generating answers with GPT models). If you plan to use these features, an OpenAI API key is required.
    *   Set your OpenAI API key as an environment variable named `OPEN_AI`. This is generally more secure than hardcoding the key in scripts.
        *   On macOS and Linux (bash/zsh):
            ```bash
            export OPEN_AI="your-api-key"
            ```
        *   On Windows (Command Prompt):
            ```bash
            set OPEN_AI=your-api-key
            ```
        *   On Windows (PowerShell):
            ```bash
            $env:OPEN_AI="your-api-key"
            ```
    *   Replace `"your-api-key"` or `your-api-key` with your actual OpenAI API key.
    *   Alternatively, you can supply the API key directly when creating the `APIClient` instance in your Python scripts, though this is less recommended for production or shared code.

6.  **Outlook Dependency (Important Note for Email Extraction):**
    *   The email extraction feature of this pipeline currently relies on `pywin32` to connect to a local Microsoft Outlook client.
    *   This means:
        *   This specific functionality is **only available on Windows systems.**
        *   You must have **Microsoft Outlook installed and configured** on that Windows system.
    *   Users on other operating systems (macOS, Linux) or those not using Outlook will not be able to use the direct email extraction features from Outlook. The rest of the pipeline (e.g., processing text from other sources, embedding, retrieval, answer generation) can still be used if data is provided through alternative means.

---

## 🚀 Running the Pipeline (Example)

Once you have completed the setup and installation, you can run the pipeline with an example script. The `main_playground.py` script demonstrates how to initialize and run the `RAGPipeline`.

1.  **Ensure your environment is set up:**
    *   Make sure your virtual environment (if you created one) is activated.
    *   Ensure your `OPEN_AI` environment variable is set if you plan to use OpenAI models.

2.  **Run the example script:**
    *   Open your terminal in the root directory of the project.
    *   You can run the `main_playground.py` script with a command like this:

        ```bash
        python scripts/pipeline/main_playground.py --config configs/tasks/test_full_api.yaml --query "How can I find deleted POLs in Alma?"
        ```

3.  **Understanding the Command:**
    *   `python scripts/pipeline/main_playground.py`: This executes the playground script.
    *   `--config configs/tasks/test_full_api.yaml`: This flag specifies the path to the YAML configuration file for the pipeline task. The `configs/tasks/` directory contains example configurations. `test_full_api.yaml` is typically configured to use API-based embeddings and generation.
    *   `--query "How can I find deleted POLs in Alma?"`: This flag sets the query you want the RAG pipeline to answer.

4.  **Expected Output:**
    *   The script will process the query through the RAG pipeline. You will see log messages in your terminal indicating the different stages (email extraction (if applicable and configured), chunking, embedding, retrieval, answer generation).
    *   The final generated answer will be printed to the terminal.
    *   Detailed outputs, including the generated answer, debug information, and logs, are typically saved in a subdirectory under `runs/<task_name>/runs/<run_id>/`, where `<task_name>` is derived from your configuration and `<run_id>` is a unique identifier for that specific execution. (Refer to the "Project Structure" section for more details).

5.  **Customization:**
    *   You can change the `--config` argument to point to other configuration files (e.g., `test_full_local.yaml` for local embeddings, or your own custom configurations).
    *   Modify the `--query` string to ask your own questions.
    *   Explore the configuration files in `configs/tasks/` to understand how different components of the pipeline are set up (e.g., email sources, embedding models, retrieval parameters).

This example provides a starting point. You can further explore the scripts in the `scripts/pipeline/` directory and the `RAGPipeline` class itself to understand the full capabilities and customize the workflow.

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
configs/            # YAML configuration files for tasks (e.g., configs/tasks/my_task.yaml)
scripts/            # All Python scripts for pipeline components and utilities
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
runs/               # Root directory for all task-specific operational data
    <task_name>/    # Each task gets its own subdirectory within runs/
        emails/         # Cleaned email data for the task
        chunks/         # Chunked text data (e.g., chunked_emails.tsv)
        embeddings/     # FAISS index and metadata for embeddings
        logs/           # Log files for the task (e.g., task.log, <run_id>.log)
        runs/           # Data specific to individual pipeline executions (runs)
            <run_id>/   # Each execution gets a unique run ID
                answer.txt          # Generated answer for this run
                query_debug.txt     # Debug information for the query and retrieved context
                run_metadata.json   # Metadata about this specific run
        updates/        # Data specific to embedding update operations
            <update_id>/
                run_metadata.json   # Metadata about this specific update run
outputs/            # General output directory
    answers/        # Final answers, typically one per task when using run_steps() (e.g., outputs/answers/<task_name>.txt)
tests/              # Test cases and sample inputs
docs/               # Original location for notes, planning, and requirements.txt (now moved)
```

---

## ⚙️ Configuration

Main settings live in YAML files under `configs/`. Task configs include:
- `config.yaml` – Core settings
- `test_full_api.yaml`, `test_full_local.yaml` – API and local embedding runs

Use `RAGPipeline.configure_task()` to create a new config dynamically.

---

## 📋 System Requirements

This section outlines the system-level requirements and recommendations for running the RAG Email Pipeline.

**Recommended:**
- **Python:** As mentioned in the "Setup and Installation" section, Python 3.10 or newer is required.
- **Operating System for Email Extraction:** As noted in the "Setup and Installation" section, Windows with Microsoft Outlook installed and configured is necessary for the email fetching features.

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
