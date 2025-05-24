import unittest
import os
import shutil
import tempfile
import pandas as pd
import yaml
from pathlib import Path
from unittest.mock import patch

from scripts.pipeline.rag_pipeline import RAGPipeline
# Assuming APIClient is in scripts.api_clients.openai.gptApiClient
# This might need adjustment based on actual project structure if different
from scripts.api_clients.openai.gptApiClient import APIClient 


# Placeholder for LoggerManager if it's not readily available or for simplicity
class LoggerManager:
    @staticmethod
    def get_logger(name, level=None, task_paths=None, run_id=None):
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG) # Default to DEBUG for tests, can be overridden
        return logger

# If RAGPipeline's logger is initialized via its own LoggerManager,
# we might need to patch scripts.pipeline.rag_pipeline.LoggerManager
# For now, this local one is for test-specific logging if needed.

SAMPLE_XML_CONTENT = """
<root>
  <item>
    <title>Test Document 1</title>
    <description>This is a sample XML for testing.</description>
  </item>
  <item>
    <title>Another Element</title>
    <text>Some more text content here.</text>
  </item>
</root>
"""

EXPECTED_CLEANED_TEXT = "Test Document 1 This is a sample XML for testing. Another Element Some more text content here."

class TestRAGPipelineXML(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.input_xml_dir = Path(self.test_dir) / "input_xml"
        self.output_chunks_dir = Path(self.test_dir) / "output_chunks"
        self.output_embeddings_dir = Path(self.test_dir) / "output_embeddings"
        self.logs_dir = Path(self.test_dir) / "logs"

        self.input_xml_dir.mkdir(parents=True, exist_ok=True)
        self.output_chunks_dir.mkdir(parents=True, exist_ok=True)
        self.output_embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.sample_xml_path = self._create_sample_xml(
            self.input_xml_dir, "sample.xml", SAMPLE_XML_CONTENT
        )
        self.test_config_path = self._create_test_config_yaml()

        self.pipeline = RAGPipeline()
        # Patch LoggerManager within the RAGPipeline module to control logging during tests
        # self.logger_patch = patch('scripts.pipeline.rag_pipeline.LoggerManager', LoggerManager)
        # self.mock_logger = self.logger_patch.start()


    def tearDown(self):
        # self.logger_patch.stop()
        shutil.rmtree(self.test_dir)

    def _create_sample_xml(self, directory: Path, filename: str, content: str) -> Path:
        file_path = directory / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path.resolve() # Return absolute path

    def _create_test_config_yaml(self) -> str:
        config_data = {
            "task_name": "xml_test_task",
            "embedding": {
                "mode": "local", # local model for testing
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dim": 384, # Dimension for all-MiniLM-L6-v2
                "output_dir": str(self.output_embeddings_dir),
                "index_filename": "test_faiss_index.idx",
                "metadata_filename": "test_metadata.json",
            },
            "chunking": {
                "max_chunk_size": 100, # Smaller chunks for easier testing
                "overlap": 10,
                "min_chunk_size": 20,
                "language_model": "en_core_web_sm", # spaCy model
                 "embedding_model": "sentence-transformers/all-MiniLM-L6-v2" # for semantic chunking if used
            },
            "xml_files": {
                "input_dir": str(self.input_xml_dir)
            },
            "paths": {
                "chunked_xml_files": str(self.output_chunks_dir / "chunked_xml_data.tsv"),
                "log_dir": str(self.logs_dir),
                # Dummy paths for other data types if RAGPipeline's config validation needs them
                "chunked_emails": str(self.output_chunks_dir / "chunked_email_data.tsv"),
                "chunked_text_files": str(self.output_chunks_dir / "chunked_text_data.tsv"),
                "email_dir": str(self.test_dir / "dummy_email_dir"), # Dummy
                "text_output_dir_raw": str(self.test_dir / "dummy_text_raw_dir"), # Dummy
            },
            "retrieval": {"top_k": 2},
            "generation": { # Dummy values, will be mocked
                "model": "gpt-dummy",
                "api_key_env": "DUMMY_OPENAI_API_KEY" 
            },
            # Dummy sections if strictly required by config validation
            "outlook": { 
                "account_name": "dummy@example.com",
                "folder_path": "Inbox",
                "days_to_fetch": 1
            },
            "text_files": {
                "input_dir": str(self.test_dir / "dummy_text_input")
            }
        }
        # Create dummy dirs referenced in paths if validation is strict
        (self.test_dir / "dummy_email_dir").mkdir(exist_ok=True)
        (self.test_dir / "dummy_text_input").mkdir(exist_ok=True)
        (self.test_dir / "dummy_text_raw_dir").mkdir(exist_ok=True)


        config_path = Path(self.test_dir) / "test_config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)
        return str(config_path)

    def test_xml_extraction_and_chunking(self):
        self.pipeline.load_config(self.test_config_path)
        
        chunked_file_path_output = self.pipeline.extract_and_chunk(data_type="xml")
        self.assertIsNotNone(chunked_file_path_output, "extract_and_chunk should return the path to the chunked file.")
        
        chunked_file_path_config = self.pipeline.config["paths"]["chunked_xml_files"]
        self.assertEqual(Path(chunked_file_path_output).resolve(), Path(chunked_file_path_config).resolve(),
                         "Returned chunked file path should match the one in config.")

        self.assertTrue(Path(chunked_file_path_config).exists(), "Chunked XML TSV file was not created.")
        
        df = pd.read_csv(chunked_file_path_config, sep="\t")
        
        expected_columns = ["File Path", "Raw Text", "Cleaned Text", "Chunk"]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Column '{col}' missing in chunked output.")
            
        self.assertGreater(len(df), 0, "Chunked DataFrame should not be empty.")
        
        # Verify "File Path"
        self.assertEqual(Path(df["File Path"].iloc[0]).resolve(), self.sample_xml_path,
                         "File Path column does not point to the correct sample XML file.")
                         
        # Verify "Raw Text"
        self.assertEqual(df["Raw Text"].iloc[0].strip(), SAMPLE_XML_CONTENT.strip(),
                         "Raw Text column does not match sample XML content.")

        # Verify "Cleaned Text" - all rows corresponding to the same original file should have the same cleaned text
        # The XMLFetcher's _extract_text_from_xml joins with " " and strips individual element texts.
        # Let's adjust EXPECTED_CLEANED_TEXT based on how XMLFetcher actually processes it.
        # XMLFetcher iterates, strips text from elements, then joins with " ".
        # Original: Test Document 1, This is a sample XML for testing., Another Element, Some more text content here.
        # Expected: "Test Document 1 This is a sample XML for testing. Another Element Some more text content here."
        actual_cleaned_text_from_df = df["Cleaned Text"].iloc[0]
        self.assertEqual(actual_cleaned_text_from_df, EXPECTED_CLEANED_TEXT,
                         f"Cleaned Text column content mismatch. Expected: '{EXPECTED_CLEANED_TEXT}', Got: '{actual_cleaned_text_from_df}'")

        # Verify "Chunk"
        self.assertTrue(all(df["Chunk"].apply(lambda x: isinstance(x, str) and len(x) > 0)),
                        "Chunk column should contain non-empty text chunks.")
        
        # Verify chunks are part of the cleaned text
        for chunk_text in df["Chunk"]:
            self.assertIn(chunk_text, actual_cleaned_text_from_df,
                          f"Chunk '{chunk_text}' is not a substring of the cleaned text '{actual_cleaned_text_from_df}'.")


    @patch.object(APIClient, 'send_completion_request')
    def test_full_pipeline_with_xml(self, mock_send_completion_request):
        # Mock the LLM call to avoid actual API interaction
        mock_send_completion_request.return_value = "This is a mocked answer based on XML content."

        self.pipeline.load_config(self.test_config_path)
        
        # 1. Extract and Chunk
        self.pipeline.extract_and_chunk(data_type="xml")
        self.assertTrue(Path(self.pipeline.chunked_file).exists(), "Chunked file not created for full pipeline test.")

        # 2. Embed Chunks (uses local model specified in config)
        # This step might be slow if spacy models or sentence-transformers need to download.
        # For true unit testing, this might also be mocked if it becomes an issue.
        # However, testing with a small local embedder is valuable.
        try:
            self.pipeline.embed_chunks()
        except Exception as e:
            # If downloads fail due to no internet in sandbox, this test might need adjustment
            # or the test environment needs to pre-cache these models.
            if "ConnectionError" in str(e) or "offline" in str(e).lower():
                self.skipTest(f"Skipping embed_chunks due to potential network issue: {e}")
            else:
                raise e
                
        self.assertTrue(Path(self.pipeline.index_path).exists(), "FAISS index file not created.")
        self.assertTrue(Path(self.pipeline.metadata_path).exists(), "Metadata file for embeddings not created.")

        # 3. Retrieve
        query = "Test Document 1" # Query related to content in sample.xml
        self.pipeline.get_user_query(query) # Set query on pipeline instance
        chunks_result = self.pipeline.retrieve(query=query)
        
        self.assertIsNotNone(chunks_result, "Retrieval result should not be None.")
        self.assertIn("top_chunks", chunks_result, "Retrieval result should contain 'top_chunks'.")
        self.assertGreater(len(chunks_result["top_chunks"]), 0, "No chunks retrieved.")
        
        # Optional: Check if retrieved content is relevant
        # This is a simple check; more sophisticated checks might parse XML content
        first_retrieved_chunk_text = chunks_result["top_chunks"][0]["text"]
        self.assertTrue(query.lower() in first_retrieved_chunk_text.lower() or \
                        "sample xml" in first_retrieved_chunk_text.lower(),
                        f"Retrieved chunk '{first_retrieved_chunk_text}' doesn't seem relevant to query '{query}'.")

        # 4. Generate Answer (with mocked LLM call)
        answer = self.pipeline.generate_answer(query=query, chunks=chunks_result)
        
        self.assertIsNotNone(answer, "Generated answer should not be None.")
        self.assertEqual(answer, "This is a mocked answer based on XML content.",
                         "Generated answer does not match mocked response.")
        mock_send_completion_request.assert_called_once()


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

```
