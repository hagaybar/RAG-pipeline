import unittest
import os
import shutil
import tempfile
import pandas as pd
import yaml
from pathlib import Path
from unittest.mock import patch
import pymarc # For creating MARC data and potential exceptions

from scripts.pipeline.rag_pipeline import RAGPipeline
from scripts.api_clients.openai.gptApiClient import APIClient # For mocking

# Updated Sample MARC data for creating test files, reflecting refined extraction
SAMPLE_MARC_RECORDS_DATA = [
    {
        "leader": "00000nam a2200000M  4500", # Corrected leader length
        "fields": [
            {"tag": "001", "data": "12345"},
            {"tag": "008", "data": "060922s2006    maua   j      000 0 eng d"}, # Date1: 2006, Lang: eng
            {"tag": "100", "indicators": ["1", " "], "subfields": [("a", "Rowling, J. K."), ("d", "1965-")]},
            {"tag": "245", "indicators": ["1", "0"], "subfields": [("a", "Harry Potter and the Half-Blood Prince"), ("c", "by J.K. Rowling.")]},
            {"tag": "520", "indicators": [" ", " "], "subfields": [("a", "Sixth-year Hogwarts student Harry Potter gains valuable insights into the boy Voldemort once was.")]},
            {"tag": "650", "indicators": [" ", "0"], "subfields": [("a", "Wizards"), ("v", "Juvenile fiction.")]}
        ],
        # Expected text based on MARCXMLFetcher's refined _extract_text_from_record:
        # Priority fields: 245a, 245c, 520a, 100a, 100d, 650a, 650v. Then 008 data.
        "expected_cleaned_text": "Harry Potter and the Half-Blood Prince by J.K. Rowling. Sixth-year Hogwarts student Harry Potter gains valuable insights into the boy Voldemort once was. Rowling, J. K. 1965- Wizards Juvenile fiction. Publication Year: 2006 Language: eng"
    },
    {
        "leader": "00001cam a2200001K  4500", # Corrected leader length
        "fields": [
            {"tag": "001", "data": "67890"},
            {"tag": "008", "data": "980115s1998    nyu    b      001 0 eng  "}, # Date1: 1998, Lang: eng
            {"tag": "245", "indicators": ["0", "0"], "subfields": [("a", "Introduction to algorithms"), ("c", "Thomas H. Cormen ... [et al.].")]},
            {"tag": "505", "indicators": ["0", " "], "subfields": [("a", "Includes bibliographical references (p. 1147-1160) and index.")]},
            # Field 260 is NOT in priority_fields_subfields in MARCXMLFetcher
            {"tag": "260", "indicators": [" ", " "], "subfields": [("a", "Cambridge, Mass. :"), ("b", "MIT Press ;"), ("c", "c1998.")]},
            {"tag": "650", "indicators": [" ", "0"], "subfields": [("a", "Algorithms"), ("x", "Data structures (Computer science)")]}
        ],
        # Expected text: 245a, 245c, 505a, 650a, 650x. Then 008 data.
        "expected_cleaned_text": "Introduction to algorithms Thomas H. Cormen ... [et al.]. Includes bibliographical references (p. 1147-1160) and index. Algorithms Data structures (Computer science) Publication Year: 1998 Language: eng"
    }
]

class TestRAGPipelineMARCXML(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.input_marcxml_dir = Path(self.test_dir) / "input_marcxml"
        self.output_chunks_dir = Path(self.test_dir) / "output_chunks"
        self.output_embeddings_dir = Path(self.test_dir) / "output_embeddings"
        self.logs_dir = Path(self.test_dir) / "logs"

        self.input_marcxml_dir.mkdir(parents=True, exist_ok=True)
        self.output_chunks_dir.mkdir(parents=True, exist_ok=True)
        self.output_embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.sample_marcxml_filepath = self.input_marcxml_dir / "sample.marc.xml"
        self._create_sample_marcxml_file(self.sample_marcxml_filepath, SAMPLE_MARC_RECORDS_DATA)
        
        self.test_config_path = self._create_test_config_yaml()

        self.pipeline = RAGPipeline()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_sample_marcxml_file(self, filepath: Path, records_list_data: list):
        # Ensure the directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        writer = pymarc.XMLWriter(open(str(filepath), 'wb'))
        for record_data in records_list_data:
            record = pymarc.Record(leader=record_data['leader'])
            for field_info in record_data['fields']:
                if 'data' in field_info: # Control field
                    record.add_field(pymarc.Field(tag=field_info['tag'], data=field_info['data']))
                else: # Data field
                    subfields_pymarc = []
                    for code, value in field_info['subfields']:
                        subfields_pymarc.extend([code, value])
                    field = pymarc.Field(
                        tag=field_info['tag'],
                        indicators=[ind for ind in field_info['indicators']],
                        subfields=subfields_pymarc
                    )
                    record.add_field(field)
            writer.write(record)
        writer.close()
        return filepath.resolve()

    def _create_test_config_yaml(self) -> str:
        config_data = {
            "task_name": "marcxml_test_task_refined",
            "embedding": {
                "mode": "local",
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dim": 384,
                "output_dir": str(self.output_embeddings_dir),
                "index_filename": "test_faiss_marc_refined_index.idx",
                "metadata_filename": "test_marc_refined_metadata.json",
            },
            "chunking": {
                "max_chunk_size": 100, # Smaller for testing, original was 150
                "overlap": 15,        # Original was 20
                "min_chunk_size": 25, # Original was 30
                "language_model": "en_core_web_sm",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "marcxml_files": {
                "input_dir": str(self.input_marcxml_dir)
            },
            "paths": {
                "chunked_marcxml_files": str(self.output_chunks_dir / "chunked_marcxml_refined_data.tsv"),
                "log_dir": str(self.logs_dir),
                "chunked_emails": str(self.output_chunks_dir / "chunked_email_data.tsv"),
                "chunked_text_files": str(self.output_chunks_dir / "chunked_text_data.tsv"),
                "chunked_xml_files": str(self.output_chunks_dir / "chunked_xml_generic_data.tsv"),
                "email_dir": str(self.test_dir / "dummy_email_dir"),
                "text_output_dir_raw": str(self.test_dir / "dummy_text_raw_dir"),
            },
            "retrieval": {"top_k": 2},
            "generation": {"model": "gpt-dummy", "api_key_env": "DUMMY_OPENAI_API_KEY"},
            "outlook": {"account_name": "d", "folder_path": "f", "days_to_fetch": 1},
            "text_files": {"input_dir": str(self.test_dir / "dummy_text_input")},
            "xml_files": {"input_dir": str(self.test_dir / "dummy_xml_input")}
        }
        (self.test_dir / "dummy_email_dir").mkdir(exist_ok=True)
        (self.test_dir / "dummy_text_input").mkdir(exist_ok=True)
        (self.test_dir / "dummy_xml_input").mkdir(exist_ok=True)
        (self.test_dir / "dummy_text_raw_dir").mkdir(exist_ok=True)

        config_path = Path(self.test_dir) / "test_config_marcxml_refined.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)
        return str(config_path)

    def test_marcxml_extraction_and_chunking(self):
        self.pipeline.load_config(self.test_config_path)
        
        chunked_file_path_output = self.pipeline.extract_and_chunk(data_type="marcxml")
        self.assertIsNotNone(chunked_file_path_output)
        
        chunked_file_path_config = self.pipeline.config["paths"]["chunked_marcxml_files"]
        self.assertEqual(Path(chunked_file_path_output).resolve(), Path(chunked_file_path_config).resolve())
        self.assertTrue(Path(chunked_file_path_config).exists())
        
        df = pd.read_csv(chunked_file_path_config, sep="\t")
        
        expected_columns = ["File Path", "Raw Text", "Cleaned Text", "Chunk"]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Column '{col}' missing.")
            
        self.assertGreater(len(df), 0, "Chunked DataFrame is empty.")
        
        self.assertEqual(df["Cleaned Text"].nunique(), len(SAMPLE_MARC_RECORDS_DATA),
                         "Number of unique 'Cleaned Text' entries should match number of MARC records.")

        for i, record_data in enumerate(SAMPLE_MARC_RECORDS_DATA):
            # Assert that the exact expected cleaned text is present for each record
            # This is the primary check for the refined extraction logic
            self.assertTrue((df["Cleaned Text"] == record_data["expected_cleaned_text"]).any(),
                            f"Expected cleaned text for record {i} ('{record_data['expected_cleaned_text']}') not found in DataFrame's 'Cleaned Text' column.")
            
            record_df_slice = df[df["Cleaned Text"] == record_data["expected_cleaned_text"]]
            self.assertGreater(len(record_df_slice), 0, f"No rows found for record {i}'s expected cleaned text.")

            self.assertEqual(Path(record_df_slice["File Path"].iloc[0]).resolve(), self.sample_marcxml_filepath.resolve(),
                             f"File Path for record {i} mismatch.")
            
            with open(self.sample_marcxml_filepath, 'r', encoding='utf-8') as f_xml:
                expected_raw_text = f_xml.read()
            self.assertEqual(record_df_slice["Raw Text"].iloc[0].strip(), expected_raw_text.strip(),
                             f"Raw Text for record {i} mismatch.")

            self.assertTrue(all(record_df_slice["Chunk"].apply(lambda x: isinstance(x, str) and len(x) > 0)),
                            f"Chunk column for record {i} should contain non-empty text chunks.")
            for chunk_text in record_df_slice["Chunk"]:
                self.assertIn(chunk_text, record_data["expected_cleaned_text"],
                              f"Chunk '{chunk_text}' for record {i} is not part of its cleaned text '{record_data['expected_cleaned_text']}'.")
    
    @patch.object(APIClient, 'send_completion_request')
    def test_full_pipeline_with_marcxml(self, mock_send_completion_request):
        # Using the first sample record's content for query and mocked answer
        mocked_answer_content = "Mocked answer: Harry Potter gains valuable insights. Published in 2006, language eng."
        mock_send_completion_request.return_value = mocked_answer_content

        self.pipeline.load_config(self.test_config_path)
        
        self.pipeline.extract_and_chunk(data_type="marcxml")
        self.assertTrue(Path(self.pipeline.chunked_file).exists())

        try:
            self.pipeline.embed_chunks()
        except Exception as e:
            if "ConnectionError" in str(e) or "offline" in str(e).lower() or "HFValidationError" in str(e):
                self.skipTest(f"Skipping embed_chunks due to potential network/HF model validation issue: {e}")
            else:
                raise e
                
        self.assertTrue(Path(self.pipeline.index_path).exists())
        self.assertTrue(Path(self.pipeline.metadata_path).exists())

        # Query related to the first sample record's refined text
        query = "Tell me about Harry Potter insights and publication year" 
        self.pipeline.get_user_query(query)
        chunks_result = self.pipeline.retrieve(query=query)
        
        self.assertIsNotNone(chunks_result)
        self.assertIn("top_chunks", chunks_result)
        self.assertGreater(len(chunks_result["top_chunks"]), 0, "No chunks retrieved for MARCXML query.")

        first_retrieved_chunk_text = chunks_result["top_chunks"][0]["text"]
        # Check if retrieved chunk relates to the expected content, e.g., from the first sample record
        self.assertTrue("harry potter" in first_retrieved_chunk_text.lower() or \
                        "voldemort" in first_retrieved_chunk_text.lower() or \
                        "publication year: 2006" in first_retrieved_chunk_text.lower(),
                        f"Retrieved chunk '{first_retrieved_chunk_text}' doesn't seem relevant to query '{query}'. Expected content related to Harry Potter 2006.")
        
        answer = self.pipeline.generate_answer(query=query, chunks=chunks_result)
        
        self.assertIsNotNone(answer)
        self.assertEqual(answer, mocked_answer_content)
        mock_send_completion_request.assert_called_once()

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
