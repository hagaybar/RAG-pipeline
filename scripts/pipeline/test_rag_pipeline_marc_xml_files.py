import unittest
import os
import shutil
import tempfile
import pandas as pd
import yaml
import json # For parsing JSON strings
from pathlib import Path
from unittest.mock import patch, call # Ensure call is imported for checking call_args
import pymarc 

from scripts.pipeline.rag_pipeline import RAGPipeline
from scripts.api_clients.openai.gptApiClient import APIClient 
from scripts.prompting.prompt_builder import TextFilePromptBuilder # For patching

# Updated Sample MARC data. `expected_text_for_embedding` needs to be carefully constructed
# based on MARCXMLProcessor's _build_text_for_embedding logic.
# For these integration tests, NER entities will be empty unless explicitly mocked during MARCXMLProcessor run.
# We assume MARCXMLProcessor is unit-tested for its NER capabilities.
# The key_metadata for _build_text_for_embedding will be what MARCXMLProcessor extracts.

SAMPLE_MARC_RECORDS_DATA_INTEGRATION = [
    {
        "leader": "00000nam a2200000M  4500",
        "fields": [
            {"tag": "001", "data": "marc001"},
            {"tag": "008", "data": "060922s2006    maua   j      000 0 eng d"}, 
            {"tag": "100", "indicators": ["1", " "], "subfields": [("a", "Rowling, J. K."), ("d", "1965-")]},
            {"tag": "245", "indicators": ["1", "0"], "subfields": [("a", "Harry Potter and the Half-Blood Prince"), ("c", "by J.K. Rowling.")]},
            {"tag": "520", "indicators": [" ", " "], "subfields": [("a", "Sixth-year Hogwarts student Harry Potter gains valuable insights into the boy Voldemort once was.")]},
            {"tag": "650", "indicators": [" ", "0"], "subfields": [("a", "Wizards"), ("v", "Juvenile fiction.")]}
        ],
        # Manually derived based on MARCXMLProcessor logic:
        # Title (245a): "Harry Potter and the Half-Blood Prince"
        # Author (100a): "Rowling, J. K."
        # Summary (520a): "Sixth-year Hogwarts student Harry Potter gains valuable insights into the boy Voldemort once was."
        # Subjects (650a, 650v): "Wizards", "Juvenile fiction." -> "Wizards Juvenile fiction."
        # Pub Year (008): "Published: 2006"
        # Languages (008): "Languages: eng"
        # Entities: Assume empty for this integration test's expected string.
        "expected_text_for_embedding": "Harry Potter and the Half-Blood Prince Rowling, J. K. Sixth-year Hogwarts student Harry Potter gains valuable insights into the boy Voldemort once was. Wizards Juvenile fiction. Published: 2006 Languages: eng"
    },
    {
        "leader": "00001cam a2200001K  4500", 
        "fields": [
            {"tag": "001", "data": "marc002"},
            {"tag": "008", "data": "980115s1998    nyu    b      001 0 eng  "}, 
            {"tag": "041", "indicators": ["0", " "], "subfields": [("a", "eng"), ("h", "ger")]}, # Lang eng, original ger
            {"tag": "245", "indicators": ["0", "0"], "subfields": [("a", "Introduction to algorithms"), ("c", "Thomas H. Cormen ... [et al.].")]},
            {"tag": "505", "indicators": ["0", " "], "subfields": [("a", "Includes bibliographical references and index.")]},
            {"tag": "650", "indicators": [" ", "0"], "subfields": [("a", "Algorithms"), ("x", "Data structures (Computer science)")]}
        ],
        # Title (245a): "Introduction to algorithms"
        # Summary (505a): "Includes bibliographical references and index."
        # Subjects (650a, 650x): "Algorithms", "Data structures (Computer science)" -> "Algorithms Data structures (Computer science)"
        # Pub Year (008): "Published: 1998"
        # Languages (008, 041): "Languages: eng, ger" (sorted)
        "expected_text_for_embedding": "Introduction to algorithms Includes bibliographical references and index. Algorithms Data structures (Computer science) Published: 1998 Languages: eng, ger"
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
        self._create_sample_marcxml_file(self.sample_marcxml_filepath, SAMPLE_MARC_RECORDS_DATA_INTEGRATION)
        
        self.test_config_path = self._create_test_config_yaml()
        self.pipeline = RAGPipeline()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_sample_marcxml_file(self, filepath: Path, records_list_data: list):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        writer = pymarc.XMLWriter(open(str(filepath), 'wb'))
        for record_data in records_list_data:
            record = pymarc.Record(leader=record_data['leader'])
            for field_info in record_data['fields']:
                if 'data' in field_info: 
                    record.add_field(pymarc.Field(tag=field_info['tag'], data=field_info['data']))
                else: 
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
        # Config remains largely the same, paths are important
        config_data = {
            "task_name": "marcxml_integration_test",
            "embedding": {
                "mode": "local", "model_name": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dim": 384,
                "output_dir": str(self.output_embeddings_dir),
                "index_filename": "test_faiss_marc_integration.idx",
                "metadata_filename": "test_marc_integration_metadata.tsv",
            },
            "chunking": { # Less relevant for MARCXML record-as-chunk, but pipeline might use defaults
                "max_chunk_size": 100, "overlap": 15, "min_chunk_size": 25,
                "language_model": "en_core_web_sm",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "marcxml_files": {"input_dir": str(self.input_marcxml_dir)},
            "paths": {
                "chunked_marcxml_files": str(self.output_chunks_dir / "chunked_marcxml_integration_data.tsv"),
                "log_dir": str(self.logs_dir),
                "chunked_emails": str(self.output_chunks_dir / "dummy_emails.tsv"), # Dummy paths
                "chunked_text_files": str(self.output_chunks_dir / "dummy_texts.tsv"),
                "chunked_xml_files": str(self.output_chunks_dir / "dummy_xml.tsv"),
                "email_dir": str(self.test_dir / "dummy_email_dir"),
                "text_output_dir_raw": str(self.test_dir / "dummy_text_raw_dir"),
            },
            "retrieval": {"top_k": 2},
            "generation": {"model": "gpt-dummy", "api_key_env": "DUMMY_KEY"},
            "outlook": {"account_name": "d", "folder_path": "f", "days_to_fetch": 1}, # Dummy sections
            "text_files": {"input_dir": str(self.test_dir / "dummy_text_input")},
            "xml_files": {"input_dir": str(self.test_dir / "dummy_xml_input")}
        }
        for dummy_path_key in ["dummy_email_dir", "dummy_text_input", "dummy_xml_input", "dummy_text_raw_dir"]:
            (self.test_dir / dummy_path_key).mkdir(exist_ok=True)
        
        config_path = Path(self.test_dir) / "test_config_marcxml_integration.yaml"
        with open(config_path, "w", encoding="utf-8") as f: yaml.dump(config_data, f)
        return str(config_path)

    def test_marcxml_extraction_and_chunking_with_processor(self):
        self.pipeline.load_config(self.test_config_path)
        chunked_file_path = self.pipeline.extract_and_chunk(data_type="marcxml")
        self.assertTrue(Path(chunked_file_path).exists())
        
        df = pd.read_csv(chunked_file_path, sep="\t")
        self.assertGreater(len(df), 0)
        
        # Expected columns from MARCXMLProcessor output + 'Chunk'
        # Note: 'File Path' and 'Raw Text' are NOT added by MARCXMLProcessor path in RAGPipeline
        expected_cols = [
            'record_id', 'publication_year_008', 'primary_language_008', 'languages',
            'key_metadata_fields', 'entities', 'raw_marc_text_representation', # These are dicts/lists before JSON
            'entities_json', 'key_metadata_fields_json', 'Chunk' 
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Column '{col}' missing from chunked TSV.")

        # Validate content for each sample record
        self.assertEqual(len(df), len(SAMPLE_MARC_RECORDS_DATA_INTEGRATION))

        for i, expected_data in enumerate(SAMPLE_MARC_RECORDS_DATA_INTEGRATION):
            record_id_to_find = expected_data["fields"][0]["data"] # Assumes 001 is first field and has 'data'
            row = df[df['record_id'] == record_id_to_find]
            self.assertEqual(len(row), 1, f"Record ID {record_id_to_find} not found or found multiple times.")
            row = row.iloc[0]

            self.assertEqual(row['Chunk'], expected_data["expected_text_for_embedding"])
            
            # Validate key_metadata_fields_json
            key_meta = json.loads(row['key_metadata_fields_json'])
            if record_id_to_find == "marc001":
                self.assertEqual(key_meta.get('title_main'), "Harry Potter and the Half-Blood Prince")
                self.assertEqual(key_meta.get('author_personal_primary'), "Rowling, J. K.")
            elif record_id_to_find == "marc002":
                self.assertEqual(key_meta.get('title_main'), "Introduction to algorithms")

            # Validate languages (stored as string representation of list by pandas to_csv)
            self.assertIsInstance(row['languages'], str) # pandas saves list as string
            languages_list = eval(row['languages']) # Convert string list back to list
            if record_id_to_find == "marc001":
                self.assertEqual(languages_list, ["eng"])
            elif record_id_to_find == "marc002":
                self.assertEqual(sorted(languages_list), sorted(["eng", "ger"]))
            
            # Validate entities_json (should be empty dict string "{}" as NER is not deeply mocked here)
            entities = json.loads(row['entities_json'])
            self.assertEqual(entities, {})
            
            self.assertEqual(row['publication_year_008'], expected_data["fields"][1]["data"][7:11].strip()) # 008, pos 7-10
            self.assertEqual(row['primary_language_008'], expected_data["fields"][1]["data"][35:38].strip()) # 008, pos 35-37

    @patch('scripts.prompting.prompt_builder.TextFilePromptBuilder.build')
    @patch.object(APIClient, 'send_completion_request')
    def test_full_pipeline_with_marcxml_processor(self, mock_send_completion, mock_prompt_builder_build):
        mock_send_completion.return_value = "Mocked LLM answer about Harry Potter."
        # The prompt builder mock will allow us to inspect the context_chunks
        # It needs to return a string itself, or the pipeline will fail.
        mock_prompt_builder_build.return_value = "This is a fully mocked prompt."

        self.pipeline.load_config(self.test_config_path)
        self.pipeline.extract_and_chunk(data_type="marcxml")
        
        try:
            self.pipeline.embed_chunks()
        except Exception as e:
            if "ConnectionError" in str(e) or "offline" in str(e).lower() or "HFValidationError" in str(e):
                self.skipTest(f"Skipping embed_chunks due to network/HF model validation: {e}")
            else: raise e
                
        self.assertTrue(Path(self.pipeline.index_path).exists())
        self.assertTrue(Path(self.pipeline.metadata_path).exists())

        query = "Harry Potter Half-Blood Prince" # Query for first sample record
        self.pipeline.get_user_query(query)
        chunks_result = self.pipeline.retrieve(query=query)
        
        self.assertIsNotNone(chunks_result)
        self.assertIn("top_chunks", chunks_result)
        self.assertGreater(len(chunks_result["top_chunks"]), 0)

        # Verify retrieved chunk metadata
        first_retrieved_chunk = chunks_result["top_chunks"][0]
        self.assertEqual(first_retrieved_chunk['record_id'], "marc001")
        self.assertIn("Harry Potter and the Half-Blood Prince", first_retrieved_chunk['text']) # 'text' is the 'Chunk'
        self.assertIsInstance(first_retrieved_chunk['key_metadata_fields_json'], str)
        self.assertIsInstance(first_retrieved_chunk['entities_json'], str)

        # Generate answer (which uses the mocked prompt builder and LLM call)
        answer = self.pipeline.generate_answer(query=query, chunks=chunks_result)
        self.assertEqual(answer, "Mocked LLM answer about Harry Potter.")
        
        mock_send_completion.assert_called_once_with("This is a fully mocked prompt.")
        
        # Verify the arguments passed to TextFilePromptBuilder.build
        mock_prompt_builder_build.assert_called_once()
        call_args_list = mock_prompt_builder_build.call_args_list
        args, _ = call_args_list[0] # Get positional arguments from the first call
        
        passed_query = args[0]
        passed_context_chunks = args[1]
        
        self.assertEqual(passed_query, query)
        self.assertIsInstance(passed_context_chunks, list)
        self.assertEqual(len(passed_context_chunks), len(chunks_result["top_chunks"]))
        
        # Check structure of the first passed context chunk
        if passed_context_chunks:
            first_passed_chunk_dict = passed_context_chunks[0]
            self.assertEqual(first_passed_chunk_dict['record_id'], "marc001")
            self.assertIn("Harry Potter", first_passed_chunk_dict['text'])
            self.assertIn('key_metadata_fields_json', first_passed_chunk_dict)
            self.assertIn('entities_json', first_passed_chunk_dict)
            self.assertEqual(first_passed_chunk_dict['publication_year_008'], "2006")


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
