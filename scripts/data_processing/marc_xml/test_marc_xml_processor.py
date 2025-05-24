import unittest
import spacy
import pandas as pd
import pymarc
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.data_processing.marc_xml.marc_xml_processor import MARCXMLProcessor

# Sample data for creating test records
# This can be expanded for more detailed tests
SAMPLE_RECORD_DATA_1 = {
    "leader": "00000nam a2200000M  4500",
    "fields": [
        {"tag": "001", "data": "unit001"},
        {"tag": "008", "data": "230101s2023    xxu||||| |||| 00| 0 eng d"}, # Year 2023, lang eng
        {"tag": "100", "indicators": ["1", " "], "subfields": [("a", "Doe, Jane"), ("d", "1970-")]},
        {"tag": "245", "indicators": ["1", "0"], "subfields": [("a", "The Art of Unit Testing"), ("b", "with examples in Python")]},
        {"tag": "520", "indicators": [" ", " "], "subfields": [("a", "A comprehensive guide by Jane Doe.")]},
        {"tag": "650", "indicators": [" ", "0"], "subfields": [("a", "Unit testing (Computer science)")]}
    ]
}
SAMPLE_RECORD_DATA_2 = {
    "leader": "00000cas a2200000M  4500",
    "fields": [
        {"tag": "001", "data": "unit002"},
        {"tag": "008", "data": "220101s2022    nyu||||| |||| 00| 0 ger d"}, # Year 2022, lang ger
        {"tag": "041", "indicators": ["0", " "], "subfields": [("a", "ger"), ("h", "eng")]}, # German, original English
        {"tag": "245", "indicators": ["0", "0"], "subfields": [("a", "MARCXML Explained")]},
        {"tag": "710", "indicators": ["2", " "], "subfields": [("a", "Library of Testing")]}
    ]
}

def create_marc_record(data: dict) -> pymarc.Record:
    record = pymarc.Record(leader=data['leader'])
    for field_info in data['fields']:
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
    return record

class TestMARCXMLProcessor(unittest.TestCase):

    @classmethod
    @patch('spacy.load') # Mock spacy.load for the whole class if processor is created in setUpClass
    def setUpClass(cls, mock_spacy_load_cls):
        # Configure the class-level mock for spacy.load
        cls.mock_nlp_cls = MagicMock()
        mock_doc_cls = MagicMock()
        cls.mock_nlp_cls.return_value = mock_doc_cls # spacy.load() returns an nlp object
        cls.mock_nlp_cls.pipe_names = ['ner'] # Simulate a model with NER
        mock_doc_cls.ents = [] # Default to no entities
        mock_spacy_load_cls.return_value = cls.mock_nlp_cls
        
        cls.processor = MARCXMLProcessor() # Now uses the mocked spacy.load

    def setUp(self):
        # Individual test records
        self.record1 = create_marc_record(SAMPLE_RECORD_DATA_1)
        self.record2 = create_marc_record(SAMPLE_RECORD_DATA_2)
        
        # Reset entities for each test if needed, or configure per test
        self.mock_nlp_cls.return_value.ents = []


    def test_extract_record_id(self):
        processed_record = self.processor.process_single_record(self.record1)
        self.assertEqual(processed_record['record_id'], "unit001")
        
        processed_record2 = self.processor.process_single_record(self.record2)
        self.assertEqual(processed_record2['record_id'], "unit002")

    def test_get_marc_languages(self):
        # Record 1: 008 has 'eng'
        self.assertEqual(self.processor._get_marc_languages(self.record1), ["eng"])
        # Record 2: 008 has 'ger', 041a has 'ger', 041h has 'eng'
        self.assertEqual(sorted(self.processor._get_marc_languages(self.record2)), sorted(["eng", "ger"]))
        
        # Test with no language info
        empty_lang_record_data = {"leader": "00000nam a2200000M  4500", "fields": [{"tag": "001", "data": "lang003"}, {"tag":"008", "data":"000000s0000    xxu||||| |||| 00| 0 --- d"}]}
        empty_lang_record = create_marc_record(empty_lang_record_data)
        self.assertEqual(self.processor._get_marc_languages(empty_lang_record), [])


    def test_key_metadata_fields_extraction(self):
        processed_r1 = self.processor.process_single_record(self.record1)
        meta_r1 = processed_r1['key_metadata_fields']
        
        self.assertEqual(meta_r1.get('author_personal_primary'), "Doe, Jane")
        self.assertEqual(meta_r1.get('title_main'), "The Art of Unit Testing")
        self.assertEqual(meta_r1.get('title_remainder'), "with examples in Python")
        # summary_or_abstract is how 520a is stored by current processor logic
        self.assertEqual(meta_r1.get('summary_or_abstract'), "A comprehensive guide by Jane Doe.") 
        self.assertIn("Unit testing (Computer science)", meta_r1.get('subject_topical_term', []))

        processed_r2 = self.processor.process_single_record(self.record2)
        meta_r2 = processed_r2['key_metadata_fields']
        self.assertEqual(meta_r2.get('title_main'), "MARCXML Explained")
        # Assuming 710a is mapped to author_corporate_primary if 110 is not present
        # The current key_metadata_structure does not map 710 to these specific keys.
        # It would be field_710: ["Library of Testing"]
        self.assertIsNone(meta_r2.get('author_corporate_primary')) # 710 is not a 1XX field.
        # It would be captured as 'field_710' by the value_array logic if 710 was in key_metadata_structure
        # Let's check based on current processor logic for 710
        # Current key_metadata_structure does not include 710. If it did, it would be under 'field_710'.
        # This test highlights a potential gap if 7XX fields are meant for these specific keys.
        # For now, this is correct based on *current* processor.key_metadata_structure.

    @patch('spacy.load') # Mock spacy.load for this specific test, separate from setUpClass
    def test_extract_entities(self, mock_spacy_load_method):
        mock_nlp_method = MagicMock()
        mock_doc_method = MagicMock()
        mock_spacy_load_method.return_value = mock_nlp_method
        mock_nlp_method.return_value = mock_doc_method
        mock_nlp_method.pipe_names = ['ner'] # Simulate a model with NER

        # Define mock entities
        mock_ent1 = MagicMock(label_="PERSON", text="Jane Doe")
        mock_ent2 = MagicMock(label_="ORG", text="Python Software Foundation")
        mock_ent3 = MagicMock(label_="PERSON", text="Jane Doe") # Duplicate for testing deduplication
        mock_doc_method.ents = [mock_ent1, mock_ent2, mock_ent3]

        entities = self.processor._extract_entities("Sample text about Jane Doe and Python Software Foundation.")
        self.assertEqual(entities.get("PERSON"), ["Jane Doe"])
        self.assertEqual(entities.get("ORG"), ["Python Software Foundation"])
        
        # Test with blank NLP (if model loading failed in __init__)
        real_nlp = self.processor.nlp # Store real nlp
        self.processor.nlp = spacy.blank("en") # Simulate blank nlp
        self.assertEqual(self.processor._extract_entities("Some text"), {})
        self.processor.nlp = real_nlp # Restore real nlp (mocked one from setUpClass)


    def test_build_text_for_embedding(self):
        key_meta = {
            'title_main': "Adventures in Code", 
            'author_personal_primary': "Alice Coder",
            'summary_or_abstract': ["A thrilling tale of bytes and bugs."],
            'subject_topical_term': ["Coding", "Adventure"],
            'publication_year_008': "2024",
            'languages': ["eng", "spa"]
        }
        entities = {"GPE": ["Virtual City"], "PERSON": ["Alice Coder", "Bob Debugger"]}
        
        # record object is not directly used by _build_text_for_embedding in current MARCXMLProcessor
        # It relies on pre-extracted key_metadata and entities.
        text = self.processor._build_text_for_embedding(None, entities, key_meta) 
        
        self.assertIn("Adventures in Code", text)
        self.assertIn("Alice Coder", text)
        self.assertIn("A thrilling tale of bytes and bugs.", text)
        self.assertIn("Coding Adventure", text) # Note: list of subjects joined
        self.assertIn("Published: 2024", text)
        self.assertIn("Languages: eng, spa", text)
        self.assertIn("Key Persons: Alice Coder, Bob Debugger", text)
        self.assertNotIn("Key Organizations", text) # No ORG entities provided


    def test_build_raw_marc_text_representation(self):
        text = self.processor._build_raw_marc_text_representation(self.record1)
        self.assertIn("Doe, Jane", text)
        self.assertIn("The Art of Unit Testing", text)
        self.assertIn("with examples in Python", text)
        self.assertIn("A comprehensive guide by Jane Doe.", text)
        self.assertIn("Unit testing (Computer science)", text)
        # Control field 001 and 008 data should not be in raw text representation
        self.assertNotIn("unit001", text) 
        self.assertNotIn("230101s2023", text)

    def test_process_single_record_orchestration(self):
        # This is more of an integration test for process_single_record
        processed_r1 = self.processor.process_single_record(self.record1)
        
        self.assertEqual(processed_r1['record_id'], "unit001")
        self.assertEqual(processed_r1['publication_year_008'], "2023")
        self.assertEqual(processed_r1['primary_language_008'], "eng")
        self.assertEqual(processed_r1['languages'], ["eng"])
        self.assertIn("The Art of Unit Testing", processed_r1['text_for_embedding'])
        self.assertTrue(isinstance(processed_r1['key_metadata_fields'], dict))
        self.assertTrue(isinstance(processed_r1['entities'], dict)) # Will be empty due to class-level mock_doc.ents = []
        self.assertIn("The Art of Unit Testing", processed_r1['raw_marc_text_representation'])

    def test_process_directory_and_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            
            # File 1 (record1 and record2)
            file1_path = input_dir / "file1.xml"
            writer1 = pymarc.XMLWriter(open(str(file1_path), 'wb'))
            writer1.write(self.record1)
            writer1.write(self.record2)
            writer1.close()

            # File 2 (empty)
            file2_path = input_dir / "file2.xml"
            writer2 = pymarc.XMLWriter(open(str(file2_path), 'wb'))
            writer2.close() # No records

            # File 3 (single record, different data)
            record3_data = {
                "leader": "00000ncm a2200000M  4500", "fields": [
                    {"tag": "001", "data": "unit003"},
                    {"tag": "245", "indicators": ["0", "0"], "subfields": [("a", "Another Book")]}
                ]
            }
            record3 = create_marc_record(record3_data)
            file3_path = input_dir / "subdir" / "file3.xml" # Test subdirectory
            (input_dir / "subdir").mkdir()
            writer3 = pymarc.XMLWriter(open(str(file3_path), 'wb'))
            writer3.write(record3)
            writer3.close()

            df = self.processor.process_directory(str(input_dir))
            
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 3) # record1, record2, record3
            
            expected_cols = ['record_id', 'publication_year_008', 'primary_language_008', 
                             'languages', 'key_metadata_fields', 'entities', 
                             'text_for_embedding', 'raw_marc_text_representation',
                             'entities_json', 'key_metadata_fields_json']
            for col in expected_cols:
                self.assertIn(col, df.columns)
            
            # Check content of JSON serialized columns for first record
            record1_df_row = df[df['record_id'] == 'unit001'].iloc[0]
            self.assertIsNotNone(record1_df_row['key_metadata_fields_json'])
            key_meta_json = json.loads(record1_df_row['key_metadata_fields_json'])
            self.assertEqual(key_meta_json.get('title_main'), "The Art of Unit Testing")
            
            # Check that entities_json is present (will be '{}' due to mock)
            self.assertIsNotNone(record1_df_row['entities_json'])
            entities_json = json.loads(record1_df_row['entities_json'])
            self.assertEqual(entities_json, {})


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

```
