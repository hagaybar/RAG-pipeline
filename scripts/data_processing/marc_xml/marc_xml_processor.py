import os
import json
import spacy
import pandas as pd
from pathlib import Path
import pymarc
from pymarc.exceptions import PymarcException
import logging
from scripts.utils.constants import COL_TEXT_FOR_EMBEDDING, COL_MARC_RECORD_ID

class LoggerManager:
    @staticmethod
    def get_logger(name, level=logging.INFO):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(level)
        return logger

logger = LoggerManager.get_logger(__name__)

class MARCXMLProcessor:
    """
    Processes MARCXML records to extract structured data, named entities,
    and generate text representations for embedding.
    """
    def __init__(self, ner_model_name: str = "en_core_web_sm"):
        """
        Initializes the MARCXMLProcessor.

        Args:
            ner_model_name (str): Name of the spaCy model for NER.
        """
        try:
            self.nlp = spacy.load(ner_model_name)
            logger.info(f"spaCy NER model '{ner_model_name}' loaded successfully.")
        except OSError:
            logger.error(f"spaCy model '{ner_model_name}' not found. Please download it (e.g., python -m spacy download {ner_model_name})")
            # Fallback to a blank English model to allow basic functionality if model is missing
            # NER will be empty in this case.
            self.nlp = spacy.blank("en") 
            logger.warning(f"Fell back to blank spaCy model. NER will not be performed.")

        # For key_metadata_fields: tag -> list of subfields to extract as simple strings or "value" for control field
        self.key_metadata_structure = {
            '001': 'value', # Control Number
            '100': {'a': 'author_personal_primary', 'd': 'author_personal_dates'},
            '110': {'a': 'author_corporate_primary', 'b': 'author_corporate_subordinate'},
            '111': {'a': 'author_meeting_primary', 'n': 'author_meeting_number_part_section'},
            '245': {'a': 'title_main', 'b': 'title_remainder', 'c': 'title_statement_of_responsibility'},
            '260': {'a': 'publication_place', 'b': 'publication_name', 'c': 'publication_date_legacy'}, # Legacy Publication
            '264': {'a': 'publication_place', 'b': 'publication_name', 'c': 'publication_date'}, # Current Publication
            '300': {'a': 'physical_description_extent', 'b': 'physical_description_other', 'c': 'physical_description_dimensions'},
            '505': 'value_array', # Formatted Contents Note (can be repeated, subfields often 'a')
            '520': 'value_array', # Summary, Abstract (can be repeated, subfield 'a' common)
            '600': 'value_array', # Subject - Personal Name
            '610': 'value_array', # Subject - Corporate Name
            '611': 'value_array', # Subject - Meeting Name
            '630': 'value_array', # Subject - Uniform Title
            '650': 'value_array', # Subject - Topical Term
            '651': 'value_array', # Subject - Geographic Name
        }
        
        logger.info("MARCXMLProcessor initialized.")

    def _extract_entities(self, text: str) -> dict:
        """
        Extracts named entities from text using the spaCy model.
        """
        if not text or not self.nlp.pipe_names: # Check if text is empty or if nlp is blank
             return {}
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)
        # Deduplicate entities within each category
        for label in entities:
            entities[label] = sorted(list(set(entities[label])))
        return entities

    def _get_marc_languages(self, record: pymarc.Record) -> list:
        """
        Extracts unique language codes from MARC fields 008 and 041.
        """
        languages = set()
        # Field 008 (positions 35-37)
        if record['008'] and len(record['008'].data) >= 38:
            lang_code = record['008'].data[35:38].strip()
            if lang_code:
                languages.add(lang_code)
        
        # Field 041
        for field in record.get_fields('041'):
            for code in ['a', 'd', 'e', 'h', 'j']: # Common subfields for language codes
                for lang_code in field.get_subfields(code):
                    if lang_code:
                        languages.add(lang_code.strip())
        return sorted(list(languages))

    def _build_text_for_embedding(self, record: pymarc.Record, extracted_entities: dict, key_metadata: dict) -> str:
        """
        Constructs a coherent textual narrative/summary for embedding.
        """
        text_parts = []
        
        # Core information from key_metadata (which should be pre-populated)
        # Title (245a, 245b)
        title_main = key_metadata.get('title_main')
        if title_main: text_parts.append(title_main)
        title_remainder = key_metadata.get('title_remainder')
        if title_remainder: text_parts.append(title_remainder)

        # Main Creator (100a, 110a, 111a) - simplified: take first available
        if key_metadata.get('author_personal_primary'):
            text_parts.append(key_metadata['author_personal_primary'])
        elif key_metadata.get('author_corporate_primary'):
            text_parts.append(key_metadata['author_corporate_primary'])
        elif key_metadata.get('author_meeting_primary'):
            text_parts.append(key_metadata['author_meeting_primary'])

        # Summary (520a) - Assuming 520 is stored as an array of strings in key_metadata
        summaries = key_metadata.get('summary_or_abstract') # Name used in schema
        if summaries and isinstance(summaries, list):
            text_parts.extend(summaries) 
        elif summaries and isinstance(summaries, str): # if only one summary
             text_parts.append(summaries)


        # Key subject terms (from various 6XX fields)
        subject_fields_to_check = ['subject_personal_name', 'subject_corporate_name', 
                                   'subject_meeting_name', 'subject_uniform_title',
                                   'subject_topical_term', 'subject_geographic_name']
        for subject_key in subject_fields_to_check:
            subjects = key_metadata.get(subject_key)
            if subjects and isinstance(subjects, list):
                text_parts.extend(subjects) # Assuming subjects are already strings
            elif subjects and isinstance(subjects, str):
                 text_parts.append(subjects)
        
        # Publication year and language (already in key_metadata from 008)
        pub_year = key_metadata.get('publication_year_008')
        if pub_year: text_parts.append(f"Published: {pub_year}")
        
        # Languages from 008/041
        languages = key_metadata.get('languages', []) # From schema
        if languages: text_parts.append(f"Languages: {', '.join(languages)}")

        # Optionally, weave in a few top entities if not already covered (e.g. from extracted_entities)
        # This is a basic example, could be more sophisticated
        persons = extracted_entities.get("PERSON", [])
        if persons: text_parts.append(f"Key Persons: {', '.join(persons[:2])}") # Add first two
        
        orgs = extracted_entities.get("ORG", [])
        if orgs: text_parts.append(f"Key Organizations: {', '.join(orgs[:2])}")

        return " ".join(filter(None, text_parts))

    def _build_raw_marc_text_representation(self, record: pymarc.Record) -> str:
        """
        Concatenates text from a broader set of data fields and their subfields.
        """
        text_parts = []
        for field in record.get_fields():
            if not field.is_control_field():
                subfield_texts = []
                for subfield in field.subfields:
                    if hasattr(subfield, 'value') and subfield.value:
                        subfield_texts.append(subfield.value.strip())
                if subfield_texts:
                    text_parts.append(" ".join(subfield_texts))
        return " ".join(filter(None, text_parts))

    def process_single_record(self, record: pymarc.Record) -> dict:
        """
        Processes a single MARC record into an enriched dictionary.
        """
        enriched_record = {}
        
        # record_id (001)
        enriched_record[COL_MARC_RECORD_ID] = record['001'].data if record['001'] else None
        
        # publication_year_008 & primary_language_008
        enriched_record['publication_year_008'] = None
        enriched_record['primary_language_008'] = None
        if record['008'] and len(record['008'].data) >= 38:
            date1 = record['008'].data[7:11].strip()
            if date1.isdigit() and len(date1) == 4:
                enriched_record['publication_year_008'] = date1
            lang_code = record['008'].data[35:38].strip()
            if lang_code:
                enriched_record['primary_language_008'] = lang_code
        
        enriched_record['languages'] = self._get_marc_languages(record)
        
        # key_metadata_fields
        key_metadata = {}
        ner_text_parts = []

        for tag, structure in self.key_metadata_structure.items():
            fields = record.get_fields(tag)
            if not fields:
                continue

            if structure == 'value': # For control fields like 001
                key_metadata[tag] = fields[0].data if fields[0].data else None
            elif structure == 'value_array': # For repeatable text fields (e.g. 5XX, 6XX)
                field_values = []
                for field in fields:
                    # Concatenate all subfields for these value_array fields
                    current_field_text_parts = [sf.value.strip() for sf in field.subfields if sf.value]
                    if current_field_text_parts:
                        field_text = " ".join(current_field_text_parts)
                        field_values.append(field_text)
                        if tag.startswith('5') or tag.startswith('6'): # Add to NER text
                            ner_text_parts.append(field_text)
                if field_values:
                    # Use a generic key name based on tag for now, or map to schema later
                    schema_key_base = f"field_{tag}" 
                    if tag == '505': schema_key_base = 'formatted_contents_note'
                    elif tag == '520': schema_key_base = 'summary_or_abstract'
                    elif tag.startswith('600'): schema_key_base = 'subject_personal_name'
                    elif tag.startswith('610'): schema_key_base = 'subject_corporate_name'
                    # ... add more specific mappings if needed for schema keys
                    key_metadata[schema_key_base] = field_values if len(field_values) > 1 else field_values[0]
            
            else: # Dictionary structure for specific subfields
                for field in fields: # Though usually these tags are not repeatable (1XX, 245, 26X, 300)
                    for sub_code, schema_key in structure.items():
                        sub_values = field.get_subfields(sub_code)
                        if sub_values:
                            # Take the first value for simplicity, or join if multiple are expected
                            val = sub_values[0].strip()
                            key_metadata[schema_key] = val
                            if tag == '245' and sub_code in ['a', 'b']: # Title for NER
                                ner_text_parts.append(val)
                            elif tag == '100' and sub_code == 'a': # Author for NER
                                ner_text_parts.append(val)
                    break # Process only the first field if non-repeatable like 1XX, 245

        enriched_record['key_metadata_fields'] = key_metadata
        
        # entities (run NER on combined relevant text)
        ner_input_text = " ".join(filter(None, ner_text_parts))
        enriched_record['entities'] = self._extract_entities(ner_input_text)
        
        # text_for_embedding
        enriched_record[COL_TEXT_FOR_EMBEDDING] = self._build_text_for_embedding(record, enriched_record['entities'], key_metadata)
        
        # raw_marc_text_representation
        enriched_record['raw_marc_text_representation'] = self._build_raw_marc_text_representation(record)
        
        return enriched_record

    def process_records_from_file(self, marc_filepath: Path) -> list[dict]:
        """
        Processes all MARC records from a single MARCXML file.
        """
        enriched_records = []
        try:
            records = pymarc.parse_xml_to_array(str(marc_filepath))
            if not records:
                logger.warning(f"No MARC records found in file: {marc_filepath}")
                return []
            
            for record in records:
                try:
                    enriched_records.append(self.process_single_record(record))
                except Exception as e:
                    record_id = record['001'].data if record['001'] else "UNKNOWN_ID"
                    logger.error(f"Error processing record {record_id} in file {marc_filepath}: {e}", exc_info=True)
            logger.info(f"Successfully processed {len(enriched_records)} records from {marc_filepath}.")
        except PymarcException as e:
            logger.error(f"Pymarc error parsing file {marc_filepath}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error reading/parsing file {marc_filepath}: {e}", exc_info=True)
        return enriched_records

    def process_directory(self, input_dir_path: str) -> pd.DataFrame:
        """
        Processes all MARCXML files in a directory and returns a DataFrame.
        """
        all_enriched_records = []
        input_path = Path(input_dir_path)
        
        if not input_path.is_dir():
            logger.error(f"Input path {input_dir_path} is not a valid directory.")
            return pd.DataFrame()

        for marc_file in input_path.rglob('*.xml'): # Can add more extensions like '*.mrcx', '*.marcxml'
            if marc_file.is_file():
                logger.info(f"Processing file: {marc_file}")
                all_enriched_records.extend(self.process_records_from_file(marc_file))
        
        if not all_enriched_records:
            logger.warning(f"No records processed from directory {input_dir_path}")
            return pd.DataFrame()

        df = pd.DataFrame(all_enriched_records)
        
        # Serialize complex fields to JSON strings
        if 'entities' in df.columns:
            df['entities_json'] = df['entities'].apply(lambda x: json.dumps(x) if x else None)
        if 'key_metadata_fields' in df.columns:
            df['key_metadata_fields_json'] = df['key_metadata_fields'].apply(lambda x: json.dumps(x) if x else None)
            
        # Drop original dict columns if JSON versions are preferred for CSV/TSV
        # For now, keep them for potential direct use, but be aware for saving.
        
        logger.info(f"Processed {len(df)} total records from directory {input_dir_path}.")
        return df


if __name__ == '__main__':
    processor = MARCXMLProcessor()
    
    # Create a dummy MARCXML file for testing
    test_dir = Path("test_marc_processor_data")
    test_dir.mkdir(exist_ok=True)
    sample_marc_file = test_dir / "sample_records.xml"

    # Using pymarc to create a valid MARCXML file with a few records
    writer = pymarc.XMLWriter(open(str(sample_marc_file), 'wb'))
    
    # Record 1
    r1 = pymarc.Record(leader='00000nam a2200000M  4500')
    r1.add_field(pymarc.Field(tag='001', data='record001'))
    r1.add_field(pymarc.Field(tag='008', data='230101s2023    xxu||||| |||| 00| 0 eng d')) # Year 2023, lang eng
    r1.add_field(pymarc.Field(tag='100', indicators=['1', ' '], subfields=['a', 'Doe, Jane']))
    r1.add_field(pymarc.Field(tag='245', indicators=['1', '0'], subfields=['a', 'The Art of Unit Testing', 'b', 'with examples in Python']))
    r1.add_field(pymarc.Field(tag='520', indicators=[' ', ' '], subfields=['a', 'A comprehensive guide to unit testing principles and practices. Jane Doe is a leading expert.']))
    r1.add_field(pymarc.Field(tag='650', indicators=[' ', '0'], subfields=['a', 'Unit testing (Computer science)']))
    r1.add_field(pymarc.Field(tag='650', indicators=[' ', '0'], subfields=['a', 'Software engineering']))
    writer.write(r1)
    
    # Record 2
    r2 = pymarc.Record(leader='00000cas a2200000M  4500')
    r2.add_field(pymarc.Field(tag='001', data='record002'))
    r2.add_field(pymarc.Field(tag='008', data='220101s2022    nyu||||| |||| 00| 0 ger d')) # Year 2022, lang ger
    r2.add_field(pymarc.Field(tag='245', indicators=['0', '0'], subfields=['a', 'MARCXML Explained']))
    r2.add_field(pymarc.Field(tag='041', indicators=['0', ' '], subfields=['a', 'ger', 'h', 'eng'])) # German, original English
    r2.add_field(pymarc.Field(tag='710', indicators=['2', ' '], subfields=['a', 'Library of Congress']))
    writer.write(r2)
    writer.close()

    logger.info(f"Created sample MARCXML file at {sample_marc_file}")

    # Process the directory
    results_df = processor.process_directory(str(test_dir))
    
    if not results_df.empty:
        print(f"\n--- Processed MARC Data (First {min(5, len(results_df))} records) ---")
        
        # Configure pandas to display more content
        pd.set_option('display.max_colwidth', 200) # Show more of long text fields
        pd.set_option('display.width', 1000)      # Wider display for DataFrame

        # Select and print relevant columns for review
        columns_to_print = [
            COL_MARC_RECORD_ID,
            COL_TEXT_FOR_EMBEDDING,
            'entities_json', 
            'key_metadata_fields_json', 
            'languages',
            'publication_year_008'
        ]
        # Filter out columns that might not exist if all records are empty for some fields
        columns_to_print = [col for col in columns_to_print if col in results_df.columns]

        print(results_df[columns_to_print].head(5).to_string())
        
        # Example of how to access a specific JSON field from the first record
        if 'key_metadata_fields_json' in results_df.columns and len(results_df) > 0:
            try:
                first_record_metadata = json.loads(results_df.iloc[0]['key_metadata_fields_json'])
                print(f"\nKey metadata from first record (001: {results_df.iloc[0][COL_MARC_RECORD_ID]}):")
                print(json.dumps(first_record_metadata, indent=2))
            except (TypeError, json.JSONDecodeError) as e:
                logger.error(f"Could not parse JSON for first record's metadata: {e}")

    else:
        print("No data processed or DataFrame is empty.")

    # Clean up test directory
    # import shutil
    # shutil.rmtree(test_dir)
    # logger.info(f"Cleaned up test directory: {test_dir}")
