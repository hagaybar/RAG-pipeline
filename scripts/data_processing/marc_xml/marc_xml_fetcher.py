import os
import pandas as pd
from pathlib import Path
import pymarc
from pymarc.exceptions import PymarcException

# Placeholder for LoggerManager until its actual location is known
# For now, define it directly in this file for simplicity.
import logging

class LoggerManager:
    @staticmethod
    def get_logger(name, level=logging.INFO): # Added level parameter
        logger = logging.getLogger(name)
        if not logger.handlers: # Avoid adding multiple handlers
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(level)
        return logger

logger = LoggerManager.get_logger(__name__)

class MARCXMLFetcher:
    """
    Fetches and processes MARCXML files from a specified directory, 
    extracting text content from MARC records.
    """
    def __init__(self, input_dir: str):
        """
        Initializes the MARCXMLFetcher.

        Args:
            input_dir (str): Path to the directory containing MARCXML files.
        """
        self.input_dir = Path(input_dir)

        if not self.input_dir.is_dir():
            logger.error(f"Input directory does not exist or is not a directory: {self.input_dir}")
            raise FileNotFoundError(f"Input directory does not exist or is not a directory: {self.input_dir}")
        
        logger.info(f"MARCXMLFetcher initialized. Input directory: {self.input_dir}")

    def _extract_text_from_record(self, record: pymarc.Record) -> str:
        """
        Extracts targeted text content from a single MARC record.
        It prioritizes specific fields and subfields and also extracts 
        publication year and language from control field 008.

        Args:
            record (pymarc.Record): The MARC record object.

        Returns:
            str: Concatenated text content from the record.
        """
        text_parts = []
        priority_fields_subfields = {
            '245': ['a', 'b', 'c', 'p', 'n'], # Title Statement
            '520': ['a'],                     # Summary, Abstract, Annotation
            '505': ['a'],                     # Formatted Contents Note
            '100': ['a', 'q', 'd'],           # Main Entry - Personal Name (name, fuller form, dates)
            '110': ['a', 'b', 'c', 'd'],      # Main Entry - Corporate Name (name, subordinate, location, date)
            '111': ['a', 'c', 'd', 'n', 'q'], # Main Entry - Meeting Name (name, location, date, number, name following jurisdiction)
            '700': ['a', 'q', 'd'],           # Added Entry - Personal Name
            '710': ['a', 'b', 'c', 'd'],      # Added Entry - Corporate Name
            '711': ['a', 'c', 'd', 'n', 'q'], # Added Entry - Meeting Name
            '600': ['a', 'x', 'y', 'z', 'v'], # Subject Added Entry - Personal Name (name, general sub, chrono sub, geo sub, form sub)
            '610': ['a', 'b', 'x', 'y', 'z', 'v'], # Subject Added Entry - Corporate Name
            '611': ['a', 'n', 'q', 'x', 'y', 'z', 'v'], # Subject Added Entry - Meeting Name
            '630': ['a', 'p', 'l', 'x', 'y', 'z', 'v'], # Subject Added Entry - Uniform Title (title, part, lang, general sub, chrono sub, geo sub, form sub)
            '650': ['a', 'x', 'y', 'z', 'v'], # Subject Added Entry - Topical Term
            '651': ['a', 'x', 'y', 'z', 'v']  # Subject Added Entry - Geographic Name
        }

        for tag, sub_codes in priority_fields_subfields.items():
            for field in record.get_fields(tag):
                field_texts = []
                for sub_code in sub_codes:
                    # field.get_subfields(sub_code) returns a list of subfield values
                    for subfield_value in field.get_subfields(sub_code):
                        if subfield_value: # Ensure value is not None
                             field_texts.append(subfield_value.strip())
                if field_texts:
                    text_parts.append(" ".join(field_texts))

        # Control Field 008 processing
        field008_list = record.get_fields('008') # get_fields returns a list
        if field008_list:
            # Assuming there's only one 008 field per record, which is standard
            data008 = field008_list[0].data 
            if len(data008) >= 38: # Ensure 008 field is long enough
                # Date 1 (positions 07-10)
                date1 = data008[7:11]
                if date1.strip().isdigit() and len(date1.strip()) == 4 : # Basic check for 4 digits
                    text_parts.append(f"Publication Year: {date1.strip()}")
                
                # Language (positions 35-37)
                lang_code = data008[35:38]
                if lang_code.strip(): # Check if not just spaces
                    text_parts.append(f"Language: {lang_code.strip()}")
            else:
                logger.debug(f"Field 008 in record (ID: {record['001'].data if record['001'] else 'N/A'}) is shorter than expected: {len(data008)} chars.")

        return " ".join(filter(None, text_parts)) # Join non-empty, stripped text parts

    def fetch_marc_xml_files(self) -> pd.DataFrame:
        """
        Reads all .xml files recursively from the input directory, 
        parses MARC records, extracts text using the refined logic, 
        and returns the data as a Pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns "File Path", "Raw Text", and "Cleaned Text".
                          Each row corresponds to one MARC record.
                          Returns an empty DataFrame if no XML files or MARC records are found.
        """
        logger.info(f"Starting to fetch MARCXML files from: {self.input_dir} using refined extraction.")
        file_data = []
        files_processed_count = 0
        records_processed_count = 0

        for filepath in self.input_dir.rglob('*.xml'):
            if filepath.is_file():
                logger.info(f"Processing MARCXML file: {filepath}")
                files_processed_count += 1
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        raw_xml_content = f.read()
                    
                    records = pymarc.parse_xml_to_array(str(filepath))
                    
                    if not records:
                        logger.warning(f"No MARC records found in file: {filepath}")
                        continue

                    for record in records:
                        records_processed_count += 1
                        cleaned_text_from_record = self._extract_text_from_record(record)
                        
                        file_data.append({
                            "File Path": str(filepath.resolve()), 
                            "Raw Text": raw_xml_content, 
                            "Cleaned Text": cleaned_text_from_record 
                        })
                    logger.debug(f"Successfully processed {len(records)} records from: {filepath.name}")

                except PymarcException as e:
                    logger.error(f"Error parsing MARCXML file {filepath} with Pymarc: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"An unexpected error occurred while processing file {filepath}: {e}", exc_info=True)
            else:
                logger.warning(f"Skipping non-file item: {filepath}")


        if not file_data:
            logger.warning(f"No MARC records extracted from any XML files in {self.input_dir}")
            return pd.DataFrame(columns=["File Path", "Raw Text", "Cleaned Text"])

        df = pd.DataFrame(file_data)
        logger.info(f"Successfully created DataFrame with {len(df)} MARC records from {files_processed_count} files using refined extraction.")
        return df

if __name__ == '__main__':
    test_dir = Path("test_marc_xml_data_refined")
    test_dir.mkdir(exist_ok=True)
    
    # Updated sample MARCXML content for refined extraction
    marc_xml_content_refined = """<?xml version="1.0" encoding="UTF-8"?>
<collection xmlns="http://www.loc.gov/MARC21/slim">
<record>
  <leader>00000nam a2200000M  4500</leader>
  <controlfield tag="001">12345</controlfield>
  <controlfield tag="008">060922s2006    maua   j      000 0 eng d</controlfield> 
  <!-- Date1: 2006, Lang: eng -->
  <datafield tag="100" ind1="1" ind2=" ">
    <subfield code="a">Rowling, J. K.</subfield>
    <subfield code="d">1965-</subfield>
  </datafield>
  <datafield tag="245" ind1="1" ind2="0">
    <subfield code="a">Harry Potter and the Half-Blood Prince /</subfield>
    <subfield code="c">by J.K. Rowling ; illustrations by Mary GrandPré.</subfield>
  </datafield>
  <datafield tag="520" ind1=" " ind2=" ">
    <subfield code="a">Sixth-year Hogwarts student Harry Potter gains valuable insights into the boy Voldemort once was.</subfield>
  </datafield>
  <datafield tag="650" ind1=" " ind2="0">
    <subfield code="a">Wizards</subfield>
    <subfield code="v">Juvenile fiction.</subfield>
  </datafield>
</record>
<record>
  <leader>00001cam a2200000K  4500</leader>
  <controlfield tag="001">67890</controlfield>
  <controlfield tag="008">980115s1998    nyu    b      001 0 eng  </controlfield>
  <!-- Date1: 1998, Lang: eng -->
  <datafield tag="245" ind1="0" ind2="0">
    <subfield code="a">Introduction to algorithms</subfield>
    <subfield code="c">Thomas H. Cormen ... [et al.].</subfield>
  </datafield>
  <datafield tag="260" ind1=" " ind2=" "> <!-- Not in priority_fields_subfields -->
    <subfield code="a">Cambridge, Mass. :</subfield>
    <subfield code="b">MIT Press ;</subfield>
    <subfield code="c">c1998.</subfield>
  </datafield>
  <datafield tag="505" ind1="0" ind2=" ">
    <subfield code="a">Includes bibliographical references (p. 1147-1160) and index.</subfield>
  </datafield>
</record>
</collection>
"""
    sample_file_path = test_dir / "refined_sample.xml"
    sample_file_path.write_text(marc_xml_content_refined, encoding='utf-8')

    logger.info("Starting MARCXMLFetcher example with refined extraction...")
    try:
        fetcher = MARCXMLFetcher(input_dir=str(test_dir))
        marc_dataframe = fetcher.fetch_marc_xml_files()
        
        if not marc_dataframe.empty:
            print("\n--- Fetched MARCXML Data (Refined Extraction) ---")
            # print(marc_dataframe) # Might be too verbose
            print(f"Total records fetched: {len(marc_dataframe)}")
            
            for index, row in marc_dataframe.iterrows():
                print(f"\n--- Record {index + 1} ---")
                print(f"File Path: {row['File Path']}")
                # print(f"Raw Text: {row['Raw Text'][:200]}...") # Show snippet of raw
                print(f"Cleaned Text: {row['Cleaned Text']}")
        else:
            print("No MARCXML data fetched.")

    except FileNotFoundError as fnf_error:
        logger.error(f"Initialization failed: {fnf_error}")
    except Exception as ex:
        logger.error(f"An error occurred during the example run: {ex}", exc_info=True)
    finally:
        # Clean up by default, but can be commented out for inspection
        import shutil
        # shutil.rmtree(test_dir)
        # logger.info(f"Example finished. Test directory '{test_dir}' {'not ' if not test_dir.exists() else ''}deleted.")
        if test_dir.exists():
             logger.info(f"Test files are in {test_dir.resolve()}")
             # To manually clean up: shutil.rmtree(test_dir)
```
