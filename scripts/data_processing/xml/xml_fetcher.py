import os
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
# Assuming LoggerManager is in a utility module, e.g., scripts.utils.logger_manager
# from scripts.utils.logger_manager import LoggerManager # Placeholder

# Placeholder for LoggerManager until its actual location is known
class LoggerManager:
    @staticmethod
    def get_logger(name):
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers: # Avoid adding multiple handlers
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

logger = LoggerManager.get_logger(__name__)

from scripts.utils.constants import COL_FILE_PATH, COL_RAW_TEXT, COL_CLEANED_TEXT

class XMLFetcher:
    """
    Fetches and processes XML files from a specified directory, extracting text content.
    """
    def __init__(self, input_dir: str):
        """
        Initializes the XMLFetcher.

        Args:
            input_dir (str): Path to the directory containing XML files.
        """
        self.input_dir = Path(input_dir)

        if not self.input_dir.is_dir():
            logger.error(f"Input directory does not exist or is not a directory: {self.input_dir}")
            raise FileNotFoundError(f"Input directory does not exist or is not a directory: {self.input_dir}")
        
        logger.info(f"XMLFetcher initialized. Input directory: {self.input_dir}")

    def _extract_text_from_xml(self, tree_root: ET.Element) -> str:
        """
        Extracts all text content from an XML tree.

        Args:
            tree_root (ET.Element): The root element of the XML tree.

        Returns:
            str: Concatenated text content from all elements.
        """
        texts = []
        for element in tree_root.iter():
            if element.text:
                texts.append(element.text.strip())
        return " ".join(filter(None, texts)) # Join non-empty, stripped text

    def fetch_xml_files(self) -> pd.DataFrame:
        """
        Reads all .xml files recursively from the input directory, extracts text,
        and returns the data as a Pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns "File Path", "Raw Text", and "Cleaned Text".
                          Returns an empty DataFrame if no XML files are found.
        """
        logger.info(f"Starting to fetch XML files from: {self.input_dir}")
        file_data = []

        for file_path in self.input_dir.rglob("*.xml"):
            if file_path.is_file():
                logger.info(f"Processing XML file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        raw_text = f.read()
                    
                    # Parse XML from string
                    tree_root = ET.fromstring(raw_text) 
                    cleaned_text = self._extract_text_from_xml(tree_root)
                    
                    file_data.append({
                        COL_FILE_PATH: str(file_path.resolve()),
                        COL_RAW_TEXT: raw_text,
                        COL_CLEANED_TEXT: cleaned_text
                    })
                    logger.debug(f"Successfully processed and added: {file_path.name}")

                except ET.ParseError as e:
                    logger.error(f"Error parsing XML file {file_path}: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"An unexpected error occurred while processing file {file_path}: {e}", exc_info=True)
            else:
                # This case should ideally not be hit with rglob("*.xml") if it only yields files,
                # but good to have as a guard.
                logger.warning(f"Skipping non-file item: {file_path}")

        if not file_data:
            logger.warning(f"No .xml files found in {self.input_dir}")
            return pd.DataFrame(columns=[COL_FILE_PATH, COL_RAW_TEXT, COL_CLEANED_TEXT])

        df = pd.DataFrame(file_data)
        logger.info(f"Successfully created DataFrame with {len(df)} XML files.")
        return df

if __name__ == '__main__':
    # Basic example of how to use XMLFetcher
    # This part is for testing and might be removed or adjusted later.
    
    # Create dummy XML files for testing
    test_dir = Path("test_xml_data")
    test_dir.mkdir(exist_ok=True)
    
    sub_dir = test_dir / "subdir"
    sub_dir.mkdir(exist_ok=True)

    xml_content1 = "<root><item>Hello</item><item>World</item></root>"
    xml_content2 = "<doc><para>This is <a>an</a> example.</para><data><value>123</value></data></doc>"
    xml_content3 = "<note><to>Tove</to><from>Jani</from><heading>Reminder</heading><body>Don't forget me this weekend!</body></note>"
    malformed_xml_content = "<root><item>Hello</item><item>World</root>" # Missing closing tag for item

    (test_dir / "file1.xml").write_text(xml_content1, encoding='utf-8')
    (sub_dir / "file2.xml").write_text(xml_content2, encoding='utf-8')
    (test_dir / "file3.xml").write_text(xml_content3, encoding='utf-8')
    (test_dir / "malformed.xml").write_text(malformed_xml_content, encoding='utf-8')
    (test_dir / "not_xml.txt").write_text("This is not an XML file.", encoding='utf-8')


    logger.info("Starting XMLFetcher example...")
    try:
        # Initialize fetcher (Update path to your test XML directory)
        fetcher = XMLFetcher(input_dir=str(test_dir))
        
        # Fetch data
        xml_dataframe = fetcher.fetch_xml_files()
        
        if not xml_dataframe.empty:
            print("\n--- Fetched XML Data ---")
            print(xml_dataframe)
            print("\n--- Cleaned Text from first file ---")
            if len(xml_dataframe) > 0:
                print(xml_dataframe.iloc[0]["Cleaned Text"])
        else:
            print("No XML data fetched.")

    except FileNotFoundError as fnf_error:
        logger.error(f"Initialization failed: {fnf_error}")
    except Exception as ex:
        logger.error(f"An error occurred during the example run: {ex}", exc_info=True)
    finally:
        # Clean up dummy files and directory
        import shutil
        # shutil.rmtree(test_dir) # Comment out if you want to inspect files
        logger.info(f"Example finished. Test directory '{test_dir}' {'not ' if test_dir.exists() else ''}deleted.")

        # Note: The above cleanup might be too aggressive if other tests depend on these files.
        # For now, it's included to show a complete example.
        # If keeping the files: logger.info(f"Test files are in {test_dir.resolve()}")
        # For safety, I will comment out the rmtree for now.
        if test_dir.exists():
            logger.info(f"Test files are in {test_dir.resolve()}")
            # To manually clean up: shutil.rmtree(test_dir)

