import unittest
import tempfile
import os
import pandas as pd
from pathlib import Path
import shutil # For cleaning up non-temporary dirs if needed, though TemporaryDirectory handles itself

# Assuming TextFileFetcher is in the same directory or scripts.data_processing.text
from scripts.data_processing.text.text_fetcher import TextFileFetcher, LoggerManager

# Mock LoggerManager if needed, or use the placeholder if it's simple enough
# For this test, the placeholder LoggerManager in TextFileFetcher should suffice.

class TestTextFileFetcher(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for input files
        self.input_dir_obj = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.input_dir_obj.name)

        # Create a temporary directory for output files
        self.output_dir_obj = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.output_dir_obj.name)

        # Sample files
        self.file1_content = "This is the first sample text file."
        self.file2_content = "  Another file, with leading/trailing spaces. \n"
        self.file3_content = "A third file for testing."

        with open(self.input_dir / "file1.txt", "w", encoding="utf-8") as f:
            f.write(self.file1_content)
        with open(self.input_dir / "file2.txt", "w", encoding="utf-8") as f:
            f.write(self.file2_content)
        with open(self.input_dir / "another.md", "w", encoding="utf-8") as f: # Non-txt file
            f.write("This is a markdown file.")

        self.config = {
            "text_files": {
                "input_dir": str(self.input_dir)
            },
            "paths": {
                "text_output_dir": str(self.output_dir)
            }
        }
        self.fetcher = TextFileFetcher(self.config)

    def tearDown(self):
        # Cleanup the temporary directories
        self.input_dir_obj.cleanup()
        self.output_dir_obj.cleanup()

    def test_fetch_successful_return_df_and_save(self):
        df = self.fetcher.fetch_text_files(return_dataframe=True, save=True)

        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2) # Only .txt files
        self.assertListEqual(list(df.columns), ["File Path", "Raw Text", "Cleaned Text"])

        # Check content (order might vary depending on glob)
        file1_path_abs = str((self.input_dir / "file1.txt").resolve())
        file2_path_abs = str((self.input_dir / "file2.txt").resolve())

        df_file1_data = df[df["File Path"] == file1_path_abs]
        self.assertEqual(len(df_file1_data), 1)
        self.assertEqual(df_file1_data["Raw Text"].iloc[0], self.file1_content)
        self.assertEqual(df_file1_data["Cleaned Text"].iloc[0], self.file1_content.strip())
        
        df_file2_data = df[df["File Path"] == file2_path_abs]
        self.assertEqual(len(df_file2_data), 1)
        self.assertEqual(df_file2_data["Raw Text"].iloc[0], self.file2_content)
        self.assertEqual(df_file2_data["Cleaned Text"].iloc[0], self.file2_content.strip())

        # Check if TSV file was saved
        saved_files = list(self.output_dir.glob("*.tsv"))
        self.assertEqual(len(saved_files), 1, "A single TSV file should have been saved.")
        
        # Verify content of the saved TSV
        saved_df = pd.read_csv(saved_files[0], sep='\t')
        self.assertEqual(len(saved_df), 2)
        self.assertListEqual(list(saved_df.columns), ["File Path", "Raw Text", "Cleaned Text"])


    def test_fetch_successful_return_path(self):
        file_path_result = self.fetcher.fetch_text_files(return_dataframe=False, save=True)

        self.assertIsInstance(file_path_result, str)
        self.assertTrue(Path(file_path_result).exists())
        self.assertTrue(Path(file_path_result).is_file())
        self.assertTrue(file_path_result.endswith(".tsv"))

        # Verify content of the saved TSV
        saved_df = pd.read_csv(file_path_result, sep='\t')
        self.assertEqual(len(saved_df), 2) # .txt files only
        self.assertListEqual(list(saved_df.columns), ["File Path", "Raw Text", "Cleaned Text"])
        
        # Check some content detail
        file1_path_abs = str((self.input_dir / "file1.txt").resolve())
        self.assertIn(file1_path_abs, saved_df["File Path"].tolist())


    def test_fetch_no_save_return_df(self):
        df = self.fetcher.fetch_text_files(return_dataframe=True, save=False)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2) # .txt files only
        
        saved_files = list(self.output_dir.glob("*.tsv"))
        self.assertEqual(len(saved_files), 0, "No TSV file should have been saved.")

    def test_fetch_no_save_no_return_df(self):
        result = self.fetcher.fetch_text_files(return_dataframe=False, save=False)
        self.assertIsNone(result, "Should return None if not saving and not returning DataFrame.")
        saved_files = list(self.output_dir.glob("*.tsv"))
        self.assertEqual(len(saved_files), 0, "No TSV file should have been saved.")

    def test_clean_text(self):
        # TextFileFetcher already instantiated in setUp
        text_to_clean = "  \n some text with spaces and newlines\t "
        cleaned = self.fetcher.clean_text(text_to_clean)
        self.assertEqual(cleaned, "some text with spaces and newlines")

    def test_empty_input_directory(self):
        with tempfile.TemporaryDirectory() as empty_input_dir:
            config_empty = {
                "text_files": {"input_dir": empty_input_dir},
                "paths": {"text_output_dir": str(self.output_dir)}
            }
            fetcher_empty = TextFileFetcher(config_empty)
            df = fetcher_empty.fetch_text_files(return_dataframe=True, save=True)
            
            self.assertIsNotNone(df)
            self.assertTrue(df.empty)
            self.assertListEqual(list(df.columns), ["File Path", "Raw Text", "Cleaned Text"])

            # Check that no file was saved (or an empty one, current behavior is no file)
            saved_files = list(self.output_dir.glob("*.tsv"))
            # If a new fetcher was used, it would write to its own output_dir.
            # Let's check the output_dir specified for fetcher_empty.
            # The current implementation of TextFileFetcher returns None for path if df is empty,
            # so no file should be saved.
            # To be sure, let's create a specific output for this test
            with tempfile.TemporaryDirectory() as specific_output_dir:
                config_empty_specific_out = {
                    "text_files": {"input_dir": empty_input_dir},
                    "paths": {"text_output_dir": specific_output_dir}
                }
                fetcher_empty_specific = TextFileFetcher(config_empty_specific_out)
                path_result = fetcher_empty_specific.fetch_text_files(return_dataframe=False, save=True)
                self.assertIsNone(path_result, "Path result should be None for empty input.")
                self.assertEqual(len(list(Path(specific_output_dir).glob("*.tsv"))), 0)


    def test_non_existent_input_directory(self):
        non_existent_dir = "/path/to/some/very/unlikely/dir_to_exist_42"
        config_non_existent = {
            "text_files": {"input_dir": non_existent_dir},
            "paths": {"text_output_dir": str(self.output_dir)}
        }
        with self.assertRaises(FileNotFoundError):
            TextFileFetcher(config_non_existent)

    def test_fetch_skip_non_txt_files(self):
        # This is implicitly tested in test_fetch_successful_return_df_and_save,
        # as it creates a .md file which should be ignored.
        # Let's make it more explicit by checking the count
        df = self.fetcher.fetch_text_files(return_dataframe=True, save=False)
        self.assertEqual(len(df), 2, "Should only process .txt files, ignoring .md file.")
        
        # Verify that the .md file path is not in the DataFrame
        md_file_path_abs = str((self.input_dir / "another.md").resolve())
        self.assertNotIn(md_file_path_abs, df["File Path"].tolist())

if __name__ == '__main__':
    unittest.main()
