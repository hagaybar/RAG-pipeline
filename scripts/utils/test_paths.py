import unittest
import os
from pathlib import Path
import shutil # For cleaning up created directories

# Add the parent directory to the Python path to allow importing paths
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.paths import TaskPaths

class TestTaskPathsGetChunkFile(unittest.TestCase):

    def setUp(self):
        # Create a temporary task name and TaskPaths instance for each test
        self.test_task_name = "test_task_for_paths"
        # Ensure the base directory for tests is unique to avoid conflicts
        # and make cleanup easier.
        self.test_base_dir = "test_runs_temp" 
        self.tp = TaskPaths(self.test_task_name, base_dir=self.test_base_dir)

    def tearDown(self):
        # Clean up the created temporary task directory structure after each test
        task_root_to_delete = os.path.join(self.test_base_dir, self.test_task_name)
        if os.path.exists(task_root_to_delete):
            shutil.rmtree(task_root_to_delete)
        # If the base_dir itself was created and is empty, remove it.
        # This handles the case where the base_dir was specific to these tests.
        if os.path.exists(self.test_base_dir) and not os.listdir(self.test_base_dir):
            shutil.rmtree(self.test_base_dir)
        elif os.path.exists(self.test_base_dir) and os.path.join(self.test_base_dir, self.test_task_name) not in os.listdir(self.test_base_dir) and len(os.listdir(self.test_base_dir))==0 :
             shutil.rmtree(self.test_base_dir)


    def test_get_chunk_file_email(self):
        expected_path = os.path.join(self.tp.chunks_dir, "chunked_emails.tsv")
        self.assertEqual(self.tp.get_chunk_file("email"), expected_path)

    def test_get_chunk_file_text_file(self):
        expected_path = os.path.join(self.tp.chunks_dir, "chunked_text_files.tsv")
        self.assertEqual(self.tp.get_chunk_file("text_file"), expected_path)

    def test_get_chunk_file_xml(self):
        expected_path = os.path.join(self.tp.chunks_dir, "chunked_xml.tsv")
        self.assertEqual(self.tp.get_chunk_file("xml"), expected_path)

    def test_get_chunk_file_marcxml(self):
        expected_path = os.path.join(self.tp.chunks_dir, "chunked_marcxml.tsv")
        self.assertEqual(self.tp.get_chunk_file("marcxml"), expected_path)

    def test_get_chunk_file_default_is_email(self):
        expected_path = os.path.join(self.tp.chunks_dir, "chunked_emails.tsv")
        self.assertEqual(self.tp.get_chunk_file(), expected_path)

    def test_get_chunk_file_unsupported(self):
        with self.assertRaises(ValueError) as context:
            self.tp.get_chunk_file("invalid_type")
        
        expected_error_message = "Unsupported data_type for chunk file: invalid_type. Supported types are 'email', 'text_file', 'xml', 'marcxml'."
        self.assertEqual(str(context.exception), expected_error_message)

if __name__ == '__main__':
    # This allows running the tests directly from this file
    unittest.main()
