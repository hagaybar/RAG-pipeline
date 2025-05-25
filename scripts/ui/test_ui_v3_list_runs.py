import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Mock win32com and pythoncom before importing ui_v3
sys.modules['win32com'] = MagicMock()
sys.modules['win32com.client'] = MagicMock()
sys.modules['pythoncom'] = MagicMock()

# Add the parent directory to the Python path to allow importing ui_v3
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui.ui_v3 import list_run_ids

class TestListRunIds(unittest.TestCase):

    @patch('os.path.isdir')
    def test_list_run_ids_none_task(self, mock_isdir):
        # This test doesn't strictly need os.path.isdir to be mocked for its direct logic,
        # but the function calls it, so we mock it to avoid unintended side effects
        # if the path "runs/None" was somehow valid in a test environment.
        mock_isdir.return_value = False # Or True, doesn't matter for None task_name
        self.assertEqual(list_run_ids(None), [])

    @patch('os.path.isdir')
    def test_list_run_ids_invalid_task_path(self, mock_isdir):
        mock_isdir.return_value = False
        self.assertEqual(list_run_ids("non_existent_task"), [])
        mock_isdir.assert_called_once_with(os.path.join("runs", "non_existent_task"))

    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_list_run_ids_task_no_runs_subdir(self, mock_isdir, mock_exists):
        # Simulate task directory 'runs/task_with_no_runs_subdir' exists
        mock_isdir.return_value = True 
        # Simulate 'runs/task_with_no_runs_subdir/runs' does NOT exist
        mock_exists.return_value = False
        
        self.assertEqual(list_run_ids("task_with_no_runs_subdir"), [])
        mock_isdir.assert_called_once_with(os.path.join("runs", "task_with_no_runs_subdir"))
        mock_exists.assert_called_once_with(os.path.join("runs", "task_with_no_runs_subdir", "runs"))

    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_list_run_ids_task_with_empty_runs_dir(self, mock_isdir, mock_exists, mock_listdir):
        # Simulate task directory 'runs/task_with_empty_runs' exists
        mock_isdir.return_value = True
        # Simulate 'runs/task_with_empty_runs/runs' exists
        mock_exists.return_value = True
        # Simulate the runs directory is empty
        mock_listdir.return_value = []
        
        self.assertEqual(list_run_ids("task_with_empty_runs"), [])
        mock_isdir.assert_called_once_with(os.path.join("runs", "task_with_empty_runs"))
        run_dir_path = os.path.join("runs", "task_with_empty_runs", "runs")
        mock_exists.assert_called_once_with(run_dir_path)
        mock_listdir.assert_called_once_with(run_dir_path)

    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_list_run_ids_task_with_runs(self, mock_isdir, mock_exists, mock_listdir):
        # Simulate task directory 'runs/task_with_runs' exists
        mock_isdir.return_value = True
        # Simulate 'runs/task_with_runs/runs' exists
        mock_exists.return_value = True
        # Simulate some run IDs in the directory
        mock_listdir.return_value = ['run1', 'run2', 'run0'] # Unsorted
        
        expected_runs = ['run2', 'run1', 'run0'] # Sorted in reverse
        self.assertEqual(list_run_ids("task_with_runs"), expected_runs)
        
        mock_isdir.assert_called_once_with(os.path.join("runs", "task_with_runs"))
        run_dir_path = os.path.join("runs", "task_with_runs", "runs")
        mock_exists.assert_called_once_with(run_dir_path)
        mock_listdir.assert_called_once_with(run_dir_path)

if __name__ == '__main__':
    unittest.main()
