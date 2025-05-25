import unittest
import sys
import os
from unittest.mock import MagicMock

# Add project root to sys.path to allow importing scripts.ui.config_form_builder
# Assuming this test file is in scripts/ui/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Mock win32com and pythoncom before importing ConfigFormBuilder, as it has deep dependencies
# that might lead to these Windows-specific modules.
# Also mock other heavy dependencies that might be pulled in by RAGPipeline.
MOCK_MODULES_FOR_CONFIG_BUILDER = [
    'win32com', 'win32com.client', 'pythoncom',
    'streamlit', # ConfigFormBuilder uses st directly
    'scripts.pipeline.rag_pipeline', # Imported by ConfigFormBuilder for constants
    'faiss', 'pandas', 'numpy',
    'spacy', 'pymarc', 'sklearn', 'transformers', 'torch', 'altair', 'pydeck'
]
original_config_builder_test_modules = {}

for module_name in MOCK_MODULES_FOR_CONFIG_BUILDER:
    if module_name not in sys.modules:
        original_config_builder_test_modules[module_name] = None
        sys.modules[module_name] = MagicMock()
    else:
        original_config_builder_test_modules[module_name] = sys.modules[module_name]
        sys.modules[module_name] = MagicMock()

# Specific mock for RAGPipeline if its class attributes are accessed upon import
if 'scripts.pipeline.rag_pipeline' in sys.modules:
    mock_rag_pipeline = sys.modules['scripts.pipeline.rag_pipeline']
    mock_rag_pipeline.RAGPipeline.DATA_TYPE_EMAIL = "email"
    mock_rag_pipeline.RAGPipeline.DATA_TYPE_TEXT = "text_file"
    mock_rag_pipeline.RAGPipeline.DATA_TYPE_XML = "xml"
    mock_rag_pipeline.RAGPipeline.DATA_TYPE_MARCXML = "marcxml"
    mock_rag_pipeline.RAGPipeline.SUPPORTED_DATA_TYPES = ["email", "text_file", "xml", "marcxml"]


from scripts.ui.config_form_builder import ConfigFormBuilder

# Restore original modules after import (mostly for interactive sessions, less critical in script)
# For robust test isolation, this kind of cleanup is better in tearDownClass if tests were in a class.
# However, for this specific script, we'll do it here if it's simple.
# For this single file test, direct mocking is fine for now.

original_config_builder_test_modules_backup = {} # Renamed to avoid conflict if script is run multiple times

class TestConfigFormBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global original_config_builder_test_modules_backup # Use the renamed global dict
        # Mock modules before any test attempts to import ConfigFormBuilder
        for module_name in MOCK_MODULES_FOR_CONFIG_BUILDER:
            if module_name in sys.modules:
                original_config_builder_test_modules_backup[module_name] = sys.modules[module_name]
                sys.modules[module_name] = MagicMock()
            else:
                original_config_builder_test_modules_backup[module_name] = None # Mark that it wasn't originally there
                sys.modules[module_name] = MagicMock()

        # Specific mock for RAGPipeline constants accessed by ConfigFormBuilder
        if 'scripts.pipeline.rag_pipeline' in sys.modules:
            mock_rag_pipeline = sys.modules['scripts.pipeline.rag_pipeline']
            mock_rag_pipeline.RAGPipeline.DATA_TYPE_EMAIL = "email"
            mock_rag_pipeline.RAGPipeline.DATA_TYPE_TEXT = "text_file"
            mock_rag_pipeline.RAGPipeline.DATA_TYPE_XML = "xml"
            mock_rag_pipeline.RAGPipeline.DATA_TYPE_MARCXML = "marcxml"
            mock_rag_pipeline.RAGPipeline.SUPPORTED_DATA_TYPES = ["email", "text_file", "xml", "marcxml"]
        
        # Mock streamlit within the class for ConfigFormBuilder instantiation in tests
        cls.st_mock = MockStreamlit()


    @classmethod
    def tearDownClass(cls):
        global original_config_builder_test_modules_backup
        # Restore original modules or remove mocks
        for module_name, original_module in original_config_builder_test_modules_backup.items():
            if original_module is None: # Was not in sys.modules before
                if module_name in sys.modules: # Check if it was added by MagicMock
                     del sys.modules[module_name]
            else:
                sys.modules[module_name] = original_module # Restore original
        original_config_builder_test_modules_backup.clear()


class MockStreamlit:
    """A minimal mock for streamlit for ConfigFormBuilder instantiation."""
    def __init__(self):
        self.session_state = {}
        # Mock other st attributes or methods if ConfigFormBuilder's __init__ or _get_key uses them.
        self.text_input = MagicMock(return_value="mock_text_input")
        self.text_area = MagicMock(return_value="mock_text_area")
        self.number_input = MagicMock(return_value=0.0)
        self.selectbox = MagicMock(return_value="mock_selectbox")
        self.checkbox = MagicMock(return_value=False)
        self.multiselect = MagicMock(return_value=[])
        self.slider = MagicMock(return_value=0)
        self.expander = MagicMock() # Can be used with 'with' statement
        self.columns = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock())) # Mock to return 4 for st.columns(4)


    def test_config_form_builder_unique_keys_for_modes(self):
        """
        Tests that ConfigFormBuilder._get_key() generates different keys for
        different modes ('create' vs 'edit').
        """
        # ConfigFormBuilder is imported at the top, after mocks are set up by setUpClass
        
        builder_create = ConfigFormBuilder(self.st_mock, mode="create")
        key_create = builder_create._get_key("task_name")

        builder_edit = ConfigFormBuilder(st_mock, mode="edit")
        key_edit = builder_edit._get_key("task_name")

        self.assertNotEqual(key_create, key_edit,
                            f"Keys should be different for create and edit modes. Got: create='{key_create}', edit='{key_edit}'")
        
        # Based on the fix, the prefixes should be mode-specific
        self.assertTrue(key_create.startswith("create_form_"),
                        f"Expected key for 'create' mode to start with 'create_form_', got '{key_create}'")
        self.assertTrue(key_edit.startswith("edit_form_"),
                        f"Expected key for 'edit' mode to start with 'edit_form_', got '{key_edit}'")
        
        # Check the full key name based on the implementation (prefix + original_key)
        self.assertEqual(key_create, "create_form_task_name",
                         f"Expected 'create_form_task_name', got '{key_create}'")
        self.assertEqual(key_edit, "edit_form_task_name",
                         f"Expected 'edit_form_task_name', got '{key_edit}'")
        
        # Test with another key to be sure
        key_create_model = builder_create._get_key("embedding_model")
        key_edit_model = builder_edit._get_key("embedding_model")

        self.assertNotEqual(key_create_model, key_edit_model,
                            "Model keys should also be different for create and edit modes.")
        self.assertEqual(key_create_model, "create_form_embedding_model")
        self.assertEqual(key_edit_model, "edit_form_embedding_model")


if __name__ == '__main__':
    unittest.main()
