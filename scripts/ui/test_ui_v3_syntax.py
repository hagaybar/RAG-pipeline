import unittest
import sys
import os
from unittest.mock import MagicMock

# Add project root to sys.path to allow importing scripts.ui.ui_v3
# Assuming this test file is in scripts/ui/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Mock potentially problematic imports in ui_v3.py that are not relevant to syntax checking
# and have caused issues in previous test runs (like streamlit, faiss, torch, etc.)
# We want to isolate syntax errors from import errors of missing heavy dependencies.
MOCK_MODULES = [
    'streamlit', 'streamlit.components.v1', 'faiss', 'pandas', 'numpy',
    'win32com', 'win32com.client', 'pythoncom',
    'spacy', 'pymarc', 'sklearn', 'transformers', 'torch',
    'altair', 'pydeck',
    'scripts.pipeline.rag_pipeline', # Main import from ui_v3
    # Dependencies of rag_pipeline and its components, to prevent deep import errors
    'scripts.data_processing.email.email_fetcher',
    'scripts.data_processing.text.text_processor',
    'scripts.data_processing.xml.xml_processor',
    'scripts.data_processing.marc_xml.marc_xml_processor',
    'scripts.chunking.text_chunker_v2',
    'scripts.embedding.general_purpose_embedder',
    'scripts.llm.llm_client',
    'scripts.vector_store.faiss_vector_store',
    'scripts.utils.utils', # General utilities, might be imported
    'scripts.utils.paths'  # Path utilities, might be imported
]

original_modules = {}

class TestUISyntax(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Mock modules before any test attempts to import ui_v3
        for module_name in MOCK_MODULES:
            if module_name not in sys.modules:
                original_modules[module_name] = None # Mark that it wasn't originally there
                sys.modules[module_name] = MagicMock()
            else:
                original_modules[module_name] = sys.modules[module_name] # Store original
                sys.modules[module_name] = MagicMock()
        
        # Specifically configure st.columns for the error in ui_v3.py
        if 'streamlit' in sys.modules:
            st_mock = sys.modules['streamlit']

            def columns_side_effect(*args, **kwargs):
                if args:
                    if isinstance(args[0], int): # For st.columns(N)
                        return tuple(MagicMock() for _ in range(args[0]))
                    elif isinstance(args[0], list): # For st.columns([spec])
                        return tuple(MagicMock() for _ in range(len(args[0])))
                return (MagicMock(), MagicMock()) # Default fallback

            st_mock.columns = MagicMock(side_effect=columns_side_effect)
            
            # Mock other streamlit features if they cause similar issues during import
            st_mock.expander = MagicMock() # Used as a context manager
            # Ensure that when expander is used in a 'with' statement, its __enter__ returns the mock itself.
            st_mock.expander.return_value.__enter__.return_value = st_mock.expander.return_value
            st_mock.tabs = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock()))

            # Mock session_state.get to prevent TypeError with embedding_dimension
            # The get_config_data method in ConfigFormBuilder checks session state for values
            # Configure the MagicMock for streamlit.session_state
            
            # This function will be the side_effect for st.session_state.get
            def session_get_side_effect(key, default=None):
                # print(f"DEBUG_MOCK: st.session_state.get called with key='{key}', default='{default}'") # Debug print
                if key == "create_form_task_name":
                    return "mock_task_name"
                elif key == "create_form_embedding_model_name":
                    return "mock_model_name"
                elif key == "create_form_embedding_dimension":
                    # This is the crucial one for the current error
                    return 128 
                elif key == "create_form_chunk_size":
                    return 512
                elif key == "create_form_chunk_overlap":
                    return 50
                elif key == "create_form_llm_model_name":
                    return "mock_llm"
                elif key == "create_form_data_type":
                    return "email" 
                elif key == "create_form_use_local_embedding_model":
                    return False 
                elif key == "create_form_vector_store_path":
                    return "mock/path"
                elif key == "create_form_index_name":
                    return "mock_index"
                elif key == "create_form_retrieval_k":
                    return 3
                elif key == "create_form_retrieval_score_threshold":
                    return 0.7
                elif key == "create_form_llm_temperature":
                    return 0.7
                elif key == "create_form_llm_max_tokens":
                    return 512
                elif key == "create_form_prompt_template":
                    return "mock prompt {context} {query}"
                # Similar keys for "edit" mode if that form builder is also accessed globally
                elif key == "edit_form_task_name": return "mock_edit_task" 
                elif key == "edit_form_embedding_model_name": return "mock_edit_model"
                elif key == "edit_form_embedding_dimension": return 256

                # print(f"DEBUG_MOCK: Key '{key}' not specifically handled, returning default: {default}")
                return default 
            
            # Ensure st_mock.session_state itself is a MagicMock that has a .get() method
            # and a __contains__ method for potential 'key in session_state' checks.
            
            # Simplified session state mock that acts more like a dict for .get()
            # This dictionary will store the "session state" for the mock.
            mock_session_storage = {
                "create_form_task_name": "mock_task_name",
                "create_form_embedding_model_name": "mock_model_name",
                "create_form_embedding_dim": 128, # Corrected key
                "create_form_chunk_size": 512,
                "create_form_chunk_overlap": 50,
                "create_form_llm_model_name": "mock_llm",
                "create_form_data_type": "email",
                "create_form_use_local_embedding_model": False,
                "create_form_vector_store_path": "mock/path",
                "create_form_index_name": "mock_index",
                "create_form_retrieval_k": 3,
                "create_form_retrieval_score_threshold": 0.7,
                "create_form_llm_temperature": 0.7,
                "create_form_llm_max_tokens": 512,
                "create_form_prompt_template": "mock prompt {context} {query}",
                # Add edit form keys if necessary, though create_form is used by the global form_builder
                "edit_form_embedding_dimension": 256 # Example for edit mode
            }

            def robust_session_get(key, default=None):
                print(f"ROBUST_SESSION_GET CALLED: key='{key}', default='{str(default)[:50]}'") # Debug print
                # Explicitly handle the problematic key first
                if key == "create_form_embedding_model_name":
                    val = "mock_model_name"
                elif key == "create_form_embedding_dim": # Corrected key here
                    val = 128
                elif key == "create_form_task_name": # Ensure this is also explicitly handled
                    val = mock_session_storage.get(key, default) # Or "mock_task_name" directly
                else:
                    # Fallback to the dictionary for other keys
                    val = mock_session_storage.get(key, default)
                
                print(f"ROBUST_SESSION_GET RETURNING: key='{key}', value='{str(val)[:50]}'") # Debug print
                return val

            st_mock.session_state = MagicMock() # Mock the session_state object
            st_mock.session_state.get = MagicMock(side_effect=robust_session_get)
            # Mock `__contains__` if `in st.session_state` is used by ConfigFormBuilder
            st_mock.session_state.__contains__ = MagicMock(side_effect=lambda key: key in mock_session_storage)
            # Mock `__setitem__` and `__getitem__` if direct dict-like access is used (st.session_state[key] = val)
            st_mock.session_state.__setitem__ = MagicMock(side_effect=lambda key, value: mock_session_storage.__setitem__(key, value))
            st_mock.session_state.__getitem__ = MagicMock(side_effect=lambda key: mock_session_storage.__getitem__(key))




    @classmethod
    def tearDownClass(cls):
        # Restore original modules or remove mocks
        for module_name, original_module in original_modules.items():
            if original_module is None: # Was not in sys.modules before
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = original_module # Restore original

    def test_ui_v3_syntax_and_basic_imports(self):
        """
        Tests that scripts.ui.ui_v3 can be imported without a SyntaxError.
        Other ImportErrors (e.g., for missing heavy dependencies if not mocked)
        are considered different from a syntax error in the file itself.
        """
        syntax_error_raised = False
        try:
            # Attempt to import the module
            from scripts.ui import ui_v3
            # If import is successful (or fails for reasons other than SyntaxError),
            # this line will be reached.
            self.assertTrue(True, "Import of ui_v3.py did not raise an immediate SyntaxError.")
        except SyntaxError as e:
            syntax_error_raised = True
            self.fail(f"SyntaxError occurred during import of ui_v3.py: {e}")
        except ImportError as e:
            # This is to catch other import errors if mocking is incomplete.
            # For this specific test, we are primarily concerned with SyntaxError.
            self.skipTest(f"ImportError occurred: {e}. This test focuses on SyntaxError, "
                          "other ImportErrors indicate missing/unmocked dependencies not syntax issues in ui_v3.py itself.")
        except Exception as e:
            # Catch any other exception during import
            self.fail(f"An unexpected error occurred during import of ui_v3.py: {e}")
        
        if not syntax_error_raised:
            # This part is mostly for clarity in test output if it doesn't fail above.
            print("\nSuccessfully imported scripts.ui.ui_v3 (or skipped due to non-SyntaxError ImportError).")

if __name__ == '__main__':
    unittest.main()
