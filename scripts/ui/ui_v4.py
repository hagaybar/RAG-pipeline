import os
import sys
import streamlit as st
import glob # For list_config_files, will be used in next step

# Calculate the project root based on the current file's location
# Assumes ui_v4.py is in scripts/ui/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Add PROJECT_ROOT to sys.path if it's not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Basic Page Configuration
st.set_page_config(page_title="RAG Pipeline UI v4", layout="wide")

st.title("📬 RAG Pipeline Control Panel v4")

# Initialize session state variables at the beginning of the script
if "selected_config_v4" not in st.session_state:
    st.session_state.selected_config_v4 = None # Stores the filename of the selected config
if "config_content_v4" not in st.session_state:
    st.session_state.config_content_v4 = "" # Stores the text content for the editor
if "selectbox_config_selector_key" not in st.session_state: # Initialize key for the selectbox itself if needed
    st.session_state.selectbox_config_selector_key = ""
if "edit_mode_v4" not in st.session_state:
    st.session_state.edit_mode_v4 = False # Manages if the text_area is editable


# Define tabs
tab_titles = ["Configuration", "Pipeline Activity", "Utilities"]
tabs = st.tabs(tab_titles)

# Add placeholder content to each tab
with tabs[0]: # Configuration
    st.header("Configuration Management") # Updated header
           
    st.subheader("Available Task Configurations")
    config_files = list_config_files() # Default path is "configs/tasks"

    # Callback function to update content when selectbox selection changes
    def _config_selection_changed():
        # selected_filename is the value from the selectbox widget
        selected_filename = st.session_state.selectbox_config_selector_key 
        st.session_state.selected_config_v4 = selected_filename # Update the central selected config state
        if selected_filename:
            st.session_state.config_content_v4 = load_config_content(selected_filename)
        else:
            st.session_state.config_content_v4 = "" # Clear content if no file is selected

    if config_files:
        # Ensure the current session state for selected_config_v4 is valid given the available files
        # This handles cases where a previously selected file might have been deleted
        if st.session_state.selected_config_v4 not in config_files:
            st.session_state.selected_config_v4 = None 
            st.session_state.config_content_v4 = "" # Clear content if selection is invalid
            st.session_state.selectbox_config_selector_key = "" # Reset selectbox widget state

        st.selectbox(
            "Select Configuration File:",
            options=[""] + config_files,  # Add an empty option for "none selected"
            key="selectbox_config_selector_key", # Key for the selectbox widget itself
            on_change=_config_selection_changed
        )
        
        # The selected_config_v4 is now primarily driven by the on_change callback setting it.
        # Content display relies on config_content_v4 which is also set in the callback.

        if st.session_state.selected_config_v4: # If a file is selected (and valid)
            # Display content if it's loaded in session state
            # The on_change callback should handle loading it.
            # If selectbox_config_selector_key is the selected_config_v4 and content is empty, it implies an error or initial state
            if st.session_state.selectbox_config_selector_key == st.session_state.selected_config_v4 and not st.session_state.config_content_v4 and st.session_state.selected_config_v4:
                 # This might happen if on_change hasn't triggered a re-run that updates config_content_v4 yet for display,
                 # or if load_config_content returned an empty string (e.g. error message that got cleared).
                 # For robustness, we can call load_config_content here again if content is unexpectedly empty for a selected file.
                 # However, Streamlit's on_change should ideally handle this by re-running and making the updated state available.
                 # Let's ensure content is loaded if selected_config_v4 is set but content is empty (e.g., after clearing due to deselection)
                 # This explicit load can be helpful on first selection if on_change doesn't immediately reflect.
                 st.session_state.config_content_v4 = load_config_content(st.session_state.selected_config_v4)


            st.text_area(
                "Configuration Content (YAML):",
                value=st.session_state.config_content_v4,
                height=300,
                disabled=True, # Read-only for now
                key="config_editor_textarea_key_v4" # A unique key for the text area
            )

            st.divider() # Visual separator

            cols = st.columns(3) # Create three columns for the buttons

            # Button 1: View/Edit (label changes based on mode)
            view_edit_button_label = "Cancel Edit" if st.session_state.edit_mode_v4 else "View/Edit YAML"
            if cols[0].button(view_edit_button_label, key="view_edit_button_v4"):
                st.session_state.edit_mode_v4 = not st.session_state.edit_mode_v4
                if not st.session_state.edit_mode_v4: # If toggled OFF (Cancel Edit)
                    # Reload content from file to discard any uncommitted changes
                    if st.session_state.selected_config_v4:
                        st.session_state.config_content_v4 = load_config_content(st.session_state.selected_config_v4)
                    else:
                        st.session_state.config_content_v4 = "" # Clear content if no file selected
                st.rerun()

            # Button 2: Save Changes (initially disabled if not in edit mode)
            if cols[1].button("Save Changes", key="save_button_v4", disabled=not st.session_state.edit_mode_v4):
                # Placeholder for save functionality
                st.toast("Save functionality not yet implemented.") 
                pass

            # Button 3: Delete (conditionally shown or enabled)
            # For safety, disable if no config is selected or if in edit mode (to prevent accidental deletion while editing)
            delete_disabled = not st.session_state.selected_config_v4 or st.session_state.selected_config_v4 == "" or st.session_state.edit_mode_v4
            if cols[2].button("Delete Config", key="delete_button_v4", disabled=delete_disabled):
                # Placeholder for delete functionality
                st.toast("Delete functionality not yet implemented.")
                pass
        else:
            st.info("No configuration file selected, or selected file was removed.")
            if st.session_state.config_content_v4: # Clear content if no file is effectively selected
                st.session_state.config_content_v4 = ""
            
    else:
        st.info("No task configurations found in the 'configs/tasks' directory.")
        st.session_state.selected_config_v4 = None # Ensure selection is cleared
        st.session_state.config_content_v4 = ""  # Ensure content is cleared
           
    # Placeholder for future actions like selecting, viewing, editing configs
    st.write("---") 
    st.write("Future actions (select, view, edit, create new) will appear here.")

with tabs[1]: # Pipeline Activity
    st.header("Pipeline Activity")
    st.write("Run pipeline steps and view results here.")

with tabs[2]: # Utilities
    st.header("Utilities")
    st.write("Access debugging tools and other utilities here.")

# Placeholder for helper functions that will be added in later steps
def list_config_files(config_dir: str = "configs/tasks") -> list:
    """
    Scan the specified directory for .yaml files.

    Args:
        config_dir (str): Path to the config folder relative to PROJECT_ROOT.

    Returns:
        List[str]: Sorted list of config file names (not full paths).
    """
    # Ensure the config_dir path is absolute or correctly relative to the project root
    search_path = os.path.join(PROJECT_ROOT, config_dir, "*.yaml")
    # Use glob to find all yaml files in the directory
    config_file_paths = glob.glob(search_path)
    # Extract just the file names and sort them
    config_file_names = sorted([os.path.basename(f) for f in config_file_paths])
    return config_file_names

if __name__ == "__main__":
    # This block can be useful for testing the UI script directly
    # For now, it's not strictly necessary for Streamlit's execution model
    pass

def load_config_content(config_filename: str) -> str:
    """Loads the raw string content of a given configuration file."""
    if not config_filename: # Handle cases where no file is selected
        return ""
    file_path = os.path.join(PROJECT_ROOT, "configs", "tasks", config_filename)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{config_filename}' not found."
    except Exception as e:
        return f"Error reading file '{config_filename}': {str(e)}"
