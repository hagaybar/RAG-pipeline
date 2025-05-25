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

# Define tabs
tab_titles = ["Configuration", "Pipeline Activity", "Utilities"]
tabs = st.tabs(tab_titles)

# Add placeholder content to each tab
with tabs[0]: # Configuration
    st.header("Configuration")
    st.write("Manage task configurations here.")

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
