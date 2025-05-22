import os
import sys # Ensure sys is imported

# Calculate the project root based on the current file's location
# Assumes ui_v3.py is in scripts/ui/
# os.path.dirname(__file__) gives scripts/ui
# os.path.join(..., "../../") goes up two levels to Rag_Project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Add PROJECT_ROOT to sys.path if it's not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# All other original imports (import streamlit as st, etc.) should follow this block.
import streamlit as st
import glob
import yaml
import os
import faiss # Added for Embedding Stats
import pandas as pd # Added for Embedding Stats
import numpy as np # Added for Chunk Metrics
from scripts.pipeline.rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG Pipeline UI", layout="wide")

# Initialize session state variables if not already present
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = []
if "show_pipeline_output_area" not in st.session_state:
    st.session_state.show_pipeline_output_area = False
if "similarity_results" not in st.session_state: # Added for Similarity Test
    st.session_state.similarity_results = []     # Added for Similarity Test
if "metadata_to_download" not in st.session_state: # Added for File Tools
    st.session_state.metadata_to_download = None   # Added for File Tools
if "metadata_download_filename" not in st.session_state: # Added for File Tools
    st.session_state.metadata_download_filename = None # Added for File Tools

# Tab 1 specific session states
if "edited_config_text" not in st.session_state: 
    st.session_state.edited_config_text = ""
if "edit_mode" not in st.session_state: 
    st.session_state.edit_mode = False
if "confirm_delete" not in st.session_state: # Also set in selectbox context, but global init is safer
    st.session_state.confirm_delete = False
if "open_preview_yaml" not in st.session_state: 
    st.session_state.open_preview_yaml = False
if "edit_mode_toggle" not in st.session_state: 
    st.session_state.edit_mode_toggle = False
if "new_task_name_input" not in st.session_state: # Cleared on success, but init is good
    st.session_state.new_task_name_input = ""
if "open_dup_box" not in st.session_state: 
    st.session_state.open_dup_box = False
if "dup_mode" not in st.session_state: # For duplicate button, related to open_dup_box
    st.session_state.dup_mode = False

# Chunk Reviewer specific session states
if "chunk_review_data" not in st.session_state: 
    st.session_state.chunk_review_data = None
if "grouped_chunk_data" not in st.session_state: 
    st.session_state.grouped_chunk_data = None
if "selected_email_id_for_review" not in st.session_state: 
    st.session_state.selected_email_id_for_review = None
if "current_email_body_for_review" not in st.session_state: 
    st.session_state.current_email_body_for_review = None
if "chunks_for_selected_email" not in st.session_state: 
    st.session_state.chunks_for_selected_email = None
if "selected_chunk_for_detail_review" not in st.session_state: # For Chunk Detail Explorer
    st.session_state.selected_chunk_for_detail_review = None
if "selected_chunk_metadata_for_detail" not in st.session_state: # For Chunk Detail Explorer
    st.session_state.selected_chunk_metadata_for_detail = None
if "run_steps_execution_count" not in st.session_state: # For Tab 3 diagnostic
    st.session_state.run_steps_execution_count = 0


st.title("📬 RAG Pipeline Control Panel")

# --- Helper Functions (Global Scope) ---
def list_config_files(config_dir: str = "configs/tasks") -> list:
    """
    Scan the task_configs directory for .yaml files.

    Args:
        config_dir (str): Path to the config folder.

    Returns:
        List[str]: Sorted list of config file names.
    """
    return sorted([f.split("/")[-1] for f in glob.glob(f"{config_dir}/*.yaml")])

def load_config(config_path: str) -> str:
    """
    Load the content of a YAML config file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        str: Raw text content of the YAML file.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading config: {e}"

# --- Helper Functions for Chunking Reviewer (Global Scope) ---

def load_chunk_data_for_task(task_config_name: str) -> pd.DataFrame | None:
    """
    Loads the chunk TSV file for a given task configuration.
    """
    config_file_path = os.path.join("configs", "tasks", task_config_name)

    def _load_yaml_config(file_path):
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            st.error(f"Error loading YAML config {file_path}: {str(e)}")
            return None
    
    task_config_data = _load_yaml_config(config_file_path)

    if task_config_data:
        chunked_emails_path = task_config_data.get("paths", {}).get("chunked_emails")
        if not chunked_emails_path:
            st.error(f"Path to 'chunked_emails' not found in config for task '{task_config_name}'.")
            return None
        
        if not os.path.exists(chunked_emails_path):
            st.error(f"Chunk file path not found or file does not exist for task '{task_config_name}'. Path: {chunked_emails_path}")
            return None
        
        try:
            df = pd.read_csv(chunked_emails_path, sep="\t", on_bad_lines='skip')
            return df
        except Exception as e:
            st.error(f"Error loading chunk TSV file {chunked_emails_path}: {str(e)}")
            return None
    else:
        return None

def group_chunks_by_email(chunk_df: pd.DataFrame, email_id_column: str = "EntryID") -> dict[str, pd.DataFrame]:
    """
    Groups the loaded chunks by a unique email identifier.
    """
    if chunk_df is None or chunk_df.empty:
        st.warning("Cannot group chunks: DataFrame is empty or None.")
        return {}
    
    if email_id_column not in chunk_df.columns:
        st.warning(f"Cannot group chunks: Email ID column '{email_id_column}' is missing from DataFrame. Available columns: {chunk_df.columns.tolist()}")
        return {}
        
    try:
        # Ensure the email_id_column is string type to avoid issues with mixed types if some IDs are numeric
        # Also handle potential NaN values if EntryID was missing for some rows, convert to a placeholder string
        chunk_df[email_id_column] = chunk_df[email_id_column].astype(str).fillna("MISSING_ENTRY_ID")
        grouped_data = {str(email_id): group_df for email_id, group_df in chunk_df.groupby(email_id_column)}
        return grouped_data
    except Exception as e:
        st.error(f"Error grouping chunks by '{email_id_column}': {str(e)}")
        return {}


def get_original_email_body(email_specific_df: pd.DataFrame, body_column_candidates: list[str] = ["Raw Body", "Cleaned Body"]) -> str | None:
    """
    Retrieves the original email body from the DataFrame corresponding to a single email.
    """
    if email_specific_df is None or email_specific_df.empty:
        return None
        
    for column_name in body_column_candidates:
        if column_name in email_specific_df.columns:
            # Assuming the body is consistent across all chunks of the same email
            body_value = email_specific_df[column_name].iloc[0]
            if pd.notna(body_value): # Check if the value is not NaN or None
                return str(body_value) 
            # If body_value is NaN/None, try next candidate
            
    st.warning(f"Could not find any of the specified body columns ({body_column_candidates}) with non-empty content in the provided data for this email.")
    return None

# Top-level tab navigation
tabs = st.tabs(["Tasks 🛠", "Runs & Logs 📊", "Pipeline Actions ⚙️", "Utilities & Tools 🧰"])

# ----------------------
# Tab 1: Task Management
# ----------------------
with tabs[0]:
    st.header("Task Management")

    # 🧩 Feature 1: Load and display available task configs from disk

    with st.expander("🔽 Select Task Config", expanded=True):
        # List all available configs
        config_list = list_config_files()
        selected_config = st.selectbox("Available Configs:", config_list)
        st.session_state.confirm_delete = False

        # Load the selected config file
        config_path = os.path.join("configs", "tasks", os.path.basename(selected_config))
        config_text = load_config(config_path)

        # Display the loaded config content
        cols = st.columns(4)
        if cols[0].button("View"):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    new_text = f.read()
                st.session_state.edited_config_text = new_text
                st.session_state.edit_mode = False
                st.session_state.open_preview_yaml = True
                st.success("🔄 Config reloaded from file.")
            except Exception as e:
                st.error(f"❌ Failed to reload config: {e}")
        # Edit
        if cols[1].button("Edit"):
            st.session_state.edit_mode_toggle = True
            st.session_state.open_preview_yaml = True
        
        # Duplicate
        with cols[2]:
            if st.button("Duplicate"):
                st.session_state.dup_mode = True
                st.session_state.open_dup_box = True

            if st.session_state.get("open_dup_box", False):
                new_name = st.text_input("New file name (without .yaml)", key="new_dup_name")
                if st.button("Confirm Duplicate", key="confirm_dup"):
                    new_file = f"configs/tasks/{new_name}.yaml"
                    if os.path.exists(new_file):
                        st.error("⚠️ File already exists.")
                    else:
                        try:
                            with open(config_path, "r", encoding="utf-8") as src, open(new_file, "w", encoding="utf-8") as dst:
                                dst.write(src.read())
                            st.success(f"✅ Duplicated to {new_file}")
                            st.session_state.open_dup_box = False
                        except Exception as e:
                            st.error(f"❌ Duplication failed: {e}")

        # Delete
        with cols[3]:
            if not st.session_state.get("confirm_delete", False):
                if st.button("Delete"):
                    st.session_state.confirm_delete = True
            else:
                st.warning(f"⚠️ Are you sure you want to delete `{selected_config}`?")
                col_yes, col_no = st.columns([1, 1])
                with col_yes:
                    if st.button("✅ Confirm Delete"):
                        try:
                            os.remove(config_path)
                            st.success(f"🗑️ Deleted {selected_config}")
                            st.session_state.confirm_delete = False
                        except Exception as e:
                            st.error(f"❌ Deletion failed: {e}")
                with col_no:
                    if st.button("❌ Cancel"):
                        st.session_state.confirm_delete = False



    with st.expander("➕ Create New Task"):
        new_task_name = st.text_input("New Task Name", key="new_task_name_input")
        if st.button("Create from Template"):
            if not new_task_name:
                st.error("Task name cannot be empty.")
            else:
                try:
                    pipeline = RAGPipeline()
                    # configure_task creates the file like "configs/tasks/new_task_name.yaml"
                    # It returns the path, but we might not need it directly if list_config_files refreshes
                    pipeline.configure_task(task_name=new_task_name, output_format="yaml")
                    st.success(f"Task '{new_task_name}' created successfully from template!")
                    # Clear input field
                    st.session_state.new_task_name_input = ""
                    # Force rerun to update the selectbox in the other expander
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to create task: {e}")

    # 🧩 Feature 2: Safe Edit Mode for YAML config
    with st.expander("📑 Preview Task YAML", expanded=st.session_state.get("open_preview_yaml", False)):
        # Use session state to remember edit mode + content
        if "edit_mode" not in st.session_state:
            st.session_state.edit_mode = False
        if "edited_config_text" not in st.session_state:
            st.session_state.edited_config_text = config_text

        # Toggle edit mode if Edit button clicked
        if st.session_state.get("edit_mode_toggle", False):
            st.session_state.edit_mode = True
            st.session_state.edited_config_text = config_text
            st.session_state.edit_mode_toggle = False

        mode = "Edit mode" if st.session_state.edit_mode else "Read-only"
        st.radio("Mode:", ["Read-only", "Edit mode"], index=(1 if st.session_state.edit_mode else 0), key="preview_mode", disabled=True)

        if st.session_state.edit_mode:
            # Editable text box
            st.session_state.edited_config_text = st.text_area("Edit Config", value=st.session_state.edited_config_text, height=300, key="config_editor")

            if st.button("💾 Save Changes"):
                try:
                    with open(config_path, "w", encoding="utf-8") as f:
                        f.write(st.session_state.edited_config_text)
                    st.success("✅ Config saved successfully.")
                    st.session_state.edit_mode = False
                except Exception as e:
                    st.error(f"❌ Failed to save config: {e}")
        else:
            # Read-only preview
            st.text_area("Config Content", value=config_text, height=300, disabled=True)
        # Reset flag after use so it doesn't persist across reruns
        st.session_state.open_preview_yaml = False


# ----------------------
# Tab 2: Runs & Logs 📊
# ----------------------
with tabs[1]:
    import glob
    import json

    st.header("Run and Log Inspection")

    def list_tasks(base_dir: str = "runs") -> list:
        """List task directories inside runs/"""
        return sorted([name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))])

    def list_run_ids(task_name: str) -> list:
        """List run folders under runs/{task_name}/runs/"""
        run_dir = os.path.join("runs", task_name, "runs")
        if not os.path.exists(run_dir):
            return []
        return sorted(os.listdir(run_dir), reverse=True)

    def load_json(path: str) -> dict:
        """Load JSON file if it exists"""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"error": f"File not found: {path}"}

    def load_text(path: str) -> str:
        """Load plain text file if it exists"""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return f"(No content found in {path})"

    # Task selector
    task_list = list_tasks()
    selected_task = st.selectbox("Select Task:", task_list)

    # Run ID selector
    run_ids = list_run_ids(selected_task)
    selected_run = st.selectbox("Run ID:", run_ids)

    # Define file paths
    run_base = os.path.join("runs", selected_task, "runs", selected_run)
    meta_path = os.path.join(run_base, "run_metadata.json")
    answer_path = os.path.join(run_base, "answer.txt")
    debug_path = os.path.join(run_base, "query_debug.txt")
    log_path = os.path.join(run_base, "log.txt")

    # Display content
    with st.expander("📄 Run Metadata"):
        st.json(load_json(meta_path))

    with st.expander("💬 Answer Output"):
        st.code(load_text(answer_path), language="text")

    with st.expander("📜 Query Debug Log"):
        st.code(load_text(debug_path), language="text")

    with st.expander("🪵 Execution Log"):
        st.code(load_text(log_path), language="text")


# ----------------------
# Tab 3: Pipeline Actions
# ----------------------
with tabs[2]:
    st.header("Manual Pipeline Execution")

    st.selectbox("Select Task Config:", list_config_files(), key="pipeline_action_selected_config")
    st.text_input("Query Text:", placeholder="Enter a natural language question...", key="pipeline_query_input")

    st.markdown("**Choose Pipeline Steps:**")
    step_cols = st.columns(3)
    step_cols[0].checkbox("Extract & Chunk", key="run_extract_and_chunk")
    step_cols[1].checkbox("Embed Chunks", key="run_embed_chunks")
    step_cols[2].checkbox("Retrieve Chunks", key="run_retrieve_chunks")
    step_cols[0].checkbox("Generate Answer", key="run_generate_answer")

    if st.button("🚀 Run Selected Steps", key="run_pipeline_button"):
        st.session_state.run_steps_execution_count += 1
        st.warning(f"Debug: 'Run Selected Steps' button logic entered. Execution count: {st.session_state.run_steps_execution_count}")
        print(f"DEBUG_UI: 'Run Selected Steps' button logic entered. Execution count: {st.session_state.run_steps_execution_count}")
        
        st.session_state.pipeline_results = []  # Clear previous results
        selected_config_name = st.session_state.get("pipeline_action_selected_config")
        query_text = st.session_state.get("pipeline_query_input", "")
        run_extract = st.session_state.get("run_extract_and_chunk", False)
        run_embed = st.session_state.get("run_embed_chunks", False)
        run_retrieve = st.session_state.get("run_retrieve_chunks", False)
        run_generate = st.session_state.get("run_generate_answer", False)

        output_messages = []

        if not selected_config_name:
            st.error("Please select a task configuration.")
            st.session_state.pipeline_results = ["Error: Please select a task configuration."]
            st.session_state.show_pipeline_output_area = True
        elif (run_retrieve or run_generate) and not query_text:
            st.error("Query text is required for 'Retrieve Chunks' or 'Generate Answer'.")
            st.session_state.pipeline_results = ["Error: Query text is required for 'Retrieve Chunks' or 'Generate Answer'."]
            st.session_state.show_pipeline_output_area = True
        else:
            # Robust path construction for Tab 3
            path_parts = selected_config_name.replace("\\", "/").split("/")
            pure_filename = path_parts[-1]
            config_path = os.path.join("configs", "tasks", pure_filename)
            
            with st.spinner("Running pipeline steps..."):
                try:
                    pipeline = RAGPipeline(config_path=config_path)

                    # Crucial for Independent Step Runs
                    if (run_embed or run_retrieve or run_generate) and not run_extract:
                        if not hasattr(pipeline, 'chunked_file') or pipeline.chunked_file is None:
                            if pipeline.config and "paths" in pipeline.config and "chunked_emails" in pipeline.config["paths"]:
                                pipeline.chunked_file = pipeline.config["paths"]["chunked_emails"]
                                if not os.path.exists(pipeline.chunked_file):
                                    output_messages.append(f"ℹ️ Note: Chunk file specified in config ({pipeline.chunked_file}) not found. 'Extract & Chunk' may be needed.")
                            else:
                                output_messages.append("⚠️ Warning: Chunk file path not found in config. 'Extract & Chunk' might be required.")
                        
                        # Ensure index_path and metadata_path are set for retrieve/generate if not embedding now
                        if not hasattr(pipeline, 'index_path') or not pipeline.index_path:
                             pipeline.index_path = os.path.join(pipeline.config.get("embedding", {}).get("output_dir", ""), 
                                                                pipeline.config.get("embedding", {}).get("index_filename", ""))
                        if not hasattr(pipeline, 'metadata_path') or not pipeline.metadata_path:
                             pipeline.metadata_path = os.path.join(pipeline.config.get("embedding", {}).get("output_dir", ""), 
                                                                   pipeline.config.get("embedding", {}).get("metadata_filename", ""))

                    # Execute Selected Steps
                    if run_extract:
                        chunk_file_path = pipeline.extract_and_chunk()
                        output_messages.append(f"✅ Extract & Chunk completed. Chunked file: {chunk_file_path}")

                    if run_embed:
                        current_chunk_file = getattr(pipeline, 'chunked_file', None)
                        if not current_chunk_file and pipeline.config and "paths" in pipeline.config and "chunked_emails" in pipeline.config["paths"]:
                            current_chunk_file = pipeline.config["paths"]["chunked_emails"]
                            pipeline.chunked_file = current_chunk_file
                        
                        if not current_chunk_file or not os.path.exists(current_chunk_file):
                            output_messages.append(f"❌ Error: Chunk file ({current_chunk_file or 'Not specified'}) not found. 'Extract & Chunk' must be run first or path in config must be valid.")
                        else:
                            pipeline.embed_chunks()
                            output_messages.append("✅ Embed Chunks completed.")

                    retrieved_context_for_generation = None
                    if run_retrieve:
                        if not query_text: # Should be caught by earlier validation, but as a safeguard
                             output_messages.append("❌ Error: Query text is required for 'Retrieve Chunks'.")
                        else:
                            retrieved_data = pipeline.retrieve(query=query_text)
                            retrieved_context_for_generation = retrieved_data
                            output_messages.append(f"✅ Retrieve Chunks completed. Found {len(retrieved_data.get('top_chunks', []))} chunks.")
                            for chunk_info in retrieved_data.get('top_chunks', [])[:3]:
                                output_messages.append(f"  - Chunk (Score: {chunk_info.get('score', 0.0):.2f}): {str(chunk_info.get('text', ''))[:100]}...")
                    
                    if run_generate:
                        if not query_text: # Should be caught by earlier validation
                            output_messages.append("❌ Error: Query text is required for 'Generate Answer'.")
                        else:
                            if retrieved_context_for_generation is None and not run_retrieve: # If retrieve wasn't just run
                                output_messages.append("ℹ️ 'Retrieve Chunks' was not selected in this run, running it now for 'Generate Answer'...")
                                try:
                                    retrieved_context_for_generation = pipeline.retrieve(query=query_text) # Assumes index exists
                                    output_messages.append(f"✅ Retrieve Chunks completed for generation. Found {len(retrieved_context_for_generation.get('top_chunks', []))} chunks.")
                                except Exception as e:
                                    output_messages.append(f"❌ Error during implicit retrieval for generation: {str(e)}")
                                    retrieved_context_for_generation = {"top_chunks": []} # Ensure it's a dict with empty chunks

                            if retrieved_context_for_generation and retrieved_context_for_generation.get('top_chunks'):
                                answer = pipeline.generate_answer(query=query_text, chunks=retrieved_context_for_generation)
                                output_messages.append(f"💬 Generated Answer: {answer}")
                            else:
                                output_messages.append("⚠️ Cannot generate answer: No chunks were retrieved or provided. Ensure 'Retrieve Chunks' runs successfully or embeddings exist.")
                
                except Exception as e:
                    output_messages.append(f"❌ Pipeline error: {str(e)}")
            
            st.session_state.pipeline_results = output_messages
            st.session_state.show_pipeline_output_area = True

    output_area_expanded = st.session_state.get("show_pipeline_output_area", False)
    with st.expander("🧠 Output Area", expanded=output_area_expanded):
        if st.session_state.get("pipeline_results"):
            for msg in st.session_state.pipeline_results:
                st.write(msg)
        else:
            st.write("(Results will be shown here after execution)")

# ----------------------
# Tab 4: Utilities & Tools
# ----------------------
with tabs[3]:
    st.header("Debugging, File Tools & Utilities")

    st.markdown("**Embedding Stats:**")
    # Ensure list_config_files() is defined globally in the script
    available_configs_for_stats = list_config_files() 
    
    if not available_configs_for_stats:
        st.info("No task configurations found to display stats.")
    else:
        selected_stat_config_name = st.selectbox(
            "Select Task Config for Stats:", 
            available_configs_for_stats, 
            key="stats_selected_config", # Unique key for this selectbox
            help="Select a task to see its embedding statistics."
        )

        if st.button("Load Embedding Stats", key="load_stats_button"):
            if not selected_stat_config_name:
                st.info("Please select a task configuration first.")
            else:
                # Robust path construction for Tab 4 - Embedding Stats
                path_parts_stats = selected_stat_config_name.replace("\\", "/").split("/")
                pure_filename_stats = path_parts_stats[-1]
                config_file_path = os.path.join("configs", "tasks", pure_filename_stats)
                
                # Helper function to load YAML (can be defined inline or globally)
                def load_yaml_for_stats(p):
                    try:
                        with open(p, 'r') as f_yaml:
                            return yaml.safe_load(f_yaml)
                    except Exception as e_yaml:
                        st.error(f"Error loading YAML config {p}: {str(e_yaml)}")
                        return None
                
                task_config_data = load_yaml_for_stats(config_file_path)

                if task_config_data:
                    embedding_config = task_config_data.get("embedding", {})
                    output_dir = embedding_config.get("output_dir")
                    # Use defaults if not specified, consistent with GeneralPurposeEmbedder/RAGPipeline
                    index_filename = embedding_config.get("index_filename", "chunks.index") 
                    metadata_filename = embedding_config.get("metadata_filename", "chunks_metadata.tsv")

                    if not output_dir:
                        st.error(f"Embedding output directory ('embedding.output_dir') not specified in {selected_stat_config_name}.")
                    else:
                        index_path = os.path.join(output_dir, index_filename)
                        metadata_path = os.path.join(output_dir, metadata_filename)

                        # Display Index Size
                        if os.path.exists(index_path):
                            try:
                                index = faiss.read_index(index_path)
                                st.write(f"Index Size ({index_filename}): {index.ntotal} vectors")
                            except Exception as e_faiss:
                                st.error(f"Error loading FAISS index {index_path}: {str(e_faiss)}")
                        else:
                            st.warning(f"FAISS index file not found: {index_path}")

                        # Display Total Chunks from metadata
                        if os.path.exists(metadata_path):
                            try:
                                meta_df = pd.read_csv(metadata_path, sep="\t", on_bad_lines='skip') # Corrected sep to "\t"
                                st.write(f"Total Chunks ({metadata_filename}): {len(meta_df)} entries")
                                
                                # Attempt to find a suitable column for unique email count
                                potential_email_id_cols = ['Email ID', 'Message-ID', 'Message ID', 'Original Email ID', 'email_id', 'Source Email', 'email_subject', 'ID'] # Added 'ID' as a common generic identifier
                                email_id_column_found = None
                                for col_option in potential_email_id_cols:
                                    if col_option in meta_df.columns:
                                        email_id_column_found = col_option
                                        break
                                
                                if email_id_column_found:
                                    unique_emails = meta_df[email_id_column_found].nunique()
                                    st.write(f"Unique Sources (estimated from '{email_id_column_found}' in metadata): {unique_emails}") # Renamed to "Unique Sources" for generality
                                else:
                                    st.write("Unique Sources/Emails count: Could not determine (no common identifier column found in metadata).")
                            except Exception as e_pd:
                                st.error(f"Error loading or processing metadata file {metadata_path}: {str(e_pd)}")
                        else:
                            st.warning(f"Metadata file not found: {metadata_path}")

    st.markdown("**Similarity Test:**")
    st.text_input(
        "Run a test query:", 
        placeholder="e.g. What is the current state of feature XYZ?", 
        key="similarity_test_query_input"
    )

    if st.button("🔍 Retrieve Similar Chunks", key="similarity_test_button"):
        st.session_state.similarity_results = [] # Clear previous results
        query_text = st.session_state.get("similarity_test_query_input", "")
        # Reuse selected_config_name from "Embedding Stats" section
        selected_config_name = st.session_state.get("stats_selected_config") 
        
        similarity_results_messages = [] # This will hold messages for the current run

        if not selected_config_name:
            st.error("Please select a task configuration under 'Embedding Stats' section first.")
            # No return here, let the error message be the primary feedback. Results will be empty.
        elif not query_text:
            st.error("Please enter a query for the similarity test.")
        else:
            # Robust path construction for Tab 4 - Similarity Test
            path_parts_similarity = selected_config_name.replace("\\", "/").split("/")
            pure_filename_similarity = path_parts_similarity[-1]
            config_path = os.path.join("configs", "tasks", pure_filename_similarity)
            
            with st.spinner("Retrieving similar chunks..."):
                try:
                    pipeline = RAGPipeline(config_path=config_path)
                    
                    index_p = getattr(pipeline, "index_path", None)
                    metadata_p = getattr(pipeline, "metadata_path", None)

                    if not index_p or not os.path.exists(index_p):
                        similarity_results_messages.append(f"❌ Error: Index file ({index_p or 'Not configured'}) not found for task '{selected_config_name}'. Embeddings may need to be generated via Tab 3.")
                    elif not metadata_p or not os.path.exists(metadata_p):
                        similarity_results_messages.append(f"❌ Error: Metadata file ({metadata_p or 'Not configured'}) not found for task '{selected_config_name}'. Embeddings may need to be generated via Tab 3.")
                    else:
                        retrieved_data = pipeline.retrieve(query=query_text)
                        if retrieved_data and retrieved_data.get('top_chunks'):
                            similarity_results_messages.append(f"**Found {len(retrieved_data['top_chunks'])} similar chunks for '{query_text}':**")
                            for i, chunk_info in enumerate(retrieved_data['top_chunks']):
                                similarity_results_messages.append(f"**Chunk {i+1} (Score: {chunk_info.get('score', 0.0):.4f})**")
                                similarity_results_messages.append(f"Text: _{chunk_info.get('text', 'N/A')}_")
                                
                                metadata_display = chunk_info.get('metadata', {})
                                source_info = ""
                                if 'Sender' in metadata_display: source_info += f"Sender: {metadata_display['Sender']} | "
                                if 'Subject' in metadata_display: source_info += f"Subject: {metadata_display['Subject']} | "
                                if 'Received' in metadata_display: source_info += f"Date: {metadata_display['Received']}"
                                # Remove trailing ' | ' if any
                                if source_info.endswith(" | "): source_info = source_info[:-3]

                                if source_info: 
                                    similarity_results_messages.append(f"Source: {source_info}")
                                similarity_results_messages.append("---") # Separator
                        else:
                            similarity_results_messages.append(f"No similar chunks found for '{query_text}'.")
                
                except Exception as e:
                    similarity_results_messages.append(f"❌ Error during similarity test: {str(e)}")
            
            st.session_state.similarity_results = similarity_results_messages

    if "similarity_results" in st.session_state and st.session_state.similarity_results:
        with st.expander("Similarity Test Results", expanded=True):
            for msg in st.session_state.similarity_results:
                st.markdown(msg)


    with st.expander("📁 File Tools"):
        st.info("Ensure a task is selected under 'Embedding Stats' above to associate file operations with.")

        # --- Upload Cleaned Email TSV ---
        uploaded_cleaned_email_file = st.file_uploader(
            "Upload Cleaned Email TSV (e.g., for a task)", 
            type=["tsv"], 
            key="upload_cleaned_email_tsv"
        )
        if uploaded_cleaned_email_file is not None:
            selected_config_name_for_upload = st.session_state.get("stats_selected_config")
            if not selected_config_name_for_upload:
                st.error("Please select a task configuration under 'Embedding Stats' to associate this upload with.")
            else:
                task_name = selected_config_name_for_upload.replace(".yaml", "")
                target_upload_dir = os.path.join("uploads", task_name, "input_data")
                os.makedirs(target_upload_dir, exist_ok=True)
                target_path = os.path.join(target_upload_dir, uploaded_cleaned_email_file.name)
                try:
                    with open(target_path, "wb") as f:
                        f.write(uploaded_cleaned_email_file.getbuffer())
                    st.success(f"Uploaded '{uploaded_cleaned_email_file.name}' to '{target_path}' for task '{task_name}'.")
                except Exception as e:
                    st.error(f"Failed to save uploaded file: {str(e)}")

        # --- Upload Chunk TSV ---
        uploaded_chunk_tsv_file = st.file_uploader(
            "Upload Chunk TSV (e.g., for direct embedding)", 
            type=["tsv"], 
            key="upload_chunk_tsv"
        )
        if uploaded_chunk_tsv_file is not None:
            selected_config_name_for_chunk_upload = st.session_state.get("stats_selected_config")
            if not selected_config_name_for_chunk_upload:
                st.error("Please select a task configuration under 'Embedding Stats' to associate this upload with.")
            else:
                task_name = selected_config_name_for_chunk_upload.replace(".yaml", "")
                target_upload_dir = os.path.join("uploads", task_name, "custom_chunks")
                os.makedirs(target_upload_dir, exist_ok=True)
                target_path = os.path.join(target_upload_dir, uploaded_chunk_tsv_file.name)
                try:
                    with open(target_path, "wb") as f:
                        f.write(uploaded_chunk_tsv_file.getbuffer())
                    st.success(f"Uploaded '{uploaded_chunk_tsv_file.name}' to '{target_path}' for task '{task_name}'. This could be used as input for an embedding process.")
                except Exception as e:
                    st.error(f"Failed to save uploaded file: {str(e)}")
        
        st.divider() # Visual separator

        # --- Download Metadata ---
        if st.button("Prepare Metadata for Download", key="prepare_metadata_download_button"):
            selected_config_name_for_download = st.session_state.get("stats_selected_config")
            if not selected_config_name_for_download:
                st.error("Please select a task configuration under 'Embedding Stats' to download its metadata.")
                st.session_state.metadata_to_download = None # Clear any previous data
            else:
                # Robust path construction for Tab 4 - File Tools (Download)
                path_parts_download = selected_config_name_for_download.replace("\\", "/").split("/")
                pure_filename_download = path_parts_download[-1]
                config_file_path = os.path.join("configs", "tasks", pure_filename_download)
                
                def load_yaml_for_file_tools(p): # Local helper
                    try:
                        with open(p, 'r') as f_yaml: return yaml.safe_load(f_yaml)
                    except Exception as e_yaml_load: 
                        st.error(f"Error loading YAML {p}: {str(e_yaml_load)}"); return None
                
                task_config_data = load_yaml_for_file_tools(config_file_path)

                if task_config_data:
                    embedding_config = task_config_data.get("embedding", {})
                    output_dir = embedding_config.get("output_dir")
                    metadata_filename = embedding_config.get("metadata_filename", "chunks_metadata.tsv")

                    if not output_dir:
                        st.error(f"Embedding output directory ('embedding.output_dir') not specified in {selected_config_name_for_download}.")
                        st.session_state.metadata_to_download = None
                    else:
                        actual_metadata_path = os.path.join(output_dir, metadata_filename)
                        if os.path.exists(actual_metadata_path):
                            try:
                                with open(actual_metadata_path, "rb") as fp: # Read as bytes
                                    data_to_download = fp.read()
                                st.session_state.metadata_to_download = data_to_download
                                st.session_state.metadata_download_filename = f"{selected_config_name_for_download.replace('.yaml', '')}_{metadata_filename}"
                                st.success(f"Metadata '{metadata_filename}' for task '{selected_config_name_for_download.replace('.yaml', '')}' is ready for download.")
                            except Exception as e_read_meta:
                                st.error(f"Error reading metadata file {actual_metadata_path}: {str(e_read_meta)}")
                                st.session_state.metadata_to_download = None
                        else:
                            st.warning(f"Metadata file not found: {actual_metadata_path}")
                            st.session_state.metadata_to_download = None
                else:
                     st.session_state.metadata_to_download = None # Ensure cleared if config load fails

        if st.session_state.get("metadata_to_download"):
            st.download_button(
                label="📥 Download Prepared Metadata",
                data=st.session_state.metadata_to_download,
                file_name=st.session_state.get("metadata_download_filename", "chunks_metadata.tsv"),
                mime="text/tab-separated-values",
                key="final_metadata_download_button"
            )
        else:
            st.caption("Click 'Prepare Metadata for Download' from a selected task to enable download.")

    st.markdown("---")
    st.markdown("### Chunking Reviewer")
    with st.expander("🔍 Review Email Chunks", expanded=False):
        selected_task_config_name_for_review = st.session_state.get("stats_selected_config")

        if selected_task_config_name_for_review:
            st.write(f"Reviewing data for task: **{selected_task_config_name_for_review}**")
        else:
            st.warning("Please select a Task Configuration in the 'Embedding Stats' section above to load data for chunk review.")
            # Using st.stop() might be too abrupt if other elements in Tab 4 should still render.
            # Consider simply not rendering the rest of this expander's UI if that's acceptable.
            # For now, we'll allow rendering to continue but subsequent widgets will check selected_task_config_name_for_review.
            
        if st.button("Load Chunk Data", key="load_chunk_review_data_button"):
            if selected_task_config_name_for_review:
                # Robust path construction for Tab 4 - Chunking Reviewer
                path_parts_chunk_review = selected_task_config_name_for_review.replace("\\", "/").split("/")
                pure_filename_chunk_review = path_parts_chunk_review[-1]
                raw_df = load_chunk_data_for_task(pure_filename_chunk_review) # Pass pure filename

                if raw_df is not None and not raw_df.empty:
                    st.session_state.chunk_review_data = raw_df
                    st.session_state.grouped_chunk_data = group_chunks_by_email(raw_df) # Uses "EntryID" by default
                    
                    if not st.session_state.grouped_chunk_data:
                        st.error("Failed to group chunk data by email ID. Ensure 'EntryID' column exists and is populated correctly in the chunk file.")
                        st.session_state.chunk_review_data = None # Clear to avoid partial state
                    else:
                        st.success(f"Loaded and grouped chunk data for {len(st.session_state.grouped_chunk_data)} unique email EntryIDs.")
                        # Reset selections for email and body if data is reloaded
                        st.session_state.selected_email_id_for_review = None
                        st.session_state.current_email_body_for_review = None
                        st.session_state.chunks_for_selected_email = None
                else:
                    st.error(f"No chunk data loaded or data is empty for task '{selected_task_config_name_for_review}'. Check the chunk file path in the task config and ensure the file is not empty.")
                    st.session_state.chunk_review_data = None
                    st.session_state.grouped_chunk_data = None
            else:
                st.error("Cannot load chunk data: No task configuration selected in 'Embedding Stats'.")

        if st.session_state.get("grouped_chunk_data"):
            email_ids = list(st.session_state.grouped_chunk_data.keys())
            if not email_ids:
                st.info("No email IDs found in the loaded chunk data.")
            else:
                current_selected_email_id = st.session_state.get("selected_email_id_for_review")
                
                # Determine index for selectbox, default to 0 if current selection not in new list or no selection
                select_email_index = 0
                if current_selected_email_id in email_ids:
                    select_email_index = email_ids.index(current_selected_email_id)
                
                # If there's no current selection, and email_ids is not empty,
                # set the first email_id as the default selection for the UI
                # but only if selected_email_id_for_review is currently None
                if not current_selected_email_id and email_ids and st.session_state.selected_email_id_for_review is None:
                     st.session_state.selected_email_id_for_review = email_ids[0] # Pre-select the first one
                     current_selected_email_id = email_ids[0] # Update for immediate use

                # The selectbox itself
                # The 'on_change' callback for selectbox is implicit; Streamlit reruns the script.
                # We will handle the logic for when the selection changes *after* the selectbox.
                newly_selected_email_id = st.selectbox(
                    "Select Email ID to Review:", 
                    email_ids, 
                    index=select_email_index, 
                    key="chunk_review_selected_email_id",
                    help="Select an Email's EntryID to see its original body and chunks."
                )

                # Logic to update details if selection has changed or if it's the first load with a default
                if newly_selected_email_id and (newly_selected_email_id != current_selected_email_id or st.session_state.current_email_body_for_review is None) :
                    st.session_state.selected_email_id_for_review = newly_selected_email_id
                    email_df = st.session_state.grouped_chunk_data[newly_selected_email_id]
                    st.session_state.current_email_body_for_review = get_original_email_body(email_df)
                    if "Chunk" in email_df.columns:
                        # Store list of dicts (each dict is a row/chunk)
                        st.session_state.chunks_for_selected_email = email_df.to_dict('records') 
                    else:
                        st.session_state.chunks_for_selected_email = []
                        st.warning("Missing 'Chunk' column in the data for this email.")
                    # When email selection changes, clear any previously selected chunk detail
                    st.session_state.selected_chunk_for_detail_review = None
                    st.session_state.selected_chunk_metadata_for_detail = None


        # Display Area for Original Email and Chunks
        if st.session_state.get("selected_email_id_for_review"):
            selected_email_id = st.session_state.selected_email_id_for_review # Convenience
            
            if st.session_state.get("current_email_body_for_review"):
                st.markdown("#### Original Email Body")
                st.text_area("Full Email Content", st.session_state.current_email_body_for_review, height=300, disabled=True, key=f"original_email_display_{selected_email_id}")
            else:
                st.warning(f"Original email body could not be retrieved for Email ID: {selected_email_id}.")

            # Display Chunks from this Email
            if st.session_state.get("chunks_for_selected_email"):
                st.markdown("#### Chunks from this Email")
                chunk_records = st.session_state.chunks_for_selected_email
                if chunk_records:
                    for i, chunk_record in enumerate(chunk_records):
                        chunk_text = chunk_record.get("Chunk", "N/A")
                        
                        col1, col2 = st.columns([0.85, 0.15]) # Adjusted column ratio
                        with col1:
                            with st.expander(f"Chunk {i+1}: {chunk_text[:50]}...", expanded=False):
                                st.markdown(chunk_text)
                        with col2:
                            # Use a unique key for each button, incorporating email_id and chunk index
                            button_key = f"detail_btn_{selected_email_id}_{i}"
                            if st.button("🔍 Details", key=button_key):
                                st.session_state.selected_chunk_for_detail_review = chunk_text
                                metadata_to_show = {k: v for k, v in chunk_record.items() if k not in ["EntryID", "Raw Body", "Cleaned Body", "Chunk"]}
                                st.session_state.selected_chunk_metadata_for_detail = metadata_to_show
                                st.experimental_rerun() # Rerun to show the details section
                else:
                     st.info("No chunk records available for this email (list of chunks is empty).")
            else:
                st.info("No chunks found or 'Chunk' column missing for this email in the data.")

            # Display Area for Selected Chunk Details
            if st.session_state.get("selected_chunk_for_detail_review"):
                st.markdown("---")
                st.markdown("#### Selected Chunk Details")
                selected_chunk_text = st.session_state.selected_chunk_for_detail_review
                selected_chunk_meta = st.session_state.get("selected_chunk_metadata_for_detail", {})
                
                st.markdown(f"**Text:**")
                st.markdown(f"> _{selected_chunk_text}_") # Display as blockquote
                st.markdown(f"**Length:** {len(selected_chunk_text)} characters")
                
                if selected_chunk_meta:
                    st.markdown("**Other Metadata:**")
                    st.json(selected_chunk_meta, expanded=True)
                
                if st.button("Clear Details", key="clear_chunk_detail_button"):
                    st.session_state.selected_chunk_for_detail_review = None
                    st.session_state.selected_chunk_metadata_for_detail = None
                    st.experimental_rerun()

            # --- Chunk Metrics for Selected Email ---
            st.markdown("---")
            st.markdown("#### Chunk Metrics for Selected Email")
            chunk_records_for_metrics = st.session_state.get("chunks_for_selected_email", [])
            num_chunks_for_metrics = len(chunk_records_for_metrics)

            st.write(f"Number of Chunks: {num_chunks_for_metrics}")

            if num_chunks_for_metrics > 0:
                chunk_lengths_for_metrics = [len(str(record.get("Chunk", ""))) for record in chunk_records_for_metrics]
                
                avg_chunk_length = np.mean(chunk_lengths_for_metrics) if chunk_lengths_for_metrics else 0
                st.write(f"Average Chunk Length: {avg_chunk_length:.2f} characters")
                
                min_chunk_length = np.min(chunk_lengths_for_metrics) if chunk_lengths_for_metrics else 0
                max_chunk_length = np.max(chunk_lengths_for_metrics) if chunk_lengths_for_metrics else 0
                st.write(f"Min/Max Chunk Length: {min_chunk_length} / {max_chunk_length} characters")

                if chunk_lengths_for_metrics:
                    st.write("**Chunk Length Distribution (Binned):**")
                    # Determine number of bins
                    if num_chunks_for_metrics > 1:
                        num_bins = int(np.ceil(np.log2(num_chunks_for_metrics) + 1)) 
                    else: # Handle case with 1 chunk
                        num_bins = 1
                    
                    if num_bins <= 0: num_bins = 5 # Fallback for very few unique values or edge cases
                    if max_chunk_length == min_chunk_length and num_bins > 1: # All chunks same length
                        num_bins = 1

                    hist_values, bin_edges = np.histogram(chunk_lengths_for_metrics, bins=num_bins)
                    
                    bin_labels = []
                    if num_bins == 1 and max_chunk_length == min_chunk_length:
                         bin_labels = [f"{int(min_chunk_length)}"]
                    else:
                        for i in range(len(bin_edges)-1):
                            # Make bin labels more readable, especially for integer lengths
                            edge_start = int(np.floor(bin_edges[i]))
                            edge_end = int(np.ceil(bin_edges[i+1]))
                            if i == len(bin_edges) - 2: # Last bin, ensure it includes the max value
                                edge_end = int(np.ceil(max_chunk_length))
                            
                            if edge_start == edge_end or (edge_end - edge_start == 1 and edge_start == bin_edges[i] and edge_end == bin_edges[i+1]) : # if bin is for a single integer value
                                bin_labels.append(f"{edge_start}")
                            else:
                                bin_labels.append(f"{edge_start}-{edge_end-1 if edge_end > edge_start else edge_end}")


                    if len(hist_values) == len(bin_labels) and sum(hist_values) > 0:
                        # Ensure bin_labels are unique if possible, or aggregate if not (though np.histogram should handle this)
                        # For st.bar_chart, index must be unique. If binning results in non-unique labels (e.g. "100-100"), aggregate.
                        # However, the above logic should try to make them distinct for ranges.
                        # If all values are the same, hist_values will have 1 element.
                        hist_df_data = {'count': hist_values}
                        if len(bin_labels) == len(hist_values):
                             hist_df = pd.DataFrame(hist_df_data, index=bin_labels)
                             hist_df.index.name = 'length_range'
                             st.bar_chart(hist_df)
                        elif len(hist_values) == 1: # Single bin case, e.g. all chunks same length
                             st.bar_chart(pd.DataFrame({'count': hist_values}, index=[f"{min_chunk_length}"]))
                        else:
                             st.caption("Could not prepare data for histogram due to binning issues.")

                    elif sum(hist_values) > 0 and len(hist_values) ==1 : 
                        st.bar_chart(pd.DataFrame({'count': hist_values}, index=[f"{min_chunk_length}-{max_chunk_length}" if min_chunk_length!=max_chunk_length else str(min_chunk_length)]))
                    else:
                        st.caption("Not enough data or variability for a binned length distribution chart.")
                else:
                    st.caption("No chunk lengths to display distribution.")
            else:
                st.write("No chunks available for this email to calculate metrics.")

        elif st.session_state.get("grouped_chunk_data"):
            st.info("Select an Email ID from the dropdown above to see its details.")
