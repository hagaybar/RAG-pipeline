import streamlit as st
import glob
import yaml
import os
import sys # For stdout manipulation, and now for sys.path

# Calculate the project root based on the current file's location
# Assumes ui_v3.py is in scripts/ui/
# os.path.dirname(__file__) gives scripts/ui
# os.path.join(..., "../../") goes up two levels to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Add PROJECT_ROOT to sys.path if it's not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from io import StringIO # For capturing stdout
from scripts.pipeline.rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG Pipeline UI", layout="wide")
st.title("📬 RAG Pipeline Control Panel")

# ----------------------
# Helper Functions
# ----------------------
def list_config_files(config_dir: str = "configs/tasks") -> list:
    """
    Scan the task_configs directory for .yaml files.

    Args:
        config_dir (str): Path to the config folder.

    Returns:
        List[str]: Sorted list of config file names.
    """
    return sorted([os.path.basename(f) for f in glob.glob(f"{config_dir}/*.yaml")])

# ----------------------
# Session State Initialization for Pipeline Actions
# ----------------------
if 'extract_and_chunk_cb' not in st.session_state:
    st.session_state.extract_and_chunk_cb = False
if 'embed_chunks_cb' not in st.session_state:
    st.session_state.embed_chunks_cb = False
# 'embed_chunks_batch_cb' session state removed
if 'retrieve_cb' not in st.session_state:
    st.session_state.retrieve_cb = False
if 'generate_answer_cb' not in st.session_state:
    st.session_state.generate_answer_cb = False
# embedding_choice_made_by_retrieve is removed as per subtask instructions
if 'pipeline_output' not in st.session_state: # For pipeline execution output
    st.session_state.pipeline_output = "(Results will be shown here after execution)"

# ----------------------
# Callback for Pipeline Step Checkboxes
# ----------------------
def handle_pipeline_step_change():
    # a. If generate_answer_cb is checked, ensure retrieve_cb is checked.
    if st.session_state.generate_answer_cb and not st.session_state.retrieve_cb:
        st.session_state.retrieve_cb = True

    # b. If retrieve_cb is *unchecked*, ensure generate_answer_cb is unchecked.
    if not st.session_state.retrieve_cb and st.session_state.generate_answer_cb:
        st.session_state.generate_answer_cb = False

    # c. If retrieve_cb is checked:
    if st.session_state.retrieve_cb:
        # The auto-checking of 'embed_chunks_cb' when 'retrieve_cb' is checked and no embedding option is selected,
        # and all related logic for 'embedding_choice_made_by_retrieve' and 'condition_c_i_met_this_run'
        # has been removed as per subtask instructions.

        # d. & e. Mutual exclusivity logic simplified as 'embed_chunks_batch_cb' is removed.
        # No specific logic needed here now for 'embed_chunks_cb' beyond its own state,
        # as there's no other embedding option to make it mutually exclusive with.
        # The 'retrieve_cb' might still imply 'embed_chunks_cb' should be checked if no other step handles embedding,
        # but that auto-checking logic was removed in the previous step.
        pass # No specific action needed for embed_chunks_cb based on other checkboxes in this simplified model.
    # No 'else' block needed here.

# ----------------------
# Callback for Running Manual Pipeline
# ----------------------
def handle_run_pipeline():
    st.session_state.pipeline_output = "" # Clear previous output
    output_messages = []

    # a. Retrieve selected task config
    selected_config_filename = st.session_state.get("manual_task_config_selector")
    if not selected_config_filename:
        output_messages.append("❌ Error: No task config selected.")
        st.session_state.pipeline_output = "\n".join(output_messages)
        return

    config_path = os.path.join("configs", "tasks", selected_config_filename)
    if not os.path.exists(config_path):
        output_messages.append(f"❌ Error: Config file not found at {config_path}")
        st.session_state.pipeline_output = "\n".join(output_messages)
        return
    
    output_messages.append(f"⚙️ Using task config: {config_path}")

    # b. Retrieve query text
    query_text = st.session_state.get("manual_query_text_input", "")

    # c. Retrieve selected pipeline steps
    steps_to_run = []
    if st.session_state.get('extract_and_chunk_cb', False):
        steps_to_run.append("extract_and_chunk")
    if st.session_state.get('embed_chunks_cb', False):
        steps_to_run.append("embed_chunks")
    # 'embed_chunks_batch_cb' removed from steps_to_run
    if st.session_state.get('retrieve_cb', False):
        steps_to_run.append("retrieve")
    if st.session_state.get('generate_answer_cb', False):
        steps_to_run.append("generate_answer")

    if not steps_to_run:
        output_messages.append("⚠️ No pipeline steps selected.")
        st.session_state.pipeline_output = "\n".join(output_messages)
        return

    output_messages.append(f"▶️ Selected steps: {', '.join(steps_to_run)}")

    # d. Perform validation for query
    requires_query = any(step in steps_to_run for step in ["retrieve", "generate_answer"])
    if requires_query and not query_text.strip():
        output_messages.append("❌ Error: Query text is required for 'retrieve' or 'generate_answer' steps.")
        st.session_state.pipeline_output = "\n".join(output_messages)
        return
    
    # e. Instantiate RAGPipeline
    try:
        pipeline = RAGPipeline(config_path=config_path)
        output_messages.append("✅ RAGPipeline instantiated.")
    except Exception as e:
        output_messages.append(f"❌ Error instantiating RAGPipeline: {e}")
        st.session_state.pipeline_output = "\n".join(output_messages)
        return

    # f. Add steps to the pipeline instance
    try:
        if requires_query:
            pipeline.get_user_query(query_text) 
            output_messages.append(f"🗣️ Query set in pipeline: '{query_text}'")
            # Formally add "get_user_query" as a step to satisfy dependencies
            try:
                pipeline.add_step("get_user_query", force=True) # force=True as get_user_query has no deps
                output_messages.append("➕ Step 'get_user_query' formally added to pipeline.")
            except Exception as e:
                output_messages.append(f"❌ Error formally adding 'get_user_query' step: {e}")
                # Update pipeline_output before returning, so user sees the message
                st.session_state.pipeline_output = "\n".join(output_messages)
                return # Stop if we can't add this conceptual step

        for step_name in steps_to_run:
            if step_name == "retrieve":
                if 'embed_chunks' not in steps_to_run:
                    pipeline.add_step(step_name, force=True)
                    output_messages.append(f"➕ Step '{step_name}' added to pipeline (force=True, assuming embeddings exist).")
                else:
                    pipeline.add_step(step_name, force=False)
                    output_messages.append(f"➕ Step '{step_name}' added to pipeline.")
            elif step_name == "generate_answer": # Depends on retrieve, should be fine without force if retrieve is added
                pipeline.add_step(step_name, force=False)
                output_messages.append(f"➕ Step '{step_name}' added to pipeline.")
            else: # For 'extract_and_chunk', 'embed_chunks'
                pipeline.add_step(step_name, force=False)
                output_messages.append(f"➕ Step '{step_name}' added to pipeline.")
    
    except Exception as e:
        output_messages.append(f"❌ Error adding steps to RAGPipeline: {e}")
        st.session_state.pipeline_output = "\n".join(output_messages)
        return

    # g. Execute pipeline
    output_messages.append("🚀 Running pipeline steps...")
    st.session_state.pipeline_output = "\n".join(output_messages) # Update UI before long run

    old_stdout = sys.stdout
    redirected_output = StringIO()
    sys.stdout = redirected_output
    try:
        pipeline.run_steps() 
        
        pipeline_stdout = redirected_output.getvalue()
        output_messages.append("✅ Pipeline execution complete.")
        if pipeline_stdout:
            output_messages.append("\n--- Pipeline Output ---")
            output_messages.append(pipeline_stdout)
        
        if steps_to_run[-1] == "generate_answer" and hasattr(pipeline, 'last_answer') and pipeline.last_answer:
            output_messages.append("\n--- Final Answer ---")
            output_messages.append(str(pipeline.last_answer))

    except Exception as e:
        output_messages.append(f"❌ Error during pipeline execution: {e}")
        pipeline_stdout = redirected_output.getvalue()
        if pipeline_stdout:
            output_messages.append("\n--- Pipeline Output (before error) ---")
            output_messages.append(pipeline_stdout)
    finally:
        sys.stdout = old_stdout # Restore stdout
    
    st.session_state.pipeline_output = "\n".join(output_messages)

# Top-level tab navigation
tabs = st.tabs(["Tasks 🛠", "Runs & Logs 📊", "Pipeline Actions ⚙️", "Utilities & Tools 🧰"])

# ----------------------
# Tab 1: Task Management
# ----------------------
with tabs[0]:
    st.header("Task Management")

    # 🧩 Feature 1: Load and display available task configs from disk

    with st.expander("🔽 Select Task Config", expanded=True):
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

        # List all available configs
        config_list = list_config_files()
        selected_config = st.selectbox("Available Configs:", config_list)

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
            if st.button("Delete"):
                try:
                    os.remove(config_path)
                    st.success(f"🗑️ Deleted {selected_config}")
                except Exception as e:
                    st.error(f"❌ Deletion failed: {e}")


    with st.expander("➕ Create New Task"):
        st.text_input("New Task Name")
        st.button("Create from Template")

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
# Tab 2: Runs & Logs
# ----------------------
with tabs[1]:
    st.header("Run and Log Inspection")

    st.selectbox("Select Task:", ["email_test", "api_test"])
    st.selectbox("Run ID:", ["20240514_153055", "20240513_184200"])

    with st.expander("📄 Run Metadata", expanded=False):
        st.json({"run_id": "20240514_153055", "embedding_model": "text-embedding-3-small"})

    with st.expander("💬 Answer Output"):
        st.code("The NERS voting enhancement is currently not available in production...", language="text")

    with st.expander("📜 Query Debug Log"):
        st.code("[1] Subject: Re: NERS update\n[2] Subject: New features...", language="text")

    with st.expander("🪵 Execution Log"):
        st.code("2024-05-14 15:30:55 | INFO | Embedding complete.", language="text")

# ----------------------
# Tab 3: Pipeline Actions
# ----------------------
with tabs[2]:
    st.header("Manual Pipeline Execution")

    config_list = list_config_files()
    selected_manual_config = st.selectbox("Select Task Config:", config_list, key="manual_task_config_selector")
    st.text_input("Query Text:", placeholder="Enter a natural language question...", key="manual_query_text_input")

    st.markdown("**Choose Pipeline Steps:**")
    # Ensure states are updated based on callback before rendering checkboxes to set disabled states correctly
    # Calling it here might be redundant if session_state is already consistent due to prior actions or initialization.
    # handle_pipeline_step_change() # Let's see if this is needed or causes issues. Usually on_change is enough.

    col1, col2 = st.columns(2)

    with col1:
        st.checkbox("Extract & Chunk", key="extract_and_chunk_cb", on_change=handle_pipeline_step_change)
       
        # 'Embed Chunks Batch' checkbox and its disabling logic removed.
        # 'Embed Chunks' checkbox no longer needs a complex 'disabled' attribute based on batch mode.
        st.checkbox("Embed Chunks", key="embed_chunks_cb", on_change=handle_pipeline_step_change) # disabled attribute removed

    with col2:
        # Retrieve cannot be unchecked if Generate Answer is checked
        disable_retrieve = st.session_state.get('generate_answer_cb', False)
        st.checkbox("Retrieve Chunks", key="retrieve_cb", on_change=handle_pipeline_step_change, disabled=disable_retrieve)
       
        # Generate Answer cannot be checked if Retrieve is not checked
        disable_generate = not st.session_state.get('retrieve_cb', False) 
        st.checkbox("Generate Answer", key="generate_answer_cb", on_change=handle_pipeline_step_change, disabled=disable_generate)

    # Re-run callback to ensure state consistency after all checkbox states might have been individually altered by user.
    # This helps ensure that complex dependencies are correctly enforced after initial rendering and any user interaction.
    # However, this can also lead to infinite loops if not careful.
    # A better approach is to ensure the callback handles all logic comprehensively.
    # For now, relying on the on_change of each checkbox.

    st.button("🚀 Run Selected Steps", on_click=handle_run_pipeline)

    with st.expander("🧠 Output Area", expanded=True): # Keep it expanded
        st.text_area("Log", value=st.session_state.get("pipeline_output", "(Results will be shown here after execution)"), height=300, disabled=True)

# ----------------------
# Tab 4: Utilities & Tools
# ----------------------
with tabs[3]:
    st.header("Debugging, File Tools & Utilities")

    st.markdown("**Embedding Stats:**")
    st.write("Index size: 238 vectors")
    st.write("Total chunks: 212, Emails: 45")

    st.markdown("**Similarity Test:**")
    st.text_input("Run a test query:", placeholder="e.g. What is the current state of feature XYZ?")
    st.button("🔍 Retrieve Similar Chunks")

    with st.expander("📁 File Tools"):
        st.file_uploader("Upload Cleaned Email TSV")
        st.file_uploader("Upload Chunk TSV")
        st.download_button("Download Metadata", data="metadata...", file_name="chunks_metadata.tsv")
