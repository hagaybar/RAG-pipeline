import streamlit as st
import glob
import yaml
import os
import sys # For stdout manipulation, and now for sys.path
import json # Added for reading JSON files

# Calculate the project root based on the current file's location
# Assumes ui_v3.py is in scripts/ui/
# os.path.dirname(__file__) gives scripts/ui
# os.path.join(..., "../../") goes up two levels to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Add PROJECT_ROOT to sys.path if it's not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from io import StringIO # For capturing stdout
from pathlib import Path # Added import
from scripts.pipeline.rag_pipeline import RAGPipeline
from scripts.utils.config_templates import get_default_config, MODEL_MODE_COMPATIBILITY, MODEL_DIMENSIONS # Added import

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
            # Remove direct call to pipeline.get_user_query(query_text)
            # Remove associated message: output_messages.append(f"🗣️ Query set in pipeline: '{query_text}'")
            # Formally add "get_user_query" as a step, passing query_text as an argument
            try:
                pipeline.add_step("get_user_query", force=True, query=query_text) 
                output_messages.append(f"➕ Step 'get_user_query' (with query: '{query_text}') formally added to pipeline.")
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
    st.session_state.pipeline_output = "\n".join(output_messages) # Update UI before starting the generator

    try:
        # pipeline.run_steps() is now a generator
        for message in pipeline.run_steps(): # Iterate through yielded messages
            output_messages.append(message)
            # Update the UI progressively
            st.session_state.pipeline_output = "\n".join(output_messages) 
        
        # pipeline.run_steps() generator is exhausted here.
        # If generate_answer was run, its result is in pipeline.last_answer.
        # Messages for it should have been yielded by generate_answer and run_steps.
        # A general "Pipeline execution complete" message should also be yielded by run_steps.
        # If RAGPipeline's run_steps doesn't yield a final "complete" message, add one here:
        # if not any("Pipeline execution complete" in msg for msg in output_messages):
        #    if not any("Error during pipeline execution" in msg for msg in output_messages):
        #        output_messages.append("✅ Pipeline execution complete.")
        
        # Check if 'generate_answer' was supposed to be run and if 'last_answer' is available
        if steps_to_run and steps_to_run[-1] == "generate_answer" and hasattr(pipeline, 'last_answer') and pipeline.last_answer:
            output_messages.append("\n--- Final Answer ---")
            output_messages.append(str(pipeline.last_answer)) # Ensure it's a string

    except Exception as e:
        # Ensure any error messages are also added to the output
        error_message = f"❌ Error during pipeline execution: {e}"
        output_messages.append(error_message)
        # Also log to console for more detailed debugging if needed (Streamlit typically logs errors to console too)
        print(f"Pipeline execution error in Streamlit UI: {e}") 
    # No finally block needed for stdout restoration
    
    # Final update to ensure everything is displayed
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
        cols = st.columns(4) # Keep 4 columns for now, can adjust later if needed
        # New "Load Selected Task to Form" button
        if cols[0].button("📝 Load Selected Task to Form", disabled=not selected_config):
            if selected_config:
                st.session_state.show_task_form = True
                st.session_state.form_mode = 'edit'
                st.session_state.form_load_config_name = selected_config # Store name to load in the form
                st.success(f"Loading '{selected_config}' into form for editing.")
            else:
                st.warning("No config selected to load.")
        
        # Placeholder for the second column, was "Edit"
        # cols[1] can be used for a new button in future or layout adjusted
        
        # Duplicate
        with cols[2]: # Duplicate remains in the 3rd column (index 2)
            if st.button("Duplicate", disabled=not selected_config):
                if selected_config: # Ensure a config is selected
                    st.session_state.prompt_for_duplicate_name = True
                    # Store the name of the config to be duplicated
                    st.session_state.config_to_duplicate_name = selected_config 
                else:
                    st.warning("Please select a configuration to duplicate.")

            if st.session_state.get("prompt_for_duplicate_name", False):
                st.markdown("---") # Visual separator
                new_task_name_for_duplicate = st.text_input(
                    "Enter new task name for the duplicate:", 
                    key="new_task_name_for_duplicate_input",
                    help="This name will be used for the new duplicated task configuration."
                )
                
                col_confirm, col_cancel_dup = st.columns(2)
                with col_confirm:
                    if st.button("✅ Confirm New Name & Load Form", key="confirm_duplicate_name_button"):
                        new_name_val = st.session_state.new_task_name_for_duplicate_input.strip()
                        original_config_to_duplicate = st.session_state.get("config_to_duplicate_name")

                        if not new_name_val:
                            st.error("New task name cannot be empty.")
                        elif original_config_to_duplicate == f"{new_name_val}.yaml":
                            st.error("New task name must be different from the original.")
                        elif os.path.exists(f"configs/tasks/{new_name_val}.yaml"):
                             st.error(f"A configuration named '{new_name_val}.yaml' already exists. Choose a different name.")
                        elif original_config_to_duplicate:
                            original_config_path = Path("configs/tasks") / original_config_to_duplicate
                            try:
                                with open(original_config_path, 'r', encoding="utf-8") as f:
                                    data_to_duplicate = yaml.safe_load(f)
                                
                                # Prepare data for the form
                                st.session_state.current_form_data = data_to_duplicate
                                st.session_state.current_form_data['task_name'] = new_name_val # Overwrite task name
                                
                                # Set form state
                                st.session_state.show_task_form = True
                                st.session_state.form_mode = 'new' # It's a new task being created from template

                                # Clear duplication prompt state
                                st.session_state.prompt_for_duplicate_name = False
                                if "config_to_duplicate_name" in st.session_state:
                                    del st.session_state.config_to_duplicate_name
                                if "new_task_name_for_duplicate_input" in st.session_state: # Clear input
                                     st.session_state.new_task_name_for_duplicate_input = ""

                                st.success(f"Loading '{original_config_to_duplicate}' as template for new task '{new_name_val}'. Please review and save.")
                                st.experimental_rerun() # Rerun to show the form, populated

                            except Exception as e:
                                st.error(f"Failed to load original configuration '{original_config_to_duplicate}': {e}")
                        else:
                            st.error("Original configuration to duplicate not specified. Please re-select and try again.")
                
                with col_cancel_dup:
                    if st.button("❌ Cancel Duplicate", key="cancel_duplicate_prompt_button"):
                        st.session_state.prompt_for_duplicate_name = False
                        if "config_to_duplicate_name" in st.session_state:
                            del st.session_state.config_to_duplicate_name
                        if "new_task_name_for_duplicate_input" in st.session_state:
                             st.session_state.new_task_name_for_duplicate_input = ""
                        st.experimental_rerun()
                st.markdown("---")


        # Delete
        with cols[3]: # Delete remains in the 4th column (index 3)
            if st.button("Delete"):
                try:
                    os.remove(config_path)
                    st.success(f"🗑️ Deleted {selected_config}")
                except Exception as e:
                    st.error(f"❌ Deletion failed: {e}")

    # New "Create New Task" button (replaces the old expander)
    if st.button("✨ Create New Task"):
        st.session_state.show_task_form = True
        st.session_state.form_mode = 'new'
        # Optionally, clear any previously loaded config data for the form
        if 'form_load_config_name' in st.session_state:
            del st.session_state.form_load_config_name 
        st.info("Proceed to the form to create a new task.")

    # The "Preview Task YAML" expander and its contents are removed.
    # The old "Create New Task" expander and its contents are removed.

    # ----------------------
    # Task Configuration Form (conditionally displayed)
    # ----------------------
    if st.session_state.get('show_task_form', False):
        st.subheader("📝 Task Configuration Form")

        # Initial data loading for the form
        if 'current_form_data' not in st.session_state: # Initialize if not present
            st.session_state.current_form_data = {}

        if st.session_state.form_mode == 'new':
            # For a new task, load default config. Use a placeholder task name.
            # The actual task_name will be set in the form.
            st.session_state.current_form_data = get_default_config("new_task_placeholder")
            # Ensure task_name in form data is empty for new tasks or uses a specific placeholder
            st.session_state.current_form_data['task_name'] = "" 
        elif st.session_state.form_mode == 'edit':
            if 'form_load_config_name' in st.session_state and st.session_state.form_load_config_name:
                try:
                    config_file_path = Path("configs/tasks") / st.session_state.form_load_config_name
                    with open(config_file_path, 'r') as f:
                        st.session_state.current_form_data = yaml.safe_load(f)
                except Exception as e:
                    st.error(f"Failed to load {st.session_state.form_load_config_name}: {e}")
                    # Fallback to default if load fails, or handle error more gracefully
                    st.session_state.current_form_data = get_default_config("error_task_placeholder")
            else:
                # Should not happen if logic is correct, but as a fallback
                st.warning("Edit mode selected but no task config specified. Loading defaults.")
                st.session_state.current_form_data = get_default_config("fallback_task_placeholder")
        
        form_data = st.session_state.current_form_data # This is the loaded dict from YAML or defaults

        # Initialize session state for form fields if not already present or if form is re-shown
        # This ensures that callbacks have values to work with and widgets can be controlled.
        
        # Task Name related
        if 'form_task_name' not in st.session_state or st.session_state.current_form_data.get("task_name") != st.session_state.get('form_task_name_loaded_from'):
            st.session_state.form_task_name = form_data.get("task_name", "")
            st.session_state.form_task_name_loaded_from = form_data.get("task_name", "") # Track source for path updates
            if st.session_state.form_mode == 'edit' and 'original_task_name_on_edit' not in st.session_state:
                 st.session_state.original_task_name_on_edit = form_data.get("task_name", "")

        # Embedding related
        default_embedding_data = form_data.get("embedding", {})
        st.session_state.form_embedding_mode = default_embedding_data.get("mode", "local")
        st.session_state.available_models_for_mode = MODEL_MODE_COMPATIBILITY.get(st.session_state.form_embedding_mode, [])
        
        current_model_name_from_data = default_embedding_data.get("model_name", "")
        if current_model_name_from_data in st.session_state.available_models_for_mode:
            st.session_state.form_embedding_model_name = current_model_name_from_data
        else:
            st.session_state.form_embedding_model_name = st.session_state.available_models_for_mode[0] if st.session_state.available_models_for_mode else None
        
        st.session_state.form_embedding_dim = default_embedding_data.get("embedding_dim", MODEL_DIMENSIONS.get(st.session_state.form_embedding_model_name, 0))

        # Path previews - Initialize based on current task name from form_data
        task_name_for_paths = form_data.get("task_name", "")
        st.session_state.form_output_dir_preview = f"runs/{task_name_for_paths}/embeddings" if task_name_for_paths else "runs/<task_name>/embeddings"
        st.session_state.form_chunked_emails_path_preview = f"runs/{task_name_for_paths}/embeddings/chunked_emails.tsv" if task_name_for_paths else "runs/<task_name>/embeddings/chunked_emails.tsv"


        # --- Callbacks ---
        def handle_task_name_change():
            # Rule 5: Path alignment
            new_task_name = st.session_state.form_task_name_input # value from text_input
            st.session_state.form_task_name = new_task_name # Update the central form_task_name state
            if new_task_name:
                st.session_state.form_output_dir_preview = f"runs/{new_task_name}/embeddings"
                st.session_state.form_chunked_emails_path_preview = f"runs/{new_task_name}/embeddings/chunked_emails.tsv"
            else:
                st.session_state.form_output_dir_preview = "runs/<task_name>/embeddings"
                st.session_state.form_chunked_emails_path_preview = "runs/<task_name>/embeddings/chunked_emails.tsv"

        def handle_embedding_mode_change():
            # Rule 1: Dynamic model_name selectbox
            selected_mode = st.session_state.form_embedding_mode_selector # value from selectbox
            st.session_state.form_embedding_mode = selected_mode # Update central state
            st.session_state.available_models_for_mode = MODEL_MODE_COMPATIBILITY.get(selected_mode, [])
            
            current_model = st.session_state.get('form_embedding_model_name')
            if current_model not in st.session_state.available_models_for_mode:
                st.session_state.form_embedding_model_name = st.session_state.available_models_for_mode[0] if st.session_state.available_models_for_mode else None
                # Trigger model change explicitly if it was reset
                handle_embedding_model_change(is_triggered_by_mode_change=True)


        def handle_embedding_model_change(is_triggered_by_mode_change=False):
            # Rule 3: Default embedding_dim
            # If called by mode change, form_embedding_model_name is already updated.
            # If called by model_name selectbox, get value from its key.
            if not is_triggered_by_mode_change:
                 st.session_state.form_embedding_model_name = st.session_state.form_embedding_model_selector

            selected_model = st.session_state.form_embedding_model_name
            if selected_model:
                st.session_state.form_embedding_dim = MODEL_DIMENSIONS.get(selected_model, 0)
            else:
                st.session_state.form_embedding_dim = 0
        
        # --- Form Definition ---
        with st.form(key='task_config_form'):
            st.markdown("#### Basic Information")
            # Task Name
            st.text_input("Task Name", key='form_task_name_input', value=st.session_state.form_task_name, on_change=handle_task_name_change, help="Unique name for the task. This will also be the YAML filename (e.g., 'my_email_task').")
            # Rule 4: Task Name Change Warning
            if st.session_state.form_mode == 'edit' and \
               st.session_state.original_task_name_on_edit and \
               st.session_state.form_task_name != st.session_state.original_task_name_on_edit:
                st.warning("Changing the task name will save this as a new task configuration and create a new output directory structure.")

            st.markdown("#### Embedding Settings")
            # Embedding Mode
            st.selectbox(
                "Mode",
                options=list(MODEL_MODE_COMPATIBILITY.keys()),
                index=list(MODEL_MODE_COMPATIBILITY.keys()).index(st.session_state.form_embedding_mode) if st.session_state.form_embedding_mode in MODEL_MODE_COMPATIBILITY else 0,
                key='form_embedding_mode_selector',
                on_change=handle_embedding_mode_change,
                help="Choose the embedding mode."
            )
            # Embedding Model Name
            st.selectbox(
                "Model Name",
                options=st.session_state.available_models_for_mode,
                index=st.session_state.available_models_for_mode.index(st.session_state.form_embedding_model_name) if st.session_state.form_embedding_model_name in st.session_state.available_models_for_mode else 0,
                key='form_embedding_model_selector',
                on_change=handle_embedding_model_change,
                help="Select the embedding model. Options depend on the selected mode."
            )
            # Embedding Dimension
            st.number_input(
                "Embedding Dimension",
                value=st.session_state.form_embedding_dim, # Controlled by session state
                key='form_embedding_dim_input', # Key for direct update if needed, though callback handles it
                step=1,
                help="Dimension of the embeddings (e.g., 384, 1536). Auto-updates based on model selection."
            )

            with st.expander("Chunking Settings"):
                chunking_data = form_data.get("chunking", {})
                max_chunk_size = st.number_input("Max Chunk Size (tokens)", value=chunking_data.get("max_chunk_size", 450), step=10)
                overlap = st.number_input("Overlap (tokens)", value=chunking_data.get("overlap", 50), step=5)
                min_chunk_size = st.number_input("Min Chunk Size (characters)", value=chunking_data.get("min_chunk_size", 150), step=10)
                similarity_threshold = st.number_input("Similarity Threshold", value=float(chunking_data.get("similarity_threshold", 0.8)), step=0.05, format="%.2f")
                language_model = st.text_input("SpaCy Language Model", value=chunking_data.get("language_model", "en_core_web_sm"))
                chunking_embedding_model = st.text_input("Chunking Embedding Model", value=chunking_data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))

            with st.expander("Retrieval Settings"):
                retrieval_data = form_data.get("retrieval", {})
                top_k = st.number_input("Top-K Chunks", value=retrieval_data.get("top_k", 5), step=1)
                strategy = st.text_input("Strategy", value=retrieval_data.get("strategy", "dense_vector"))

            with st.expander("Generation Settings"):
                generation_data = form_data.get("generation", {})
                gen_model = st.text_input("OpenAI Model", value=generation_data.get("model", "openai-gpt-4"))
                prompt_template = st.text_input("Prompt Template Name", value=generation_data.get("prompt_template", "standard_qa"))

            with st.expander("Outlook Settings"):
                outlook_data = form_data.get("outlook", {})
                account_name = st.text_input("Account Name", value=outlook_data.get("account_name", "YOUR_ACCOUNT_NAME"))
                folder_path = st.text_input("Folder Path", value=outlook_data.get("folder_path", "Inbox"))
                days_to_fetch = st.number_input("Days to Fetch", value=outlook_data.get("days_to_fetch", 3), step=1)
            
            with st.expander("Derived Paths (Read-only)"):
                # These paths are typically derived from task_name or fixed.
                # For the form, we use the preview session state variables.
                # The actual config values for `output_dir` and `chunked_emails` might be simpler 
                # (e.g., "embeddings", "chunked_emails.tsv") and combined with task_name at runtime by the pipeline.
                # However, Rule 5 asks for dynamic updates like "runs/<task_name>/embeddings".
                
                # Displaying the "config value" for output_dir (usually "embeddings")
                st.text_input("Embedding Config: Output Dir", value=form_data.get("embedding", {}).get("output_dir", "embeddings"), disabled=True, help="This is the 'output_dir' value stored in the config (typically 'embeddings'). The full path will be 'runs/<task_name>/<output_dir_value>'.")
                # Displaying the dynamic preview path
                st.text_input("Full Output Directory (Preview)", value=st.session_state.form_output_dir_preview, disabled=True, help="Dynamically updated based on Task Name. This is where outputs will be stored.")
                
                # Displaying the "config value" for chunked_emails path
                default_paths_structure = get_default_config("temp").get("paths", {})
                chunked_emails_config_value = form_data.get("paths", {}).get("chunked_emails", default_paths_structure.get("chunked_emails"))
                st.text_input("Chunked Emails Config: Path", value=chunked_emails_config_value, disabled=True, help="This is the 'chunked_emails' path value stored in the config. The full path will be 'runs/<task_name>/<path_value>'.")
                # Displaying the dynamic preview path
                st.text_input("Full Chunked Emails Path (Preview)", value=st.session_state.form_chunked_emails_path_preview, disabled=True, help="Dynamically updated based on Task Name.")

            # Form submission button
            submitted = st.form_submit_button("💾 Save Task Config")
            if submitted:
                # --- 1. Retrieve Form Data ---
                retrieved_task_name = st.session_state.form_task_name_input # This is the most up-to-date task name from the input field
                
                # Validate task_name (must not be empty)
                if not retrieved_task_name or not retrieved_task_name.strip():
                    st.error("Task Name cannot be empty.")
                else:
                    retrieved_embedding_mode = st.session_state.form_embedding_mode_selector
                    retrieved_embedding_model_name = st.session_state.form_embedding_model_selector
                    retrieved_embedding_dim = st.session_state.form_embedding_dim_input # Direct from widget
                    
                    # --- 2. Model-Mode Validation ---
                    compatible_models = MODEL_MODE_COMPATIBILITY.get(retrieved_embedding_mode, [])
                    if retrieved_embedding_model_name not in compatible_models:
                        st.error(f"Validation Error: Model '{retrieved_embedding_model_name}' is not compatible with mode '{retrieved_embedding_mode}'. Please select a compatible model.")
                    else:
                        # --- 3. Construct Configuration Dictionary ---
                        # Actual paths based on the final task name
                        final_output_dir = f"runs/{retrieved_task_name}/embeddings"
                        final_chunked_emails_path = f"runs/{retrieved_task_name}/embeddings/chunked_emails.tsv"

                        config_dict = {
                            "task_name": retrieved_task_name,
                            "embedding": {
                                "mode": retrieved_embedding_mode,
                                "model_name": retrieved_embedding_model_name,
                                "embedding_dim": retrieved_embedding_dim,
                                # --- 4. Path Construction ---
                                "output_dir": final_output_dir, # Rule 5
                                "index_filename": "chunks.index", # Default
                                "metadata_filename": "chunks_metadata.tsv" # Default
                            },
                            "chunking": {
                                "max_chunk_size": st.session_state.max_chunk_size_input,
                                "overlap": st.session_state.overlap_input,
                                "min_chunk_size": st.session_state.min_chunk_size_input,
                                "similarity_threshold": float(st.session_state.similarity_threshold_input),
                                "language_model": st.session_state.language_model_input,
                                "embedding_model": st.session_state.chunking_embedding_model_input
                            },
                            "retrieval": {
                                "top_k": st.session_state.top_k_input,
                                "strategy": st.session_state.strategy_input
                            },
                            "generation": {
                                "model": st.session_state.gen_model_input,
                                "prompt_template": st.session_state.prompt_template_input
                            },
                            "outlook": {
                                "account_name": st.session_state.account_name_input,
                                "folder_path": st.session_state.folder_path_input,
                                "days_to_fetch": st.session_state.days_to_fetch_input
                            },
                            "paths": {
                                "chunked_emails": final_chunked_emails_path # Rule 5
                            }
                        }

                        # --- 5. Save to YAML File ---
                        config_dir = Path("configs/tasks")
                        config_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
                        
                        # Determine the actual filename. If task_name changed in edit mode, it's a new file.
                        # The 'original_task_name_on_edit' helps decide if it's a rename-as-new-file scenario.
                        # However, we just save with the new name. The old file is not touched/deleted here.
                        file_path = config_dir / f"{retrieved_task_name}.yaml"

                        try:
                            with open(file_path, "w", encoding="utf-8") as f:
                                yaml.safe_dump(config_dict, f, sort_keys=False, default_flow_style=False)
                            st.success(f"✅ Task configuration '{retrieved_task_name}.yaml' saved successfully!")
                            
                            # --- 6. Post-Save Actions ---
                            st.session_state.show_task_form = False # Hide the form
                            # Clear form-specific session state to ensure fresh state next time
                            keys_to_clear = [
                                'current_form_data', 'form_task_name', 'original_task_name_on_edit',
                                'form_embedding_mode', 'available_models_for_mode', 
                                'form_embedding_model_name', 'form_embedding_dim',
                                'form_output_dir_preview', 'form_chunked_emails_path_preview',
                                'form_task_name_loaded_from', 'form_task_name_input',
                                'form_embedding_mode_selector', 'form_embedding_model_selector', 
                                'form_embedding_dim_input',
                                # Keys for chunking settings
                                'max_chunk_size_input', 'overlap_input', 'min_chunk_size_input', 
                                'similarity_threshold_input', 'language_model_input', 'chunking_embedding_model_input',
                                # Keys for retrieval settings
                                'top_k_input', 'strategy_input',
                                # Keys for generation settings
                                'gen_model_input', 'prompt_template_input',
                                # Keys for outlook settings
                                'account_name_input', 'folder_path_input', 'days_to_fetch_input'
                            ]
                            # For widgets that were not explicitly controlled via st.session_state.form_*,
                            # their keys (like 'max_chunk_size_input') will be cleared too if listed.
                            # It's safer to list them or have a prefix.
                            # For now, clear the main ones. Streamlit might clear widget state on hide/rerun anyway.
                            for key in keys_to_clear:
                                if key in st.session_state:
                                    del st.session_state[key]
                            # Also ensure the main selectbox for configs updates:
                            if 'selected_config' in st.session_state: #This is for the main task selector
                                st.session_state.selected_config = f"{retrieved_task_name}.yaml"

                            st.experimental_rerun()

                        except Exception as e:
                            st.error(f"❌ Failed to save task configuration: {e}")
        
        # Cancel button (outside the form)
        if st.button("❌ Cancel"):
            st.session_state.show_task_form = False
            # Clear form data if canceling
            if 'current_form_data' in st.session_state:
                del st.session_state.current_form_data
            # Clear other form-specific session state variables as done in post-save
            # This list should mirror the one in the submit block
            keys_to_clear_on_cancel = [
                'current_form_data', # Also clear current_form_data on cancel
                'form_task_name', 'original_task_name_on_edit',
                'form_embedding_mode', 'available_models_for_mode', 
                'form_embedding_model_name', 'form_embedding_dim',
                'form_output_dir_preview', 'form_chunked_emails_path_preview',
                'form_task_name_loaded_from', 'form_task_name_input',
                'form_embedding_mode_selector', 'form_embedding_model_selector', 
                'form_embedding_dim_input',
                # Keys for chunking settings
                'max_chunk_size_input', 'overlap_input', 'min_chunk_size_input', 
                'similarity_threshold_input', 'language_model_input', 'chunking_embedding_model_input',
                # Keys for retrieval settings
                'top_k_input', 'strategy_input',
                # Keys for generation settings
                'gen_model_input', 'prompt_template_input',
                # Keys for outlook settings
                'account_name_input', 'folder_path_input', 'days_to_fetch_input'
            ]
            for key in keys_to_clear_on_cancel: # Ensure current_form_data is also cleared if present
                if key == 'current_form_data' and key not in st.session_state: # Already deleted above
                    continue 
                if key in st.session_state:
                    del st.session_state[key]
            st.experimental_rerun() # To immediately hide the form and clear states

# ----------------------
# Tab 2: Runs & Logs
# ----------------------
with tabs[1]:
    st.header("Run and Log Inspection")

    # Populate 'Select Task' dropdown
    task_config_files = list_config_files()
    task_names = [os.path.splitext(f)[0] for f in task_config_files]
    # Initialize selected_log_task if not present, to avoid errors on first run
    if 'selected_log_task' not in st.session_state:
        st.session_state.selected_log_task = None
    st.selectbox("Select Task:", task_names, key="selected_log_task", index=task_names.index(st.session_state.selected_log_task) if st.session_state.selected_log_task and st.session_state.selected_log_task in task_names else 0 if task_names else 0)

    # Populate 'Run ID' dropdown based on selected task
    run_ids = []
    if st.session_state.selected_log_task:
        # Construct path using PROJECT_ROOT for robustness
        runs_dir_path = os.path.join(PROJECT_ROOT, "runs", st.session_state.selected_log_task, "runs")
        if os.path.exists(runs_dir_path) and os.path.isdir(runs_dir_path):
            try:
                run_ids = sorted([name for name in os.listdir(runs_dir_path) if os.path.isdir(os.path.join(runs_dir_path, name))], reverse=True)
            except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard
                st.warning(f"Run directory not found for task '{st.session_state.selected_log_task}'. It might have been moved or deleted.")
                run_ids = [] # Ensure run_ids is empty
            except Exception as e:
                st.error(f"Error listing run IDs for task '{st.session_state.selected_log_task}': {e}")
                run_ids = [] # Ensure run_ids is empty
        else:
            # This case handles when the specific task's run directory doesn't exist.
            # No explicit message here as the empty dropdown implies no runs or selectable task.
            pass # run_ids remains empty
    
    # Initialize selected_log_run_id if not present
    if 'selected_log_run_id' not in st.session_state:
        st.session_state.selected_log_run_id = None

    # Ensure the selected_log_run_id is valid for the current run_ids
    # If the previously selected run_id is not in the new list, reset it
    if st.session_state.selected_log_run_id not in run_ids:
        st.session_state.selected_log_run_id = run_ids[0] if run_ids else None

    st.selectbox("Run ID:", run_ids, key="selected_log_run_id", index=run_ids.index(st.session_state.selected_log_run_id) if st.session_state.selected_log_run_id and st.session_state.selected_log_run_id in run_ids else 0 if run_ids else 0,
                 help="Select a Run ID. Run IDs are sorted reverse chronologically. If empty, select a task or the task has no runs yet.")


    # Get selected task and run ID
    selected_task_name = st.session_state.get("selected_log_task")
    selected_run_id = st.session_state.get("selected_log_run_id")

    # Placeholder for messages if task/run not selected or file not found
    default_no_selection_message = "Select a task and a run ID to view details."
    default_not_found_message = "File not found."

    with st.expander("📄 Run Metadata", expanded=True): # Expanded by default
        if selected_task_name and selected_run_id:
            metadata_path = os.path.join(PROJECT_ROOT, "runs", selected_task_name, "runs", selected_run_id, "run_metadata.json")
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata_content = json.load(f)
                st.json(metadata_content)
            except FileNotFoundError:
                st.info(f"Run metadata file not found at: {metadata_path}")
            except json.JSONDecodeError:
                st.error(f"Error decoding JSON from metadata file at: {metadata_path}")
            except Exception as e:
                st.error(f"An error occurred while reading metadata: {e}")
        else:
            st.info(default_no_selection_message)

    with st.expander("💬 Answer Output"):
        if selected_task_name and selected_run_id:
            answer_path = os.path.join(PROJECT_ROOT, "runs", selected_task_name, "runs", selected_run_id, "answer.txt")
            try:
                with open(answer_path, "r", encoding="utf-8") as f:
                    answer_content = f.read()
                st.text_area("Answer", value=answer_content, height=200, disabled=True)
            except FileNotFoundError:
                st.info(f"Answer file not found at: {answer_path}")
            except Exception as e:
                st.error(f"An error occurred while reading the answer file: {e}")
        else:
            st.info(default_no_selection_message)

    with st.expander("📜 Query Debug Log"):
        if selected_task_name and selected_run_id:
            debug_log_path = os.path.join(PROJECT_ROOT, "runs", selected_task_name, "runs", selected_run_id, "query_debug.txt")
            try:
                with open(debug_log_path, "r", encoding="utf-8") as f:
                    debug_log_content = f.read()
                st.text_area("Debug Log", value=debug_log_content, height=300, disabled=True)
            except FileNotFoundError:
                st.info(f"Query debug log file not found at: {debug_log_path}")
            except Exception as e:
                st.error(f"An error occurred while reading the query debug log: {e}")
        else:
            st.info(default_no_selection_message)

    with st.expander("🪵 Execution Log"):
        if selected_task_name and selected_run_id:
            # Note: Path for execution log is different
            execution_log_path = os.path.join(PROJECT_ROOT, "runs", selected_task_name, "logs", f"{selected_run_id}.log")
            try:
                with open(execution_log_path, "r", encoding="utf-8") as f:
                    execution_log_content = f.read()
                st.text_area("Execution Log", value=execution_log_content, height=400, disabled=True)
            except FileNotFoundError:
                st.info(f"Execution log file not found at: {execution_log_path}")
            except Exception as e:
                st.error(f"An error occurred while reading the execution log: {e}")
        else:
            st.info(default_no_selection_message)

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
        st.code(st.session_state.get("pipeline_output", "(Results will be shown here after execution)"), language=None)

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
