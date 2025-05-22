import streamlit as st
import glob
import yaml
import os

st.set_page_config(page_title="RAG Pipeline UI", layout="wide")
st.title("📬 RAG Pipeline Control Panel")

# Top-level tab navigation
tabs = st.tabs(["Tasks 🛠", "Runs & Logs 📊", "Pipeline Actions ⚙️", "Utilities & Tools 🧰"])

# ----------------------
# Tab 1: Task Management
# ----------------------
with tabs[0]:
    st.header("Task Management")

    # 🧩 Feature 1: Load and display available task configs from disk

    with st.expander("🔽 Select Task Config", expanded=True):
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

        # List all available configs
        config_list = list_config_files()
        selected_config = st.selectbox("Available Configs:", config_list)
        st.session_state.confirm_delete = False

        # Load the selected config file
        config_path = os.path.join("configs", "tasks", os.path.basename(selected_config))
        config_text = load_config(config_path)
        st.write("🔍 confirm_delete state:", st.session_state.get("confirm_delete", False))

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
            # Safely check click and update session state immediately
            delete_clicked = st.button("Delete")
            if delete_clicked:
                st.session_state.confirm_delete = True

            # Now show confirmation dialog
            if st.session_state.get("confirm_delete", False):
                st.warning(f"⚠️ Are you sure you want to delete `{selected_config}`?")
                col_yes, col_no = st.columns(2)
                if col_yes.button("✅ Confirm Delete"):
                    try:
                        os.remove(config_path)
                        st.success(f"🗑️ Deleted {selected_config}")
                    except Exception as e:
                        st.error(f"❌ Deletion failed: {e}")
                    st.session_state.confirm_delete = False
                if col_no.button("❌ Cancel"):
                    st.session_state.confirm_delete = False



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
        st.text_area("Answer", value=load_text(answer_path), height=250, disabled=True)

    with st.expander("📜 Query Debug Log"):
        st.code(load_text(debug_path), language="text")

    with st.expander("🪵 Execution Log"):
        st.code(load_text(log_path), language="text")


# ----------------------
# Tab 3: Pipeline Actions
# ----------------------
with tabs[2]:
    st.header("Manual Pipeline Execution")

    st.selectbox("Select Task Config:", ["email_test.yaml"])
    st.text_input("Query Text:", placeholder="Enter a natural language question...")

    st.markdown("**Choose Pipeline Steps:**")
    step_cols = st.columns(3)
    step_cols[0].checkbox("Extract & Chunk")
    step_cols[1].checkbox("Embed Chunks")
    step_cols[2].checkbox("Retrieve Chunks")
    step_cols[0].checkbox("Generate Answer")

    st.button("🚀 Run Selected Steps")

    with st.expander("🧠 Output Area"):
        st.write("(Results will be shown here after execution)")

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
