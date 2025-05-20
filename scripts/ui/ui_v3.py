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

        # Load the selected config file
        config_path = os.path.join("configs", "tasks", os.path.basename(selected_config))
        config_text = load_config(config_path)

        cols = st.columns(4)
        cols[0].button("View")
        cols[1].button("Edit")
        cols[2].button("Duplicate")
        cols[3].button("Delete")


    with st.expander("➕ Create New Task"):
        st.text_input("New Task Name")
        st.button("Create from Template")

    with st.expander("📑 Preview Task YAML"):
        st.radio("Mode:", ["Read-only", "Edit mode"], horizontal=True, key="preview_mode")
        st.text_area("Config Content", value=config_text, height=300, disabled=True)






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
