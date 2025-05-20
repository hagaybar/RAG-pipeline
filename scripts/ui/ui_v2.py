import streamlit as st

st.set_page_config(page_title="RAG Pipeline UI", layout="wide")
st.title("📬 RAG Pipeline Control Panel")

# Top-level tab navigation
tabs = st.tabs(["Tasks 🛠", "Runs & Logs 📊", "Pipeline Actions ⚙️", "Utilities & Tools 🧰"])

# ----------------------
# Tab 1: Task Management
# ----------------------
with tabs[0]:
    st.header("Task Management")

    with st.expander("🔽 Select Task Config", expanded=True):
        st.selectbox("Available Configs:", ["email_test.yaml", "api_test.yaml"])
        cols = st.columns(4)
        cols[0].button("View")
        cols[1].button("Edit")
        cols[2].button("Duplicate")
        cols[3].button("Delete")

    with st.expander("➕ Create New Task"):
        st.text_input("New Task Name")
        st.button("Create from Template")

    with st.expander("📑 Preview Task YAML"):
        st.radio("Mode:", ["Read-only", "Edit mode"], horizontal=True)
        st.text_area("Config Content", """task_name: email_test
embedding:
  mode: local
  model_name: sentence-transformers/all-MiniLM-L6-v2
...""", height=250)

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
