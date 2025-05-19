# ui.py
# Add project root to sys.path
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from scripts.pipeline.rag_pipeline import RAGPipeline

# Initialize or reload pipeline
@st.cache_resource
def load_pipeline(config_path: str):
    return RAGPipeline(config_path=config_path)

st.title("📬 RAG Email QA Interface (Local Test UI)")

# Config path selector
config_path = st.text_input("Path to Config File:", value="configs/tasks/api_full_180525.yaml")
pipeline = load_pipeline(config_path)

# Query input
query = st.text_area("Enter your question:", height=100)

if st.button("Run RAG Query"):
    try:
        with st.spinner("Running pipeline..."):
            pipeline.get_user_query(query)
            chunks = pipeline.retrieve()
            answer = pipeline.generate_answer(query=query, chunks=chunks)

        st.success("✅ Answer generated!")
        st.subheader("💬 Answer")
        st.write(answer)

        st.subheader("🔍 Retrieved Chunks")
        for c in chunks["top_chunks"]:
            st.markdown(f"**[{c['rank']}] {c['metadata'].get('Subject', 'No Subject')}**")
            st.markdown(c["text"])
            st.markdown("---")

        with st.expander("🧾 Raw Debug File"):
            st.code(chunks["debug_file"])
    except Exception as e:
        st.error(f"❌ Error: {e}")
