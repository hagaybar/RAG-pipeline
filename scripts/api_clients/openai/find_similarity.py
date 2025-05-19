import faiss
import numpy as np
import pandas as pd
from scripts.api_clients.openai.gptApiClient import APIClient

# Load API Client
api_client = APIClient()

# File paths
TSV_FILE = "emails.tsv"  # Your emails dataset
OUTPUT_TSV = "similarity_results.tsv"  # Output file for results
FAISS_INDEX_FILE = "email_faiss.index"  # Your FAISS index file

def load_faiss_index(index_file: str):
    """Load FAISS index from disk."""
    return faiss.read_index(index_file)

def load_emails(file_path: str):
    """Load email dataset."""
    return pd.read_csv(file_path, sep="\t")

def search_similar_emails(query_text: str, df: pd.DataFrame, index: faiss.IndexFlatL2, k: int = 5, threshold: float = 0.5):
    """
    Search for the top-k similar emails to the given query text and filter by similarity threshold.

    Args:
        query_text (str): The search query.
        df (pd.DataFrame): DataFrame containing emails.
        index (faiss.IndexFlatL2): FAISS index storing embeddings.
        k (int): Number of closest matches to return before filtering.
        threshold (float): Maximum similarity score (lower is better).

    Returns:
        pd.DataFrame: Filtered DataFrame with relevant emails.
    """
    query_embedding = np.array([api_client.get_embedding(query_text)], dtype=np.float32)
    distances, indices = index.search(query_embedding, k)

    # Retrieve matching emails
    results = df.iloc[indices[0]].copy()
    results["similarity_score"] = distances[0]

    # Filter by similarity threshold
    filtered_results = results[results["similarity_score"] <= threshold]

    return filtered_results.sort_values(by="similarity_score", ascending=True)

# --- Execution ---
df = load_emails(TSV_FILE)  # Load original emails
index = load_faiss_index(FAISS_INDEX_FILE)  # Load FAISS index

# Example query
query = "What are the news regarding NERS enhancement votint?"  # Replace with your test query
similar_emails = search_similar_emails(query, df, index, k=10, threshold=0.8)
# Add the query to the DataFrame
similar_emails["query"] = query  # Add query as a new colum

# 🔹 Check if any results were found
if similar_emails is None or similar_emails.empty:
    print("⚠️ No similar emails found. Try increasing 'k' or adjusting the 'threshold'.")
else:
    # ✅ Adjust column names
    print("\nMost similar emails:")
    print(similar_emails[["query","Subject", "Sender", "Body", "similarity_score"]])  # Now using correct column names

    # ✅ Save results to a TSV file
    similar_emails.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"\nResults saved to {OUTPUT_TSV}")
