"""
This utility script analyzes a TSV file of chunked text data (e.g., emails).
It reads the specified file, often using a hardcoded path for specific
analyses, and computes basic statistics, including the total number of chunks,
unique chunk count, unique email subject count, and the average chunks per
email, printing these to the console.
"""
import pandas as pd

# Load the actual TSV used for embedding
df_chunks = pd.read_csv(r"C:\Users\hagaybar\OneDrive - Tel-Aviv University\My Personal files\systems\Rag_Project\data\chunks\debug_test\chunked_emails.tsv", sep="\t")

print("✅ Total rows in file:", len(df_chunks))
print("🧠 Unique chunks:", df_chunks["Chunk"].nunique())
print("📬 Unique emails:", df_chunks["Subject"].nunique())
print("📊 Chunks per email (avg):", len(df_chunks) / df_chunks["Subject"].nunique())
