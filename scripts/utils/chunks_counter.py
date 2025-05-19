import pandas as pd

# Load the actual TSV used for embedding
df_chunks = pd.read_csv(r"C:\Users\hagaybar\OneDrive - Tel-Aviv University\My Personal files\systems\Rag_Project\data\chunks\debug_test\chunked_emails.tsv", sep="\t")

print("✅ Total rows in file:", len(df_chunks))
print("🧠 Unique chunks:", df_chunks["Chunk"].nunique())
print("📬 Unique emails:", df_chunks["Subject"].nunique())
print("📊 Chunks per email (avg):", len(df_chunks) / df_chunks["Subject"].nunique())
