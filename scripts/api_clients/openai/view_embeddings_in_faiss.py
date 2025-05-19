import faiss
import numpy as np

# Load the FAISS index

full_path = r"C:\Users\hagaybar\OneDrive - Tel-Aviv University\My Personal files\systems\AI Project\embeddings\email_chunks.index"
faiss_index_file = full_path
index = faiss.read_index(faiss_index_file)

# Get number of stored vectors
num_vectors = index.ntotal
print(f"Number of stored vectors: {num_vectors}")

# Retrieve embeddings one by one (since reconstruct_n doesn't work with IndexFlatL2)
stored_vectors = []
for i in range(num_vectors):
    vector = np.zeros((index.d,), dtype=np.float32)  # Correct dtype for FAISS
    index.reconstruct(i, vector)
    stored_vectors.append(vector)

# Convert list to NumPy array
stored_vectors = np.array(stored_vectors, dtype=np.float32)

# Print first 5 embeddings
print("First 5 stored embeddings:")
print(stored_vectors[:5])
print("Shape of stored vectors:", stored_vectors.shape)

