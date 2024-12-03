from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import numpy as np
import pinecone

index_name = "text-similarity" # first create this in Pinecone with dim=384
PINECONE_API_KEY = "YOUR-PINECONE_API_KEY"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name) # NOTE: make sure this exist on the server

#########################################
# Step 2: Create or Connect to a Pinecone Index
if index_name not in pc.list_indexes().names():
    pinecone.create_index(index_name, dimension=384, metric="cosine")

# Step 3: Generate Sample Text Documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaps over a sleepy dog.",
    "The lazy dog is jumped over by a quick fox.",
    "The brown fox is fast and jumps over the dog.",
    "An entirely unrelated sentence about something else.",
    "Another sentence completely unrelated to the fox and dog."
]

# Step 4: Convert Text to Embeddings using Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

#Find the dimension of the embedding
# embedding = model.encode("This is a sample sentence.")
# print("Embedding Dimension:", embedding.shape[0])  # Outputs 384

# Step 5: Insert Embeddings into Pinecone
# Use document index as the vector ID for simplicity
ids = [f"doc_{i}" for i in range(len(documents))]
vectors = [{"id": ids[i], "values": embeddings[i]} for i in range(len(documents))]
index.upsert(vectors)

# Step 6: Perform Similarity Search
threshold = 0.8
near_duplicates = []

for i, doc_id in enumerate(ids):
    query_vector = embeddings[i]
    results = index.query(vector=query_vector.tolist(), top_k=len(documents), include_metadata=False)


    for match in results["matches"]:
        if match["id"] != doc_id and match["score"] > threshold:
            near_duplicates.append((doc_id, match["id"], match["score"]))

# Step 7: Print Results
print("Near-duplicate documents (threshold = 0.8):\n")
for doc1, doc2, score in near_duplicates:
    idx1 = int(doc1.split("_")[1])
    idx2 = int(doc2.split("_")[1])
    print(f"Document {idx1}: {documents[idx1]}")
    print(f"Document {idx2}: {documents[idx2]}")
    print(f"Similarity Score: {score:.2f}")
    print("-" * 50)

# Step 8: Clean up (Optional)
# Delete index if you want to clean up resources
# pinecone.delete_index(index_name)

"""Near-duplicate documents (threshold = 0.8):

Document 0: The quick brown fox jumps over the lazy dog.
Document 2: The lazy dog is jumped over by a quick fox.
Similarity Score: 0.88
--------------------------------------------------
Document 0: The quick brown fox jumps over the lazy dog.
Document 3: The brown fox is fast and jumps over the dog.
Similarity Score: 0.87
--------------------------------------------------
Document 0: The quick brown fox jumps over the lazy dog.
Document 1: A fast brown fox leaps over a sleepy dog.
Similarity Score: 0.85
--------------------------------------------------
Document 1: A fast brown fox leaps over a sleepy dog.
Document 0: The quick brown fox jumps over the lazy dog.
Similarity Score: 0.85
--------------------------------------------------
Document 1: A fast brown fox leaps over a sleepy dog.
Document 3: The brown fox is fast and jumps over the dog.
Similarity Score: 0.84
--------------------------------------------------
Document 2: The lazy dog is jumped over by a quick fox.
Document 0: The quick brown fox jumps over the lazy dog.
Similarity Score: 0.88
--------------------------------------------------
Document 3: The brown fox is fast and jumps over the dog.
Document 0: The quick brown fox jumps over the lazy dog.
Similarity Score: 0.87
--------------------------------------------------
Document 3: The brown fox is fast and jumps over the dog.
Document 1: A fast brown fox leaps over a sleepy dog.
Similarity Score: 0.84
--------------------------------------------------
"""