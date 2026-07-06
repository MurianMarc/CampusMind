from sentence_transformers import SentenceTransformer as st
import chromadb
import pandas as pd

# Model for generating embeddings
model = st('all-MiniLM-L6-v2')

# This creates a place to store our documents
client = chromadb.Client()
collection = client.create_collection("school_docs")

# Load documents from a CSV file
documents = pd.read_csv("DocIngestion Sample\\docs\\sample.csv")

# Generate embeddings and add documents to the collection
for i, doc in documents.iterrows():
    embedding = model.encode(doc['text'])
    collection.add(
        documents=[doc['text']],
        metadatas=[{"source": doc['source'], "audience": doc['audience']}],
        ids=[f"doc_{i}"]
    )

# Function to find the answer to a question based on the ingested documents
def find_answer(question):
    # Generate embedding for the question
    question_embedding = model.encode(question)

    # Finds the most similar document(s) to the question embedding
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=1  # Number of similar documents to retrieve
    )

    return results['documents'][0][0] if results['documents'] else "I don't know"

print(find_answer(input("Ask a question: ")))