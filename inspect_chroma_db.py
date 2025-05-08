from langchain_community.vectorstores import Chroma

# Connect to your existing Chroma DB
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=None,  # Not needed for just viewing records
)

# Fetch all records
collection = vector_store._collection
results = collection.get(include=["documents", "metadatas"])

docs = results["documents"]
metas = results["metadatas"]

for i, (doc, meta) in enumerate(zip(docs, metas)):
    print(f"Record {i + 1}:")
    print(f"  Document: {doc[:200]}...")  # Print first 200 chars
    print(f"  Metadata: {meta}")
    print("-" * 40)
