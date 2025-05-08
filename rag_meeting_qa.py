import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM


# 1. Load and process meeting transcripts
def load_transcripts(directory):
    """
    Loads all .txt files from the specified directory as LangChain documents, adding the filename as metadata.
    """
    documents = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(directory, file))
            docs = loader.load()
            # Add filename as metadata to each document
            for doc in docs:
                doc.metadata["source"] = file
            documents.extend(docs)
    return documents


# 2. Chunk the transcripts
def chunk_documents(documents):
    """
    Splits documents into manageable chunks for embedding and retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks


# 3. Create embeddings and store in vector DB
def create_vector_store(chunks):
    """
    Embeds the document chunks and stores them in a local Chroma vector database.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory="./chroma_db"
    )
    return vector_store


# 4. Create the RAG query system
def setup_rag_system(vector_store):
    """
    Sets up a Retrieval-Augmented Generation (RAG) system using a local Ollama LLM and the vector store.
    """
    llm = OllamaLLM(model="qwen3:8b")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
    )
    return qa_chain


# Main function to process transcripts and set up system
def main():
    """
    Main entry point: loads transcripts, creates vector store, and starts an interactive QA loop.
    """
    transcript_dir = "./meeting_transcripts"  # Directory containing .txt transcripts
    documents = load_transcripts(transcript_dir)
    chunks = chunk_documents(documents)
    vector_store = create_vector_store(chunks)
    qa_system = setup_rag_system(vector_store)

    print("\nRAG Meeting QA System Ready! Type your questions or 'exit' to quit.")
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        result = qa_system.invoke({"query": query})
        print("\nAnswer:", result["result"])
        print("\nSources:")
        for doc in result["source_documents"]:
            source = doc.metadata.get("source", "Unknown file")
            print(f"- [{source}] {doc.page_content[:100]}...\n")


if __name__ == "__main__":
    main()
