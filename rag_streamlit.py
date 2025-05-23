import streamlit as st
from rag_meeting_qa import (
    load_transcripts,
    chunk_documents,
    create_vector_store,
    setup_rag_system,
)
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(
    page_title="RAG Meeting Transcripts QA System", layout="centered"
)
st.title("RAG Meeting Transcripts QA System")
st.write("Ask questions about your meeting transcripts. All data stays local.")


# Load and prepare the RAG system (cache to avoid reloading on every run)
@st.cache_resource
def get_qa_system():
    transcript_dir = "./meeting_transcripts"
    documents = load_transcripts(transcript_dir)
    chunks = chunk_documents(documents)

    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Pass embeddings to create_vector_store
    # Based on previous subtask, create_vector_store expects (chunks, embeddings_model)
    vector_store = create_vector_store(chunks, embeddings_model=embeddings)

    # Define retrieval and re-ranking parameters
    # Using the same values as in rag_meeting_qa.py main for consistency
    initial_retrieval_k = 5 # Number of docs to retrieve initially
    final_reranked_n = 3   # Number of docs after re-ranking

    # Pass all required arguments to setup_rag_system
    # Based on previous subtask, setup_rag_system expects 
    # (vector_store, embeddings_model, top_k_retrieval, top_n_reranked)
    qa_system = setup_rag_system(
        vector_store=vector_store,
        embeddings_model=embeddings,
        top_k_retrieval=initial_retrieval_k,
        top_n_reranked=final_reranked_n
    )
    return qa_system


qa_system = get_qa_system()

# User input
question = st.text_input("Enter your question:", "")

if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        result = qa_system.invoke({"query": question})
        st.markdown("### Answer:")
        st.write(result["result"])
        st.markdown("### Sources:")
        seen = set()
        for doc in result["source_documents"]:
            snippet = doc.page_content[:200]
            source = doc.metadata.get("source", "Unknown file")
            key = (source, snippet)
            if key in seen:
                continue
            seen.add(key)
            st.write(f"**[{source}]** {snippet}...")
