import streamlit as st
from rag_meeting_qa import (
    load_transcripts,
    chunk_documents,
    create_vector_store,
    setup_rag_system,
)

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
    vector_store = create_vector_store(chunks)
    qa_system = setup_rag_system(vector_store)
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
