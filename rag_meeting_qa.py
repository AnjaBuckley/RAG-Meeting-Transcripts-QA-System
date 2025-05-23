import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document


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


from sentence_transformers import CrossEncoder

# 4. Create the RAG query system
def setup_rag_system(vector_store, embeddings_model, top_k_retrieval=5, top_n_reranked=3):
    """
    Sets up a Retrieval-Augmented Generation (RAG) system using a local Ollama LLM,
    the vector store, and HyDE.
    """
    llm = OllamaLLM(model="qwen3:8b")

    # HyDE: Prompt to generate a hypothetical document
    hypothetical_doc_prompt_template = """Based on the following question, generate a short, relevant document that you believe would provide a good answer.
Question: {question}
Hypothetical Document:"""
    hypothetical_doc_prompt = PromptTemplate(
        input_variables=["question"], template=hypothetical_doc_prompt_template
    )

    # Chain to generate the hypothetical document
    hyde_chain = hypothetical_doc_prompt | llm

    # Function to retrieve documents using the hypothetical document
    def get_hyde_retrieved_docs(input_dict):
        user_query = input_dict["question"]
        # Generate hypothetical document
        hypothetical_document_content = hyde_chain.invoke({"question": user_query})
        
        # Embed the hypothetical document
        hypothetical_embedding = embeddings_model.embed_query(hypothetical_document_content)
        
        # Retrieve documents from vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        # Note: Chroma retriever expects a string query, not an embedding directly for its .get_relevant_documents method
        # So we'll use the hypothetical document content as the query string for the retriever.
        # Alternatively, some vector stores allow direct vector search.
        # For Chroma's as_retriever(), it will handle embedding the text query internally.
        # To use the embedding directly, we'd use vector_store.similarity_search_by_vector.
        # Let's adjust to use the content string for retriever for simplicity with as_retriever()
        # relevant_docs = vector_store.similarity_search_by_vector(embedding=hypothetical_embedding, k=5)
        relevant_docs = retriever.get_relevant_documents(hypothetical_document_content) # Query with hypothetical doc text

        return {"context": relevant_docs, "question": user_query, "source_documents": relevant_docs}


    # Define the prompt for the final QA
    qa_prompt_template = """Answer the question based only on the following context:
Context:
{context}

Question: {question}

Answer:"""
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"], template=qa_prompt_template
    )

    # Create the final QA chain
    # The chain structure will be:
    # 1. User query passthrough and HyDE document generation + retrieval
    # 2. Combine retrieved docs and original query
    # 3. Pass to LLM for final answer
    
    # This custom retriever function will be part of the chain
    hyde_retrieval_chain = RunnableLambda(get_hyde_retrieved_docs)

    # Final answer generation chain
    answer_chain = qa_prompt | llm

    # Full RAG chain with HyDE
    # The output of hyde_retrieval_chain is {"context": docs, "question": query, "source_documents": docs}
    # This output is directly compatible with what `answer_chain` expects if we map keys correctly.
    # However, RetrievalQA is a convenient class. Let's try to make a custom retriever.

    from langchain_core.retrievers import BaseRetriever
    from typing import List
    
    class HyDERetriever(BaseRetriever):
        vectorstore: Chroma
        llm_chain: any # The hyde_chain (hypothetical_doc_prompt | llm)
        embeddings: HuggingFaceEmbeddings # The embedding model
        cross_encoder: CrossEncoder
        top_k_initial_retrieval: int
        top_n_final_rerank: int


        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            hypothetical_document_content = self.llm_chain.invoke({"question": query})
            hypothetical_embedding = self.embeddings.embed_query(hypothetical_document_content)
            
            # Initial retrieval using HyDE
            initially_retrieved_docs = self.vectorstore.similarity_search_by_vector(
                embedding=hypothetical_embedding, k=self.top_k_initial_retrieval 
            )

            if not initially_retrieved_docs:
                return []

            # Re-ranking using CrossEncoder
            doc_contents = [doc.page_content for doc in initially_retrieved_docs]
            pairs = [[query, doc_content] for doc_content in doc_contents]
            
            scores = self.cross_encoder.predict(pairs)
            
            # Combine documents with scores and sort
            scored_documents = list(zip(scores, initially_retrieved_docs))
            scored_documents.sort(key=lambda x: x[0], reverse=True)
            
            # Select top N re-ranked documents
            reranked_documents = [doc for score, doc in scored_documents[:self.top_n_final_rerank]]
            
            return reranked_documents

    # Initialize the CrossEncoder model
    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    custom_hyde_retriever = HyDERetriever(
        vectorstore=vector_store, 
        llm_chain=hyde_chain, 
        embeddings=embeddings_model,
        cross_encoder=cross_encoder_model,
        top_k_initial_retrieval=top_k_retrieval, # How many docs to fetch initially
        top_n_final_rerank=top_n_reranked # How many docs to keep after reranking
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" chain type will use the qa_prompt implicitly if not provided, but we can customize
        retriever=custom_hyde_retriever,
        return_source_documents=True,
        # chain_type_kwargs={"prompt": qa_prompt} # Optional: if we want to enforce our specific QA prompt
    )
    # The "stuff" chain will internally format context and question.
    # If we need more control, we'd build it with LCEL:
    # rag_chain_passthrough = RunnablePassthrough.assign(
    #     context=hyde_retrieval_chain # This would need to be adjusted to just return docs
    # )
    # final_rag_chain = (
    #     rag_chain_passthrough
    #     | qa_prompt
    #     | llm
    # )
    # For now, using RetrievalQA with a custom retriever is cleaner.

    return qa_chain


# Main function to process transcripts and set up system
def main():
    """
    Main entry point: loads transcripts, creates vector store, and starts an interactive QA loop.
    """
    transcript_dir = "./meeting_transcripts"  # Directory containing .txt transcripts
    
    # It's good practice to initialize embeddings once and pass them around
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = load_transcripts(transcript_dir)
    chunks = chunk_documents(documents)
    vector_store = create_vector_store(chunks, embeddings_model) 
    
    # Configure how many documents to retrieve initially and how many to keep after re-ranking
    # These values can be adjusted as needed.
    # For example, retrieve 10 documents initially, then re-rank and keep the top 3.
    # The task description implies vector_store.as_retriever(search_kwargs={"k": 5}) was used.
    # So HyDE retriever was fetching 5. Let's use that as initial k.
    # And re-rank to top 3.
    initial_k = 5 # Number of docs from HyDE before re-ranking
    rerank_top_n = 3 # Number of docs after re-ranking

    qa_system = setup_rag_system(
        vector_store, 
        embeddings_model, 
        top_k_retrieval=initial_k, 
        top_n_reranked=rerank_top_n
    )

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
