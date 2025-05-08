# RAG Meeting QA System

Welcome to your local Retrieval-Augmented Generation (RAG) Meeting QA system! This project lets you ask questions about your meeting transcripts using state-of-the-art language models and embeddings—all running locally on your machine.

## 🚀 What does it do?
- Loads your meeting transcripts (as `.txt` files)
- Splits them into smart, searchable chunks
- Embeds them using HuggingFace models
- Stores them in a local Chroma vector database
- Lets you ask questions and get answers (with sources!) using a local Ollama LLM (like qwen3)
- **Now includes a Streamlit interface for easy local interaction!**

## 🛠️ Setup Instructions

### 1. Clone this repository

```
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Install dependencies

It's best to use a virtual environment. With [uv](https://github.com/astral-sh/uv):

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Prepare your meeting transcripts
- Create a folder called `meeting_transcripts` in the project root.
- Add your `.txt` transcript files to this folder. Each file should contain the text of a meeting.

### 4. Start your Ollama model
- Make sure you have [Ollama](https://ollama.com/) installed and running locally.
- Pull a model (e.g., qwen3:8b):
  ```
  ollama pull qwen3:8b
  ollama serve
  ```

### 5. Run the QA system (CLI or Streamlit UI)

#### **A. Command Line Interface**
```
python rag_meeting_qa.py

```
## 💡 Example Usage
```
Enter your question (or 'exit' to quit): What did we decide about the Q3 roadmap?

Answer: ...

Sources:
- [First 100 characters of the relevant transcript chunk...]

```
#### **B. Streamlit Interface (Recommended for local use!)**

```
pip install streamlit  # if not already installed
streamlit run rag_streamlit.py
```
- This will open a browser window at http://localhost:8501 where you can interact with your RAG system using a simple UI.

```
## 🧩 Troubleshooting
- **Ollama not found?** Make sure it's installed and running (`ollama serve`).
- **No transcripts found?** Ensure your `.txt` files are in the `meeting_transcripts` folder.
- **Dependency errors?** Double-check you're using the right Python version (3.8+ recommended) and that your virtual environment is activated.
- **Streamlit not opening?** Make sure you installed it (`pip install streamlit`) and that your browser allows popups.

## 📚 How it works
This project uses [LangChain](https://python.langchain.com/), [ChromaDB](https://www.trychroma.com/), [HuggingFace Sentence Transformers](https://www.sbert.net/), [Ollama](https://ollama.com/), and [Streamlit](https://streamlit.io/) to create a fully local, private, and powerful QA system for your meeting notes.

---

Enjoy asking questions about your meetings! If you have ideas or run into issues, feel free to open an issue or contribute. 
