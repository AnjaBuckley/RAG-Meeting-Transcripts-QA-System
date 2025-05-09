# Use Python 3.11 as base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy application files
COPY requirements.txt .
COPY rag_streamlit.py .
COPY rag_meeting_qa.py .

# Create directory for meeting transcripts
RUN mkdir -p /app/meeting_transcripts
RUN mkdir -p /app/chroma_db

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Streamlit
EXPOSE 8501

# Create a startup script with better service management
RUN echo '#!/bin/bash\n\
echo "Starting Ollama service..."\n\
ollama serve &\n\
\n\
# Wait for Ollama to be ready\n\
echo "Waiting for Ollama to be ready..."\n\
timeout=30\n\
while ! nc -z localhost 11434; do\n\
  if [ "$timeout" -le 0 ]; then\n\
    echo "Timeout waiting for Ollama"\n\
    exit 1\n\
  fi\n\
  echo "Waiting for Ollama to be ready... ($timeout seconds left)"\n\
  sleep 1\n\
  timeout=$((timeout-1))\n\
done\n\
\n\
echo "Pulling Ollama model..."\n\
ollama pull qwen3:8b\n\
\n\
echo "Starting Streamlit application..."\n\
streamlit run rag_streamlit.py --server.address=0.0.0.0 --server.port=8501' > /app/start.sh

RUN chmod +x /app/start.sh

# Set volumes for meeting transcripts and vector store
VOLUME ["/app/meeting_transcripts", "/app/chroma_db"]

# Run the application
CMD ["/app/start.sh"] 