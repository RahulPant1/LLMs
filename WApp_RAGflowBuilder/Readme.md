# RAG Flow Web App

## Overview
The **RAG Flow Web App** is a Retrieval-Augmented Generation (RAG) system built using **LangChain**, **Ollama**, and **Streamlit**. It enables document upload, vector database storage, and querying using LLMs (Large Language Models).

## Features
- Upload and process documents (**PDF, DOCX, TXT**)
- Generate embeddings using **HuggingFace Embeddings**
- Store and retrieve document chunks using **ChromaDB**
- Query documents using **Ollama LLM**
- Interactive Streamlit-based UI

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip
- Ollama server running on `http://localhost:11434`

### Setup
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
### Upload Documents
- Upload **PDF, DOCX, or TXT** files.
- Documents are split into chunks and stored in **ChromaDB**.

### Querying the Documents
- Enter a query in the text box.
- The system retrieves relevant document chunks.
- The **Ollama LLM** generates a response.

### Model Selection
- Choose embedding and LLM models via the sidebar.
- Default LLM: `deepseek-r1:8b`

## File Structure
```
📂 RAG-Flow-Web-App
├── 📄 app.py                # Streamlit UI & Logic
├── 📄 requirements.txt      # Dependencies
├── 📂 data                  # Uploaded documents
├── 📂 chroma_db             # Vector database
└── 📄 README.md             # Documentation
```

## Configuration
You can modify the following parameters in the UI:
- **Embedding Model:** Choose between `sentence-transformers` and `openai`.
- **Chunk Size & Overlap:** Adjust chunking parameters.
- **LLM Model:** Select from available Ollama models.

## Troubleshooting
### Common Issues
- **"Error fetching models"** → Ensure Ollama server is running.
- **"No documents available"** → Upload valid document files.

## License
MIT License

## Contributors
Rahul

