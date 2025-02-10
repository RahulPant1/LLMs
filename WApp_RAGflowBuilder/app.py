import os
from langchain.document_loaders import Docx2txtLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader
import tempfile

def upload_files(uploaded_files):
    """Handles file upload and parsing."""
    from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

    docs = []
    for file in uploaded_files:
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        # Determine the file type and load the document
        ext = os.path.splitext(file.name)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(temp_file_path)
        elif ext == ".txt":
            loader = TextLoader(temp_file_path)
        else:
            os.remove(temp_file_path)  # Clean up the temporary file
            raise ValueError(f"Unsupported file type: {ext}")

        docs.extend(loader.load())
        os.remove(temp_file_path)  # Clean up the temporary file after loading

    return docs


#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

def initialize_embeddings(embedding_model):
    """Initializes embedding model."""
    if embedding_model == "sentence-transformers":
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    elif embedding_model == "openai":
        raise NotImplementedError("OpenAI embeddings can be added here.")
    else:
        raise ValueError("Unsupported embedding model!")



#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def setup_vector_store(docs, embeddings, persist_dir="./chroma_db", chunk_size=1000, chunk_overlap=200):
    """Creates and persists a vector store, and returns both the store and chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)  # Generate the document chunks
    
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vector_store.persist()
    return vector_store, split_docs  # Return both the vector store and chunks


from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

def initialize_ollama_llm(model_name: str = "deepseek-r1:8b") -> OllamaLLM:
    """Initializes Ollama LLM."""
    return OllamaLLM(
        model=model_name,
        base_url="http://localhost:11434",  # Ensure the Ollama server is running
        temperature=0.7,
    )


def query_vector_store_with_ollama(vector_store, query, llm):
    """Queries the vector store using Ollama LLM."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = qa_chain.run(query)
    return response


from langchain.prompts import PromptTemplate

def create_agent_prompt():
    """Defines a custom prompt for agents."""
    return PromptTemplate(
        template="""You must answer questions using the provided document context. 
If the answer isn't in the documents, say: "I don't know based on the documents."

Question: {input}
Document Context:
{context}
Answer:""",
        input_variables=["input", "context"]
    )



import streamlit as st
#from streamlit.columns import columns

import ollama

# def list_ollama_models():
#     """Fetch and display available models from local Ollama."""
#     models = ollama.list()
#     return [model['NAME'] for model in models['models']]

def list_ollama_models():
    """Fetch and return available models from local Ollama."""
    try:
        models = ollama.list()  # Fetch models from Ollama
        if hasattr(models, "models"):
            return [m.model for m in models.models]  # Extract model names
        else:
            return ["No models found."]
    except Exception as e:
        return [f"Error: {str(e)}"]

# Page Config
st.set_page_config(page_title="RAG Flow Web App", layout="wide")
st.title("RAG Flow Builder 🌐")

# Sidebar Configuration
# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    embedding_model = st.selectbox("Embedding Model", ["sentence-transformers", "openai"])
    vector_store_dir = st.text_input("Vector Store Directory", "./chroma_db")
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    st.header("Ollama Models")
    try:
        available_models = list_ollama_models()
        selected_llm_model = st.selectbox("Select LLM Model", available_models)
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        selected_llm_model = "deepseek-r1:8b"  # Default model if error occurs

    st.write("---")
    #llm_model = st.selectbox("Select LLM Model", ["deepseek-r1:8b", "gpt-3.5-turbo", "gpt-4"])
    llm_model = selected_llm_model
    st.write("---")

    # Chunking Configuration
    st.header("Chunking Configuration")
    chunk_size = st.slider("Chunk Size (number of characters)", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap (number of characters)", 0, 500, 200, step=50)
    st.write("---")


# State Initialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "response" not in st.session_state:
    st.session_state.response = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

# File Upload and Embedding Initialization
if uploaded_files:
    with st.spinner("Processing documents..."):
        
        # Parse documents and generate chunks
        docs = upload_files(uploaded_files)
        embeddings = initialize_embeddings(embedding_model)

        # Generate Chunks for Validation
        #splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        #st.session_state.chunks = splitter.split_documents(docs)

        # Store Chunks in Vector Store
        vector_store, chunks = setup_vector_store(
            docs, embeddings, persist_dir=vector_store_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        st.session_state.vector_store = vector_store
        st.session_state.chunks = chunks
        st.success("Documents processed, split into chunks, and stored in vector database!")


# Initialize Ollama LLM
if "llm" not in st.session_state:
    st.session_state.llm = initialize_ollama_llm(llm_model)

# Main Content Layout
col1, col2 = st.columns([1, 2])  # Two columns, with col2 being twice as wide as col1

# Chunk Preview Section
with col1:
    st.write("### Chunk Preview")
    if st.session_state.chunks:
        num_chunks_to_show = st.slider("Number of Chunks to Preview", 1, len(st.session_state.chunks), 5)
        for i, chunk in enumerate(st.session_state.chunks[:num_chunks_to_show]):
            st.write(f"#### Chunk {i + 1}")
            st.text(chunk.page_content)
    else:
        st.write("No document chunks available yet. Please upload and process documents.")


with col2:
    st.write("### Query Input")
    query_input = st.text_area("Enter your query:")
    if st.button("Search"):  # Add a search button
        if query_input and st.session_state.vector_store:
            with st.spinner("Fetching response..."):
                st.session_state.response = query_vector_store_with_ollama(
                    vector_store=st.session_state.vector_store,
                    query=query_input,
                    llm=st.session_state.llm
                )
                st.write("### Results")
                st.write(st.session_state.response)
                st.success("Response generated!")

# with col2:
#     st.write("### Results")
#     if st.session_state.response:
#         st.write(st.session_state.response)
#     else:
#         st.write("No results to display yet.")

