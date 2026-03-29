import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Gen Z Cognitive RAG", page_icon="🧠")
st.title("🧠 Gen Z Cognitive Engagement Platform")

# Get API Key from Environment (Local .env or Streamlit Secrets)
groq_api_key = os.getenv("GROQ_API_KEY")
persist_dir = "Vector"
pdf_path = "Docs/fabric onelake.pdf" # Make sure this file is in your GitHub!

# --- 2. THE "BRAIN" (Embeddings) ---
@st.cache_resource # This saves memory so it doesn't reload every click
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # If the database doesn't exist yet, create it on the fly
    if not os.path.exists(persist_dir):
        st.info("First time setup: Processing documents...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=persist_dir
        )
    else:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vectorstore

# --- 3. THE UI ---

try:
    st.info("Loading embedding model...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.info("Creating vector database...")
    
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="/tmp/chroma_db"
    )

    st.success("Setup complete!")

except Exception as e:
    st.error(f"Error during setup: {e}")
