import streamlit as st
import os

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

st.title("🧠 Rag application")

try:
    # Step 1: Load and split documents only once
    if "docs" not in st.session_state:
        st.info("Loading documents...")

        loader = DirectoryLoader("Docs")
        documents = loader.load()

        st.info("Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        st.session_state.docs = splitter.split_documents(documents)

    # Step 2: Load embedding model only once
    if "embeddings" not in st.session_state:
        st.info("Loading embedding model...")

        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    # Step 3: Create vector database
    if "vectorstore" not in st.session_state:
        st.info("Creating vector database...")

        st.session_state.vectorstore = Chroma.from_documents(
            st.session_state.docs,
            st.session_state.embeddings,
            persist_directory="/tmp/chroma_db"
        )

    st.success("System ready! Ask your questions below 👇")

except Exception as e:
    st.error(f"Error during setup: {e}")
