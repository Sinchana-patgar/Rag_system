import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="🧠 A RAG system")
st.title("🧠 A RAG system")

try:
    # -------------------------
    # 1. Check Docs folder
    # -------------------------
    docs_folder = "Docs"

    if not os.path.exists(docs_folder):
        st.error("Docs folder not found. Please ensure a folder named 'Docs' exists in the repository.")
        st.stop()

    # -------------------------
    # 2. Load PDF documents
    # -------------------------
    if "docs" not in st.session_state:
        st.info("Loading PDF documents...")

        documents = []
        pdf_files = [f for f in os.listdir(docs_folder) if f.lower().endswith(".pdf")]

        if len(pdf_files) == 0:
            st.error("No PDF files found inside Docs folder.")
            st.stop()

        for file in pdf_files:
            try:
                loader = PyPDFLoader(os.path.join(docs_folder, file))
                documents.extend(loader.load())
            except Exception as pdf_error:
                st.error(f"Failed to load {file}: {pdf_error}")
                st.stop()

        # -------------------------
        # 3. Split documents
        # -------------------------
        st.info("Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        st.session_state.docs = splitter.split_documents(documents)

    # -------------------------
    # 4. Load embedding model
    # -------------------------
    if "embeddings" not in st.session_state:
        st.info("Loading embedding model...")

        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    # -------------------------
    # 5. Create vector database
    # -------------------------
    if "vectorstore" not in st.session_state:
        st.info("Creating vector database...")

        st.session_state.vectorstore = Chroma.from_documents(
            st.session_state.docs,
            st.session_state.embeddings,
            persist_directory="/tmp/chroma_db"
        )

    st.success("System ready! Ask your questions below 👇")

    # -------------------------
    # 6. Query interface
    # -------------------------
    query = st.text_input("Ask a question from the PDFs")

    if query:
        with st.spinner("Searching documents..."):
            results = st.session_state.vectorstore.similarity_search(query, k=3)

            st.subheader("Top relevant chunks:")
            for doc in results:
                st.write(doc.page_content)
                st.write("---")

except Exception as e:
    st.error(f"Fatal error: {e}")
