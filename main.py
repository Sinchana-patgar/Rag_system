import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline


st.set_page_config(page_title="Rag-system")

st.title("🧠 A RAG system")

try:
    # -------------------------
    # 1. Load and split PDFs
    # -------------------------
    if "docs" not in st.session_state:
        st.info("Loading PDF documents...")

        documents = []
        docs_folder = "Docs"

        for file in os.listdir(docs_folder):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(docs_folder, file))
                documents.extend(loader.load())

        st.info("Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        st.session_state.docs = splitter.split_documents(documents)

    # -------------------------
    # 2. Load embedding model
    # -------------------------
    if "embeddings" not in st.session_state:
        st.info("Loading embedding model...")
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    # -------------------------
    # 3. Create vector database
    # -------------------------
    if "vectorstore" not in st.session_state:
        st.info("Creating vector database...")
        st.session_state.vectorstore = Chroma.from_documents(
            st.session_state.docs,
            st.session_state.embeddings,
            persist_directory="/tmp/chroma_db"
        )

    # -------------------------
    # 4. Create retriever
    # -------------------------
    retriever = st.session_state.vectorstore.as_retriever()

    st.success("System ready! Ask your questions below 👇")

    # -------------------------
    # 5. User query interface
    # -------------------------
    query = st.text_input("Ask a question from the PDFs")

    if query:
        with st.spinner("Searching documents..."):
            results = retriever.get_relevant_documents(query)

            st.subheader("Relevant document chunks:")
            for doc in results:
                st.write(doc.page_content)
                st.write("---")

except Exception as e:
    st.error(f"Error during setup: {e}")
