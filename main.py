import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Page Config ---
st.set_page_config(page_title="RAG Intelligence", page_icon="🧠", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_base_content=True)

# --- Sidebar for Knowledge Base Management ---
with st.sidebar:
    st.title("📚 Knowledge Base")
    docs_folder = "Docs"
    
    if not os.path.exists(docs_folder):
        st.error(f"❌ '{docs_folder}' folder not found!")
        st.stop()
    
    pdf_files = [f for f in os.listdir(docs_folder) if f.lower().endswith(".pdf")]
    st.write(f"Found {len(pdf_files)} documents.")
    
    # Progress Indicators in Sidebar
    status_placeholder = st.empty()

# --- Backend Logic (Cached in session_state) ---
if "vectorstore" not in st.session_state:
    try:
        with st.sidebar:
            with st.status("Initializing Engine...", expanded=True) as status:
                st.write("Reading PDFs...")
                documents = []
                for file in pdf_files:
                    try:
                        file_path = os.path.join(docs_folder, file)
                        loader = PyPDFLoader(file_path)
                        documents.extend(loader.load())
                    except Exception as e:
                        st.warning(f"Skipping {file}: {e}")

                st.write("Splitting text...")
                splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                docs = splitter.split_documents(documents)

                st.write("Loading Embeddings...")
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                st.write("Building Vector Store...")
                st.session_state.vectorstore = Chroma.from_documents(
                    docs, embeddings, persist_directory="./chroma_db"
                )
                status.update(label="System Ready!", state="complete", expanded=False)
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

# --- Main Chat Interface ---
st.title("🧠 AI Doc Assistant")
st.caption("Ask questions based on your uploaded research papers and documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How does OneLake handle data fabric?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Similarity Search
            docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
            
            # Formulating a clean display (In a real app, you'd pass this to an LLM)
            if docs:
                response = f"**Top Result:**\n\n {docs[0].page_content}"
                st.markdown(response)
                
                with st.expander("View Source Chunks"):
                    for i, doc in enumerate(docs):
                        st.info(f"Chunk {i+1}: {doc.metadata.get('source', 'Unknown')}")
                        st.write(doc.page_content)
            else:
                response = "I couldn't find any relevant information in the documents."
                st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
