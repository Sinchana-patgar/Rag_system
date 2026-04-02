import streamlit as st
import tempfile
import os
import sys

# ── STABILITY FIXES ──────────────────────────────────────────────────────────
# This prevents the "Descriptor cannot be created directly" Protobuf error
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

# ── PAGE CONFIG & UI STYLE ────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG System", 
    page_icon="🧠", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0d0f14; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1e2330; }
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
.rag-title { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; color: #38bdf8; letter-spacing: -0.5px; margin-bottom: 0; }
.rag-subtitle { font-size: 0.85rem; color: #475569; font-family: 'IBM Plex Mono', monospace; margin-bottom: 2rem; }
.chat-user { background: #1e293b; border: 1px solid #334155; border-radius: 12px 12px 2px 12px; padding: 14px 18px; margin: 10px 0; font-size: 0.95rem; color: #e2e8f0; }
.chat-bot { background: #0f1929; border: 1px solid #1e3a5f; border-left: 3px solid #38bdf8; border-radius: 2px 12px 12px 12px; padding: 14px 18px; margin: 10px 0; font-size: 0.95rem; color: #e2e8f0; line-height: 1.7; }
.chat-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; font-weight: 600; letter-spacing: 1px; margin-bottom: 6px; }
.label-user { color: #64748b; }
.label-bot { color: #38bdf8; }
.source-card { background: #111827; border: 1px solid #1e2a3a; border-radius: 8px; padding: 12px 16px; margin: 6px 0; font-size: 0.82rem; color: #94a3b8; font-family: 'IBM Plex Mono', monospace; line-height: 1.6; }
.source-header { font-size: 0.7rem; color: #475569; margin-bottom: 6px; letter-spacing: 0.5px; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR SETTINGS ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    groq_api_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...", help="Free key at console.groq.com")
    model_name = st.selectbox("🦙 Model", ["llama3-8b-8192", "llama3-70b-8192", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
    
    st.markdown("🔍 **Retrieval Settings**")
    top_k = st.slider("Retrieved chunks (k)", 1, 8, 3)
    chunk_size = st.slider("Chunk size", 200, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 100, 50)
    
    st.markdown("---")
    uploaded_files = st.file_uploader("📄 Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if st.button("🔄 Reset & Reload"):
        for k in ["docs", "embeddings", "vectorstore", "chat_history", "loaded_files"]:
            st.session_state.pop(k, None)
        st.rerun()

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="rag-title">🧠 RAG System</p>', unsafe_allow_html=True)
st.markdown('<p class="rag-subtitle">// Retrieval-Augmented Generation · Groq LLaMA 3 · PDF Knowledge Base</p>', unsafe_allow_html=True)

if not groq_api_key:
    st.info("👈 Enter your **Groq API key** in the sidebar to start.")
    st.stop()

if not uploaded_files:
    st.info("👈 Upload **PDF files** in the sidebar to begin indexing.")
    st.stop()

# ── RAG PIPELINE ───────────────────────────────────────────────────────────────
try:
    file_names = sorted([f.name for f in uploaded_files])

    # Check if files changed
    if st.session_state.get("loaded_files") != file_names:
        for k in ["docs", "embeddings", "vectorstore"]:
            st.session_state.pop(k, None)
        st.session_state.loaded_files = file_names

    # 1. Load and Split Documents
    if "docs" not in st.session_state:
        with st.spinner("📄 Analyzing PDFs..."):
            all_docs = []
            for uploaded in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                
                try:
                    loader = PyPDFLoader(tmp_path)
                    all_docs.extend(loader.load())
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.session_state.docs = splitter.split_documents(all_docs)

    # 2. Initialize Embeddings (Local HF Model)
    if "embeddings" not in st.session_state:
        with st.spinner("🔢 Loading local embedding model..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Build Vector Store
    if "vectorstore" not in st.session_state:
        with st.spinner("🗄️ Building vector database..."):
            st.session_state.vectorstore = Chroma.from_documents(
                documents=st.session_state.docs, 
                embedding=st.session_state.embeddings
            )

    # 4. Show Stats
    c1, c2, c3 = st.columns(3)
    c1.metric("📄 PDFs", len(uploaded_files))
    c2.metric("🧩 Chunks", len(st.session_state.docs))
    c3.metric("🦙 Model", model_name.split("-")[0].upper())
    st.markdown("---")

    # ── CHAT INTERFACE ──────────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display History
    for turn in st.session_state.chat_history:
        st.markdown(f'<div class="chat-user"><div class="chat-label label-user">YOU</div>{turn["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bot"><div class="chat-label label-bot">RAG · {turn["model"]}</div>{turn["answer"]}</div>', unsafe_allow_html=True)
        with st.expander(f"📎 Sources ({len(turn['sources'])} chunks)"):
            for i, src in enumerate(turn["sources"], 1):
                st.markdown(f'<div class="source-card"><div class="source-header">SOURCE {i} · Page {src.metadata.get("page", "?")}</div>{src.page_content}</div>', unsafe_allow_html=True)

    # User Input Form
    with st.form("query_form", clear_on_submit=True):
        query = st.text_input("Question", placeholder="What would you like to know?", label_visibility="collapsed")
        submitted = st.form_submit_button("⮐ Send Message")

    if submitted and query.strip():
        with st.spinner("🔍 Searching documents & generating answer..."):
            # Search
            results = st.session_state.vectorstore.similarity_search(query, k=top_k)
            context = "\n\n---\n\n".join([doc.page_content for doc in results])
            
            # Groq Generation
            client = Groq(api_key=groq_api_key)
            chat_completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a professional research assistant. Answer the question using ONLY the provided context. If the answer is not in the context, say: 'I cannot find that information in the uploaded documents.'"},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            answer = chat_completion.choices[0].message.content

            # Save to History
            st.session_state.chat_history.append({
                "question": query, 
                "answer": answer, 
                "sources": results, 
                "model": model_name
            })
            st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()

except Exception as e:
    st.error(f"⚠️ A system error occurred: {e}")
    st.info("Try resetting the application using the button in the sidebar.")
