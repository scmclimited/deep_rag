"""
Streamlit Frontend for Deep RAG
ChatGPT-like interface for document ingestion and querying
"""
import streamlit as st
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import os
from pathlib import Path
from api_client import DeepRAGClient

# Page config
st.set_page_config(
    page_title="Deep RAG Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Initialize API client
@st.cache_resource
def get_api_client():
    return DeepRAGClient(base_url=API_BASE_URL)

api_client = get_api_client()

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "threads" not in st.session_state:
    st.session_state.threads = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "streamlit_user"  # In production, get from auth
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


def check_health() -> bool:
    """Check if API is available"""
    try:
        return api_client.health_check()
    except Exception as e:
        st.error(f"Health check error: {str(e)}")
        return False


def get_threads() -> List[Dict]:
    """Get list of all threads for current user"""
    try:
        return api_client.get_threads(user_id=st.session_state.user_id)
    except Exception as e:
        st.error(f"Error getting threads: {str(e)}")
        return []


def get_documents() -> List[Dict]:
    """Get list of all documents"""
    try:
        return api_client.get_documents()
    except Exception as e:
        st.error(f"Error getting documents: {str(e)}")
        return []


def ingest_file(file, title: Optional[str] = None) -> Optional[Dict]:
    """Ingest a single file"""
    try:
        return api_client.ingest_file_obj(file, file.name, title)
    except Exception as e:
        st.error(f"Upload Error: {str(e)}")
        return None


def ingest_batch_files(files: List, titles: Optional[List[str]] = None) -> List[Dict]:
    """Ingest multiple files (batch)"""
    results = []
    for idx, file in enumerate(files):
        title = titles[idx] if titles and idx < len(titles) else None
        result = ingest_file(file, title)
        if result:
            results.append(result)
    return results


def send_message(question: str, attachment=None, doc_id: Optional[str] = None, 
                 cross_doc: bool = False) -> Optional[Dict]:
    """Send a message to the API"""
    try:
        if attachment:
            # Use infer-graph endpoint (ingest + query)
            title = Path(attachment.name).stem if attachment.name else None
            return api_client.infer_graph(
                question=question,
                thread_id=st.session_state.thread_id,
                file_obj=attachment,
                filename=attachment.name,
                title=title,
                cross_doc=cross_doc
            )
        else:
            # Use ask-graph endpoint (query only)
            return api_client.ask_graph(
                question=question,
                thread_id=st.session_state.thread_id,
                doc_id=doc_id,
                cross_doc=cross_doc
            )
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def create_new_thread():
    """Create a new thread"""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.uploaded_files = []
    st.rerun()


def load_thread(thread_id: str):
    """Load a thread's messages"""
    try:
        thread_data = api_client.get_thread(thread_id)
        if thread_data:
            st.session_state.thread_id = thread_id
            # Convert thread messages to session messages format
            st.session_state.messages = []
            for msg in thread_data.get("messages", []):
                st.session_state.messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("created_at", ""),
                    "doc_id": msg.get("doc_id")
                })
            st.rerun()
    except Exception as e:
        st.error(f"Error loading thread: {str(e)}")


# Sidebar
with st.sidebar:
    st.title("🧠 Deep RAG")
    
    # Health check
    if st.button("🔄 Check API"):
        if check_health():
            st.success("API is online")
        else:
            st.error("API is offline")
    
    st.divider()
    
    # Thread Management
    st.subheader("💬 Threads")
    if st.button("➕ New Thread", use_container_width=True):
        create_new_thread()
    
    st.caption(f"Current Thread: `{st.session_state.thread_id[:8]}...`")
    
    # Load existing threads (when endpoint is available)
    # threads = get_threads()
    # if threads:
    #     for thread in threads:
    #         if st.button(f"📝 {thread.get('title', thread['thread_id'][:8])}", 
    #                     key=f"thread_{thread['thread_id']}", use_container_width=True):
    #             load_thread(thread['thread_id'])
    
    st.divider()
    
    # Document Management
    st.subheader("📚 Documents")
    if st.button("🔄 Refresh Documents", use_container_width=True):
        st.session_state.documents = get_documents()
    
    documents = st.session_state.documents
    if documents:
        for doc in documents[:10]:  # Show first 10
            st.caption(f"📄 {doc.get('title', 'Untitled')}")
            if st.button("🗑️", key=f"del_{doc.get('doc_id')}", help="Delete document"):
                st.info("Delete functionality coming soon")
    else:
        st.caption("No documents found")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    cross_doc = st.checkbox("Cross-Document Search", value=False, 
                           help="Enable cross-document retrieval")
    selected_doc_id = st.selectbox(
        "Filter by Document",
        options=[None] + [doc.get('doc_id') for doc in documents],
        format_func=lambda x: "All Documents" if x is None else 
                             next((d.get('title', 'Untitled') for d in documents if d.get('doc_id') == x), x)
    )
    
    st.divider()
    
    # File Upload Section
    st.subheader("📤 Upload Files")
    uploaded_files = st.file_uploader(
        "Choose files to ingest",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or image files"
    )
    
    if uploaded_files:
        if st.button("📥 Ingest Files", use_container_width=True):
            with st.spinner("Ingesting files..."):
                results = ingest_batch_files(uploaded_files)
                if results:
                    st.success(f"✅ Ingested {len(results)} file(s)")
                    st.session_state.documents = get_documents()
                    st.session_state.uploaded_files = []
                    st.rerun()
                else:
                    st.error("Failed to ingest files")


# Main Chat Interface
st.title("💬 Deep RAG Chat")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("attachment"):
                st.caption(f"📎 {message['attachment']}")
            st.markdown(message["content"])
            if message.get("doc_id"):
                st.caption(f"📄 Document: {message['doc_id'][:8]}...")

# Chat input
if prompt := st.chat_input("Ask a question or upload a file..."):
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if there's a file in session state to attach
    attachment = None
    if st.session_state.uploaded_files:
        attachment = st.session_state.uploaded_files[0]
        st.session_state.uploaded_files = []
    
    # Get selected document ID
    doc_id = selected_doc_id if selected_doc_id else None
    
    # Send message and get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_message(
                prompt, 
                attachment=attachment,
                doc_id=doc_id,
                cross_doc=cross_doc
            )
            
            if response:
                answer = response.get("answer", "No answer received")
                st.markdown(answer)
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now().isoformat(),
                    "doc_id": response.get("doc_id"),
                    "thread_id": response.get("thread_id"),
                    "mode": response.get("mode"),
                    "pipeline": response.get("pipeline")
                })
                
                # Show metadata
                with st.expander("📊 Response Details"):
                    st.json(response)
            else:
                error_msg = "Sorry, I encountered an error processing your request."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat(),
                    "error": True
                })

# File upload in main area (alternative to sidebar)
with st.expander("📎 Attach File to Next Message"):
    file = st.file_uploader(
        "Upload a file to include with your next question",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
        help="The file will be ingested and used to answer your question"
    )
    if file:
        st.session_state.uploaded_files = [file]
        st.success(f"✅ {file.name} ready to attach")

# Footer
st.divider()
st.caption(f"Thread ID: `{st.session_state.thread_id}` | API: `{API_BASE_URL}`")

