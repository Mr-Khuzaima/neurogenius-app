import streamlit as st
import requests
import os
from typing import Dict, List

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# UI Setup
st.set_page_config(
    page_title="NeuroGenius",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background: #f8f9fa; }
.stChatMessage { border-radius: 15px; padding: 1.5rem; }
[data-testid="stChatMessage-user"] {
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    margin-left: 10%;
}
[data-testid="stChatMessage-assistant"] {
    background: white;
    margin-right: 10%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Session State
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_feature" not in st.session_state:
    st.session_state.active_feature = "chat"

# Sidebar
with st.sidebar:
    st.title("🧠 NeuroGenius")
    st.markdown("---")
    
    # Document Upload
    uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing..."):
            response = requests.post(
                f"{BACKEND_URL}/upload",
                files={"file": ("document.pdf", uploaded_file.getvalue())}
            )
            if response.status_code == 200:
                st.session_state.doc_id = response.json()["doc_id"]
                st.success("Document processed!")
    
    st.markdown("---")
    
    # Feature Selection
    st.subheader("✨ Features")
    features = {
        "💬 Chat": "chat",
        "🌐 Web Search": "search",
        "📝 Summarize": "summarize",
        "📊 Extract Tables": "tables",
        "🌍 Translate": "translate"
    }
    
    for display_name, feature_id in features.items():
        if st.button(display_name, key=feature_id, use_container_width=True):
            st.session_state.active_feature = feature_id
    
    st.markdown("---")
    
    # Voice Input
    st.subheader("🎙 Voice Input")
    audio_bytes = st.audio_recorder("Press to speak")
    if audio_bytes:
        with st.spinner("Transcribing..."):
            response = requests.post(
                f"{BACKEND_URL}/transcribe",
                files={"file": ("audio.wav", audio_bytes)}
            )
            if response.status_code == 200:
                st.session_state.voice_input = response.json()["text"]

# Main App
def main():
    st.title("Academic Assistant")
    
    if st.session_state.active_feature == "chat":
        render_chat()
    elif st.session_state.active_feature == "search":
        render_search()
    elif st.session_state.active_feature == "summarize":
        render_summarize()
    elif st.session_state.active_feature == "tables":
        render_tables()
    elif st.session_state.active_feature == "translate":
        render_translate()

def render_chat():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    prompt = st.chat_input("Ask something...") or getattr(st.session_state, "voice_input", "")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.doc_id:
                        response = requests.post(
                            f"{BACKEND_URL}/query",
                            json={"doc_id": st.session_state.doc_id, "question": prompt}
                        )
                        reply = response.json()["response"] if response.ok else f"Error: {response.text}"
                    else:
                        response = requests.post(
                            f"{BACKEND_URL}/search",
                            json={"query": prompt}
                        )
                        reply = f"🔍 {response.json()['results']}" if response.ok else f"Error: {response.text}"
                    
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def render_search():
    st.subheader("🌐 Web Search")
    query = st.text_input("Search query")
    
    if st.button("Search") or getattr(st.session_state, "voice_input", ""):
        query = query or st.session_state.voice_input
        if query:
            with st.spinner("Searching..."):
                response = requests.post(f"{BACKEND_URL}/search", json={"query": query})
                if response.ok:
                    st.markdown(response.json()["results"])
                else:
                    st.error("Search failed")

def render_summarize():
    st.subheader("📝 Document Summary")
    if not st.session_state.doc_id:
        st.warning("Upload a document first")
        return
    
    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            response = requests.post(
                f"{BACKEND_URL}/summarize",
                json={"doc_id": st.session_state.doc_id}
            )
            if response.ok:
                st.markdown(response.json()["summary"])
            else:
                st.error("Summary failed")

def render_tables():
    st.subheader("📊 Document Tables")
    if not st.session_state.doc_id:
        st.warning("Upload a document first")
        return
    
    if st.button("Extract Tables"):
        with st.spinner("Extracting..."):
            response = requests.post(
                f"{BACKEND_URL}/tables",
                json={"doc_id": st.session_state.doc_id}
            )
            if response.ok:
                tables = response.json()["tables"]
                for i, table in enumerate(tables, 1):
                    st.markdown(f"### Table {i}")
                    st.text(table["content"])
            else:
                st.error("Extraction failed")

def render_translate():
    st.subheader("🌍 Translation")
    col1, col2 = st.columns(2)
    with col1:
        text = st.text_area("Text to translate")
    with col2:
        lang = st.selectbox("Language", ["French", "Spanish", "German"])
    
    if st.button("Translate") or getattr(st.session_state, "voice_input", ""):
        text = text or st.session_state.voice_input
        if text:
            with st.spinner("Translating..."):
                response = requests.post(
                    f"{BACKEND_URL}/translate",
                    json={"text": text, "target_language": lang}
                )
                if response.ok:
                    st.markdown(response.json()["translation"])
                else:
                    st.error("Translation failed")

if _name_ == "_main_":
    main()
