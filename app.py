import os
import streamlit as st
from groq import Groq
from document_loader import load_uploaded_pdfs, load_default_pdfs
from rag_pipeline import build_index, retrieve_chunks, answer_question

st.set_page_config(page_title="IBS Department AI Assistant", page_icon="🎓", layout="wide")

st.title("🎓 IBS Department AI Assistant")
st.caption("Ask questions about department timetables, teachers, courses, rooms, and other information in the uploaded documents.")

def get_secret_api_key():
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return ""

with st.sidebar:
    st.header("Setup")
    secret_key = get_secret_api_key()
    api_key = secret_key or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Groq API key", type="password", help="For private testing only. On Streamlit Cloud, add GROQ_API_KEY under App settings → Secrets.")
    st.divider()
    st.subheader("Department documents")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload official timetable, course, teacher, or department information PDFs."
    )
    st.caption("For the first version, upload the relevant PDFs whenever you start a new session. PDFs are processed in memory and are not permanently stored by this app.")
    load_button = st.button("📚 Process documents", type="primary", use_container_width=True)

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "messages" not in st.session_state:
    st.session_state.messages = []

if load_button:
    if not uploaded_files:
        st.warning("Please upload at least one PDF first.")
    else:
        with st.spinner("Reading PDFs and building the search index. The first run may take a little longer while the embedding model downloads..."):
            try:
                chunks, names = load_uploaded_pdfs(uploaded_files)
                if not chunks:
                    st.error("No readable text was found. Check that the PDFs contain selectable text rather than only scanned images.")
                else:
                    index = build_index(chunks)
                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.doc_names = names
                    st.session_state.messages = []
                    st.success(f"Processed {len(names)} PDF(s) and created {len(chunks)} searchable text sections.")
            except Exception as exc:
                st.error(f"Could not process the PDFs: {exc}")

# Load bundled PDFs from data/ if no uploaded documents have been processed.
if st.session_state.index is None:
    default_chunks, default_names = load_default_pdfs("data")
    if default_chunks:
        with st.spinner("Preparing department documents included with the app..."):
            try:
                st.session_state.chunks = default_chunks
                st.session_state.index = build_index(default_chunks)
                st.session_state.doc_names = default_names
            except Exception as exc:
                st.warning(f"Could not prepare the default PDFs: {exc}")

if st.session_state.doc_names:
    st.info("Knowledge base: " + ", ".join(st.session_state.doc_names))
else:
    st.warning("No documents loaded yet. Upload your department PDFs in the left sidebar and click **Process documents**.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for source in msg["sources"]:
                    st.markdown(f"- **{source['document']}**, page {source['page']}")
                    st.caption(source["text"])

question = st.chat_input("Ask a question about the IBS Department...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key:
            response = "Please configure your Groq API key in Streamlit Secrets as `GROQ_API_KEY`, or enter it in the sidebar for a private test."
            st.warning(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        elif st.session_state.index is None or not st.session_state.chunks:
            response = "I don't have any department documents to search yet. Please upload official PDF documents in the sidebar and click **Process documents**."
            st.info(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            try:
                with st.spinner("Searching documents and preparing an answer..."):
                    retrieved = retrieve_chunks(
                        question,
                        st.session_state.index,
                        st.session_state.chunks,
                        top_k=5
                    )
                    client = Groq(api_key=api_key)
                    response = answer_question(client, question, retrieved)
                st.markdown(response)
                sources = []
                seen = set()
                for item in retrieved:
                    key = (item["document"], item["page"])
                    if key not in seen:
                        seen.add(key)
                        sources.append(item)
                if sources:
                    with st.expander("Sources used"):
                        for source in sources:
                            st.markdown(f"- **{source['document']}**, page {source['page']}")
                            st.caption(source["text"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.caption("Check your API key, internet connection, Groq model availability, and whether the PDFs contain readable text.")

st.divider()
st.caption("Important: Answers depend on the documents provided. Confirm official schedules and policy decisions with the IBS Department.")
