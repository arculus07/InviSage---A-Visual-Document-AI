import os
import streamlit as st
import tempfile
import shutil
import fitz
from ingestion import DocumentLoader
from embeddings import EmbeddingManager

try:
    from langchain.chat_models import ChatOpenAI
except Exception:
    from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Visual Document RAG", layout="wide", page_icon="📄")

st.markdown("""
<style>
  body { background-color: #f5f7fa; }
  .main { background-color: #ffffff; padding: 20px; border-radius: 12px; }
  .scrollable-box { height: 300px; overflow-y: scroll; border: 1px solid #e6e8eb; padding: 12px; border-radius: 10px; background-color: #f9fafb; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.92rem; }
  .answer-box { border-left: 4px solid #6c63ff; background: #f3f2ff; padding: 16px; border-radius: 8px; }
  .pill { display:inline-block; padding:4px 10px; border-radius:999px; background:#eef2ff; margin-right:6px; font-size:12px; border:1px solid #dfe3ff; }
  .text-ocr { color: #1d4ed8; font-weight: 600; }
  .vision-analysis { color: #9333ea; font-weight: 600; }
  .vision-error { color: #dc2626; font-weight: 600; }
  .swatch { display:inline-block; width:14px; height:14px; border-radius:3px; border:1px solid #e5e7eb; margin-right:6px; vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

st.title("📄 InviSAGE")
st.caption("Upload → Extract (OCR + Vision) → Semantic Search → AI Answer with Sources")

# ---------------------- Initialize session state ----------------------
if "last_file_token" not in st.session_state:
    st.session_state.last_file_token = None
if "text_docs" not in st.session_state:
    st.session_state.text_docs = []
if "tables" not in st.session_state:
    st.session_state.tables = []
if "vision" not in st.session_state:
    st.session_state.vision = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_question" not in st.session_state:
    st.session_state.qa_question = ""
if "qa_answer" not in st.session_state:
    st.session_state.qa_answer = ""
if "db_path" not in st.session_state:
    st.session_state.db_path = None


def _reset_state():
    st.session_state.text_docs = []
    st.session_state.tables = []
    st.session_state.vision = []
    st.session_state.qa_question = ""
    st.session_state.qa_answer = ""
    st.session_state.vectorstore = None
    st.session_state.db_path = None


# ---------------------- Sidebar Title ----------------------
st.sidebar.markdown("""
<div style="text-align:center; margin-bottom:10px;">
    <h1 style="font-size:28px; margin:0;">Visual Document AI</h1>
    <p style="font-size:12px; color:#00bcd4; margin:0;">— A Visual Document AI by Ayush Ranjan</p>
</div>
""", unsafe_allow_html=True)

# ---------------------- Sidebar Upload ----------------------
st.sidebar.header("📂 Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or Image",
                                         type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"])

# ---------------------- About Section ----------------------
st.sidebar.header("ℹ About")
with st.sidebar.expander("About & Project", expanded=False):
    st.markdown("""
**Ayush Ranjan** 
- Email: [ranjanayush918@gmail.com](mailto:ranjanayush918@gmail.com)  
- LinkedIn: [Ayush Ranjan](https://www.linkedin.com/in/ayushxranjan/)  
- GitHub: [Ayush Ranjan](https://github.com/arculus07)

**InviSage — Visual Document AI**  
This project, *InviSage*, demonstrates a **Retrieval-Augmented Generation (RAG)** system that can process **text and visual content** in documents. The architecture and logic were developed independently, with external AI tools like Gemini, ChatGPT, Perplexity, and Grok used only for brainstorming and debugging.

**⚠ Important Disclaimer**  
This is a public demo using a personal OpenAI API key.  
1. Use images smaller than 500x500 pixels to minimize costs.  
2. Excessive or repeated use may result in key revocation to prevent financial strain.
""", unsafe_allow_html=True)

# ---------------------- File Upload Handling ----------------------
if uploaded_file:
    token = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.last_file_token != token:
        _reset_state()
        st.session_state.last_file_token = token

        loader = DocumentLoader()
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        if uploaded_file.type == "application/pdf":
            try:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    if len(doc) > 10:
                        st.error(f"⚠ PDF has {len(doc)} pages. Please upload a file with 10 pages or less.")
                        st.stop()
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                st.stop()
        uploaded_file.seek(0)

        with st.spinner("🔍 Analyzing document..."):
            try:
                st.session_state.db_path = tempfile.mkdtemp()
                result = loader.load_file(uploaded_file, uploaded_file.name)
                st.session_state.text_docs = result["text_docs"]
                st.session_state.tables = result["tables"]
                st.session_state.vision = result.get("vision", [])

                if st.session_state.text_docs:
                    embed_manager = EmbeddingManager()
                    st.session_state.vectorstore = embed_manager.build_vectorstore(
                        st.session_state.text_docs, st.session_state.db_path
                    )

                st.sidebar.success(f"✅ {len(st.session_state.text_docs)} page(s) processed")
                if st.session_state.tables:
                    st.sidebar.write(f"📊 {len(st.session_state.tables)} table(s) found")
                if st.session_state.vision:
                    st.sidebar.write(f"🖼 {len(st.session_state.vision)} visual(s) analyzed")

            except Exception as e:
                _reset_state()
                st.error(f"❌ Error during ingestion: {e}")

# ---------------------- Display Document Content ----------------------
if st.session_state.text_docs:
    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.subheader("📄 Document Text")
        for doc in st.session_state.text_docs:
            page_num = doc.metadata.get("page", 1)
            content = (doc.page_content
                       .replace("[TEXT OCR]", "<span class='text-ocr'>[TEXT OCR]</span>")
                       .replace("[VISION ANALYSIS]", "<span class='vision-analysis'>[VISION ANALYSIS]</span>")
                       .replace("[VISION ERROR]", "<span class='vision-error'>[VISION ERROR]</span>"))
            with st.expander(f"Page {page_num}", expanded=False):
                st.markdown(f"<div class='scrollable-box'>{content.strip().replace('\n', '<br>')}</div>",
                            unsafe_allow_html=True)
    with col2:
        if st.session_state.tables:
            st.subheader("📊 Extracted Tables")
            for i, df in enumerate(st.session_state.tables, start=1):
                st.markdown(f"<span class='pill'>Table {i}</span>", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)
        if st.session_state.vision:
            st.subheader("🖼 Visual Findings (Structured)")
            for i, item in enumerate(st.session_state.vision, start=1):
                page, fig = item.get("page", "?"), item.get("figure", "?")
                data = item.get("data", {}) or {}
                st.markdown(
                    f"Figure {fig} — Page {page}  \n*{data.get('chart_type', 'chart')}* — {data.get('title', '(no title)')}")
                slices = data.get("slices", [])
                if slices:
                    rows = []
                    for s in slices:
                        hexv = s.get("color", {}).get("hex")
                        swatch = f"<span class='swatch' style='background:{hexv or '#fff'}'></span>" if hexv else ""
                        rows.append([swatch + (s.get("color", {}).get("name") or ""), s.get("label"), s.get("percent"),
                                     s.get("value"), hexv])
                    st.markdown("| Color | Label | % | Value | Hex |\n|---|---:|---:|---:|---|\n" +
                                "\n".join([
                                              f"| {r[0]} | {r[1] or ''} | {r[2] if r[2] is not None else ''} | {r[3] if r[3] is not None else ''} | {r[4] or ''} |"
                                              for r in rows]),
                                unsafe_allow_html=True)
                st.markdown("---")

# ---------------------- RAG Q&A ----------------------
if st.session_state.vectorstore:
    st.subheader("🤖 Ask the Document (RAG)")

    st.text_input(
        "Your question",
        placeholder="e.g., Summarize this Text / CLICK ON GET ANSWER",
        key="qa_question"
    )

    col1_rag, col2_rag = st.columns([1, 4])
    with col1_rag:
        k = st.slider("Chunks to retrieve", 2, 6, 3)
    with col2_rag:
        if st.button("Get Answer", use_container_width=True):
            if st.session_state.qa_question:
                try:
                    with st.spinner("🧠 Thinking..."):
                        retrieved = st.session_state.vectorstore.similarity_search(st.session_state.qa_question, k=k)
                        context = "\n\n---\n\n".join(
                            f"[Chunk {i + 1} | Page {d.metadata.get('page', '?')}] {d.page_content.strip()}"
                            for i, d in enumerate(retrieved)
                        )
                        system = "Answer strictly from the context. If missing, say you can't find it. Cite page numbers."
                        prompt = f"{system}\n\nQ: {st.session_state.qa_question}\n\nContext:\n{context}\n\nAnswer:"

                        model = os.getenv("RAG_MODEL", "gpt-4o-mini")
                        llm = ChatOpenAI(model_name=model, temperature=0.2)
                        st.session_state.qa_answer = llm.predict(prompt)
                except Exception as e:
                    st.error(f"❌ RAG error: {e}")
            else:
                st.warning("Please enter a question.")

    if st.session_state.qa_answer:
        st.markdown("### 🧠 Answer")
        st.markdown(f"<div class='answer-box'>{st.session_state.qa_answer}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Stateless between uploads. Vision Model: gpt-4o-mini")
