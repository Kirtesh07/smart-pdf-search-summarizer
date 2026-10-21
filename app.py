import re
from collections import Counter

import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Smart PDF Study Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Smart PDF Search & Summarizer")
st.write(
    "Upload a PDF to extract its text, create a concise extractive summary, "
    "and search for the most relevant passages using TF-IDF."
)
st.caption("Privacy note: this demo processes the uploaded PDF in the running app session. Avoid uploading confidential documents.")

def extract_pdf(uploaded_file):
    """Extract text page by page from a text-based PDF."""
    pdf_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_number, "text": text})
    doc.close()
    return pages

def split_sentences(text):
    """Simple sentence splitter suitable for a beginner project."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]

def summarize_text(text, sentence_count=5):
    """Extract the highest-scoring sentences using TF-IDF sentence vectors."""
    sentences = split_sentences(text)
    if len(sentences) <= sentence_count:
        return sentences
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(sentences)
        # Sentences with higher total TF-IDF weight tend to contain informative terms.
        scores = matrix.sum(axis=1).A1
        # Keep original document order after selecting the top-scoring sentences.
        chosen = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:sentence_count]
        chosen.sort()
        return [sentences[i] for i in chosen]
    except ValueError:
        return sentences[:sentence_count]

def split_passages(text, words_per_passage=110):
    words = text.split()
    passages = []
    for start in range(0, len(words), words_per_passage):
        passage = " ".join(words[start:start + words_per_passage])
        if passage.strip():
            passages.append(passage)
    return passages

def search_passages(pages, query, top_k=5):
    rows = []
    for page in pages:
        for passage in split_passages(page["text"]):
            rows.append({"page": page["page"], "passage": passage})
    if not rows:
        return []
    corpus = [row["passage"] for row in rows]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        doc_matrix = vectorizer.fit_transform(corpus)
        query_vector = vectorizer.transform([query])
    except ValueError:
        return []
    scores = cosine_similarity(query_vector, doc_matrix).ravel()
    ranked = scores.argsort()[::-1]
    results = []
    for idx in ranked:
        if scores[idx] <= 0:
            continue
        results.append({**rows[idx], "score": float(scores[idx])})
        if len(results) >= top_k:
            break
    return results

uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])

if uploaded:
    try:
        pages = extract_pdf(uploaded)
    except Exception as exc:
        st.error(f"Could not read this PDF: {exc}")
        st.stop()

    if not pages:
        st.warning(
            "No selectable text was found. This may be a scanned PDF. "
            "This beginner version does not include OCR; try a text-based PDF."
        )
        st.stop()

    full_text = "\n\n".join(page["text"] for page in pages)
    total_words = len(full_text.split())
    total_pages = len(pages)

    st.success(f"Extracted text from {total_pages} page(s).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pages with text", total_pages)
    c2.metric("Words extracted", f"{total_words:,}")
    c3.metric("Characters", f"{len(full_text):,}")

    tab_summary, tab_search, tab_text = st.tabs(["📝 Summary", "🔎 Search PDF", "📄 Extracted text"])

    with tab_summary:
        st.subheader("Create a concise summary")
        sentence_count = st.slider("Summary length (sentences)", min_value=3, max_value=12, value=5)
        if st.button("Generate summary", type="primary"):
            summary = summarize_text(full_text, sentence_count)
            if summary:
                st.session_state["pdf_summary"] = summary
            else:
                st.warning("There was not enough sentence-like text to summarize.")
        if st.session_state.get("pdf_summary"):
            summary_text = "\n\n".join(f"{i+1}. {sentence}" for i, sentence in enumerate(st.session_state["pdf_summary"]))
            st.markdown(summary_text)
            st.download_button(
                "Download summary as TXT",
                data=summary_text,
                file_name="pdf_summary.txt",
                mime="text/plain",
            )
        st.caption(
            "This is an extractive summary: it selects sentences from the original PDF. "
            "It does not write new sentences or guarantee that every key idea is included."
        )

    with tab_search:
        st.subheader("Find relevant passages")
        query = st.text_input("What do you want to find?", placeholder="e.g. Explain normalization in databases")
        top_k = st.selectbox("Number of results", [3, 5, 8], index=1)
        if st.button("Search PDF"):
            if not query.strip():
                st.warning("Enter a search question or keywords first.")
            else:
                results = search_passages(pages, query.strip(), top_k=top_k)
                st.session_state["pdf_search_results"] = results
                st.session_state["pdf_search_query"] = query.strip()
        if "pdf_search_results" in st.session_state:
            results = st.session_state["pdf_search_results"]
            if results:
                st.caption(f"Results for: {st.session_state.get('pdf_search_query', '')}")
                for i, result in enumerate(results, start=1):
                    with st.container(border=True):
                        st.markdown(f"**Result {i} · Page {result['page']} · Relevance {result['score']:.2f}**")
                        st.write(result["passage"])
            else:
                st.info("No relevant passage found. Try different keywords or a more specific question.")
        st.caption("Search uses TF-IDF and cosine similarity. It finds passages with overlapping important terms; it does not truly understand meaning.")

    with tab_text:
        st.subheader("Extracted PDF text")
        st.text_area("Text content", value=full_text, height=400)
        st.download_button(
            "Download extracted text",
            data=full_text,
            file_name="extracted_pdf_text.txt",
            mime="text/plain",
        )
else:
    st.info("Upload a PDF above to begin.")
    st.markdown(
        """
        **What this app can do**
        - Extract selectable text from PDF pages
        - Create a short extractive summary
        - Search for relevant passages and show page numbers
        - Download the summary or extracted text
        """
    )

st.divider()
with st.expander("AICA / viva notes"):
    st.markdown(
        """
        - **PDF text extraction:** PyMuPDF reads selectable text from each page.
        - **Preprocessing:** whitespace is normalized and text is split into sentences/passages.
        - **TF-IDF:** gives higher weights to terms that are informative within the document.
        - **Cosine similarity:** compares the query vector with passage vectors and ranks the closest matches.
        - **Summary method:** selects high-scoring sentences and returns them in the original order.
        - **Limitations:** scanned PDFs need OCR; the summarizer is extractive; keyword search can miss semantically related passages.
        """
    )
