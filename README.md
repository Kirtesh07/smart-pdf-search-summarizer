# Smart PDF Search & Summarizer — AICA Mini Project

A beginner-friendly Python app that extracts text from a PDF, makes an extractive summary, and searches for relevant passages.

## Features
- Upload a text-based PDF
- Extract selectable text and show page/word counts
- Generate an extractive summary using TF-IDF sentence scoring
- Search for relevant passages using TF-IDF and cosine similarity
- Display page numbers for search results
- Download the summary and extracted text

## Run on macOS
1. Extract this ZIP file.
2. Open the folder in VS Code.
3. Open Terminal in that folder and run:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

4. Open the local URL printed by Streamlit if the browser does not open automatically.

## Notes and limitations
- Works best with PDFs that contain selectable text.
- Scanned/image-only PDFs need OCR, which is not included in this starter version.
- Summarization is extractive: it selects sentences from the source rather than generating new prose.
- Search is based on TF-IDF and cosine similarity, so it ranks keyword overlap rather than understanding meaning.
- Avoid uploading confidential documents.

## Viva points
- **Libraries:** Streamlit, PyMuPDF, scikit-learn, pandas
- **TF-IDF:** Term Frequency–Inverse Document Frequency
- **Cosine similarity:** measures similarity between text vectors
- **Extractive summarization:** selects important source sentences
- **Future improvements:** OCR for scanned PDFs, semantic embeddings, downloadable PDF report, multi-PDF search.
