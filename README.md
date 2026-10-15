# IBS Department AI Assistant

A Retrieval-Augmented Generation (RAG) chatbot for the Institute of Business Studies (IBS). Students can ask questions about department information found in official PDF documents, such as timetables, teachers, subjects, rooms, and schedules.

## Technology
- Python
- Streamlit
- Groq API
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS vector search
- pypdf for PDF text extraction

## Features
- Upload one or more PDFs in the sidebar
- Search relevant text sections using embeddings and FAISS
- Generate answers using Groq
- Show source document names and page numbers
- Say when the documents do not contain enough information

## Run locally
Python 3.10 or 3.11 is recommended.

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment (optional but recommended).
4. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Set your API key as an environment variable, or enter it in the sidebar for local testing.

   Windows PowerShell:
   ```powershell
   $env:GROQ_API_KEY="your_groq_api_key"
   ```

6. Run:
   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud
1. Upload these project files to a GitHub repository.
2. Open Streamlit Community Cloud and create a new app from that repository.
3. Set the main file path to `app.py`.
4. In the app's settings, open **Secrets** and add:

   ```toml
   GROQ_API_KEY = "your_real_groq_api_key"
   ```

5. Save and deploy/reboot the app.

**Never commit your real API key to GitHub.** If a key has accidentally been published, revoke it and create a new one.

## Add official PDFs permanently
You can place approved PDFs inside the `data/` folder and commit them to the repository, provided you have permission and the documents are suitable for public access. The app will attempt to load PDFs from that folder automatically.

Alternatively, users can upload PDFs in the sidebar. Uploaded PDFs are processed in the current app session and are not permanently saved.

## Important limitations
- The first launch may take time because the sentence-transformer model must download.
- PDFs must contain selectable text. Scanned/image-only PDFs need OCR, which is not included in this first version.
- Timetable answers are only as accurate as the uploaded documents.
- The app does not authenticate users or provide an admin role yet. Do not upload private student records or confidential documents.
- Free hosting has resource and sleep limitations; large PDFs may exceed available memory or processing time.

## Suggested test questions
- When is Corporate Finance scheduled?
- Who teaches Financial Statement Analysis?
- Which room is assigned to the class?
- What subjects are listed for my semester?
- Is the exam date mentioned in these documents?
- What is the department contact information?

Use questions whose answers you can verify directly from the official PDFs.
