import os
import re
from pypdf import PdfReader

def _split_text(text, chunk_size=900, overlap=150):
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer ending a chunk at a sentence/space boundary.
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end), text.rfind(" ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def _read_pdf(file_obj, name):
    reader = PdfReader(file_obj)
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for piece in _split_text(text):
            chunks.append({
                "text": piece,
                "document": name,
                "page": page_number
            })
    return chunks

def load_uploaded_pdfs(uploaded_files):
    all_chunks = []
    names = []
    for file_obj in uploaded_files:
        names.append(file_obj.name)
        all_chunks.extend(_read_pdf(file_obj, file_obj.name))
    return all_chunks, names

def load_default_pdfs(folder="data"):
    all_chunks = []
    names = []
    if not os.path.isdir(folder):
        return all_chunks, names
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder, filename)
            try:
                with open(path, "rb") as pdf_file:
                    all_chunks.extend(_read_pdf(pdf_file, filename))
                names.append(filename)
            except Exception:
                # A single unreadable PDF should not prevent other documents loading.
                continue
    return all_chunks, names
