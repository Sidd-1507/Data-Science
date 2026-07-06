"""
loader.py
Loads raw text from PDF or plain text files.
"""
import os
from pypdf import PdfReader


def load_document(file_path: str) -> str:
    """Read a .pdf, .txt, or .md file and return its full text content."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf, .txt, or .md")
