"""
chunker.py
Splits long text into overlapping chunks for better retrieval accuracy.

Why overlap? Important sentences can fall right at a chunk boundary.
A small overlap ensures that context isn't lost between chunks.
"""
import re


def clean_text(text: str) -> str:
    """Collapse extra whitespace/newlines into single spaces."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into chunks of roughly `chunk_size` words each,
    with `overlap` words shared between consecutive chunks.
    """
    text = clean_text(text)
    words = text.split(" ")
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap  # step forward while keeping overlap

    return chunks
