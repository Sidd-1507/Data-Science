"""
embedder.py
Wraps a sentence-transformers model to convert text into vector embeddings.

We use "all-MiniLM-L6-v2": small, fast, runs on CPU, and gives strong
semantic search quality for its size - a good default for a beginner RAG system.
"""
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        """Convert a string or list of strings into a numpy array of embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.astype("float32")
