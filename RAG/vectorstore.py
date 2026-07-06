"""
vectorstore.py
A simple vector database built on FAISS for fast similarity search.

Vectors are L2-normalized so that inner product search behaves like
cosine similarity - the standard choice for semantic text search.
"""
import faiss
import numpy as np
import pickle
import os


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []
        self.metadata = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    def add(self, embeddings: np.ndarray, texts: list, metadata: list = None):
        """Add new chunk embeddings + their source text to the index."""
        embeddings = self._normalize(embeddings)
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        """Return the top_k most similar chunks to the query embedding."""
        query_embedding = self._normalize(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.texts[idx],
                "score": float(score),
                "metadata": self.metadata[idx],
            })
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "store.pkl"), "wb") as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata, "dim": self.dim}, f)

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, "store.pkl"), "rb") as f:
            data = pickle.load(f)
        store = cls(data["dim"])
        store.index = faiss.read_index(os.path.join(path, "index.faiss"))
        store.texts = data["texts"]
        store.metadata = data["metadata"]
        return store
