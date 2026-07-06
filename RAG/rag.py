"""
rag.py
The main RAG pipeline: document ingestion, retrieval, and grounded answer generation.
"""
from loader import load_document
from chunker import chunk_text
from embedder import Embedder
from vectorstore import VectorStore
from generator import get_generator


class RAGPipeline:
    def __init__(self, chunk_size: int = 500, overlap: int = 50, top_k: int = 3):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.embedder = Embedder()
        self.generator = get_generator()
        self.vector_store = None

    def ingest(self, file_paths: list):
        """Load, chunk, embed, and store one or more documents."""
        all_chunks, all_metadata = [], []

        for path in file_paths:
            print(f"[Ingest] Loading {path}")
            text = load_document(path)
            chunks = chunk_text(text, self.chunk_size, self.overlap)
            print(f"[Ingest] Split into {len(chunks)} chunks")
            all_chunks.extend(chunks)
            all_metadata.extend([{"source": path} for _ in chunks])

        print(f"[Ingest] Embedding {len(all_chunks)} chunks...")
        embeddings = self.embedder.embed(all_chunks)

        self.vector_store = VectorStore(dim=embeddings.shape[1])
        self.vector_store.add(embeddings, all_chunks, all_metadata)
        print("[Ingest] Done. Ready for queries.")

    def query(self, question: str) -> dict:
        """Retrieve relevant chunks and generate a context-grounded answer."""
        if self.vector_store is None:
            raise RuntimeError("No documents ingested yet. Call ingest() first.")

        query_embedding = self.embedder.embed(question)
        results = self.vector_store.search(query_embedding, top_k=self.top_k)

        context = "\n\n---\n\n".join(r["text"] for r in results)
        answer = self.generator.generate(context, question)

        return {
            "question": question,
            "answer": answer,
            "sources": results,
        }

    def save(self, path: str = "rag_index"):
        self.vector_store.save(path)

    def load(self, path: str = "rag_index"):
        self.vector_store = VectorStore.load(path)
