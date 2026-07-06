# Simple RAG System — Question Answering Over Your Own Documents

A beginner-friendly Retrieval-Augmented Generation (RAG) pipeline. Point it at your
own PDFs, notes, resumes, or research papers, and ask questions grounded in that
content instead of the model's general knowledge.

## How it works

```
PDF/TXT files
      │
      ▼
1. Document Loading    (loader.py)      → extract raw text
2. Chunking             (chunker.py)     → split into overlapping ~500-word pieces
3. Embedding            (embedder.py)    → convert chunks to vectors (MiniLM)
4. Vector Store         (vectorstore.py) → store vectors in a FAISS index
      │
      ▼  user asks a question
5. Query Embedding      → embed the question the same way
6. Retrieval            → find top-k most similar chunks (cosine similarity)
7. Augmentation         → stitch retrieved chunks into a context block
8. Generation           (generator.py)   → LLM answers using ONLY that context
```

This mirrors the classic RAG architecture: **retrieve → augment → generate**,
so answers are grounded in your actual documents rather than hallucinated.

## Project structure

```
rag_system/
├── loader.py        # reads .pdf / .txt / .md files into raw text
├── chunker.py        # splits text into overlapping chunks
├── embedder.py        # sentence-transformers wrapper (all-MiniLM-L6-v2)
├── vectorstore.py     # FAISS-based similarity search + save/load
├── generator.py       # answer generation (Anthropic / OpenAI / local, auto-picked)
├── rag.py             # RAGPipeline class - wires all the above together
├── main.py            # command-line interface
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

No API key is required. If you don't set one, the system automatically falls
back to a small local model (`google/flan-t5-base`) that runs entirely on your
own machine.

**Optional — use a stronger hosted model instead:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # will use Claude
# or
export OPENAI_API_KEY="sk-..."          # will use GPT-4o-mini
```
The pipeline checks for these automatically — no code changes needed.

## Usage

**1. Ingest your documents** (build the searchable index):
```bash
python main.py ingest my_notes.pdf resume.pdf research_paper.txt
```
This loads each file, splits it into chunks, embeds them, and saves a FAISS
index to `rag_index/` so you don't have to re-embed every time.

**2. Ask a question:**
```bash
python main.py ask "What is the main contribution of this paper?"
```

**3. Or chat interactively:**
```bash
python main.py interactive
```

Each answer is printed along with the source chunks that were retrieved, so
you can verify exactly what the model based its answer on.

## Example

```
$ python main.py ingest resume.pdf
[Ingest] Loading resume.pdf
[Ingest] Split into 6 chunks
[Ingest] Embedding 6 chunks...
[Ingest] Done. Ready for queries.

$ python main.py ask "What programming languages does this person know?"
[Generator] No API key found - using local flan-t5-base model

Answer:
 Based on the document, the candidate has experience with Python, JavaScript, and SQL.

Sources used:
  [1] (score=0.812) Skills: Python, JavaScript, SQL, React, Docker, AWS...
  [2] (score=0.654) Experience: Built backend services in Python using FastAPI...
```

## Tuning knobs

In `rag.py`, `RAGPipeline(chunk_size=500, overlap=50, top_k=3)`:
- **chunk_size** — larger chunks give more context per retrieval, but dilute
  relevance; smaller chunks are more precise but may lose surrounding context.
- **overlap** — words shared between consecutive chunks, so answers that span
  a chunk boundary aren't lost.
- **top_k** — how many chunks are retrieved per question. Increase for
  broader questions ("summarize the document"), decrease for narrow factual ones.

## Ideas for extending this project

- **Hybrid search**: combine this vector search with keyword search (e.g. BM25)
  and merge the results for better recall.
- **Re-ranking**: after retrieving top-k chunks, re-score them with a
  cross-encoder model for more precise ordering.
- **Better chunking**: split on sentence/paragraph boundaries (e.g. with
  `nltk` or `spacy`) instead of raw word counts.
- **Different embedding models**: try `all-mpnet-base-v2` for higher quality,
  or a domain-specific embedding model for scientific/legal/medical text.
- **Multi-document metadata**: extend `metadata` to include page numbers, so
  answers can cite "page 4 of resume.pdf".
- **Swap vector stores**: try Chroma, Qdrant, or Pinecone for larger-scale or
  persistent, multi-user deployments.

## Key concepts recap

- **Retrieval** — find the most semantically relevant chunks using vector
  similarity search over embeddings.
- **Augmentation** — insert those chunks into the LLM's prompt as context.
- **Generation** — the LLM answers using that context, which keeps responses
  grounded in your actual data instead of relying purely on what it memorized
  during training.

This pattern is the backbone of most real-world "chat with your documents"
tools, knowledge-base assistants, and enterprise search systems.
