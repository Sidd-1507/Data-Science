"""
main.py
Command-line interface for the RAG system.

Usage:
    python main.py ingest doc1.pdf doc2.txt      # build the vector index
    python main.py ask "What is the main idea?"  # ask a one-off question
    python main.py interactive                    # chat-style loop
"""
import sys
from rag import RAGPipeline

INDEX_PATH = "rag_index"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]
    pipeline = RAGPipeline()

    if command == "ingest":
        files = sys.argv[2:]
        if not files:
            print("Please provide at least one file to ingest.")
            return
        pipeline.ingest(files)
        pipeline.save(INDEX_PATH)

    elif command == "ask":
        question = " ".join(sys.argv[2:])
        if not question:
            print("Please provide a question.")
            return
        pipeline.load(INDEX_PATH)
        result = pipeline.query(question)
        print("\nAnswer:\n", result["answer"])
        print("\nSources used:")
        for i, src in enumerate(result["sources"], 1):
            preview = src["text"][:150].replace("\n", " ")
            print(f"  [{i}] (score={src['score']:.3f}) {preview}...")

    elif command == "interactive":
        pipeline.load(INDEX_PATH)
        print("Type your questions (or 'exit' to quit):")
        while True:
            q = input("\n> ")
            if q.lower() in ("exit", "quit"):
                break
            result = pipeline.query(q)
            print("Answer:", result["answer"])

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
