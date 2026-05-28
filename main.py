"""
main.py — CLI runner: ingest documents and test the full RAG pipeline
Usage:
    python main.py --ingest --pdf path/to/file.pdf --url https://example.com
    python main.py --query "What is RAG?" --mode agentic
    python main.py --evaluate
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from loguru import logger
from src.config import settings
from src.document_loader import build_chunks
from src.vector_store import get_vector_store
from src.rag_chain import BasicRAGChain, AgenticRAG
from src.evaluation import run_batch_evaluation, print_scores


def ingest(args):
    logger.info("=== INGESTION ===")
    chunks = build_chunks(
        pdf_paths=args.pdf or None,
        web_urls=args.url or None,
        docx_paths=args.docx or None,
        text_paths=args.text or None,
        directories=args.dir or None,
        strategy=args.chunk_strategy,
    )

    vs = get_vector_store("chroma", persist_dir="./chroma_db")
    vs.build(chunks)
    logger.info(f"Ingestion complete. {len(chunks)} chunks stored.")
    return vs


def query(args):
    logger.info("=== QUERY ===")
    vs = get_vector_store("chroma", persist_dir="./chroma_db").load()
    retriever = vs.as_retriever()

    if args.mode == "agentic":
        chain = AgenticRAG(retriever)
    else:
        chain = BasicRAGChain(retriever)

    result = chain.invoke(args.question)

    print("\n" + "═" * 60)
    print(f"Question: {args.question}")
    print("─" * 60)
    print(f"Answer:\n{result['answer']}")
    print("─" * 60)
    print(f"Sources ({len(result['source_documents'])}):")
    for i, doc in enumerate(result["source_documents"], 1):
        src = doc.metadata.get("file_name") or doc.metadata.get("url", "unknown")
        print(f"  [{i}] {src}  — {doc.page_content[:120]}...")
    print("═" * 60 + "\n")


def evaluate(args):
    logger.info("=== EVALUATION ===")
    vs = get_vector_store("chroma", persist_dir="./chroma_db").load()
    retriever = vs.as_retriever()
    chain = AgenticRAG(retriever)

    # Load eval dataset
    eval_file = args.eval_file or "evaluation/eval_dataset.json"
    with open(eval_file) as f:
        eval_data = json.load(f)

    report = run_batch_evaluation(chain, eval_data)
    print_scores(report["scores"])
    logger.info(f"Full report saved to evaluation/results.json")


def interactive(args):
    """Interactive chat loop in the terminal."""
    logger.info("=== INTERACTIVE MODE ===")
    vs = get_vector_store("chroma", persist_dir="./chroma_db").load()
    retriever = vs.as_retriever()
    chain = AgenticRAG(retriever) if args.mode == "agentic" else BasicRAGChain(retriever)

    print("\n📚 RAG Research Assistant (type 'exit' to quit)\n")
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break

        result = chain.invoke(question)
        print(f"\nAssistant: {result['answer']}\n")
        if result.get("source_documents"):
            print(f"[{len(result['source_documents'])} source(s) retrieved]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Research Assistant CLI")
    subparsers = parser.add_subparsers(dest="command")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--pdf",   nargs="+", help="PDF file paths")
    ingest_parser.add_argument("--url",   nargs="+", help="Web URLs")
    ingest_parser.add_argument("--docx",  nargs="+", help="DOCX file paths")
    ingest_parser.add_argument("--text",  nargs="+", help="TXT/MD file paths")
    ingest_parser.add_argument("--dir",   nargs="+", help="Directory paths")
    ingest_parser.add_argument("--chunk-strategy", default="recursive",
                               choices=["recursive", "token", "semantic"])

    # query
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="Your question")
    query_parser.add_argument("--mode", default="agentic", choices=["agentic", "basic"])

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run RAGAS evaluation")
    eval_parser.add_argument("--eval-file", help="Path to eval JSON dataset")

    # interactive
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument("--mode", default="agentic", choices=["agentic", "basic"])

    args = parser.parse_args()

    if args.command == "ingest":     ingest(args)
    elif args.command == "query":    query(args)
    elif args.command == "evaluate": evaluate(args)
    elif args.command == "chat":     interactive(args)
    else:
        parser.print_help()
