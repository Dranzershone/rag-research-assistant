"""
tests/test_pipeline.py — Unit tests for the RAG pipeline
Run: pytest tests/ -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain.schema import Document


# ── document_loader ────────────────────────────────────────────────────────────

def test_chunk_recursive_basic():
    from document_loader import chunk_recursive, enrich_metadata, deduplicate
    docs = [Document(page_content="Hello world. " * 100, metadata={"source": "test.txt"})]
    chunks = chunk_recursive(docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 110 for c in chunks)


def test_enrich_metadata():
    from document_loader import enrich_metadata
    docs = [Document(page_content="Test content", metadata={"source": "test.pdf", "page": 1})]
    enriched = enrich_metadata(docs)
    assert "chunk_id" in enriched[0].metadata
    assert "char_count" in enriched[0].metadata
    assert enriched[0].metadata["char_count"] == len("Test content")


def test_deduplicate_removes_exact_duplicates():
    from document_loader import deduplicate
    doc = Document(page_content="Duplicate content", metadata={})
    chunks = [doc, doc, doc]
    unique = deduplicate(chunks)
    assert len(unique) == 1


def test_deduplicate_keeps_unique():
    from document_loader import deduplicate
    chunks = [
        Document(page_content="First unique chunk", metadata={}),
        Document(page_content="Second unique chunk", metadata={}),
        Document(page_content="Third unique chunk", metadata={}),
    ]
    unique = deduplicate(chunks)
    assert len(unique) == 3


def test_chunk_by_tokens():
    from document_loader import chunk_by_tokens
    docs = [Document(page_content="Token chunking test. " * 50, metadata={})]
    chunks = chunk_by_tokens(docs, chunk_size=50, chunk_overlap=5)
    assert len(chunks) > 1


# ── format_docs ────────────────────────────────────────────────────────────────

def test_format_docs():
    from rag_chain import format_docs
    docs = [
        Document(page_content="Content one", metadata={"file_name": "doc1.pdf", "page": 1}),
        Document(page_content="Content two", metadata={"url": "https://example.com"}),
    ]
    formatted = format_docs(docs)
    assert "Source 1" in formatted
    assert "doc1.pdf" in formatted
    assert "Content one" in formatted
    assert "Content two" in formatted


# ── config ─────────────────────────────────────────────────────────────────────

def test_config_defaults():
    from config import Settings
    s = Settings()
    assert s.chunk_size == 512
    assert s.chunk_overlap == 64
    assert s.retriever_k == 5
    assert s.chunk_strategy == "recursive"
