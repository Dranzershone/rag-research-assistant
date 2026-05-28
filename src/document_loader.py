"""
document_loader.py — Multi-source document loading + 3 chunking strategies
Sources: PDF, Web, CSV, DOCX, plain text, entire directories
"""

import hashlib
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from loguru import logger

from config import settings


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_pdf(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import PyMuPDFLoader
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({"source_type": "pdf", "file_name": Path(file_path).name})
    logger.info(f"[PDF] {len(docs)} pages ← {file_path}")
    return docs


def load_web(url: str) -> List[Document]:
    import bs4
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs={"parse_only": bs4.SoupStrainer(["p", "h1", "h2", "h3", "article", "section"])},
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({"source_type": "web", "url": url})
    logger.info(f"[WEB] {len(docs)} doc(s) ← {url}")
    return docs


def load_csv(file_path: str, content_col: str, metadata_cols: List[str] = None) -> List[Document]:
    from langchain_community.document_loaders.csv_loader import CSVLoader
    loader = CSVLoader(
        file_path=file_path,
        source_column=content_col,
        metadata_columns=metadata_cols or [],
        encoding="utf-8",
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({"source_type": "csv", "file_name": Path(file_path).name})
    logger.info(f"[CSV] {len(docs)} rows ← {file_path}")
    return docs


def load_docx(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({"source_type": "docx", "file_name": Path(file_path).name})
    logger.info(f"[DOCX] {len(docs)} doc(s) ← {file_path}")
    return docs


def load_text(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({"source_type": "text", "file_name": Path(file_path).name})
    logger.info(f"[TEXT] {len(docs)} doc(s) ← {file_path}")
    return docs


def load_directory(dir_path: str) -> List[Document]:
    from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader, Docx2txtLoader
    all_docs: List[Document] = []
    loaders_map = {".pdf": PyMuPDFLoader, ".txt": TextLoader, ".md": TextLoader, ".docx": Docx2txtLoader}
    for ext, cls in loaders_map.items():
        loader = DirectoryLoader(dir_path, glob=f"**/*{ext}", loader_cls=cls, show_progress=True)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_type"] = ext.lstrip(".")
        all_docs.extend(docs)
    logger.info(f"[DIR] {len(all_docs)} total docs ← {dir_path}")
    return all_docs


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def chunk_recursive(docs: List[Document], chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
    """Default strategy — splits on paragraphs → sentences → words → chars."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"[CHUNK:recursive] {len(docs)} docs → {len(chunks)} chunks")
    return chunks


def chunk_by_tokens(docs: List[Document], chunk_size: int = 256, chunk_overlap: int = 32) -> List[Document]:
    """Token-aware chunking — use when fitting tight LLM context windows."""
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"[CHUNK:token] {len(docs)} docs → {len(chunks)} chunks")
    return chunks


def chunk_semantic(docs: List[Document], threshold: float = 95.0) -> List[Document]:
    """Semantic chunking — splits on topic boundaries using embeddings."""
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"[CHUNK:semantic] {len(docs)} docs → {len(chunks)} chunks")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def enrich_metadata(chunks: List[Document]) -> List[Document]:
    for i, chunk in enumerate(chunks):
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
        source = chunk.metadata.get("source", "unknown")
        page   = chunk.metadata.get("page", i)
        chunk.metadata["chunk_id"]    = f"{Path(str(source)).stem}_p{page}_{content_hash}"
        chunk.metadata["chunk_index"] = i
        chunk.metadata["char_count"]  = len(chunk.page_content)
    return chunks


def deduplicate(chunks: List[Document]) -> List[Document]:
    seen, unique = set(), []
    for chunk in chunks:
        h = hashlib.md5(" ".join(chunk.page_content.split()).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    removed = len(chunks) - len(unique)
    if removed:
        logger.info(f"[DEDUP] Removed {removed} duplicates → {len(unique)} chunks remain")
    return unique


def print_stats(chunks: List[Document]) -> None:
    lengths = [len(c.page_content) for c in chunks]
    logger.info(
        f"Chunk stats | total={len(chunks)} | "
        f"min={min(lengths)} | max={max(lengths)} | avg={sum(lengths)//len(lengths)} chars"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def build_chunks(
    pdf_paths:   List[str] = None,
    web_urls:    List[str] = None,
    csv_paths:   List[Tuple[str, str]] = None,   # (file_path, content_col)
    docx_paths:  List[str] = None,
    text_paths:  List[str] = None,
    directories: List[str] = None,
    strategy:    str = None,
) -> List[Document]:
    """
    One-call pipeline: load all sources → chunk → enrich metadata → deduplicate.
    Returns chunks ready for embedding + vector storage.
    """
    all_docs: List[Document] = []

    for p in (pdf_paths or []):    all_docs.extend(load_pdf(p))
    for u in (web_urls or []):     all_docs.extend(load_web(u))
    for e in (csv_paths or []):    all_docs.extend(load_csv(*e))
    for p in (docx_paths or []):   all_docs.extend(load_docx(p))
    for p in (text_paths or []):   all_docs.extend(load_text(p))
    for d in (directories or []):  all_docs.extend(load_directory(d))

    if not all_docs:
        raise ValueError("No documents loaded — check your paths/URLs.")

    logger.info(f"Total raw docs: {len(all_docs)}")

    strat = strategy or settings.chunk_strategy
    if strat == "recursive":   chunks = chunk_recursive(all_docs)
    elif strat == "token":     chunks = chunk_by_tokens(all_docs)
    elif strat == "semantic":  chunks = chunk_semantic(all_docs)
    else:
        raise ValueError(f"Unknown strategy '{strat}'. Use: recursive | token | semantic")

    chunks = enrich_metadata(chunks)
    chunks = deduplicate(chunks)
    print_stats(chunks)
    return chunks
