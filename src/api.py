"""
api.py — FastAPI REST backend for the RAG Research Assistant
Endpoints: ingest, query, evaluate, health
"""

import os
import uuid
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from config import settings
from document_loader import build_chunks
from vector_store import get_vector_store
from rag_chain import BasicRAGChain, AgenticRAG
from evaluation import evaluate_single, print_scores

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Research Assistant API",
    description="Multi-source RAG pipeline with agentic retrieval and RAGAS evaluation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
vector_store = None
rag_chain = None
chat_history = []


# ── Pydantic models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    mode: str = "agentic"          # "basic" | "agentic"
    k: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    question: str

class IngestResponse(BaseModel):
    status: str
    chunks_created: int
    message: str

class EvalRequest(BaseModel):
    question: str
    answer: str
    ground_truth: Optional[str] = None
    source_texts: List[str] = []

class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
    llm_provider: str
    embedding_provider: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_chain(mode: str):
    global vector_store, rag_chain
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No documents ingested yet. POST /ingest first.")
    retriever = vector_store.as_retriever()
    if mode == "agentic":
        return AgenticRAG(retriever)
    return BasicRAGChain(retriever)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        vector_store_ready=vector_store is not None,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
    )


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_files(files: List[UploadFile] = File(...)):
    """Upload PDF/DOCX/TXT files and ingest them into the vector store."""
    global vector_store

    saved_paths = {"pdf": [], "docx": [], "text": []}
    tmp_dir = Path(tempfile.mkdtemp())

    try:
        for file in files:
            dest = tmp_dir / file.filename
            dest.write_bytes(await file.read())
            ext = dest.suffix.lower()
            if ext == ".pdf":    saved_paths["pdf"].append(str(dest))
            elif ext == ".docx": saved_paths["docx"].append(str(dest))
            else:                saved_paths["text"].append(str(dest))

        chunks = build_chunks(
            pdf_paths=saved_paths["pdf"] or None,
            docx_paths=saved_paths["docx"] or None,
            text_paths=saved_paths["text"] or None,
        )

        vector_store = get_vector_store("chroma", persist_dir="./chroma_db")
        vector_store.build(chunks)

        return IngestResponse(
            status="success",
            chunks_created=len(chunks),
            message=f"Ingested {len(files)} file(s) → {len(chunks)} chunks stored.",
        )

    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/urls", response_model=IngestResponse)
async def ingest_urls(urls: List[str]):
    """Ingest web pages by URL."""
    global vector_store

    try:
        chunks = build_chunks(web_urls=urls)
        vector_store = get_vector_store("chroma", persist_dir="./chroma_db")
        vector_store.build(chunks)

        return IngestResponse(
            status="success",
            chunks_created=len(chunks),
            message=f"Ingested {len(urls)} URL(s) → {len(chunks)} chunks stored.",
        )
    except Exception as e:
        logger.error(f"URL ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Ask a question. Returns answer + source documents."""
    global chat_history

    try:
        chain = get_chain(req.mode)
        result = chain.invoke(req.question, chat_history if req.mode == "agentic" else None)

        # Update chat history
        from langchain_core.messages import HumanMessage, AIMessage
        chat_history.append(HumanMessage(content=req.question))
        chat_history.append(AIMessage(content=result["answer"]))
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

        sources = [
            {
                "content": doc.page_content[:300],
                "source": doc.metadata.get("file_name") or doc.metadata.get("url", "unknown"),
                "page": doc.metadata.get("page", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
            }
            for doc in result.get("source_documents", [])
        ]

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            question=result.get("question", req.question),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate(req: EvalRequest):
    """Evaluate a single Q&A pair with RAGAS metrics."""
    try:
        from langchain_core.documents import Document
        fake_docs = [Document(page_content=t, metadata={}) for t in req.source_texts]
        scores = evaluate_single(
            question=req.question,
            answer=req.answer,
            source_docs=fake_docs,
            ground_truth=req.ground_truth,
        )
        return {"scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset")
async def reset():
    """Clear chat history and vector store."""
    global vector_store, rag_chain, chat_history
    vector_store = None
    rag_chain = None
    chat_history = []
    return {"status": "reset", "message": "Vector store and chat history cleared."}


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=settings.app_host, port=settings.app_port, reload=True)
