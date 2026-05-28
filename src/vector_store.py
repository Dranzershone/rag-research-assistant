"""
vector_store.py — Embeddings + Vector DB (Qdrant / ChromaDB)
Supports dense search, hybrid search (dense + BM25), and cross-encoder re-ranking.
"""

from typing import List, Tuple
from langchain_core.documents import Document
from loguru import logger

from config import settings


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

def get_embeddings():
    """
    Return embedding model based on config.
    Default: HuggingFace BGE (free, runs locally, no API key needed).
    Switch to OpenAI in .env if you want cloud embeddings.
    """
    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info(f"Using HuggingFace embeddings: {settings.embedding_model}")
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    elif settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        logger.info(f"Using OpenAI embeddings: {settings.embedding_model}")
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}. Use: huggingface | openai")


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE — QDRANT
# ══════════════════════════════════════════════════════════════════════════════

class QdrantVectorStore:
    """
    Qdrant-based vector store with:
     - Dense vector search
     - Hybrid search (dense + BM25 sparse)
     - Cross-encoder re-ranking (Cohere)
    """

    def __init__(self):
        self.embeddings = get_embeddings()
        self.vectorstore = None
        self.collection_name = settings.qdrant_collection

    def build(self, chunks: List[Document], in_memory: bool = False) -> "QdrantVectorStore":
        """Create a new Qdrant collection from document chunks."""
        from langchain_community.vectorstores import Qdrant

        location = ":memory:" if in_memory else None
        url = None if in_memory else f"http://{settings.qdrant_host}:{settings.qdrant_port}"

        logger.info(f"Building Qdrant store: {len(chunks)} chunks → '{self.collection_name}'")

        self.vectorstore = Qdrant.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            location=location,
            url=url,
            collection_name=self.collection_name,
            force_recreate=True,
        )
        logger.info("Qdrant vector store ready.")
        return self

    def load(self) -> "QdrantVectorStore":
        """Connect to an existing Qdrant collection."""
        from langchain_community.vectorstores import Qdrant
        from qdrant_client import QdrantClient

        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.vectorstore = Qdrant(
            client=client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
        )
        logger.info(f"Connected to existing Qdrant collection '{self.collection_name}'")
        return self

    def dense_search(self, query: str, k: int = None) -> List[Document]:
        """Standard dense (vector similarity) search."""
        k = k or settings.retriever_k
        results = self.vectorstore.similarity_search(query, k=k)
        logger.debug(f"Dense search: '{query[:60]}' → {len(results)} results")
        return results

    def hybrid_search(self, query: str, k: int = None) -> List[Document]:
        """
        Hybrid search: combine dense vector search + BM25 sparse retrieval.
        Then merge using Reciprocal Rank Fusion (RRF).
        Significantly outperforms dense-only on keyword-heavy queries.
        """
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever

        k = k or settings.retriever_k

        dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # BM25 needs documents — fetch all stored docs first
        all_docs = self.vectorstore.similarity_search("", k=1000)
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = k

        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.4, 0.6],   # tune: BM25 weight vs dense weight
        )
        results = ensemble.get_relevant_documents(query)
        logger.debug(f"Hybrid search: '{query[:60]}' → {len(results)} results")
        return results

    def rerank(self, query: str, docs: List[Document], top_n: int = None) -> List[Document]:
        """
        Cross-encoder re-ranking using Cohere Rerank v3.
        Takes initial retrieval results and re-scores them for relevance.
        This alone can boost answer quality significantly.
        """
        import cohere

        top_n = top_n or settings.retriever_k
        co = cohere.Client(api_key=settings.cohere_api_key)

        texts = [d.page_content for d in docs]
        response = co.rerank(
            model=settings.reranker_model,
            query=query,
            documents=texts,
            top_n=top_n,
        )

        reranked = [docs[r.index] for r in response.results]
        logger.debug(f"Re-ranked {len(docs)} → top {len(reranked)} results")
        return reranked

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """
        Full retrieval pipeline:
        1. Hybrid search (or dense-only if disabled)
        2. Cross-encoder re-ranking (if enabled)
        Returns final top-k documents.
        """
        k = k or settings.retriever_k

        if settings.use_hybrid_search:
            docs = self.hybrid_search(query, k=k * 2)   # over-fetch for re-ranking
        else:
            docs = self.dense_search(query, k=k * 2)

        if settings.use_reranker and settings.cohere_api_key:
            docs = self.rerank(query, docs, top_n=k)
        else:
            docs = docs[:k]

        return docs

    def as_retriever(self, k: int = None):
        """LangChain-compatible retriever for use in chains."""
        return self.vectorstore.as_retriever(
            search_kwargs={"k": k or settings.retriever_k}
        )


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE — CHROMADB (easy local alternative)
# ══════════════════════════════════════════════════════════════════════════════

class ChromaVectorStore:
    """Chroma-based vector store — great for quick local prototyping."""

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.embeddings = get_embeddings()
        self.persist_dir = persist_dir
        self.vectorstore = None

    def build(self, chunks: List[Document]) -> "ChromaVectorStore":
        from langchain_community.vectorstores import Chroma
        logger.info(f"Building Chroma store: {len(chunks)} chunks → '{self.persist_dir}'")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        self.vectorstore.persist()
        logger.info("Chroma store saved to disk.")
        return self

    def load(self) -> "ChromaVectorStore":
        from langchain_community.vectorstores import Chroma
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )
        logger.info(f"Loaded Chroma store from '{self.persist_dir}'")
        return self

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        k = k or settings.retriever_k
        return self.vectorstore.similarity_search(query, k=k)

    def as_retriever(self, k: int = None):
        return self.vectorstore.as_retriever(
            search_kwargs={"k": k or settings.retriever_k}
        )


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY — pick your vector store in one line
# ══════════════════════════════════════════════════════════════════════════════

def get_vector_store(backend: str = "qdrant", **kwargs):
    """
    Usage:
        vs = get_vector_store("qdrant").build(chunks)
        vs = get_vector_store("chroma").build(chunks)
        vs = get_vector_store("qdrant").load()   # connect to existing
    """
    if backend == "qdrant":
        return QdrantVectorStore(**kwargs)
    elif backend == "chroma":
        return ChromaVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use: qdrant | chroma")
