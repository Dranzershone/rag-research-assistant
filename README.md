# 📚 RAG Research Assistant

A production-grade **Multi-Source Agentic RAG** pipeline built for resume showcase.  
Demonstrates: LangChain · LangGraph · Qdrant · RAGAS · FastAPI · Streamlit · Docker

---

## 🏗️ Architecture

```
Documents (PDF/Web/CSV/DOCX)
        ↓
   Chunking (Recursive / Token / Semantic)
        ↓
   Embeddings (OpenAI / BGE)
        ↓
   Vector Store (Qdrant / Chroma)
        ↓
   Hybrid Search (Dense + BM25)
        ↓
   Cross-Encoder Re-ranking (Cohere)
        ↓
   Agentic RAG (LangGraph ReAct)
        ↓
   Cited Answer + RAGAS Evaluation
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourusername/rag-research-assistant
cd rag-research-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add your Google API key — get one free at https://aistudio.google.com/apikey
# Edit .env and set: GOOGLE_API_KEY=AIza...
# That's the ONLY key you need to get started!
```

### 3. Run with Docker (recommended)

```bash
docker-compose up --build
# API:       http://localhost:8000/docs
# Streamlit: http://localhost:8501
```

### 4. Run locally (without Docker)

```bash
# Terminal 1 — start FastAPI
python src/api.py

# Terminal 2 — start Streamlit
streamlit run app.py
```

---

## 💻 CLI Usage

```bash
# Ingest documents
python main.py ingest --pdf paper.pdf --url https://en.wikipedia.org/wiki/RAG

# Ask a question (agentic mode)
python main.py query "What is RAG and how does it work?" --mode agentic

# Interactive chat
python main.py chat

# Run RAGAS evaluation
python main.py evaluate --eval-file evaluation/eval_dataset.json
```

---

## 📁 Project Structure

```
rag-research-assistant/
├── src/
│   ├── config.py            # Pydantic settings (reads .env)
│   ├── document_loader.py   # PDF/Web/CSV/DOCX loaders + 3 chunking strategies
│   ├── vector_store.py      # Qdrant + Chroma + hybrid search + re-ranking
│   ├── rag_chain.py         # BasicRAGChain + AgenticRAG (LangGraph)
│   ├── evaluation.py        # RAGAS metrics evaluation
│   └── api.py               # FastAPI REST backend
├── app.py                   # Streamlit frontend
├── main.py                  # CLI runner
├── tests/
│   └── test_pipeline.py     # Pytest unit tests
├── evaluation/
│   ├── eval_dataset.json    # Sample Q&A eval pairs
│   └── results.json         # Generated after evaluation
├── data/
│   ├── raw/                 # Drop your source documents here
│   └── processed/           # Intermediate outputs
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## 🔑 Key Features

| Feature | Details |
|---|---|
| **Multi-source ingestion** | PDF, DOCX, TXT, CSV, Web URLs, entire directories |
| **3 chunking strategies** | Recursive (default), Token-aware, Semantic |
| **Hybrid search** | Dense vectors + BM25 sparse, merged with RRF |
| **Cross-encoder re-ranking** | Cohere Rerank v3 |
| **Agentic RAG** | LangGraph ReAct: grades question, retrieves, grades docs, rewrites query if needed |
| **Conversation memory** | Sliding window (last 5 turns), question condensation |
| **RAGAS evaluation** | Faithfulness, Answer Relevancy, Context Precision, Context Recall |
| **Multi-LLM support** | OpenAI / Anthropic Claude / Google Gemini |
| **REST API** | FastAPI with Swagger docs at `/docs` |
| **Streamlit UI** | Chat interface with source citations + live eval scores |
| **Docker** | One-command deployment with Qdrant |

---

## 📊 Evaluation Metrics

| Metric | Description | Good score |
|---|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? | > 0.85 |
| **Answer Relevancy** | Does the answer address the question? | > 0.80 |
| **Context Precision** | Are retrieved chunks relevant? | > 0.75 |
| **Context Recall** | Is the ground truth info in the context? | > 0.70 |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System status |
| `POST` | `/ingest/upload` | Upload files |
| `POST` | `/ingest/urls` | Ingest URLs |
| `POST` | `/query` | Ask a question |
| `POST` | `/evaluate` | RAGAS eval for one Q&A pair |
| `DELETE` | `/reset` | Clear index + chat history |

Full interactive docs: `http://localhost:8000/docs`

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Swap LLM Provider

Edit `.env`:

```bash
# Google Gemini (default — free tier available)
LLM_PROVIDER=google
LLM_MODEL=gemini-2.5-flash        # fast + free tier
# LLM_MODEL=gemini-1.5-pro        # better quality

# OpenAI (optional)
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o-mini

# Anthropic Claude (optional)
# LLM_PROVIDER=anthropic
```

### Embeddings (no API key needed by default)

```bash
# HuggingFace BGE — free, runs locally (default)
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# OpenAI embeddings (optional, needs OPENAI_API_KEY)
# EMBEDDING_PROVIDER=openai
# EMBEDDING_MODEL=text-embedding-3-small
```

---

## 📄 License

MIT — free to use, modify, and showcase in your resume / portfolio.
