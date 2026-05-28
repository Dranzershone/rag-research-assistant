"""
rag_chain.py — RAG pipeline with LangChain + Agentic RAG with LangGraph
Compatible with LangChain 0.3.x / langchain-core 0.3.x

Two modes:
  1. BasicRAGChain — retrieve → generate with cited answer + chat history
  2. AgenticRAG    — LangGraph ReAct agent that grades, retrieves, rewrites, generates
"""

from typing import List, Dict, Any, Annotated, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from config import settings


# ══════════════════════════════════════════════════════════════════════════════
# LLM FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def get_llm():
    """Return LLM based on config provider. Defaults to Google Gemini."""
    if settings.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_tokens,
            google_api_key=settings.google_api_key,  # explicit — never falls back to ADC
        )
    elif settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            openai_api_key=settings.openai_api_key,
        )
    elif settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            anthropic_api_key=settings.anthropic_api_key,
        )
    else:
        raise ValueError(f"Unknown provider: {settings.llm_provider}. Use: google | openai | anthropic")


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

RAG_SYSTEM_PROMPT = """You are an expert research assistant. Answer the user's question
using ONLY the provided context. If the context does not contain enough information,
say so honestly — do not make up information.

Always cite which document or source your answer comes from.

Context:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and follow-up question, rephrase it as a "
               "standalone question that captures all context. Return only the question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Follow-up: {question}"),
])


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def format_docs(docs: List[Document]) -> str:
    """Format retrieved docs into a numbered context string with source labels."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("file_name") or doc.metadata.get("url", "unknown")
        page   = doc.metadata.get("page", "")
        label  = f"[Source {i}: {source}" + (f" page {page}" if page else "") + "]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASIC RAG CHAIN
# ══════════════════════════════════════════════════════════════════════════════

class BasicRAGChain:
    """
    Standard RAG: retrieve → format context → LLM → answer.
    Uses a plain list for chat history — no deprecated memory classes.
    """

    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = get_llm()
        self.chat_history: List[BaseMessage] = []   # simple list, no LangChain memory

    def _condense_question(self, question: str) -> str:
        """Rephrase follow-up questions as standalone using chat history."""
        if not self.chat_history:
            return question
        chain = CONDENSE_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"question": question, "chat_history": self.chat_history})

    def invoke(self, question: str, _chat_history=None) -> Dict[str, Any]:
        """Run the RAG chain. Returns {answer, source_documents, standalone_question}."""

        # Step 1 — condense with history
        standalone_q = self._condense_question(question)
        logger.debug(f"Standalone question: {standalone_q}")

        # Step 2 — retrieve
        source_docs = self.retriever.invoke(standalone_q)
        context = format_docs(source_docs)
        logger.info(f"Retrieved {len(source_docs)} chunks")

        # Step 3 — generate
        chain = RAG_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({
            "question": standalone_q,
            "context": context,
            "chat_history": self.chat_history[-10:],   # last 5 turns (10 messages)
        })

        # Step 4 — update history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "source_documents": source_docs,
            "standalone_question": standalone_q,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2. AGENTIC RAG — LangGraph ReAct
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """State that flows between LangGraph nodes."""
    messages:        List[BaseMessage]
    question:        str
    context:         str
    answer:          str
    sources:         List[Document]
    needs_retrieval: bool
    iterations:      int


class AgenticRAG:
    """
    ReAct-style agentic RAG using LangGraph.

    Flow:
      grade_question
          ↓ yes (needs docs)       ↓ no (conversational)
        retrieve               generate_direct
          ↓
        grade_documents
          ↓ relevant    ↓ not relevant (max 2 retries)
        generate        rewrite → retrieve
          ↓
         END
    """

    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = get_llm()
        self.graph = self._build_graph()

    # ── Nodes ──────────────────────────────────────────────────────────────

    def _grade_question(self, state: AgentState) -> AgentState:
        """Decide if this question needs document retrieval."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Does this question need document retrieval to answer?\n"
                       "Reply ONLY with 'yes' or 'no'.\n"
                       "'yes' → needs specific facts from documents\n"
                       "'no'  → conversational, greeting, or general knowledge"),
            ("human", "{question}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"question": state["question"]}).strip().lower()
        state["needs_retrieval"] = result.startswith("yes")
        logger.info(f"Grade question → needs_retrieval={state['needs_retrieval']}")
        return state

    def _retrieve(self, state: AgentState) -> AgentState:
        """Retrieve relevant document chunks."""
        docs = self.retriever.invoke(state["question"])
        state["sources"] = docs
        state["context"] = format_docs(docs)
        logger.info(f"Retrieved {len(docs)} chunks")
        return state

    def _grade_documents(self, state: AgentState) -> AgentState:
        """Check if retrieved documents are relevant to the question."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Are these documents relevant to the question? Reply ONLY 'yes' or 'no'."),
            ("human", "Question: {question}\n\nDocuments:\n{context}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "question": state["question"],
            "context":  state["context"][:2000],
        }).strip().lower()
        # if not relevant, flag for query rewrite
        state["needs_retrieval"] = not result.startswith("yes")
        logger.info(f"Grade docs → relevant={result.startswith('yes')}")
        return state

    def _rewrite_query(self, state: AgentState) -> AgentState:
        """Rewrite the query to improve retrieval quality."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the question to improve document retrieval. "
                       "Make it more specific with relevant keywords. Return only the question."),
            ("human", "Original: {question}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        new_q = chain.invoke({"question": state["question"]}).strip()
        logger.info(f"Rewrote query: '{state['question']}' → '{new_q}'")
        state["question"]   = new_q
        state["iterations"] = state.get("iterations", 0) + 1
        return state

    def _generate(self, state: AgentState) -> AgentState:
        """Generate answer using retrieved context."""
        chain = RAG_PROMPT | self.llm | StrOutputParser()
        state["answer"] = chain.invoke({
            "question":     state["question"],
            "context":      state["context"],
            "chat_history": state.get("messages", [])[-10:],
        })
        return state

    def _generate_direct(self, state: AgentState) -> AgentState:
        """Answer directly without retrieval for conversational questions."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer concisely and clearly."),
            ("human", "{question}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        state["answer"]  = chain.invoke({"question": state["question"]})
        state["sources"] = []
        return state

    # ── Routing ────────────────────────────────────────────────────────────

    def _route_after_grade(self, state: AgentState) -> str:
        return "retrieve" if state["needs_retrieval"] else "generate_direct"

    def _route_after_doc_grade(self, state: AgentState) -> str:
        if not state["needs_retrieval"]:
            return "generate"
        if state.get("iterations", 0) >= 2:
            logger.warning("Max retries reached — generating with available context")
            return "generate"
        return "rewrite"

    # ── Build graph ────────────────────────────────────────────────────────

    def _build_graph(self):
        from langgraph.graph import StateGraph, END

        g = StateGraph(AgentState)

        g.add_node("grade_question",  self._grade_question)
        g.add_node("retrieve",        self._retrieve)
        g.add_node("grade_documents", self._grade_documents)
        g.add_node("rewrite",         self._rewrite_query)
        g.add_node("generate",        self._generate)
        g.add_node("generate_direct", self._generate_direct)

        g.set_entry_point("grade_question")

        g.add_conditional_edges("grade_question", self._route_after_grade, {
            "retrieve":        "retrieve",
            "generate_direct": "generate_direct",
        })
        g.add_edge("retrieve", "grade_documents")
        g.add_conditional_edges("grade_documents", self._route_after_doc_grade, {
            "generate": "generate",
            "rewrite":  "rewrite",
        })
        g.add_edge("rewrite",         "retrieve")
        g.add_edge("generate",        END)
        g.add_edge("generate_direct", END)

        return g.compile()

    def invoke(self, question: str, chat_history: list = None) -> Dict[str, Any]:
        """Run the full agentic RAG pipeline."""
        initial: AgentState = {
            "messages":        chat_history or [],
            "question":        question,
            "context":         "",
            "answer":          "",
            "sources":         [],
            "needs_retrieval": True,
            "iterations":      0,
        }
        final = self.graph.invoke(initial)
        return {
            "answer":           final["answer"],
            "source_documents": final.get("sources", []),
            "question":         final["question"],
        }