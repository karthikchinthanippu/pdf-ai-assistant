from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import AIMessage, BaseMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import re
import os

from langchain_groq import ChatGroq


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "faiss_index"

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ---------------------------------------
# Load Local Embeddings + FAISS
# ---------------------------------------

embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

vectordb = FAISS.load_local(
    DB_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,
)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})


# ---------------------------------------
# Helpers: citations
# ---------------------------------------

def _get_page(d) -> str:
    """
    Try common metadata keys for page.
    PyPDFLoader usually uses 'page' (0-indexed).
    We'll display as 1-indexed when possible.
    """
    page = d.metadata.get("page", None)
    if isinstance(page, int):
        return str(page + 1)
    if page is None:
        return "unknown"
    return str(page)


def _get_source(d) -> str:
    """
    PyPDFLoader often sets metadata['source'] to the file path.
    We'll display just the filename if it's a path.
    """
    src = d.metadata.get("source", None)
    if not src:
        return "unknown_source"
    return os.path.basename(str(src))


def format_sources(docs) -> str:
    """
    Create a clean, de-duplicated Sources section like:
    Sources:
    - file.pdf (page 1)
    - file.pdf (page 2)
    """
    seen = set()
    lines = []
    for d in docs:
        src = _get_source(d)
        page = _get_page(d)
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {src} (page {page})")

    if not lines:
        return ""

    return "Sources:\n" + "\n".join(lines)


# ---------------------------------------
# Safe Retrieval Logic (LangChain 0.2+)
# ---------------------------------------

def retrieve_docs(query: str):
    """
    Normalize retriever output to list[Document].
    Some LC versions return a list, others return a dict with 'documents'.
    """
    result = retriever.invoke(query)

    if isinstance(result, dict) and "documents" in result:
        return result["documents"]

    if isinstance(result, list):
        return result

    raise ValueError(f"Unexpected retriever output: {type(result)}")


# ---------------------------------------
# Extraction Logic (EIN, Receipt, Invoice, Dates, etc.)
# NOTE: Issue 1 only -> we add citations to the response, but keep extraction unchanged.
# ---------------------------------------

def extract_fields(text: str):
    patterns = {
        "EIN": r"\b\d{2}-\d{7}\b",
        "Receipt Number": r"\b(?:Receipt|Receipt No\.?|Receipt Number)[:\s#]*([A-Za-z0-9\-]+)\b",
        "Invoice Number": r"\b(?:Invoice|Invoice No\.?|Invoice Number)[:\s#]*([A-Za-z0-9\-]+)\b",
        "Date": r"\b(?:\d{1,2}/\d{1,2}/\d{2,4})\b",
        "Amount": r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
    }

    extracted = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            extracted[field] = match.group(1) if match.lastindex else match.group(0)

    return extracted


# ---------------------------------------
# LangGraph State
# ---------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    mode: str  # "qa" or "summary"


# ---------------------------------------
# Router node
# ---------------------------------------

def router_node(state: AgentState):
    user_input = state["messages"][-1].content.lower()

    if any(word in user_input for word in ["summary", "summarize", "overview"]):
        return {"mode": "summary"}

    return {"mode": "qa"}


# ---------------------------------------
# QA Node — RAG + Extraction + LLM Reasoning (+ Citations)
# ---------------------------------------

def rag_qa_node(state: AgentState):
    user_query = state["messages"][-1].content
    docs = retrieve_docs(user_query)

    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # Extraction first (for numeric fields)
    extracted = extract_fields(context)

    lowered = user_query.lower()
    for field, value in extracted.items():
        if field.lower() in lowered:
            # Issue 1: add citations for extracted answers too
            sources = format_sources(docs)
            citation_line = sources if sources else "Sources: unknown"
            return {
                "messages": [
                    AIMessage(content=f"{field}: {value}\n\n{citation_line}")
                ]
            }

    # Otherwise use LLM for full reasoning
    llm_prompt = f"""
You are a helpful assistant. Use ONLY the following context to answer.

Context:
{context}

Question:
{user_query}

If the answer is not present in the context, say "Not found in document."
"""

    # Ollama returns a string; ChatGroq returns an AIMessage
    resp = llm.invoke(llm_prompt)
    answer_text = resp.content if hasattr(resp, "content") else str(resp)

    # Issue 1: append citations
    sources = format_sources(docs)
    if sources:
        answer_text = f"{answer_text}\n\n{sources}"

    return {"messages": [AIMessage(content=answer_text)]}


# ---------------------------------------
# Summary Node — Uses LLM (+ Citations)
# ---------------------------------------

def rag_summary_node(state: AgentState):
    user_query = state["messages"][-1].content
    docs = retrieve_docs(user_query)

    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = f"""
Provide a clear summary of the following document sections.
Use ONLY the provided text.

{context}
"""

    resp = llm.invoke(prompt)
    summary_text = resp.content if hasattr(resp, "content") else str(resp)

    sources = format_sources(docs)
    if sources:
        summary_text = f"{summary_text}\n\n{sources}"

    return {"messages": [AIMessage(content=summary_text)]}


# ---------------------------------------
# Routing Logic
# ---------------------------------------

def route(state: AgentState) -> str:
    return state["mode"]


# ---------------------------------------
# Build LangGraph App
# ---------------------------------------

def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag_qa", rag_qa_node)
    graph.add_node("rag_summary", rag_summary_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        route,
        {
            "qa": "rag_qa",
            "summary": "rag_summary",
        },
    )

    graph.add_edge("rag_qa", END)
    graph.add_edge("rag_summary", END)

    return graph.compile()
