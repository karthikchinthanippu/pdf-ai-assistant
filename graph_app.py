# graph_app.py

from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

import re


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"   # Change if needed
DB_PATH = "faiss_index"


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

llm = Ollama(model=LLM_MODEL)


# ---------------------------------------
# Safe Retrieval Logic (LangChain 0.2+)
# ---------------------------------------

def retrieve_docs(query):
    """Retrieves documents safely from retriever output."""
    result = retriever.invoke(query)

    if isinstance(result, dict) and "documents" in result:
        return result["documents"]

    if isinstance(result, list):
        return result

    raise ValueError(f"Unexpected retriever output: {type(result)}")


# ---------------------------------------
# Extraction Logic (EIN, Receipt, Invoice, Dates, etc.)
# ---------------------------------------

def extract_fields(text):
    """Extracts EIN, receipt numbers, invoice numbers, dates, etc."""

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
# QA Node — Full RAG + Extraction + LLM Reasoning
# ---------------------------------------

def rag_qa_node(state: AgentState):
    user_query = state["messages"][-1].content
    docs = retrieve_docs(user_query)

    context = "\n\n---\n\n".join([d.page_content for d in docs])

    # Extraction first (for numeric fields)
    extracted = extract_fields(context)

    # If user asks specifically for EIN, receipt number, etc.
    lowered = user_query.lower()
    for field, value in extracted.items():
        if field.lower() in lowered:
            return {"messages": [AIMessage(content=f"{field}: {value}")]}

    # Otherwise use LLM for full reasoning
    llm_prompt = f"""
You are a helpful assistant. Use ONLY the following context to answer:

Context:
{context}

Question:
{user_query}

Give a precise answer.
"""

    answer = llm.invoke(llm_prompt)

    return {"messages": [AIMessage(content=answer)]}


# ---------------------------------------
# Summary Node — Uses Local LLM
# ---------------------------------------

def rag_summary_node(state: AgentState):
    user_query = state["messages"][-1].content
    docs = retrieve_docs(user_query)

    context = "\n\n---\n\n".join([d.page_content for d in docs])

    prompt = f"""
Provide a clear summary of the following document sections:

{context}
"""

    summary = llm.invoke(prompt)

    return {"messages": [AIMessage(content=summary)]}


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
