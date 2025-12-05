# app.py

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from graph_app import build_app

st.set_page_config(page_title="PDF AI Assistant (Local LLM)", layout="wide")

st.title("📄 PDF AI Assistant (Local LLM + Extraction)")
st.write("Ask questions about EINs, receipt numbers, totals, dates, and more.")


if "app" not in st.session_state:
    st.session_state.app = build_app()

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)


user_msg = st.chat_input("Ask something...")

if user_msg:
    st.session_state.messages.append(HumanMessage(content=user_msg))
    st.chat_message("user").write(user_msg)

    result = st.session_state.app.invoke(
        {"messages": st.session_state.messages, "mode": "qa"}
    )

    reply = result["messages"][-1]
    st.session_state.messages.append(reply)

    st.chat_message("assistant").write(reply.content)
