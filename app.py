import json
from datetime import datetime
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from main import run_chat

st.set_page_config(page_title="💬 BardAgent Chat", page_icon="💬", layout="centered")

st.title("💬 BardAgent Chat")
st.caption("Talk with a Gemini-powered LangChain agent.")

if "history" not in st.session_state:
    st.session_state.history: List[AnyMessage] = []
if "pending" not in st.session_state:
    st.session_state.pending = False

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []
        st.session_state.pending = False
        st.rerun()
    st.divider()

# Render chat history
for msg in st.session_state.history:
    timestamp = datetime.now().strftime("%H:%M")

    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(f"{msg.content}  \n*{timestamp}*")
    else:
        # parse structured output
        try:
            resp = msg.content
            parsed = json.loads(resp)
            typ = parsed.get("type", "text")
            data = parsed.get("data")
        except Exception:
            typ = "text"
            data = msg.content

        if typ == "text":
            st.chat_message("assistant").write(f"{data}  \n*{timestamp}*")

        elif typ == "markdown":
            st.chat_message("assistant").markdown(f"{data}  \n*{timestamp}*")

        elif typ == "table":
            st.chat_message("assistant").write(pd.DataFrame(data))

        elif typ == "dataframe":
            st.chat_message("assistant").dataframe(pd.DataFrame(data))

        elif typ == "chart":
            fig = px.line(x=data["x"], y=data["y"])
            st.chat_message("assistant").plotly_chart(fig)

        else:
            st.chat_message("assistant").write(f"{msg.content}  \n*{timestamp}*")

# Step 1: capture prompt
prompt: str = st.chat_input("Ask anything...", key="chat_input") or ""

if prompt:
    st.session_state.history.append(HumanMessage(content=prompt))
    st.session_state.pending = True
    st.rerun()

# Step 2: if pending, run the model
if st.session_state.pending:
    with st.spinner("Thinking..."):
        try:
            ai_msg: AIMessage = run_chat(st.session_state.history)
        except Exception as err:
            st.error(f"Model error: {err}")
            ai_msg = AIMessage(content="Sorry, I hit a model error.")
        st.session_state.history.append(ai_msg)

    st.session_state.pending = False
    st.rerun()
