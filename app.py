import json
from datetime import datetime
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from main import run_chat

st.set_page_config(page_title="💬 BardAgent Chat", page_icon="💬", layout="centered")

st.title("💬 BardAgent Chat")
st.caption("Talk with a Gemini-powered LangChain agent.")

if "history" not in st.session_state:
    # Contains only HumanMessage and AIMessage entries in order
    st.session_state.history: List[AnyMessage] = []
if "pending" not in st.session_state:
    st.session_state.pending = False
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""
if "tool_calls" not in st.session_state:
    # Parallel list aligned to AI messages; each entry is list[str]
    st.session_state.tool_calls: List[List[str]] = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []
        st.session_state.pending = False
        st.rerun()
    st.divider()

# Render chat history (user + AI); attach tool calls to AI turns
ai_idx = 0
for msg in st.session_state.history:
    timestamp = datetime.now().strftime("%I:%M %p").lower()

    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(f"{msg.content}  \n*{timestamp}*")
    else:
        # parse structured output from AI; tolerate dicts or lists of dicts
        try:
            parsed = json.loads(msg.content)
        except Exception:
            parsed = msg.content

        payloads = parsed if isinstance(parsed, list) else [parsed]

        assistant_block = st.chat_message("assistant")

        for payload in payloads:
            if isinstance(payload, dict):
                typ = payload.get("type", "text")
                data = payload.get("text") or payload.get("data") or payload
            else:
                typ = "text"
                data = payload

            if typ == "text":
                assistant_block.write(f"{data}  \n*{timestamp}*")
            elif typ == "markdown":
                assistant_block.markdown(f"{data}  \n*{timestamp}*")
            elif typ == "table":
                assistant_block.write(pd.DataFrame(data))
            elif typ == "dataframe":
                assistant_block.dataframe(pd.DataFrame(data))
            elif typ == "chart":
                fig = px.line(x=data["x"], y=data["y"])
                assistant_block.plotly_chart(fig)
            else:
                assistant_block.write(f"{data}  \n*{timestamp}*")

        # Show tool calls for this AI turn, if any
        if ai_idx < len(st.session_state.tool_calls):
            tools = st.session_state.tool_calls[ai_idx]
            if tools:
                unique_tools = ", ".join(sorted(set(tools)))
                assistant_block.caption(f"Tools used: {unique_tools}")
        ai_idx += 1

# Step 1: capture prompt
prompt: str = st.chat_input("Ask anything...", key="chat_input") or ""

if prompt:
    st.session_state.pending_prompt = prompt
    st.session_state.pending = True
    st.rerun()

# Step 2: if pending, run the model
if st.session_state.pending:
    with st.spinner("Thinking..."):
        prompt = st.session_state.pending_prompt
        try:
            progress = run_chat(prompt, st.session_state.history)
        except Exception as err:
            st.error(f"Model error: {err}")
            progress = [AIMessage(content="Sorry, I hit a model error.")]

        # Persist only user and AI messages in history
        st.session_state.history.append(HumanMessage(content=prompt))
        ai_msg = next(
            (m for m in reversed(progress) if isinstance(m, AIMessage)), progress[-1]
        )
        st.session_state.history.append(ai_msg)

        # Record tool calls from this turn
        tool_calls = [m.name for m in progress if isinstance(m, ToolMessage)]
        st.session_state.tool_calls.append(tool_calls)

    st.session_state.pending = False
    st.session_state.pending_prompt = ""
    st.rerun()
