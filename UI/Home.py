import json
from datetime import datetime, timezone
from typing import List
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from main import run_chat
from utilities.history import (
    clear_history,
    get_history_meta,
    read_history,
    save_chat_history,
)
from utilities.logger import logging
from utilities.utils import AGENT_NAME, random_title

# st.set_page_config is removed from here as it should be in the main app entry point or configured per page if supported/needed.
# But usually st.navigation handles the page context.
# However, for title and icon, we can still set it.
# Let's keep it for now, it might override the main one or just work.
# Actually, if this is a page, st.set_page_config might throw if called twice.
# I will comment it out and let app.py handle the global config or let this page handle it if it's the only one running.
# But for now, I'll comment it out to be safe and set it in app.py.

st.title(f"💬 {AGENT_NAME} Chat")
st.caption(f"Talk with a Gemini-powered LangChain agent: {AGENT_NAME}.")

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
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "session_title" not in st.session_state:
    st.session_state.session_title = random_title()


# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []
        st.session_state.tool_calls = []
        st.session_state.pending = False
        st.session_state.pending_prompt = ""
        st.session_state.session_id = str(uuid4())
        st.session_state.session_title = random_title()
        st.session_state.pop("history_id_select", None)
        st.rerun()
    if st.button("➕ New Chat"):
        st.session_state.session_id = str(uuid4())
        st.session_state.session_title = random_title()
        st.session_state.history = []
        st.session_state.tool_calls = []
        st.session_state.pending = False
        st.session_state.pending_prompt = ""
        st.session_state.pop("history_id_select", None)
        st.rerun()
    st.divider()
    st.header("📜 Chat History")
    history_meta = get_history_meta()
    history_options = [("", "— new chat —")] + [
        (item["id"], item["title"]) for item in history_meta
    ]
    selected_history = st.selectbox(
        "Select past chat session",
        history_options,
        index=0,
        key="history_id_select",
        help="Select a previous chat session to load its history.",
        format_func=lambda opt: opt[1],
    )

# Load selected history into the chat window
if selected_history and selected_history[0]:
    sel_id, sel_title = selected_history
    if sel_id != st.session_state.get("session_id"):
        st.session_state.session_id = sel_id
        st.session_state.session_title = sel_title
        st.session_state.history = read_history(sel_id)
        # tool_calls cannot be reconstructed; align empty entries to AI messages
        ai_count = sum(1 for m in st.session_state.history if isinstance(m, AIMessage))
        st.session_state.tool_calls = [[] for _ in range(ai_count)]
        st.session_state.pending = False
        st.session_state.pending_prompt = ""
        st.rerun()


# Render chat history (user + AI); attach tool calls to AI turns
ai_idx = 0
for msg in st.session_state.history:
    sent_ts = (
        getattr(msg, "additional_kwargs", {}).get("sent_time")
        or datetime.now().isoformat()
    )
    timestamp = (
        datetime.fromisoformat(sent_ts.replace("Z", "+00:00"))
        .strftime("%I:%M %p")
        .lower()
    )

    if isinstance(msg, HumanMessage):
        block = st.chat_message("user")
        block.write(msg.content)
        block.caption(timestamp)
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
                assistant_block.write(data)
            elif typ == "markdown":
                assistant_block.markdown(data)
            elif typ == "table":
                assistant_block.write(pd.DataFrame(data))
            elif typ == "dataframe":
                assistant_block.dataframe(pd.DataFrame(data))
            elif typ == "chart":
                fig = px.line(x=data["x"], y=data["y"])
                assistant_block.plotly_chart(fig)
            else:
                assistant_block.write(data)

        # Show tool calls for this AI turn, if any
        if ai_idx < len(st.session_state.tool_calls):
            tools = st.session_state.tool_calls[ai_idx]
            if tools:
                badges = " ".join(
                    f":blue-badge[{t.replace('_', ' ').title()}]"
                    for t in sorted(set(tools))
                )
                assistant_block.markdown(
                    f"Tools used: {badges}", unsafe_allow_html=False
                )
        assistant_block.caption(timestamp)
        ai_idx += 1

# Step 1: capture prompt
prompt_raw = st.chat_input(
    "Ask anything...",
    key="chat_input",
    accept_file=True,
    file_type=["pdf", "txt", "docx", "md"],
)
# Streamlit may return a ChatInputValue; safely extract text without touching __repr__
if prompt_raw is None:
    prompt: str = ""
elif isinstance(prompt_raw, str):
    prompt = prompt_raw
else:
    prompt = (
        getattr(prompt_raw, "text", None) or getattr(prompt_raw, "message", "") or ""
    )

if prompt:
    st.session_state.pending_prompt = prompt
    st.session_state.pending = True
    # Show the user message immediately before running the model
    st.session_state.history.append(
        HumanMessage(
            content=prompt,
            additional_kwargs={"sent_time": datetime.now(timezone.utc).isoformat()},
        )
    )
    st.rerun()

# Step 2: if pending, run the model
if st.session_state.pending:
    prompt = str(st.session_state.pending_prompt)
    tool_calls = []
    with st.status("Thinking...", state="running", expanded=True) as status:
        with st.spinner("Contacting model..."):
            try:
                progress, tool_calls = run_chat(prompt, st.session_state.history)
            except Exception as err:
                st.error(f"Model error: {err}")
                progress = [AIMessage(content="Sorry, I hit a model error.")]

        ai_msg = next(
            (m for m in reversed(progress) if isinstance(m, AIMessage)), progress[-1]
        )
        st.session_state.history.append(ai_msg)

        # Record tool calls from this AI turn
        st.session_state.tool_calls.append(set([ttl["name"] for ttl in tool_calls]))

        # Persist history to disk (replace existing session content)
        clear_history(st.session_state.session_id)
        save_chat_history(
            st.session_state.session_id,
            st.session_state.history,
            title=st.session_state.session_title,
        )
        status.update(label="Response ready", state="complete", expanded=False)

    st.session_state.pending = False
    st.session_state.pending_prompt = ""
    st.rerun()
