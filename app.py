import streamlit as st

from utilities.utils import AGENT_NAME

st.set_page_config(page_title=AGENT_NAME, page_icon="🤖", layout="wide")

with st.sidebar:
    st.title(f"🤖 {AGENT_NAME}")
    st.divider()

pg = st.navigation(
    [
        st.Page("UI/Home.py", title="Home", icon="🏠"),
        st.Page("UI/Evals.py", title="Evals", icon="📊"),
    ]
)

pg.run()
