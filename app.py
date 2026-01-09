import streamlit as st

from utilities.utils import AGENT_NAME

st.set_page_config(
    page_title=AGENT_NAME,
    page_icon="🤖",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/adiagarwalrock",
        "Report a bug": "https://github.com/adiagarwalrock/bardagent/issues",
        "About": """
### BardAgent
Created by **Aditya Agarwal**.

- [GitHub](https://github.com/adiagarwalrock)
- [LinkedIn](https://www.linkedin.com/in/adityaagarwal1999/)
- [Portfolio](https://adityaagarwal.me)
""",
    },
)

# Inject fixed footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: #808495;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 1000;
        border-top: 1px solid #262730;
    }
    .footer a {
        color: #808495;
        text-decoration: none;
    }
    .footer a:hover {
        color: #fff;
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        © 2026 <a href="https://adityaagarwal.me" target="_blank">Aditya Agarwal</a> | 
        <a href="https://github.com/adiagarwalrock" target="_blank">GitHub</a> | 
        <a href="https://www.linkedin.com/in/adityaagarwal1999/" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)

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
