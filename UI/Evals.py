import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from evals.run_eval import run_evaluations
from utilities.utils import AGENT_NAME

st.title("📊 Evaluation Results")
st.caption(f"Run and view evaluations for the agent: {AGENT_NAME}")

EVAL_RESULT_FILE = "eval_result.json"
DATASET_FILE = "evals/dataset.jsonl"

with st.sidebar:
    st.header("Configuration")
    max_workers = st.slider("Max Workers", min_value=1, max_value=16, value=8)

    if st.button("🚀 Run Evaluations"):
        st.session_state.running_evals = True

    with st.expander("ℹ️ About", expanded=False):
        st.markdown(
            """
            Created by **Aditya Agarwal**.
            
            - [GitHub](https://github.com/adiagarwalrock)
            - [LinkedIn](https://www.linkedin.com/in/adityaagarwal1999/)
            - [Portfolio](https://adityaagarwal.me)
            """
        )


if st.session_state.get("running_evals", False):
    progress_bar = st.progress(0, text="Starting evaluations...")
    status_text = st.empty()

    def update_progress(current, total):
        pct = current / total
        progress_bar.progress(pct, text=f"Evaluated {current}/{total}")

    try:
        run_evaluations(DATASET_FILE, max_workers, progress_callback=update_progress)
        status_text.success("Evaluations complete!")
        # Clear the flag so we don't re-run on simple interactions, but keep results shown
        st.session_state.running_evals = False
        st.rerun()
    except Exception as e:
        st.error(f"Error running evaluations: {e}")
        st.session_state.running_evals = False

if os.path.exists(EVAL_RESULT_FILE):
    try:
        with open(EVAL_RESULT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = data.get("summary", {})
        results = data.get("results", [])

        # Top level metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tests", summary.get("total", 0))
        col1.metric("Passed", summary.get("passed", 0))
        col2.metric("Pass Rate", f"{summary.get('pass_rate', 0)}%")

        latency = summary.get("latency_ms", {})
        col3.metric("Avg Latency (ms)", latency.get("avg", 0))
        col3.metric("P90 Latency (ms)", latency.get("p90", 0))

        if "tool_compliance" in summary:
            tc = summary["tool_compliance"]
            col4.metric("Tool Compliance", f"{tc.get('pass_rate', 0)}%")

        st.divider()

        # Category breakdown
        st.subheader("Category Breakdown")
        by_category = summary.get("by_category", {})
        if by_category:
            cat_data = []
            for cat, stats in by_category.items():
                cat_data.append(
                    {
                        "Category": cat,
                        "Total": stats.get("total", 0),
                        "Passed": stats.get("passed", 0),
                        "Pass Rate (%)": stats.get("pass_rate", 0),
                    }
                )
            df_cat = pd.DataFrame(cat_data)
            st.dataframe(df_cat, width="stretch")

            # Chart
            fig = px.bar(
                df_cat,
                x="Category",
                y="Pass Rate (%)",
                title="Pass Rate by Category",
                color="Category",
            )
            st.plotly_chart(fig, width="stretch")

        st.divider()

        # Detailed Results
        st.subheader("Detailed Results")

        # Filter options
        filter_status = st.radio(
            "Filter by Status", ["All", "Passed", "Failed"], horizontal=True
        )

        filtered_results = results
        if filter_status == "Passed":
            filtered_results = [r for r in results if r.get("pass")]
        elif filter_status == "Failed":
            filtered_results = [r for r in results if not r.get("pass")]

        for res in filtered_results:
            with st.expander(
                f"{'✅' if res.get('pass') else '❌'} {res.get('id')} ({res.get('category')})"
            ):
                st.markdown(f"**Prompt:** {res.get('prompt')}")
                st.markdown(f"**Answer:** {res.get('answer')}")
                st.markdown(f"**Latency:** {res.get('latency_sec')}s")
                st.markdown(f"**Tools Used:** {', '.join(res.get('tools_used', []))}")

                if not res.get("pass"):
                    st.error(f"Reason: {res.get('reason')}")
                    if res.get("tool_reason"):
                        st.warning(f"Tool Issue: {res.get('tool_reason')}")

                st.json(res)

    except json.JSONDecodeError:
        st.error(f"Error reading {EVAL_RESULT_FILE}. The file might be corrupted.")
else:
    st.info("No evaluation results found. Click 'Run Evaluations' to generate them.")
