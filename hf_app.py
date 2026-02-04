"""
HuggingFace Space Application for GAIA Benchmark Assessment.

This Gradio app allows users to:
1. Log in with their HuggingFace account
2. Fetch GAIA benchmark questions
3. Run the bardagent on all questions
4. Submit answers and view their score
"""

import json
import os
from typing import List, Tuple

import gradio as gr

from gaia_runner import (
    fetch_questions,
    run_all_questions,
    submit_answers,
)


# Global state for questions and results
_questions = []
_results = []


def format_questions_display(questions: List[dict]) -> str:
    """Format questions for display in the UI."""
    if not questions:
        return "No questions loaded. Click 'Fetch Questions' to load them."

    lines = []
    for i, q in enumerate(questions, 1):
        task_id = q.get("task_id", "")[:8]
        question = q.get("question", "")[:100]
        file_name = q.get("file_name", "")

        line = f"**{i}. [{task_id}...]** {question}"
        if len(q.get("question", "")) > 100:
            line += "..."
        if file_name:
            line += f" 📎 `{file_name}`"
        lines.append(line)

    return "\n\n".join(lines)


def format_results_display(results: List[dict]) -> str:
    """Format results for display in the UI."""
    if not results:
        return "No results yet. Click 'Run Agent' to process questions."

    lines = []
    for i, r in enumerate(results, 1):
        task_id = r.get("task_id", "")[:8]
        question = r.get("question", "")[:60]
        answer = r.get("submitted_answer", "")[:80]

        lines.append(f"**{i}. [{task_id}...]**")
        lines.append(f"   Q: {question}...")
        lines.append(f"   A: `{answer}`")
        lines.append("")

    return "\n".join(lines)


def load_questions() -> Tuple[str, str]:
    """Fetch questions from the GAIA API."""
    global _questions

    try:
        _questions = fetch_questions()
        status = f"✅ Loaded {len(_questions)} questions successfully!"
        display = format_questions_display(_questions)
        return status, display
    except Exception as e:
        status = f"❌ Failed to load questions: {str(e)}"
        return status, ""


def run_agent_on_questions(progress=gr.Progress()) -> Tuple[str, str]:
    """Run the agent on all loaded questions."""
    global _questions, _results

    if not _questions:
        return "❌ No questions loaded. Please fetch questions first.", ""

    try:

        def progress_callback(current, total):
            progress(current / total, desc=f"Processing {current}/{total}")

        _results = run_all_questions(_questions, progress_callback=progress_callback)

        status = f"✅ Processed {len(_results)} questions!"
        display = format_results_display(_results)
        return status, display
    except Exception as e:
        status = f"❌ Error running agent: {str(e)}"
        return status, ""


def submit_to_leaderboard(
    username: str, space_url: str, profile: gr.OAuthProfile | None
) -> str:
    """Submit answers to the GAIA leaderboard."""
    global _results

    if not _results:
        return "❌ No results to submit. Run the agent first."

    # Use OAuth profile username if available
    if profile:
        username = profile.username

    if not username:
        return "❌ Please provide your HuggingFace username or log in."

    if not space_url or len(space_url) < 10:
        return "❌ Please provide your HuggingFace Space URL (e.g., https://huggingface.co/spaces/username/bardagent/tree/main)"

    try:
        result = submit_answers(username, space_url, _results)

        score = result.get("score", 0)
        correct = result.get("correct_count", 0)
        total = result.get("total_attempted", 0)
        message = result.get("message", "")

        status = f"""
## 🎉 Submission Successful!

**Username:** {username}

**Score:** {score:.1f}% ({correct}/{total} correct)

**Message:** {message}

{"🏆 **Congratulations! You've earned your certificate!**" if score >= 30 else "Keep improving to reach 30%!"}

Check the leaderboard: [Students Leaderboard](https://huggingface.co/spaces/agents-course/Students_leaderboard)
"""
        return status
    except Exception as e:
        return f"❌ Submission failed: {str(e)}"


def create_app():
    """Create the Gradio application."""

    with gr.Blocks(title="BardAgent - GAIA Benchmark", theme=gr.themes.Soft()) as app:

        gr.Markdown(
            """
# 🤖 BardAgent - GAIA Benchmark Assessment

This application runs BardAgent on the GAIA benchmark questions for the 
[HuggingFace Agents Course](https://huggingface.co/learn/agents-course/en/unit4/introduction).

**Goal:** Score ≥30% to earn your Certificate of Excellence!

---
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔐 Authentication")
                login_btn = gr.LoginButton()
                username_input = gr.Textbox(
                    label="HuggingFace Username",
                    placeholder="Your HF username (auto-filled if logged in)",
                    interactive=True,
                )
                space_url_input = gr.Textbox(
                    label="Space Code URL",
                    placeholder="https://huggingface.co/spaces/YOUR_USERNAME/bardagent/tree/main",
                    interactive=True,
                )

        gr.Markdown("---")

        with gr.Row():
            fetch_btn = gr.Button("📥 Fetch Questions", variant="secondary", size="lg")
            run_btn = gr.Button("🚀 Run Agent", variant="primary", size="lg")
            submit_btn = gr.Button("📤 Submit Answers", variant="primary", size="lg")

        with gr.Row():
            status_output = gr.Markdown(
                "Ready to start. Click 'Fetch Questions' to begin."
            )

        with gr.Tabs():
            with gr.TabItem("📋 Questions"):
                questions_display = gr.Markdown("No questions loaded yet.")

            with gr.TabItem("📊 Results"):
                results_display = gr.Markdown("No results yet.")

            with gr.TabItem("🏆 Submission"):
                submission_output = gr.Markdown(
                    "Submit your answers to see your score."
                )

        # Event handlers
        fetch_btn.click(fn=load_questions, outputs=[status_output, questions_display])

        run_btn.click(
            fn=run_agent_on_questions, outputs=[status_output, results_display]
        )

        submit_btn.click(
            fn=submit_to_leaderboard,
            inputs=[username_input, space_url_input],
            outputs=[submission_output],
        )

        gr.Markdown(
            """
---

### 📚 About

This application uses **BardAgent**, an AI agent built with LangChain and LangGraph, 
featuring tools for:
- 🔍 Web search and scraping
- 📖 Wikipedia search
- 🧮 Mathematical calculations
- 🎵 Audio transcription
- 🖼️ Image analysis
- 📊 Excel file processing
- 🐍 Python code execution
- 📺 YouTube video analysis

**Source Code:** [View on HuggingFace](https://huggingface.co/spaces/agents-course/Final_Assignment_Template)
"""
        )

    return app


# Create and launch the app
app = create_app()

if __name__ == "__main__":
    app.launch()
