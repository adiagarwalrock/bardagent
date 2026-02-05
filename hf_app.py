"""
HuggingFace Space Application for GAIA Benchmark Assessment.

Based on the official template from:
https://huggingface.co/spaces/agents-course/Final_Assignment_Template
"""

import os
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
import requests

# Import agent functionality
from main import run_chat
from langchain_core.messages import AIMessage
from utilities.answer_extractor import clean_for_exact_match, extract_clean_answer
from utilities.utils import normalize_content

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


def download_file(task_id: str, file_name: str, output_dir: Path) -> Path | None:
    """Download a file associated with a task."""
    if not file_name:
        return None

    url = f"{DEFAULT_API_URL}/files/{task_id}"
    output_path = output_dir / file_name

    print(f"Downloading file for task {task_id}: {file_name}")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        print(f"Downloaded file to {output_path}")
        return output_path
    except Exception as e:
        print(f"Failed to download file: {e}")
        return None


def build_prompt_for_question(question: str, file_path: Path | None = None) -> str:
    """Build a prompt for the agent based on the question and any attached file."""
    base_question = question

    if file_path:
        file_ext = file_path.suffix.lower()
        abs_path = str(file_path.absolute())

        if file_ext in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
            base_question = f"""An audio file has been downloaded to this EXACT path: {abs_path}

You MUST call the `transcribe_audio` tool with file_path="{abs_path}" (use this exact path, do not modify it).

After transcribing, answer this question based on the audio content:

{question}"""

        elif file_ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            base_question = f"""An image file has been downloaded to this EXACT path: {abs_path}

You MUST call the `analyze_image` tool with file_path="{abs_path}" (use this exact path, do not modify it).

Use the image analysis to answer this question:

{question}"""

        elif file_ext in {".xlsx", ".xls"}:
            base_question = f"""An Excel file has been downloaded to this EXACT path: {abs_path}

You MUST call the `read_excel` tool with file_path="{abs_path}" (use this exact path, do not modify it).

Use the Excel data to answer this question:

{question}"""

        elif file_ext == ".py":
            base_question = f"""A Python file has been downloaded to this EXACT path: {abs_path}

You MUST call the `execute_python` tool with file_path="{abs_path}" (use this exact path, do not modify it).

Use the Python output to answer this question:

{question}"""

        else:
            base_question = f"""A file has been downloaded to this EXACT path: {abs_path}

Please analyze this file to answer the following question:

{question}"""

    return f"""{base_question}

IMPORTANT: Provide ONLY the direct answer. No explanations, no "The answer is...", just the answer itself.
- For lists: use comma-separated format
- For numbers: just the number
- For names: just the name as requested
- Be precise and concise."""


# ----- FILE PRE-PROCESSING ------
def preprocess_file(file_path: Path) -> str | None:
    """Pre-process a file and return its content/analysis for injection into prompt."""
    if not file_path or not file_path.exists():
        return None

    file_ext = file_path.suffix.lower()
    abs_path = str(file_path.absolute())

    try:
        # Audio transcription
        if file_ext in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
            from tools.multimodal_tools import transcribe_audio

            result = transcribe_audio.invoke({"file_path": abs_path})
            return f"[AUDIO TRANSCRIPTION]:\n{result}"

        # Image analysis
        elif file_ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            from tools.multimodal_tools import analyze_image

            result = analyze_image.invoke(
                {
                    "file_path": abs_path,
                    "question": "Describe this image in detail, including all text, numbers, positions, and visual elements.",
                }
            )
            return f"[IMAGE ANALYSIS]:\n{result}"

        # Excel processing
        elif file_ext in {".xlsx", ".xls"}:
            from tools.multimodal_tools import read_excel

            result = read_excel.invoke({"file_path": abs_path})
            return f"[EXCEL DATA]:\n{result}"

        # Python execution
        elif file_ext == ".py":
            from tools.multimodal_tools import execute_python

            result = execute_python.invoke({"file_path": abs_path})
            return f"[PYTHON OUTPUT]:\n{result}"

    except Exception as e:
        print(f"Error pre-processing file {file_path}: {e}")
        return f"[FILE PROCESSING ERROR]: {e}"

    return None


# ----- BARDAGENT IMPLEMENTATION ------
class BardAgent:
    """GAIA Benchmark Agent using BardAgent's capabilities."""

    def __init__(self):
        print("BardAgent initialized.")

    def __call__(self, question: str, file_path: Path | None = None) -> str:
        """Run the agent on a question and return a clean answer."""
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        if file_path:
            print(f"  With file: {file_path}")

        try:
            # Pre-process file if present
            file_content = None
            if file_path and file_path.exists():
                print(f"  Pre-processing file: {file_path}")
                file_content = preprocess_file(file_path)
                if file_content:
                    print(f"  File processed: {len(file_content)} chars")

            # Build prompt with file content injected
            if file_content:
                prompt = f"""{file_content}

Based on the above file content, answer the following question:

{question}

IMPORTANT: Provide ONLY the direct answer. No explanations, no "The answer is...", just the answer itself.
- For lists: use comma-separated format, alphabetized
- For numbers: just the number
- For names: just the name as requested
- Be precise and concise."""
            else:
                prompt = build_prompt_for_question(question, file_path)

            # Run the agent
            messages, _ = run_chat(prompt)

            # Extract final AI message
            ai_msg = next(
                (m for m in reversed(messages) if isinstance(m, AIMessage)),
                None,
            )
            raw_response = normalize_content(ai_msg.content) if ai_msg else ""

            # Clean for exact match
            clean_answer = clean_for_exact_match(
                extract_clean_answer(raw_response, question)
            )

            print(f"Agent returning answer: {clean_answer[:100]}...")
            return clean_answer

        except Exception as e:
            print(f"Agent error: {e}")
            return f"Error: {e}"


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs BardAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = profile.username
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent
    try:
        agent = BardAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    # Agent code URL for the leaderboard
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(f"Agent code URL: {agent_code}")

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run Agent on all questions (with file downloads)
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for item in questions_data:
            task_id = item.get("task_id")
            question_text = item.get("question")
            file_name = item.get("file_name", "")

            if not task_id or question_text is None:
                print(f"Skipping item with missing task_id or question: {item}")
                continue

            # Download file if present
            file_path = None
            if file_name:
                file_path = download_file(task_id, file_name, temp_path)

            try:
                submitted_answer = agent(question_text, file_path)
                answers_payload.append(
                    {
                        "task_id": task_id,
                        "submitted_answer": submitted_answer,
                    }
                )
                results_log.append(
                    {
                        "Task ID": task_id,
                        "Question": (
                            question_text[:80] + "..."
                            if len(question_text) > 80
                            else question_text
                        ),
                        "File": file_name or "-",
                        "Submitted Answer": submitted_answer,
                    }
                )
            except Exception as e:
                print(f"Error running agent on task {task_id}: {e}")
                results_log.append(
                    {
                        "Task ID": task_id,
                        "Question": (
                            question_text[:80] + "..."
                            if len(question_text) > 80
                            else question_text
                        ),
                        "File": file_name or "-",
                        "Submitted Answer": f"AGENT ERROR: {e}",
                    }
                )

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=120)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 BardAgent - GAIA Benchmark Evaluation")
    gr.Markdown(
        """
        **Instructions:**

        1. Log in to your Hugging Face account using the button below.
        2. Click 'Run Evaluation & Submit All Answers' to fetch questions, run BardAgent, submit answers, and see the score.

        ---
        **About BardAgent:**
        An AI agent built with LangChain and LangGraph featuring web search, Wikipedia, math, audio transcription, 
        image analysis, Excel processing, Python execution, and YouTube analysis capabilities.

        **Goal:** Score ≥30% to earn your Certificate of Excellence!
        """
    )

    gr.LoginButton()

    run_button = gr.Button("🚀 Run Evaluation & Submit All Answers", variant="primary")

    status_output = gr.Textbox(
        label="Run Status / Submission Result", lines=5, interactive=False
    )
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
    )


if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")

    if space_host:
        print(f"✅ SPACE_HOST found: {space_host}")
        print(f"   Runtime URL should be: https://{space_host}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id:
        print(f"✅ SPACE_ID found: {space_id}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?).")

    print("-" * (60 + len(" App Starting ")) + "\n")

    print("Launching BardAgent GAIA Evaluation Interface...")
    demo.launch(debug=True, share=False)
