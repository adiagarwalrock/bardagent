"""Multimodal tools for processing audio, images, Excel files, and Python code."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from utilities.logger import logger

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

# Constants
GEMINI_MODEL = "gemini-2.5-flash"
SCRIPT_TIMEOUT_SECONDS = 30


def _get_genai_client():
    """Configure and return a Google GenAI Client instance."""
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Audio Transcription
# ---------------------------------------------------------------------------


class TranscribeAudioArgs(BaseModel):
    """Arguments for audio transcription."""

    model_config = ConfigDict(extra="forbid")
    file_path: str = Field(..., description="Path to the audio file (MP3, WAV, etc.)")


@tool(args_schema=TranscribeAudioArgs)
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe an audio file to text using Gemini's audio understanding.

    Args:
        file_path: Path to the audio file

    Returns:
        Transcribed text from the audio
    """
    from google.genai import types

    try:
        client = _get_genai_client()
        logger.info(f"Uploading audio file: {file_path}")

        # Read audio file and determine mime type
        audio_path = Path(file_path)
        mime_types = {
            ".mp3": "audio/mp3",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
        }
        mime_type = mime_types.get(audio_path.suffix.lower(), "audio/mpeg")

        with open(file_path, "rb") as f:
            audio_bytes = f.read()

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                "Transcribe this audio file exactly. Provide only the transcription, "
                "no additional commentary.",
            ],
        )

        transcription = response.text.strip()
        logger.info(f"Audio transcription completed: {len(transcription)} chars")
        return transcription

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}", exc_info=True)
        return f"Error transcribing audio: {e}"


# ---------------------------------------------------------------------------
# Image Analysis
# ---------------------------------------------------------------------------


class AnalyzeImageArgs(BaseModel):
    """Arguments for image analysis."""

    model_config = ConfigDict(extra="forbid")
    file_path: str = Field(..., description="Path to the image file (PNG, JPG, etc.)")
    question: str = Field(..., description="Question to answer about the image")


@tool(args_schema=AnalyzeImageArgs)
def analyze_image(file_path: str, question: str) -> str:
    """
    Analyze an image and answer a question about it using Gemini Vision.

    Args:
        file_path: Path to the image file
        question: Question to answer about the image

    Returns:
        Answer to the question based on image analysis
    """
    from PIL import Image

    try:
        client = _get_genai_client()
        logger.info(f"Loading image: {file_path}")

        img = Image.open(file_path)
        prompt = (
            f"Analyze this image and answer the following question. "
            f"Provide only the direct answer, no explanation.\n\nQuestion: {question}"
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[img, prompt],
        )
        answer = response.text.strip()

        logger.info(f"Image analysis completed: {answer[:100]}...")
        return answer

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Image analysis failed: {e}", exc_info=True)
        return f"Error analyzing image: {e}"


# ---------------------------------------------------------------------------
# Excel File Processing
# ---------------------------------------------------------------------------


class ReadExcelArgs(BaseModel):
    """Arguments for Excel file reading."""

    model_config = ConfigDict(extra="forbid")
    file_path: str = Field(..., description="Path to the Excel file (.xlsx)")
    query: Optional[str] = Field(None, description="Optional query about the data")


@tool(args_schema=ReadExcelArgs)
def read_excel(file_path: str, query: Optional[str] = None) -> str:
    """
    Read an Excel file and optionally answer a query about its contents.

    Args:
        file_path: Path to the Excel file
        query: Optional question about the data

    Returns:
        Excel data summary or query answer
    """
    import pandas as pd

    try:
        logger.info(f"Reading Excel file: {file_path}")
        xlsx = pd.ExcelFile(file_path)

        # Build data summary for all sheets
        result_parts = []
        for sheet_name in xlsx.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet_name)
            logger.info(
                f"Read sheet '{sheet_name}': {df.shape[0]} rows, {df.shape[1]} cols"
            )

            result_parts.extend(
                [
                    f"Sheet: {sheet_name}",
                    f"Columns: {list(df.columns)}",
                    f"Shape: {df.shape}",
                    f"Data:\n{df.to_string()}",
                    "",
                ]
            )

        full_data = "\n".join(result_parts)

        # Use Gemini to answer query if provided
        if query:
            try:
                client = _get_genai_client()
                prompt = (
                    f"Given this Excel data:\n\n{full_data}\n\n"
                    f"Answer this question with ONLY the answer, no explanation: {query}"
                )
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[prompt],
                )
                return response.text.strip()
            except ValueError:
                pass  # Fall through to return raw data

        return full_data[:5000]  # Truncate if too long

    except Exception as e:
        logger.error(f"Excel read failed: {e}", exc_info=True)
        return f"Error reading Excel: {e}"


# ---------------------------------------------------------------------------
# Python Code Execution
# ---------------------------------------------------------------------------


class ExecutePythonArgs(BaseModel):
    """Arguments for Python code execution."""

    model_config = ConfigDict(extra="forbid")
    file_path: str = Field(..., description="Path to the Python file (.py)")


@tool(args_schema=ExecutePythonArgs)
def execute_python(file_path: str) -> str:
    """
    Execute a Python file and return its output.

    Args:
        file_path: Path to the Python file

    Returns:
        Standard output from the Python script
    """
    try:
        logger.info(f"Executing Python file: {file_path}")

        # Log code preview
        with open(file_path, encoding="utf-8") as f:
            code = f.read()
        logger.debug(f"Python code to execute:\n{code[:500]}")

        # Run in subprocess with timeout
        cwd = Path(file_path).parent or Path(".")
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT_SECONDS,
            cwd=cwd,
            check=False,
        )

        output = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            logger.warning(f"Python script exited with code {result.returncode}")
            if stderr:
                return f"Error (exit code {result.returncode}): {stderr}"

        if stderr:
            output = f"{output}\n[stderr]: {stderr}"

        logger.info(f"Python execution completed: {output[:100]}...")
        return output or "(no output)"

    except subprocess.TimeoutExpired:
        logger.error("Python execution timed out")
        return f"Error: Script execution timed out ({SCRIPT_TIMEOUT_SECONDS} seconds)"
    except Exception as e:
        logger.error(f"Python execution failed: {e}", exc_info=True)
        return f"Error executing Python: {e}"


# ---------------------------------------------------------------------------
# YouTube Video Analysis
# ---------------------------------------------------------------------------


class AnalyzeYouTubeArgs(BaseModel):
    """Arguments for YouTube video analysis."""

    model_config = ConfigDict(extra="forbid")
    url: str = Field(..., description="YouTube video URL")
    question: str = Field(..., description="Question to answer about the video")


@tool(args_schema=AnalyzeYouTubeArgs)
def analyze_youtube(url: str, question: str) -> str:
    """
    Analyze a YouTube video and answer a question about it.

    Args:
        url: YouTube video URL
        question: Question to answer about the video

    Returns:
        Answer based on video analysis
    """
    import re

    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        # Extract video ID from URL
        video_id_match = re.search(r"(?:v=|/)([a-zA-Z0-9_-]{11})", url)
        if not video_id_match:
            return f"Error: Could not extract video ID from URL: {url}"

        video_id = video_id_match.group(1)
        logger.info(f"Fetching transcript for video: {video_id}")

        # Try to get transcript
        transcript = None
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join(entry["text"] for entry in transcript_list)
            logger.info(f"Got transcript: {len(transcript)} chars")
        except Exception as e:
            logger.warning(f"Could not get transcript: {e}")

        # Use Gemini to answer
        client = _get_genai_client()

        if transcript:
            prompt = (
                f"Based on this YouTube video transcript:\n\n{transcript[:8000]}\n\n"
                f"Answer this question with ONLY the direct answer, no explanation: {question}"
            )
        else:
            prompt = (
                f"For the YouTube video at {url}, answer this question. "
                f"If you cannot access the video, say so.\n\n"
                f"Question: {question}\n\n"
                f"Provide ONLY the direct answer, no explanation."
            )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
        )
        return response.text.strip()

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"YouTube analysis failed: {e}", exc_info=True)
        return f"Error analyzing YouTube video: {e}"


# ---------------------------------------------------------------------------
# Tool List Export
# ---------------------------------------------------------------------------


def get_multimodal_tools() -> List["BaseTool"]:
    """Return list of all multimodal tools."""
    return [
        transcribe_audio,
        analyze_image,
        read_excel,
        execute_python,
        analyze_youtube,
    ]
