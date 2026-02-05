---
title: BardAgent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.29.0
app_file: hf_app.py
pinned: false
license: mit
hf_oauth: true
---

# 🤖 BardAgent

BardAgent is a versatile AI assistant designed to leverage various tools and multimodal capabilities to help users solve complex tasks.

> **Note:** This deployment is currently configured for the [HuggingFace Agents Course](https://huggingface.co/learn/agents-course/en/unit4/introduction) Unit 4 Final Assessment to evaluate its performance on the GAIA benchmark.

## 🎯 Benchmark Goal

The current objective in this environment is to score ≥30% on the GAIA benchmark Level 1 questions using BardAgent's reasoning and tool-use capabilities.

## 🛠️ Features

BardAgent is powered by LangChain and LangGraph with the following capabilities:

| Tool | Description |
|------|-------------|
| 🔍 Web Search | DuckDuckGo search with Playwright scraping |
| 📖 Wikipedia | Wikipedia article search and retrieval |
| 🧮 Math | Mathematical calculations with SymPy |
| 🎵 Audio | Audio transcription (Gemini) |
| 🖼️ Vision | Image analysis (Gemini Vision) |
| 📊 Excel | Excel file reading and analysis |
| 🐍 Python | Python code execution |
| 📺 YouTube | Video transcript analysis |

## 🚀 Usage

1. Click **"Fetch Questions"** to load the 20 GAIA benchmark questions
2. Click **"Run Agent"** to process all questions with BardAgent
3. Enter your HuggingFace username and Space URL
4. Click **"Submit Answers"** to submit and see your score

## 📝 Configuration

This Space requires the following secrets:

- `GOOGLE_API_KEY` - Your Google AI (Gemini) API key

## 📚 About GAIA

[GAIA](https://huggingface.co/papers/2311.12983) is a benchmark designed to evaluate AI assistants on real-world tasks requiring reasoning, multimodal understanding, web browsing, and tool use.

## 🔗 Links

- [Agents Course](https://huggingface.co/learn/agents-course/)
- [Student Leaderboard](https://huggingface.co/spaces/agents-course/Students_leaderboard)
- [GAIA Paper](https://huggingface.co/papers/2311.12983)
