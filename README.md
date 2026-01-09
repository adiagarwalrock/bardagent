# Question-Answering Agent (BardAgent)

A robust LLM-powered chatbot capable of answering open-ended questions by leveraging external tools (Search, Wikipedia, Math, Finance). This project satisfies the requirements of building a tool-using agent, an evaluation pipeline, and a web-based dashboard.

## 🚀 Features

- **Intelligent Tool Usage**: Dynamically selects tools like `DuckDuckGoSearch`, `Wikipedia`, `Calculator`, and `YahooFinance` to answer complex queries.
- **Dual-Model Architecture**:
  - **Agent**: Runs on `gpt-5.2-mini` for speed and efficiency.
  - **Judge**: Evaluations run on `gpt-5.2` for high-fidelity quality assessment.
- **Evaluation Pipeline**: Automated testing suite measuring accuracy, helpfulness, factuality, and tool compliance.
- **Web Dashboard**: Interactive Streamlit UI for chatting and viewing evaluation results.

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) (Recommended for fast dependency management) OR `pip`

### Installation Steps

1. **Prepare the project**
   Unzip the provided package and navigate into the project root:

   ```bash
   cd bardagent
   ```

2. **Configure Environment**
   Copy the example environment file and add your API keys.

   ```bash
   cp .env.example .env
   ```

   *Edit `.env` and set `OPENAI_API_KEY` (required).*

3. **Install Dependencies**

   **Using `uv` (Recommended):**

   ```bash
   uv sync
   # Optional: Install playwright browser for web scraping capabilities
   uv run python -m playwright install chromium
   ```

   **Using `pip`:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt # or pip install -e .
   python -m playwright install chromium
   ```

---

## 🏃 Usage

### 1. Run the Agent (CLI)

Interact with the chatbot directly in your terminal.

```bash
uv run python main.py
```

### 2. Run the Web Dashboard

Launch the web interface to chat and visualize evaluation metrics.

```bash
uv run streamlit run app.py
```

*Access the UI at `http://localhost:8501`*

### 3. Run Evaluations

Execute the evaluation suite to benchmark agent performance.

```bash
uv run python evals/run_eval.py --max-workers 4
```

*Results are saved to `eval_result.json`.*

---

## 📊 Evaluation Methodology

The agent is evaluated against a curated dataset (`evals/dataset.jsonl`) using the following metrics:

1. **Content Accuracy**: Checks for exact matches or presence of required keywords/numbers.
2. **LLM-as-a-Judge**: A `gpt-5.2` judge scores the response (1-5) on:
    - **Helpfulness**: Does the answer directly address the user's intent?
    - **Factuality**: Is the information accurate?
    - **Groundedness**: Does the agent cite its sources?
3. **Tool Compliance**: Verifies that mandatory tools (e.g., Math for calculations) were actually called.
4. **Latency**: Measures end-to-end response time.

---

## 📂 Project Structure

```text
/
├── core/               # Core agent logic and model configuration
│   └── agent.py        # LangChain agent setup
├── tools/              # Tool definitions
│   ├── action_tool.py  # Calculator, etc.
│   ├── info_tools.py   # Search, Wikipedia, Finance
│   └── web_fetch.py    # Web scraping
├── evals/              # Evaluation framework
│   ├── run_eval.py     # Evaluation runner (metrics & judge)
│   └── dataset.jsonl   # Test cases
├── UI/                 # Streamlit pages
│   ├── Home.py         # Chat interface
│   └── Evals.py        # Evaluation dashboard
├── utilities/          # Helper functions
│   ├── prompts.py      # System prompts (Identity, Instructions)
│   └── history.py      # Chat persistence
├── app.py              # Streamlit entry point
└── main.py             # CLI entry point
```
