import sys

sys.path.append(".")

import argparse
import functools
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from tqdm import tqdm

import core.agent as core_agent
import main as agent_main
from utilities.prompts import JUDGE_PROMPT, JudgeScores
from utilities.utils import normalize_content

JUDGE_AGENT = "gemini-2.5-flash"


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def call_model_local(prompt: str) -> Tuple[str, float, List[str]]:
    start = time.time()
    tools_used: List[str] = []
    try:
        msgs, tool_calls = agent_main.run_chat(prompt)
        tools_used = [str(t.get("name", "")).lower() for t in tool_calls]
        ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
        raw_content = ai.content if ai else ""
        text = normalize_content(raw_content) if ai else ""
    except Exception as e:
        text = f"[error] {e}"
    latency = time.time() - start
    return text, latency, tools_used


def check_expected(answer: str | Any, record: Dict[str, Any]) -> Tuple[bool, str]:
    """Check an answer against eval expectations.

    Designed to be robust to punctuation, casing, and minor formatting differences.
    """
    ans_raw = str(answer)
    ans_l = ans_raw.lower().strip()

    # numeric tolerant exact
    if "expected_exact" in record:
        target_raw = str(record["expected_exact"]).strip()
        target_l = target_raw.lower()
        try:
            a = float(ans_l)
            b = float(target_l)
            if abs(a - b) < 1e-3:
                return True, "expected_exact_num"
        except Exception:
            pass
        return ans_l == target_l, "expected_exact"

    def _normalize_text(s: str) -> str:
        # Lowercase, remove punctuation, normalize whitespace.
        return " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())

    def _stem(w: str) -> str:
        # Tiny stemmer to reduce brittleness: grounding vs ground, edits vs editing, etc.
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                return w[: -len(suf)]
        return w

    def _wordset(s: str) -> set[str]:
        return {_stem(w) for w in _normalize_text(s).split() if w}

    ans_norm = _normalize_text(ans_l)
    ans_words = _wordset(ans_norm)

    # If expected_contains contains numeric tokens, allow numeric comparison too.
    def _is_number_token(t: str) -> bool:
        return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", t.strip()))

    expected_contains = record.get("expected_contains") or []
    missing: list[str] = []

    for token in expected_contains:
        tok_s = str(token).strip()
        tok_norm = _normalize_text(tok_s)

        # Numeric token: compare by value if we can extract a number from the answer.
        if _is_number_token(tok_s):
            try:
                expected_val = float(tok_s)
                # Extract first number-like substring from answer.
                m_num = re.search(r"-?\d+(?:\.\d+)?", ans_l)
                if m_num:
                    got_val = float(m_num.group(0))
                    if abs(got_val - expected_val) < 1e-2:
                        continue
                # Fallback to textual containment if numeric parse fails
            except Exception:
                pass
            if tok_norm not in ans_norm:
                missing.append(tok_s)
            continue

        # Multi-word expectation: require all words (stemmed) present anywhere in the answer.
        tok_words = _wordset(tok_norm)
        if tok_words and not tok_words.issubset(ans_words):
            missing.append(tok_s)

    if missing:
        return False, f"missing: {missing}"

    unexpected = record.get("unexpected_contains") or []
    hit: list[str] = []
    for token in unexpected:
        tok_s = str(token).strip()
        tok_norm = _normalize_text(tok_s)
        tok_words = _wordset(tok_norm)
        if tok_words and tok_words.issubset(ans_words):
            hit.append(tok_s)

    if hit:
        return False, f"unexpected: {hit}"

    return True, "expected_contains"


def check_tools(tools_used: List[str], record: Dict[str, Any]) -> Tuple[bool, str]:
    # normalize tool labels (duckduckgo_search -> duckduckgo, wikipedia_search -> wikipedia)
    normalized = []
    for t in tools_used:
        t_l = t.lower()
        if "duckduckgo" in t_l:
            normalized.append("duckduckgo")
        elif "wikipedia" in t_l:
            normalized.append("wikipedia")
        else:
            normalized.append(t_l)

    required = [t.lower() for t in record.get("must_use_tools", [])]
    if not required:
        return True, ""
    missing = [req for req in required if req not in normalized]
    if missing:
        return False, f"missing_tools: {missing}"
    return True, "tools_ok"


def format_tool_list(tools: List[str]) -> str:
    if not tools:
        return ""
    if len(tools) == 1:
        return tools[0]
    if len(tools) == 2:
        return " and ".join(tools)
    return ", ".join(tools[:-1]) + f", and {tools[-1]}"


def build_prompt(record: Dict[str, Any], add_tool_hint: bool) -> str:
    base = record["prompt"]
    if not add_tool_hint:
        return base
    tools = record.get("must_use_tools") or []
    if not tools:
        return base
    return f"{base.rstrip()}\nYou must use: {format_tool_list(tools)}."


# -----------------------
# LLM-as-a-judge (structured)
# -----------------------


def get_judge_model():
    """Structured-output judge model.

    Uses a cheaper/faster model for judging when available.
    """
    try:
        llm = core_agent.get_model(model_name=JUDGE_AGENT)  # type: ignore[arg-type]
    except TypeError:
        llm = core_agent.get_model()  # fallback if signature differs
    return llm.with_structured_output(JudgeScores)


def judge_answer(prompt: str, answer: str) -> Dict[str, Any]:
    judge_llm = get_judge_model()
    start = time.time()
    jp = JUDGE_PROMPT.format(prompt=prompt, answer=answer)
    resp: JudgeScores = judge_llm.invoke(jp)
    latency = time.time() - start
    data = resp.model_dump()
    data["judge_latency_sec"] = round(latency, 3)
    return data


# -----------------------
# LangGraph pipeline
# -----------------------


class EvalState(TypedDict, total=False):
    record: Dict[str, Any]
    answer: str
    latency_sec: float
    tools_used: List[str]
    pass_: bool
    reason: str
    tool_reason: str
    judge: Dict[str, Any]


def _build_eval_graph():
    def answer_node(state: EvalState) -> EvalState:
        record = state["record"]
        prompt = record.get("_prompt_for_eval", record["prompt"])
        answer, latency, tools_used = call_model_local(prompt)
        ok, reason = check_expected(answer, record)
        tools_ok, tool_reason = check_tools(tools_used, record)
        return {
            "answer": answer,
            "latency_sec": round(latency, 3),
            "tools_used": tools_used,
            # Tool compliance is tracked separately; functional pass is content-only.
            "pass_": ok,
            "reason": reason,
            "tool_reason": tool_reason if not tools_ok else "",
        }

    def judge_node(state: EvalState) -> EvalState:
        record = state["record"]
        judge = judge_answer(
            record.get("_prompt_for_eval", record["prompt"]), state["answer"]
        )
        return {"judge": judge}

    graph: StateGraph[EvalState] = StateGraph(EvalState)
    graph.add_node("answer", answer_node)
    graph.add_node("judge", judge_node)
    graph.set_entry_point("answer")
    graph.add_edge("answer", "judge")
    graph.set_finish_point("judge")
    return graph.compile()


EVAL_GRAPH = _build_eval_graph()


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    by_category: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat = r["category"]
        agg = by_category.setdefault(cat, {"total": 0, "passed": 0})
        agg["total"] += 1
        agg["passed"] += 1 if r["pass"] else 0
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total * 100, 2) if total else 0.0,
        "by_category": {
            cat: {
                "total": v["total"],
                "passed": v["passed"],
                "pass_rate": (
                    round(v["passed"] / v["total"] * 100, 2) if v["total"] else 0.0
                ),
            }
            for cat, v in by_category.items()
        },
    }


def eval_record(
    record: Dict[str, Any], use_judge: bool = True, tool_mode: str = "separate"
) -> Dict[str, Any]:
    # Defensive: ensure prompt variant exists even if run_evaluations didn't pre-process.
    record.setdefault("_prompt_for_eval", record["prompt"])
    if use_judge:
        state: EvalState = {"record": record}
        out = EVAL_GRAPH.invoke(state)
        answer = out["answer"]
        latency = out["latency_sec"]
        tools_used = out.get("tools_used", [])
        ok, reason = check_expected(answer, record)
        tools_ok, tool_reason = check_tools(tools_used, record)
        passed = ok if tool_mode == "separate" else ok and tools_ok
        base = {
            "id": record.get("id"),
            "category": record.get("category"),
            "tools_required": record.get("tools_required"),
            "prompt": record["prompt"],
            "must_use_tools": record.get("must_use_tools", []),
            "answer": answer,
            "latency_sec": latency,
            "tools_used": tools_used,
            "pass": passed,
            "reason": reason,
            "tool_reason": tool_reason if not tools_ok else "",
            "judge": out.get("judge", {}),
        }
        return base

    # fallback to non-judge path
    answer, latency, tools_used = call_model_local(record["_prompt_for_eval"])
    ok, reason = check_expected(answer, record)
    tools_ok, tool_reason = check_tools(tools_used, record)
    passed = ok if tool_mode == "separate" else ok and tools_ok
    return {
        "id": record.get("id"),
        "category": record.get("category"),
        "tools_required": record.get("tools_required"),
        "prompt": record["prompt"],
        "must_use_tools": record.get("must_use_tools", []),
        "answer": answer,
        "latency_sec": round(latency, 3),
        "tools_used": tools_used,
        "pass": passed,
        "reason": reason,
        "tool_reason": tool_reason if not tools_ok else "",
    }


def run_evaluations(
    dataset_path: str,
    max_workers: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    use_judge: bool = True,
    tool_mode: str = "separate",
    tool_prompt_hint: bool = False,
) -> None:
    # Ensure the agent is created (and logged) before evaluation.
    agent_main.get_agent()

    if tool_mode not in {"separate", "gate"}:
        raise ValueError(f"tool_mode must be 'separate' or 'gate', got {tool_mode}")

    dataset = load_dataset(dataset_path)
    for record in dataset:
        record["_prompt_for_eval"] = build_prompt(record, tool_prompt_hint)

    total = len(dataset)
    results = []
    with (
        ThreadPoolExecutor(max_workers=max_workers) as executor,
        tqdm(total=total, desc="Running evals") as pbar,
    ):
        eval_fn = functools.partial(
            eval_record, use_judge=use_judge, tool_mode=tool_mode
        )
        future_to_record = {
            executor.submit(eval_fn, record): record for record in dataset
        }
        for i, future in enumerate(as_completed(future_to_record)):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"\n[Error] Exception during evaluation: {e}")
                print("[Info] Saving partial results to temp_results.json...")
                with open("temp_results.json", "w", encoding="utf-8") as f:
                    json.dump({"partial_results": results, "error": str(e)}, f, indent=2)
                raise e
            pbar.update(1)
            if progress_callback:
                progress_callback(i + 1, total)

    summary = summarize(results)
    latencies = [r["latency_sec"] for r in results]
    latencies_sorted = sorted(latencies)

    def pct(p: float) -> float:
        if not latencies_sorted:
            return 0.0
        idx = max(0, min(len(latencies_sorted) - 1, int(len(latencies_sorted) * p)))
        return latencies_sorted[idx]

    summary["latency_ms"] = {
        "avg": round(sum(latencies) / len(latencies) * 1000, 1) if latencies else 0,
        "p50": round(pct(0.5) * 1000, 1),
        "p90": round(pct(0.9) * 1000, 1),
    }

    must_use = [r for r in results if (r.get("must_use_tools"))]
    if must_use:
        tool_fail = sum(1 for r in must_use if r.get("tool_reason"))
        tool_pass = len(must_use) - tool_fail
        summary["tool_compliance"] = {
            "required": len(must_use),
            "passed": tool_pass,
            "pass_rate": round(tool_pass / len(must_use) * 100, 2),
        }
    summary["tool_policy"] = {"mode": tool_mode, "prompt_hint": tool_prompt_hint}

    # Judge aggregates
    judged = [r for r in results if r.get("judge")]
    if judged:
        summary["judge_avg_helpfulness"] = round(
            sum(r["judge"].get("helpfulness", 0) for r in judged) / len(judged), 2
        )
        summary["judge_avg_factuality"] = round(
            sum(r["judge"].get("factuality", 0) for r in judged) / len(judged), 2
        )

    output = {"summary": summary, "results": results}
    with open("eval_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evals against the local agent.")
    parser.add_argument(
        "--dataset", default="evals/dataset.jsonl", help="Path to JSONL dataset."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of concurrent evaluation threads.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-a-judge scoring (faster, legacy mode).",
    )
    parser.add_argument(
        "--tool-mode",
        choices=["separate", "gate"],
        default="separate",
        help="How to treat tool compliance: 'separate' keeps it out of pass/fail; 'gate' preserves legacy gating.",
    )
    parser.add_argument(
        "--tool-prompt-hint",
        action="store_true",
        help="Append a must-use tools hint to prompts during eval.",
    )
    args = parser.parse_args()

    run_evaluations(
        args.dataset,
        args.max_workers,
        use_judge=not args.no_judge,
        tool_mode=args.tool_mode,
        tool_prompt_hint=args.tool_prompt_hint,
    )


if __name__ == "__main__":
    main()
