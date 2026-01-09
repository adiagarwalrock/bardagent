import sys

sys.path.append(".")

import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

import core.agent as core_agent
import main as agent_main
from utilities.prompts import JUDGE_PROMPT, JudgeScores
from utilities.utils import normalize_content

JUDGE_LLM = "gpt-5.1"


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _run_batch_safe(prompts: List[str]) -> List[Tuple[List[Any], List]]:
    return agent_main.run_chat_batch(prompts)


def call_model_local_batch(
    prompts: List[str],
    max_concurrency: int,
    system_date: Optional[str] = None,
    retry_count: int = 1,
    retry_wait: float = 1.5,
) -> List[Tuple[str, float, List[str]]]:
    """Run multiple prompts via the agent with tools, using proper batching and retries."""

    if not prompts:
        return []

    start = time.time()
    results: List[Tuple[str, float, List[str]]] = []

    try:
        batch_results = _run_batch_safe(prompts)
    except Exception as e:
        # Fallback if even retries fail hard (e.g. auth error)
        return [(f"[error] {e}", 0.0, []) for _ in prompts]

    # Approximate per-item latency as average of batch
    batch_latency = (time.time() - start) / len(prompts) if prompts else 0.0

    for msgs, tool_calls in batch_results:
        tools_used = [str(t.get("name", "")).lower() for t in tool_calls]
        ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
        raw_content = ai.content if ai else ""
        text = normalize_content(raw_content) if ai else ""
        results.append((text, batch_latency, tools_used))

    return results


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

    # Count how many expected tokens are actually found
    found_count = 0

    for token in expected_contains:
        tok_s = str(token).strip()
        tok_norm = _normalize_text(tok_s)

        is_found = False
        # Numeric token: compare by value if we can extract a number from the answer.
        if _is_number_token(tok_s):
            try:
                expected_val = float(tok_s)
                # Extract first number-like substring from answer.
                m_num = re.search(r"-?\d+(?:\.\d+)?", ans_l)
                if m_num:
                    got_val = float(m_num.group(0))
                    if abs(got_val - expected_val) < 1e-2:
                        is_found = True
            except Exception:
                pass

            if not is_found and tok_norm in ans_norm:
                is_found = True
        else:
            # Multi-word expectation: require all words (stemmed) present anywhere in the answer.
            tok_words = _wordset(tok_norm)
            if not tok_words or tok_words.issubset(ans_words):
                is_found = True

        if is_found:
            found_count += 1
        else:
            missing.append(tok_s)

    # If expected_min_count is set, pass if we found enough tokens
    min_required = record.get("expected_min_count")
    if min_required is not None:
        if found_count < min_required:
            return False, f"found {found_count}/{min_required}, missing: {missing}"
    # Otherwise, require all tokens (legacy behavior)
    elif missing:
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
        if "wikipedia_search" in t_l:
            normalized.append("wikipedia")
        elif "calculator" in t_l:
            normalized.append("math")
        elif "duckduckgo" in t_l:
            normalized.append("web_search")
        else:
            normalized.append(t_l)

    required = []
    for t in record.get("must_use_tools", []):
        t_l = t.lower()
        if "wikipedia_search" in t_l:
            required.append("wikipedia")
        elif "calculator" in t_l:
            required.append("math")
        elif "duckduckgo" in t_l:
            required.append("web_search")
        else:
            required.append(t_l)
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

    Uses the default model from the active provider.
    """
    llm = core_agent.get_model(model_name=JUDGE_LLM)
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

        judge_data = out.get("judge", {})
        judge_verdict = judge_data.get("verdict", "").lower()

        # Hybrid Pass: Keywords OK OR Judge says "pass"
        content_pass = ok or (judge_verdict == "pass")

        if not ok and content_pass:
            reason = f"rescued_by_judge (was: {reason})"

        passed = (
            content_pass if tool_mode == "separate" else (content_pass and tools_ok)
        )
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
            "judge": judge_data,
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
    batch_size: int = 4,
    workers: Optional[int] = None,
    judge_sample_rate: float = 1.0,
) -> None:
    # Ensure the agent is created (and logged) before evaluation.
    agent_main.get_agent()

    if tool_mode not in {"separate", "gate"}:
        raise ValueError(f"tool_mode must be 'separate' or 'gate', got {tool_mode}")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not (0 < judge_sample_rate <= 1):
        raise ValueError("judge_sample_rate must be in (0, 1]")

    dataset = load_dataset(dataset_path)
    for record in dataset:
        record["_prompt_for_eval"] = build_prompt(record, tool_prompt_hint)

    total = len(dataset)
    results: List[Dict[str, Any]] = []
    workers = workers or max_workers

    with tqdm(total=total, desc="Running evals") as pbar:
        for start_idx in range(0, total, batch_size):
            batch = dataset[start_idx : start_idx + batch_size]
            prompts = [r["_prompt_for_eval"] for r in batch]

            batch_outputs = call_model_local_batch(
                prompts,
                max_concurrency=workers,
                retry_count=1,
                retry_wait=1.5,
            )

            # Build base results (without judge) first
            pending_judge: List[Tuple[int, Dict[str, Any]]] = []
            for offset, (record, (answer, latency, tools_used)) in enumerate(
                zip(batch, batch_outputs)
            ):
                ok, reason = check_expected(answer, record)
                tools_ok, tool_reason = check_tools(tools_used, record)
                passed = ok if tool_mode == "separate" else ok and tools_ok
                base: Dict[str, Any] = {
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
                if use_judge and random.random() <= judge_sample_rate:
                    pending_judge.append((start_idx + offset, base))
                else:
                    results.append(base)
                    pbar.update(1)
                    if progress_callback:
                        progress_callback(len(results), total)

            # Judge in parallel using same worker pool
            if use_judge and pending_judge:
                with ThreadPoolExecutor(max_workers=workers) as judge_pool:
                    future_to_idx = {
                        judge_pool.submit(
                            judge_answer,
                            dataset[idx].get(
                                "_prompt_for_eval", dataset[idx]["prompt"]
                            ),
                            base["answer"],
                        ): (idx, base)
                        for idx, base in pending_judge
                    }
                    for future in as_completed(future_to_idx):
                        idx, base = future_to_idx[future]
                        try:
                            judge_res = future.result()
                            base["judge"] = judge_res

                            # Hybrid Pass Logic
                            judge_verdict = judge_res.get("verdict", "").lower()
                            # Check if originally passed by keywords
                            original_pass = base["pass"]
                            # Re-evaluate content pass: (Keywords OK) OR (Judge says Pass)
                            # We can infer Keywords OK if (original_pass is True) AND (tools_ok is True or mode is separate)
                            # But simpler: we have the 'reason' field.
                            # If reason is 'expected_contains' or 'expected_exact...', it passed keywords.
                            # If reason starts with 'missing:', it failed keywords.

                            keywords_ok = not base["reason"].startswith(
                                "missing"
                            ) and not base["reason"].startswith("unexpected")
                            content_pass = keywords_ok or (judge_verdict == "pass")

                            if not keywords_ok and content_pass:
                                base["reason"] = (
                                    f"rescued_by_judge (was: {base['reason']})"
                                )

                            # Re-calculate final pass
                            # Need to know tools status. We can infer from tool_reason
                            tools_ok = not bool(base["tool_reason"])

                            if tool_mode == "separate":
                                base["pass"] = content_pass
                            else:
                                base["pass"] = content_pass and tools_ok

                        except Exception as e:  # pragma: no cover - defensive
                            base["judge"] = {"error": str(e)}
                        results.append(base)
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(len(results), total)

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
    summary["tool_policy"] = {
        "mode": tool_mode,
        "prompt_hint": tool_prompt_hint,
        "batch_size": batch_size,
        "workers": workers,
        "judge_sample_rate": judge_sample_rate,
        "use_langchain_batch": True,
    }

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
        help="Maximum number of concurrent workers (used for model and judge).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for model calls (>=1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override worker count (default: max-workers).",
    )
    parser.add_argument(
        "--judge-sample-rate",
        type=float,
        default=1.0,
        help="Fraction of evals to send to judge (0-1]; lowers cost/latency when <1.",
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
        batch_size=args.batch_size,
        workers=args.workers,
        judge_sample_rate=args.judge_sample_rate,
    )


if __name__ == "__main__":
    main()
