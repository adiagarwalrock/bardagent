import sys

sys.path.append(".")

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage
from tqdm import tqdm

import main as agent_main
from utilities.utils import normalize_content


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
    ans_l = str(answer).lower()
    if "expected_exact" in record:
        match = ans_l.strip() == str(record["expected_exact"]).lower().strip()
        return match, "expected_exact"
    expected_contains = record.get("expected_contains") or []
    missing = [s for s in expected_contains if str(s).lower() not in ans_l]
    if missing:
        return False, f"missing: {missing}"
    unexpected = record.get("unexpected_contains") or []
    hit = [s for s in unexpected if str(s).lower() in ans_l]
    if hit:
        return False, f"unexpected: {hit}"
    return True, "expected_contains"


def check_tools(tools_used: List[str], record: Dict[str, Any]) -> Tuple[bool, str]:
    required = [t.lower() for t in record.get("must_use_tools", [])]
    if not required:
        return True, ""
    missing = [req for req in required if not any(req in t for t in tools_used)]
    if missing:
        return False, f"missing_tools: {missing}"
    return True, "tools_ok"


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


def eval_record(record: Dict[str, Any]) -> Dict[str, Any]:
    answer, latency, tools_used = call_model_local(record["prompt"])
    ok, reason = check_expected(answer, record)
    tools_ok, tool_reason = check_tools(tools_used, record)
    return {
        "id": record.get("id"),
        "category": record.get("category"),
        "tools_required": record.get("tools_required"),
        "prompt": record["prompt"],
        "must_use_tools": record.get("must_use_tools", []),
        "answer": answer,
        "latency_sec": round(latency, 3),
        "tools_used": tools_used,
        "pass": ok and tools_ok,
        "reason": reason if ok else reason,
        "tool_reason": tool_reason if not tools_ok else "",
    }


def run_evaluations(dataset_path: str, max_workers: int) -> None:
    # Ensure the agent is created (and logged) before evaluation.
    agent_main.get_agent()

    dataset = load_dataset(dataset_path)
    results = []
    with (
        ThreadPoolExecutor(max_workers=max_workers) as executor,
        tqdm(total=len(dataset), desc="Running evals") as pbar,
    ):
        future_to_record = {
            executor.submit(eval_record, record): record for record in dataset
        }
        for future in as_completed(future_to_record):
            results.append(future.result())
            pbar.update(1)

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
    args = parser.parse_args()

    run_evaluations(args.dataset, args.max_workers)


if __name__ == "__main__":
    main()
