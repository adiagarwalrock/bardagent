import logging
import time
from datetime import datetime
from typing import Any, List, Tuple

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)

from core import get_agent, get_model
from utilities.logger import logger as logging
from utilities.prompts import (
    QUERY_MESSAGE_TEMPLATE,
    PLANNING_INSTRUCTIONS,
    SimpleCheck,
    SIMPLE_CHECK_TEMPLATE,
)
from utilities.utils import normalize_content

SIMPLE_WORDS_THRESHOLD = 6


def _is_simple_query(text: str, model) -> bool:
    # Fast heuristic gate to avoid LLM call for trivial cases
    tokens = text.strip().split()
    if len(tokens) <= SIMPLE_WORDS_THRESHOLD:
        return True
    lowered = text.lower()
    if any(greet in lowered for greet in {"hi", "hello", "thanks", "thank you"}):
        return True

    prompt: str = SIMPLE_CHECK_TEMPLATE.format(query=text)
    res = model.with_structured_output(SimpleCheck).invoke(prompt)
    return bool(res.is_simple)


def _plan_needed(user_text: str, model) -> bool:
    if _is_simple_query(user_text, model):
        return False
    return True


def _generate_plan(model, user_text: str) -> AIMessage:
    prompt = PLANNING_INSTRUCTIONS.format(query=user_text)
    logging.info(f"Generating plan...")
    plan_msg = model.invoke(prompt)
    # Ensure it's an AIMessage
    if isinstance(plan_msg, AIMessage):
        return plan_msg
    return AIMessage(content=str(getattr(plan_msg, "content", plan_msg)))


def run_chat(
    user_message: str, history: List[AnyMessage] | None = None
) -> Tuple[List[AnyMessage], List]:
    """
    Run a chat turn with the agent.
    Adds an optional planning step for non-trivial queries.

    Args:
        user_message: The user's input message.
        history: The chat history as a list of messages.

    Returns:
        A tuple containing the list of new messages from the agent and a list of tool calls made.

    """

    if not user_message.strip():
        return [AIMessage(content="Please provide a non-empty message.")], []
    if history is None:
        history = []

    model = get_model()
    agent = get_agent()

    plan: str = ""
    if _plan_needed(user_message, model):
        plan_msg = _generate_plan(model, user_message)
        plan = normalize_content([plan_msg]) or ""

    user_msg = QUERY_MESSAGE_TEMPLATE.format(
        query=user_message,
        current_date=datetime.now().strftime("%Y-%m-%d"),
        planning_instructions=plan,
    )

    turn_input: List[AnyMessage] = list(history)

    turn_input.append(HumanMessage(content=user_msg))
    pre_len = len(turn_input)

    state: dict[str, Any] = agent.invoke({"messages": list(turn_input)})

    msgs: list[AnyMessage] = list(state.get("messages") or [])
    delta: list[AnyMessage] = msgs[pre_len:]

    logging.info(f"Agent produced {len(delta)} new messages this turn")

    if not delta:
        logging.warning("Agent returned no new messages; synthesizing fallback reply")
        delta = [AIMessage(content="We ran into an issue generating a response.")]

    if not isinstance(delta[-1], AIMessage):
        logging.warning("Final message is not AIMessage; appending normalized fallback")
        delta.append(
            AIMessage(content=normalize_content(getattr(delta[-1], "content", "")))
        )

    tool_calls: list = [
        {"name": tool.name, "content": tool.content, "status": tool.status}
        for tool in delta
        if isinstance(tool, ToolMessage)
    ]

    return delta, tool_calls


def run_chat_batch(
    user_messages: List[str], histories: List[List[AnyMessage]] | None = None
) -> List[Tuple[List[AnyMessage], List]]:
    """
    Run a batch of chat turns with the agent, optimizing LLM calls.
    """
    if not user_messages:
        return []

    count = len(user_messages)
    if histories is None:
        histories = [[] for _ in range(count)]

    model = get_model()
    agent = get_agent()

    # 1. Determine "is_simple" (Batch LLM check)
    is_simple_results = [False] * count
    needs_llm_check_indices = []
    llm_check_prompts = []

    for i, text in enumerate(user_messages):
        if not text.strip():
            continue

        tokens = text.strip().split()
        if len(tokens) <= SIMPLE_WORDS_THRESHOLD:
            is_simple_results[i] = True
            continue
        lowered = text.lower()
        if any(greet in lowered for greet in {"hi", "hello", "thanks", "thank you"}):
            is_simple_results[i] = True
            continue

        prompt = SIMPLE_CHECK_TEMPLATE.format(query=text)
        needs_llm_check_indices.append(i)
        llm_check_prompts.append(prompt)

    if llm_check_prompts:
        try:
            results = model.with_structured_output(SimpleCheck).batch(llm_check_prompts)
            for idx, res in zip(needs_llm_check_indices, results):
                is_simple_results[idx] = bool(res.is_simple) if res else False
        except Exception as e:
            logging.error(f"Batch simple check failed: {e}")

    # 2. Plan generation (Batch)
    needs_plan_indices = []
    plan_prompts = []

    for i in range(count):
        if is_simple_results[i]:
            continue
        hist = histories[i]
        has_plan = False
        for msg in reversed(hist):
            if isinstance(msg, AIMessage) and "Plan:" in (msg.content or ""):
                has_plan = True
                break
        if not has_plan:
            needs_plan_indices.append(i)
            prompt = PLANNING_INSTRUCTIONS.format(query=user_messages[i])
            plan_prompts.append(prompt)

    plans = {}
    if plan_prompts:
        try:
            plan_msgs = model.batch(plan_prompts)
            for idx, msg in zip(needs_plan_indices, plan_msgs):
                if isinstance(msg, AIMessage):
                    plans[idx] = msg
                else:
                    plans[idx] = AIMessage(content=str(getattr(msg, "content", msg)))
        except Exception as e:
            logging.error(f"Batch plan generation failed: {e}")

    # 3. Agent Execution (Batch)
    agent_inputs = []
    input_lengths = []

    for i in range(count):
        user_msg_text = user_messages[i]
        if not user_msg_text.strip():
            agent_inputs.append({"messages": []})
            input_lengths.append(0)
            continue

        plan_content = ""
        if i in plans:
            plan_content = normalize_content([plans[i]]) or ""

        user_msg = QUERY_MESSAGE_TEMPLATE.format(
            query=user_msg_text,
            current_date=datetime.now().strftime("%Y-%m-%d"),
            planning_instructions=plan_content,
        )

        turn = list(histories[i]) + [HumanMessage(content=user_msg)]

        agent_inputs.append({"messages": turn})
        input_lengths.append(len(turn))

    results_out = []
    try:
        # Use return_exceptions=True to prevent one failure from crashing all
        states = agent.batch(
            agent_inputs, config={"max_concurrency": 8}, return_exceptions=True
        )
    except Exception as e:
        logging.error(f"Agent batch failed hard: {e}")
        states = [e] * count

    for i, state in enumerate(states):
        if not user_messages[i].strip():
            results_out.append(
                ([AIMessage(content="Please provide a non-empty message.")], [])
            )
            continue

        if isinstance(state, Exception):
            logging.error(f"Item {i} failed: {state}")
            results_out.append(([AIMessage(content=f"Error: {state}")], []))
            continue

        msgs = list(state.get("messages") or [])
        pre_len = input_lengths[i]
        delta = msgs[pre_len:]

        if not delta:
            delta = [AIMessage(content="We ran into an issue generating a response.")]

        if not isinstance(delta[-1], AIMessage):
            delta.append(
                AIMessage(content=normalize_content(getattr(delta[-1], "content", "")))
            )

        tool_calls = [
            {"name": tool.name, "content": tool.content, "status": tool.status}
            for tool in delta
            if isinstance(tool, ToolMessage)
        ]
        results_out.append((delta, tool_calls))

    return results_out


if __name__ == "__main__":
    from uuid import uuid4

    session_id = str(uuid4())

    print(f"Starting bardagent chat session {session_id}. Type 'exit' to quit.")
    print("")

    chat_history: List[AnyMessage] = []
    total_history: List[AnyMessage] = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"} or not user_input.strip():
            print("Exiting chat. Goodbye!")
            break

        start = time.perf_counter()
        progress, tool_calls = run_chat(user_input, chat_history)

        # Persist only the user and AI messages in history
        ai_msg = next(
            (m for m in reversed(progress) if isinstance(m, AIMessage)), progress[-1]
        )
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(ai_msg)

        total_history.append(HumanMessage(content=user_input))
        total_history.extend(progress)

        print(
            f"BardAgent (tt: {time.perf_counter() - start:.3f}s) : {normalize_content(ai_msg.content)}"
        )

        print(f"Tools used: {set([tc['name'] for tc in tool_calls])}")