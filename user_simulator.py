import json
import re

DECISIVE_CONCLUSION_ALL_INFO = (
    "I've provided all the information I have, please make a decisive conclusion based on the information you have so far."
)
DECISIVE_CONCLUSION_NOT_SURE = (
    "I'm not sure, please make a decisive conclusion based on the information you have so far."
)
ROUTING_MODEL = "anthropic/claude-opus-4-8"
YES_NO_REWRITE_MODEL = "anthropic/claude-opus-4-8"


def _build_tool_selection_prompt(assistant_message: str, tools_block: str) -> str:
    """Build a prompt for selecting which fact tools to call."""
    return (
        "You are selecting variable keys for a user simulator.\n\n"
        "You receive:\n"
        "1) The assistant's latest message\n"
        "2) Available variable keys, each with a key question and key fact answer\n\n"
        "IMPORTANT MATCHING RULES:\n"
        "- Evaluate EACH key independently and decide whether it should be selected.\n"
        "- Select a key only when the assistant explicitly asks about that key's fact topic, or a clear semantic equivalent of that exact fact.\n"
        "- Use BOTH the key question and key answer meaning when deciding.\n"
        "- If the assistant asks compound questions spanning multiple topics, select all and only the directly asked topics.\n"
        "- Do NOT select keys that are merely related downstream implications of a broader question.\n"
        "- Do NOT select keys for generic discussion, process-only suggestions, or legal consequence summaries.\n"
        "- If uncertain whether a key was directly asked, set select=false.\n\n"
        f"Assistant message:\n\"\"\"\n{assistant_message}\n\"\"\"\n\n"
        f"Available keys (question + answer descriptions):\n{tools_block}\n\n"
        "Respond ONLY with valid JSON in this format:\n"
        "{\n"
        "  \"decisions\": [\n"
        "    {\"key\": \"key_1\", \"select\": true, \"reason\": \"short reason\"},\n"
        "    {\"key\": \"key_2\", \"select\": false, \"reason\": \"short reason\"}\n"
        "  ],\n"
        "  \"selected_keys\": [\"key_1\"],\n"
        "  \"reason\": \"brief explanation\"\n"
        "}\n"
    )


def _build_variable_tools(scenario: dict) -> dict:
    """Build tool metadata keyed by tool name."""
    tools = {}
    for var_key, var_info in scenario["variables"].items():
        tool_name = f"check_{var_key}"
        tools[tool_name] = {
            "variable_key": var_key,
            "question": var_info["question"],
            "answer": var_info["answer"],
        }
    return tools


def _build_key_profiles_block(tools: dict) -> str:
    """Build key profiles for LLM semantic matching."""
    return "\n".join(
        f'- "{tool_info["variable_key"]}": '
        f'question="{tool_info["question"]}" ; '
        f'answer="{tool_info["answer"]}"'
        for tool_info in tools.values()
    )


def _select_tools_from_assistant_message(
    assistant_message: str,
    tools: dict,
    call_llm_fn,
    selector_model: str = ROUTING_MODEL,
) -> list[str]:
    """Use an LLM to select which variable keys should be invoked."""
    tools_block = _build_key_profiles_block(tools)
    prompt = _build_tool_selection_prompt(assistant_message, tools_block)
    try:
        raw = call_llm_fn(
            selector_model,
            "You are a strict JSON key selector. Respond only with valid JSON.",
            [{"role": "user", "content": prompt}],
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        parsed = json.loads(cleaned)
        selected_keys = []
        decisions = parsed.get("decisions", [])
        if isinstance(decisions, list):
            for d in decisions:
                if not isinstance(d, dict):
                    continue
                key = d.get("key")
                select = bool(d.get("select", False))
                if isinstance(key, str) and select:
                    selected_keys.append(key)
        # Backward compatibility with older selector schema.
        selected_from_legacy = parsed.get("selected_keys", parsed.get("tool_calls", []))
        if isinstance(selected_from_legacy, list):
            for k in selected_from_legacy:
                if isinstance(k, str):
                    selected_keys.append(k)
        if not selected_keys:
            return []
        valid_tool_names = []
        seen = set()
        key_to_tool = {tool["variable_key"]: name for name, tool in tools.items()}
        for key in selected_keys:
            if not isinstance(key, str):
                continue
            normalized = key.strip()
            tool_name = key_to_tool.get(normalized)
            if tool_name and tool_name not in seen:
                valid_tool_names.append(tool_name)
                seen.add(tool_name)
        return valid_tool_names
    except Exception:
        return []


def _rephrase_binary_answer_with_llm(
    question: str,
    answer: str,
    call_llm_fn,
    model: str = YES_NO_REWRITE_MODEL,
) -> str:
    """
    Rephrase bare yes/no answer into a complete sentence using a lightweight LLM.
    Falls back to a generic statement if the LLM call fails.
    """
    yn = answer.strip().lower().replace("’", "'")
    prompt = (
        "Rewrite the binary answer into one short, natural sentence.\n"
        "Use the same subject/perspective as the question.\n"
        "Do not add facts. Preserve only what is implied by the question and yes/no answer.\n"
        "Return plain text only.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
    )
    try:
        rewritten = call_llm_fn(
            model,
            "You rewrite yes/no facts into concise sentences while preserving the question subject.",
            [{"role": "user", "content": prompt}],
        ).strip()
        if rewritten:
            return rewritten
    except Exception:
        pass

    if yn in {"yes", "yes."}:
        return "Yes, that's correct."
    return "No, that's not the case."


def _rephrase_nonbinary_answer_with_llm(
    question: str,
    answer: str,
    call_llm_fn,
    model: str = YES_NO_REWRITE_MODEL,
) -> str:
    """
    Rephrase non-binary answers into complete sentences using question context.
    Falls back to a deterministic question-grounded sentence if the LLM call fails.
    """
    cleaned_answer = " ".join(answer.strip().split())
    prompt = (
        "Rewrite the answer into one short, natural, complete sentence.\n"
        "Use the question text for context so the sentence is self-contained.\n"
        "Do not add new facts beyond the question and answer.\n"
        "Keep uncertainty if the answer is uncertain.\n"
        "Return plain text only.\n\n"
        f"Question: {question}\n"
        f"Answer: {cleaned_answer}\n"
    )
    try:
        rewritten = call_llm_fn(
            model,
            "You rewrite short fact answers into concise, self-contained sentences.",
            [{"role": "user", "content": prompt}],
        ).strip()
        if rewritten:
            return rewritten
    except Exception:
        pass

    if not cleaned_answer:
        return answer
    if cleaned_answer[-1] in ".!?":
        cleaned_answer = cleaned_answer[:-1].strip()
    question_stub = " ".join(question.strip().rstrip("?").split())
    if question_stub:
        return f"For \"{question_stub}\", the answer is {cleaned_answer}."
    return f"The answer is {cleaned_answer}."


def _render_tool_answer(question: str, answer: str, call_llm_fn) -> str:
    """Turn raw tool answer text into clearer user-facing text."""
    normalized = answer.strip().lower().replace("’", "'")
    if normalized in {"yes", "yes."}:
        return _rephrase_binary_answer_with_llm(question, answer, call_llm_fn)
    if normalized in {"no", "no."}:
        return _rephrase_binary_answer_with_llm(question, answer, call_llm_fn)
    return _rephrase_nonbinary_answer_with_llm(question, answer, call_llm_fn)


def _execute_tools(tool_calls: list[str], tools: dict, call_llm_fn) -> list[str]:
    """Execute selected tools and return clearer fact strings."""
    responses = []
    for tool_name in tool_calls:
        tool = tools.get(tool_name)
        if tool:
            responses.append(_render_tool_answer(tool["question"], tool["answer"], call_llm_fn))
    return responses


def _build_user_response_from_tool_outputs(tool_outputs: list[str]) -> str:
    """Render tool outputs into a concise layperson response."""
    if not tool_outputs:
        return "I'm not sure about that."
    return "\n".join(tool_outputs)


def _build_classifier_prompt(assistant_message: str) -> str:
    """Build prompt for classifying assistant message intent."""
    return (
        "Classify the following assistant message from a legal consultation.\n\n"
        f"Message:\n\"\"\"\n{assistant_message}\n\"\"\"\n\n"
        "Respond with a JSON object containing:\n"
        "- \"has_clarifying_question\": true ONLY if the assistant message includes at least one "
        "CONCRETE question requesting particular facts about the "
        "user's situation. Set to false for "
        "general invitations like 'feel free to share more details' or "
        "'if you have more information, let me know' — these are NOT "
        "clarifying questions.\n"
        "- Treat numbered or bulleted fact requests as clarifying questions even "
        "if there is no literal '?' character, as long as they are "
        "CONCRETE questions requesting particular facts about the user's situation.\n"
        "- If the assistant asks the user to answer specific factual items "
        "(e.g., dates, participants, events, amounts, prior actions), "
        "set \"has_clarifying_question\" to true.\n"
        "- \"has_definitive_answer\": true if the assistant provides a "
        "case-specific conclusion, including probabilistic but committed conclusions "
        "(e.g., 'likely valid/invalid', 'likely liable/not liable', "
        "'the claim likely succeeds/fails'). Set to false if the assistant only lists "
        "generic possibilities without connecting to the user's facts.\n\n"
        "Respond ONLY with the JSON object, no other text."
    )


def classify_assistant_message(
    response_text: str,
    call_llm_fn,
    classifier_model: str = "claude-opus-4-6",
) -> dict:
    """
    Use an LLM to classify whether the assistant's response contains
    a clarifying question, a definitive answer, or neither.
    """
    try:
        raw = call_llm_fn(
            classifier_model,
            "You are a precise classifier. Respond only with valid JSON.",
            [{"role": "user", "content": _build_classifier_prompt(response_text)}],
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        parsed = json.loads(cleaned)
        has_q = bool(parsed.get("has_clarifying_question", False))
        has_answer = bool(parsed.get("has_definitive_answer", False))
        return {
            "has_clarifying_question": has_q,
            "has_definitive_answer": has_answer,
        }
    except Exception:
        # Conservative fallback if classifier fails:
        # detect concrete fact requests, including numbered/bulleted lists
        # that may omit '?' punctuation.
        text = response_text.lower()
        has_q_mark = "?" in text
        has_listed_fact_requests = (
            bool(re.search(r"\n\s*\d+[\.\)]\s", text))
            and any(
                cue in text
                for cue in [
                    "please answer",
                    "please provide",
                    "to assess",
                    "to determine",
                    "i need",
                    "clarify",
                ]
            )
        )
        has_q = has_q_mark or has_listed_fact_requests
        return {
            "has_clarifying_question": has_q,
            "has_definitive_answer": False,
        }


def _is_not_sure_response(text: str) -> bool:
    """Detect canonical uncertainty replies for loop stopping."""
    normalized = text.strip().lower().replace("’", "'")
    return (
        normalized == "i'm not sure about that."
        or normalized == "i'm not sure."
        or normalized == "i am not sure."
    )


def _is_repetitive_response(
    new_response: str,
    selected_variable_keys: set[str],
    previous_simulator_response: str,
    previous_simulator_keys: set[str],
) -> bool:
    """
    Repetition trigger for decisive handoff:
    - two identical uncertainty responses in a row, or
    - current selected keys were already requested in the previous simulator turn.
    """
    if (
        previous_simulator_response
        and _is_not_sure_response(new_response)
        and _is_not_sure_response(previous_simulator_response)
    ):
        return True

    if selected_variable_keys and previous_simulator_keys and selected_variable_keys.issubset(previous_simulator_keys):
        return True

    return False


def _all_variable_keys_covered(
    used_variable_keys: set[str],
    tools: dict,
) -> bool:
    """Check if all scenario variable keys have been covered by explicit state."""
    all_keys = {tool["variable_key"] for tool in tools.values()}
    return all_keys.issubset(used_variable_keys)


def simulate_user_response(
    conversation: list[dict],
    scenario: dict,
    call_llm_fn,
    used_variable_keys: set[str],
    previous_simulator_response: str,
    previous_simulator_keys: set[str],
    simulator_model: str = "anthropic/claude-opus-4-8",
) -> tuple[str, set[str], bool]:
    """
    Generate a user response using an LLM simulator grounded in the
    scenario's point-instance values.

    Returns:
        - response: the simulator's response text
        - selected_variable_keys: keys selected in this simulator turn
        - handoff_injected: whether a decisive-conclusion handoff message was injected
    """
    # Fully-specified scenarios have no clarifying-question variables to draw
    # from. If the model still chose INSUFFICIENT_FACTS at the unified gate and
    # is asking clarifying questions, jump straight to the canonical "all info
    # provided, please conclude" handoff on the very first simulator turn —
    # rather than emitting "I'm not sure" first and only handing off on the
    # repeat. Saves a round-trip and avoids a misleading simulator utterance.
    if not scenario.get("variables"):
        return DECISIVE_CONCLUSION_ALL_INFO, set(), True

    # Tool-gated simulator: choose fact tools using only the latest assistant turn.
    # The selector sees tool descriptions but never sees fact values.
    latest_assistant_message = ""
    for turn in reversed(conversation):
        if turn["role"] == "assistant":
            latest_assistant_message = turn["content"]
            break

    tools = _build_variable_tools(scenario)
    tool_calls = _select_tools_from_assistant_message(
        latest_assistant_message,
        tools,
        call_llm_fn,
        selector_model=ROUTING_MODEL,
    )
    selected_variable_keys = {
        tools[tool_name]["variable_key"]
        for tool_name in tool_calls
        if tool_name in tools
    }
    tool_outputs = _execute_tools(tool_calls, tools, call_llm_fn)
    response = _build_user_response_from_tool_outputs(tool_outputs)

    # If simulator is repeating, send a decisive-conclusion handoff message
    # instead of stopping immediately.
    if _is_repetitive_response(
        response, selected_variable_keys, previous_simulator_response, previous_simulator_keys
    ):
        if _all_variable_keys_covered(used_variable_keys, tools):
            return DECISIVE_CONCLUSION_ALL_INFO, selected_variable_keys, True
        return DECISIVE_CONCLUSION_NOT_SURE, selected_variable_keys, True

    used_variable_keys.update(selected_variable_keys)
    return response, selected_variable_keys, False
