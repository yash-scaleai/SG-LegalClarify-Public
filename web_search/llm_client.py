"""Web search enabled LLM client: native (OpenAI/Anthropic/Gemini) and custom (Serper+Jina)
tool-calling paths, plus optional forced tool-use. Talks to models through a LiteLLM proxy
(OpenAI-compatible chat completions / Responses API).

Requires in your .env: LITELLM_API_KEY, LITELLM_BASE_URL. For the custom path:
SERPER_API_KEY, JINA_API_KEY (see web_search_tools.py).

See README.md for a usage example.
"""

import os
import time
import random
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from dataclasses import dataclass
from typing import List, Optional, Any
from dotenv import load_dotenv
from openai import OpenAI

from config import (
    WEB_SEARCH_MODEL_METHOD,
    CUSTOM_WEB_SEARCH_ENABLE_VISIT,
    CUSTOM_WEB_SEARCH_MAX_ROUNDS,
    CUSTOM_WEB_SEARCH_MAX_SEARCHES_TOTAL,
    CUSTOM_WEB_SEARCH_MAX_VISITS_TOTAL,
    NATIVE_ANTHROPIC_MAX_USES,
    FORCE_TOOL_CHOICE_SUPPORT,
    NATIVE_NO_TOOL_FORCE,
)

# Forced-tool-call wording. Used for both the custom and native web search paths, for
# consistency across all models -- wording-only, layered underneath API-level tool_choice
# forcing where available (see FORCE_TOOL_CHOICE_SUPPORT / NATIVE_NO_TOOL_FORCE).
FORCE_SEARCH_INSTRUCTION = (
    "This question cannot be answered from memory alone and requires a live search. "
    "Your ONLY valid first action is a web search tool call. Any response that is not a "
    "web search tool call will count as a failed task. "
    "Call the web search tool now, before writing anything else."
)

load_dotenv()
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")

if not LITELLM_API_KEY:
    raise RuntimeError("Missing LITELLM_API_KEY in your .env")
if not LITELLM_BASE_URL:
    raise RuntimeError("Missing LITELLM_BASE_URL in your .env")
if not LITELLM_BASE_URL.endswith("/"):
    LITELLM_BASE_URL = f"{LITELLM_BASE_URL}/"

# --- Local vLLM SEA-LION endpoints (mirror run_eval._build_llm_request routing) ---
# The two SEA-LION checkpoints are served locally via vLLM (OpenAI-compatible), so
# their web-search requests must NOT go through the LiteLLM proxy. Overridable via the
# same env vars run_eval uses (SEALION_BASE_URL / GEMMA_SEALION_BASE_URL).
SEALION_MODEL_ID = os.getenv("SEALION_MODEL_ID", "sea-lion-v4.5-27b")
SEALION_BASE_URL = os.getenv("SEALION_BASE_URL", "http://localhost:8001/v1")
SEALION_API_KEY = os.getenv("SEALION_API_KEY", "dummy")
GEMMA_SEALION_MODEL_ID = os.getenv("GEMMA_SEALION_MODEL_ID", "gemma-sea-lion-v4-27b")
GEMMA_SEALION_BASE_URL = os.getenv("GEMMA_SEALION_BASE_URL", "http://localhost:8002/v1")
GEMMA_SEALION_API_KEY = os.getenv("GEMMA_SEALION_API_KEY", "dummy")
GEMMA3_MODEL_ID = os.getenv("GEMMA3_MODEL_ID", "gemma-3-27b-it")

# Local vLLM servers accept a per-request `seed` on chat.completions; API providers on the
# custom path do not all accept it, so the custom loop only forwards `seed` for these ids.
# Added 2026-09-14, after the seed-42/43/44 web-search runs, which therefore sampled unseeded.
LOCAL_VLLM_MODEL_IDS = {SEALION_MODEL_ID, GEMMA_SEALION_MODEL_ID, GEMMA3_MODEL_ID}
GEMMA3_BASE_URL = os.getenv("GEMMA3_BASE_URL", "http://localhost:8003/v1")
GEMMA3_API_KEY = os.getenv("GEMMA3_API_KEY", "dummy")


def _client_for_model(model: str) -> "OpenAI":
    """OpenAI client routed to the correct endpoint for ``model``.

    SEA-LION variants hit their local vLLM servers directly; everything else goes
    through the LiteLLM proxy. Mirrors the local-routing branch of
    run_eval._build_llm_request so the web-search path stays consistent.
    """
    if model == SEALION_MODEL_ID:
        return OpenAI(base_url=SEALION_BASE_URL, api_key=SEALION_API_KEY)
    if model == GEMMA_SEALION_MODEL_ID:
        return OpenAI(base_url=GEMMA_SEALION_BASE_URL, api_key=GEMMA_SEALION_API_KEY)
    if model == GEMMA3_MODEL_ID:
        return OpenAI(base_url=GEMMA3_BASE_URL, api_key=GEMMA3_API_KEY)
    return OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)

MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 500

MAX_RETRIES = 8
INITIAL_BACKOFF_S = 2.0
MAX_BACKOFF_S = 60.0
JITTER_RANGE = 0.5  # +/- 50% randomization

# Newer Anthropic models reject thinking.type.enabled and require the adaptive
# thinking API (thinking.type.adaptive + output_config.effort). Add model IDs as needed.
ADAPTIVE_THINKING_MODELS = {
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-4-8",
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
}


def _anthropic_thinking_extra(model: str) -> dict:
    """extra_body thinking config for an Anthropic model (thinking assumed wanted)."""
    if model in ADAPTIVE_THINKING_MODELS:
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}
    return {"thinking": {"type": "enabled", "budget_tokens": 4096}}


def _jittered_backoff(base_s: float) -> float:
    """Add random jitter to backoff to prevent thundering herd."""
    jitter = random.uniform(1 - JITTER_RANGE, 1 + JITTER_RANGE)
    return base_s * jitter


@dataclass
class WebSearchResponse:
    """Response from a web search enabled LLM call."""
    content: str
    urls_cited: List[str]
    num_urls: int
    # True when the raw response shows a real tool invocation (e.g. Anthropic's
    # message.tool_calls with a server_tool_use id) even if no inline citation URL was
    # extracted -- num_urls alone undercounts real searches that weren't textually cited.
    tool_called: bool = False
    # Custom path only: per-round, per-tool-call event log of what the model actually did.
    # Each entry is a dict:
    #   {"round": int, "kind": "search", "query": str, "n_results": int, "urls_returned": [str,...]}
    #   {"round": int, "kind": "visit",  "url": str,   "success": bool, "summary_chars": int}
    #   {"round": int, "kind": "refused", "reason": str}   # cap hit
    # Empty on the native path (provider runs the loop server-side; we can only observe
    # citation URLs, not the individual queries).
    tool_uses: List[dict] = None  # type: ignore[assignment]
    # Number of tool-call rounds the model actually executed (custom path).
    # Native path: number of visible search-call items in the response payload.
    n_rounds: int = 0
    # For reasoning models: the model's chain-of-thought if the provider exposed one.
    # None if not a reasoning model or provider didn't return one on this call.
    reasoning_trace: Optional[str] = None

    def __post_init__(self):
        if self.tool_uses is None:
            self.tool_uses = []


# =============================================================================
# URL extraction helpers (native path)
# =============================================================================

def _extract_urls_from_annotations(resp_dict: dict) -> List[str]:
    """Extract unique URLs from url_citation annotations in an OpenAI Responses API payload."""
    urls = set()
    for item in resp_dict.get("output", []) or []:
        if item.get("type") == "message":
            for content in item.get("content", []) or []:
                for ann in content.get("annotations", []) or []:
                    if ann.get("type") == "url_citation":
                        if url := ann.get("url"):
                            base_url = url.split("?")[0]
                            urls.add(base_url)
    return sorted(urls)


def _strip_query(url: str) -> str:
    return url.split("?", 1)[0]


def _extract_anthropic_citation_urls(chat_resp_dict: dict) -> List[str]:
    """Extract URLs from LiteLLM's Anthropic provider_specific_fields.citations (chat endpoint).

    Only catches URLs the model inline-cited in its text. A model can search and use results
    without citing -- check WebSearchResponse.tool_called for a reliable "did it search" signal.
    """
    urls = set()
    try:
        msg = (chat_resp_dict.get("choices") or [])[0].get("message") or {}
        psf = msg.get("provider_specific_fields") or {}
        citations = psf.get("citations") or []
        for group in citations:
            for c in group or []:
                u = c.get("url")
                if isinstance(u, str) and u.startswith("http"):
                    urls.add(u.split("?")[0])
    except Exception:
        return []
    return sorted(urls)


def _extract_openai_web_search_urls(resp_dict: dict) -> List[str]:
    """Extract URLs from OpenAI Responses API payload, without relying on model output text."""
    urls = set(_extract_urls_from_annotations(resp_dict))
    for item in resp_dict.get("output", []) or []:
        if (item.get("type") or "").lower() == "web_search_call":
            action = item.get("action") or {}
            u = action.get("url")
            if isinstance(u, str) and u.startswith("http"):
                urls.add(_strip_query(u))
    return sorted(urls)


def _resolve_vertex_redirect(url: str, timeout_s: int = 15) -> str:
    """Resolve Vertex grounding redirect URLs to their final destination URL."""
    try:
        p = urlparse(url)
        if p.netloc != "vertexaisearch.cloud.google.com":
            return url
        if "/grounding-api-redirect/" not in p.path:
            return url
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": "https://vertexaisearch.cloud.google.com/",
            },
        )
        with urlopen(req, timeout=timeout_s) as resp:
            final = resp.geturl()
        return _strip_query(final)
    except Exception:
        return url


def _extract_gemini_grounded_urls(chat_resp_dict: dict, content_text: str) -> List[str]:
    """Extract URLs for Gemini models from grounding metadata (not model output text)."""
    urls: set = set()

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if lk in ("url", "uri") and isinstance(v, str) and v.startswith("http"):
                    urls.add(_strip_query(v))
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    for k in (
        "vertex_ai_grounding_metadata",
        "vertex_ai_url_context_metadata",
        "vertex_ai_citation_metadata",
    ):
        walk(chat_resp_dict.get(k))

    resolved = [_resolve_vertex_redirect(u) for u in sorted(urls)]
    return sorted(set(resolved))


# =============================================================================
# NATIVE web search (OpenAI Responses API / Anthropic / Gemini)
# =============================================================================


# =============================================================================
# CUSTOM web search (Serper + Jina, for models without native support)
# =============================================================================

def _capture_reasoning(msg) -> Optional[str]:
    """Pull a reasoning trace off a chat-completion message if the provider exposed one.

    Handles the common places providers put chain-of-thought:
      - Fireworks / DeepSeek / Kimi / Qwen: msg.reasoning_content
      - Anthropic via LiteLLM: msg.provider_specific_fields.thinking (list of blocks)
      - vLLM SEA-LION (qwen3 parser): msg.reasoning_content
    Returns a plain string or None.
    """
    if msg is None:
        return None
    rc = getattr(msg, "reasoning_content", None)
    if isinstance(rc, str) and rc.strip():
        return rc
    psf = getattr(msg, "provider_specific_fields", None)
    if isinstance(psf, dict):
        thinking = psf.get("thinking")
        if isinstance(thinking, list) and thinking:
            parts = []
            for blk in thinking:
                if isinstance(blk, dict):
                    t = blk.get("thinking") or blk.get("text") or ""
                    if t: parts.append(t)
            if parts:
                return "\n\n".join(parts)
        elif isinstance(thinking, str) and thinking.strip():
            return thinking
    return None


# =============================================================================
# MESSAGES-INPUT VARIANTS — for integration into multi-turn conversations
# =============================================================================
#
# The prompt-string functions above build a fresh 2-message conversation each
# call. For integration into run_eval.py's multi-turn loop, we need to accept
# the FULL messages list (system + prior turns + trailing user) and return
# the model's next assistant reply after any tool use.

def chat_with_web_search_messages(
    model: str,
    messages: list,
    reasoning_mode: Optional[str] = None,
    force: bool = False,
    seed: Optional[int] = None,
) -> WebSearchResponse:
    """Route to native or custom path based on config registration.

    `seed` is forwarded only to local vLLM models on the custom path (LOCAL_VLLM_MODEL_IDS);
    the native path and API providers ignore it.
    """
    if model in WEB_SEARCH_MODEL_METHOD:
        return _native_web_search_from_messages(model, messages, reasoning_mode, force)
    return _custom_web_search_from_messages(model, messages, reasoning_mode, force, seed=seed)


def _native_web_search_from_messages(
    model: str, messages: list, reasoning_mode: Optional[str], force: bool,
) -> WebSearchResponse:
    """Native path with messages input."""
    client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)
    method = WEB_SEARCH_MODEL_METHOD.get(model)
    if not method:
        raise RuntimeError(f"Model not configured for native web search: {model}")

    backoff = INITIAL_BACKOFF_S
    for attempt in range(MAX_RETRIES):
        try:
            if method == "openai_responses":
                # Responses API accepts a list of input items with roles.
                responses_input = [
                    {"role": m.get("role"), "content": m.get("content", "") or ""}
                    for m in messages if m.get("role") in ("system", "user", "assistant")
                ]
                responses_params = {
                    "model": model, "input": responses_input,
                    "tools": [{"type": "web_search"}], "timeout": DEFAULT_TIMEOUT,
                }
                if force and model not in NATIVE_NO_TOOL_FORCE:
                    responses_params["tool_choice"] = "required"
                response = client.responses.create(**responses_params)
                content = getattr(response, "output_text", "") or ""
                resp_dict = response.model_dump()
                urls = _extract_openai_web_search_urls(resp_dict)
                n_search_calls = sum(
                    1 for item in resp_dict.get("output", []) or []
                    if (item.get("type") or "").lower() == "web_search_call"
                )
                tool_called = n_search_calls > 0
                reasoning = None
                for item in resp_dict.get("output", []) or []:
                    if (item.get("type") or "").lower() == "reasoning":
                        summary = item.get("summary") or []
                        parts = [s.get("text","") for s in summary if isinstance(s, dict)]
                        joined = "\n\n".join(p for p in parts if p)
                        if joined: reasoning = joined; break
                return WebSearchResponse(
                    content=content, urls_cited=urls, num_urls=len(urls),
                    tool_called=tool_called, tool_uses=[], n_rounds=n_search_calls,
                    reasoning_trace=reasoning,
                )

            extra_body: dict = {"web_search_options": {"search_context_size": "medium"}}
            if method == "chat_web_search_options" and (
                model.startswith("anthropic/") or "claude" in model.lower()
            ):
                extra_body["web_search_options"] = {
                    "search_context_size": "medium",
                    "max_uses": NATIVE_ANTHROPIC_MAX_USES,
                }
                if reasoning_mode and reasoning_mode != "off" and not force:
                    extra_body.update(_anthropic_thinking_extra(model))
            chat_params = {
                "model": model, "messages": messages,
                "max_tokens": MAX_TOKENS, "timeout": DEFAULT_TIMEOUT,
            }
            if force:
                if model.startswith("anthropic/"):
                    chat_params["tool_choice"] = "required"
                elif model.startswith("gemini/"):
                    extra_body["tool_config"] = {"function_calling_config": {"mode": "ANY"}}
            chat_params["extra_body"] = extra_body
            response = client.chat.completions.create(**chat_params)
            choice = response.choices[0] if response.choices else None
            content = (choice.message.content or "") if choice else ""
            finish = choice.finish_reason if choice else None
            if finish in ("refusal", "content_filter") or (not content and finish != "stop"):
                return WebSearchResponse(
                    content="<answer>MODEL_REFUSED</answer><reasoning>Model refused to answer this question.</reasoning>",
                    urls_cited=[], num_urls=0,
                )
            resp_dict = response.model_dump()
            if model.startswith("anthropic/"):
                urls = _extract_anthropic_citation_urls(resp_dict)
            elif model.startswith("gemini/"):
                urls = _extract_gemini_grounded_urls(resp_dict, content)
            else:
                urls = []
            reasoning = _capture_reasoning(choice.message) if choice else None
            n_rounds_est = 0
            if model.startswith("gemini/"):
                tool_called = any(
                    resp_dict.get(k)
                    for k in ("vertex_ai_grounding_metadata", "vertex_ai_url_context_metadata",
                              "vertex_ai_citation_metadata")
                )
                if tool_called: n_rounds_est = 1
            else:
                tc = choice.message.tool_calls if choice else None
                tool_called = bool(tc)
                if tc: n_rounds_est = len(tc)
            return WebSearchResponse(
                content=content, urls_cited=urls, num_urls=len(urls),
                tool_called=tool_called, tool_uses=[], n_rounds=n_rounds_est,
                reasoning_trace=reasoning,
            )
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(_jittered_backoff(backoff))
            backoff = min(backoff * 2, MAX_BACKOFF_S)
    return WebSearchResponse(content="", urls_cited=[], num_urls=0)


def _custom_web_search_from_messages(
    model: str, messages: list, reasoning_mode: Optional[str], force: bool,
    seed: Optional[int] = None,
) -> WebSearchResponse:
    """Custom-path multi-round tool loop with pre-built messages input.

    The caller passes the full conversation context. We APPEND a web-search
    awareness line to the caller's system message (or prepend a new one if
    there is none) so prompt-literal models like Mistral 3 Large actually
    invoke the tool schema — otherwise Mistral reproducibly ignores the tools
    param and answers from training data alone. Verified by an A/B/C/D
    diagnostic; empirical impact: Mistral 0% → 100% tool_called, other custom
    models unaffected. See scripts/diagnose_mistral_no_search.py.
    """
    from web_search_tools import (
        serper_search, jina_fetch, get_available_tools,
        create_summarization_messages, extract_urls_from_search_results,
        truncate_to_words, MAX_SUMMARY_WORDS,
    )
    import json as _json

    client = _client_for_model(model)
    tools = get_available_tools(enable_visit=CUSTOM_WEB_SEARCH_ENABLE_VISIT)

    # Inject web-search awareness into (or prepend to) the system message.
    # This is idempotent — if the hint substring is already present we skip.
    WEB_SEARCH_HINT = (
        "\n\nYou have access to web search (search + visit tools) to help answer questions. "
        "Use them to find relevant, up-to-date information when needed. "
        f"You may call the search tool up to {CUSTOM_WEB_SEARCH_MAX_SEARCHES_TOTAL} times "
        f"across up to {CUSTOM_WEB_SEARCH_MAX_ROUNDS} rounds, and visit up to "
        f"{CUSTOM_WEB_SEARCH_MAX_VISITS_TOTAL} webpages in total."
    )
    _hint_marker = "You have access to web search"
    messages = list(messages)  # copy so we don't mutate caller's list
    if messages and messages[0].get("role") == "system":
        sys_content = messages[0].get("content", "") or ""
        if _hint_marker not in sys_content:
            messages[0] = {"role": "system", "content": sys_content + WEB_SEARCH_HINT}
    else:
        messages = [{"role": "system", "content": "You are a helpful assistant." + WEB_SEARCH_HINT}] + messages

    def _extras(first_round: bool):
        eb: dict = {}
        if "openai/" in model and reasoning_mode:
            eb["reasoning_effort"] = reasoning_mode
        return eb

    def _call_model(msgs, allow_tools: bool, first_round: bool):
        p = {"model": model, "messages": msgs,
             "max_tokens": MAX_TOKENS, "timeout": DEFAULT_TIMEOUT}
        if seed is not None and model in LOCAL_VLLM_MODEL_IDS:
            p["seed"] = seed
        extra = _extras(first_round)
        if allow_tools:
            p["tools"] = tools
            if first_round and force:
                support = FORCE_TOOL_CHOICE_SUPPORT.get(model, "none")
                if support in ("native", "bypass"):
                    p["tool_choice"] = {"type": "function", "function": {"name": "search"}}
                if support == "bypass":
                    extra["allowed_openai_params"] = ["tool_choice"]
        if extra: p["extra_body"] = extra
        return client.chat.completions.create(**p)

    backoff = INITIAL_BACKOFF_S
    for attempt in range(MAX_RETRIES):
        try:
            all_urls: List[str] = []
            tool_uses: List[dict] = []
            total_searches = 0
            total_visits = 0
            rounds_executed = 0
            reasoning_trace: Optional[str] = None
            final_content = ""
            conv_msgs = list(messages)

            for round_idx in range(1, CUSTOM_WEB_SEARCH_MAX_ROUNDS + 1):
                is_first = (round_idx == 1)
                response = _call_model(conv_msgs, allow_tools=True, first_round=is_first)
                choice = response.choices[0]
                r = _capture_reasoning(choice.message)
                if r:
                    reasoning_trace = (
                        (reasoning_trace + "\n\n---\n\n" + r) if reasoning_trace else r
                    )
                if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                    final_content = choice.message.content or ""
                    break
                rounds_executed += 1
                conv_msgs.append(choice.message.model_dump())
                tool_results = []
                for tool_call in choice.message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = _json.loads(tool_call.function.arguments)
                    except _json.JSONDecodeError:
                        tool_results.append({"tool_call_id": tool_call.id, "role": "tool",
                                             "content": "[Error] Invalid tool arguments"})
                        tool_uses.append({"round": round_idx, "kind": "refused",
                                          "reason": "invalid_tool_args"})
                        continue
                    if func_name == "search":
                        if total_searches >= CUSTOM_WEB_SEARCH_MAX_SEARCHES_TOTAL:
                            tool_results.append({"tool_call_id": tool_call.id, "role": "tool",
                                                 "content": f"[Budget exhausted] search cap {CUSTOM_WEB_SEARCH_MAX_SEARCHES_TOTAL} hit"})
                            tool_uses.append({"round": round_idx, "kind": "refused",
                                              "reason": "search_budget_exhausted"})
                            continue
                        query = func_args.get("query", "")
                        result = serper_search(query)
                        urls = extract_urls_from_search_results(result)
                        all_urls.extend(urls)
                        total_searches += 1
                        tool_uses.append({"round": round_idx, "kind": "search",
                                          "query": query, "n_results": len(urls),
                                          "urls_returned": urls})
                        tool_results.append({"tool_call_id": tool_call.id, "role": "tool", "content": result})
                    elif func_name == "visit" and CUSTOM_WEB_SEARCH_ENABLE_VISIT:
                        if total_visits >= CUSTOM_WEB_SEARCH_MAX_VISITS_TOTAL:
                            tool_results.append({"tool_call_id": tool_call.id, "role": "tool",
                                                 "content": f"[Budget exhausted] visit cap {CUSTOM_WEB_SEARCH_MAX_VISITS_TOTAL} hit"})
                            tool_uses.append({"round": round_idx, "kind": "refused",
                                              "reason": "visit_budget_exhausted"})
                            continue
                        url = func_args.get("url", "")
                        goal = func_args.get("goal", "Extract relevant information")
                        page = jina_fetch(url)
                        if page.startswith("[Fetch Error]"):
                            tool_results.append({"tool_call_id": tool_call.id, "role": "tool", "content": page})
                            tool_uses.append({"round": round_idx, "kind": "visit",
                                              "url": url, "success": False, "summary_chars": 0})
                        else:
                            sm = create_summarization_messages(page, goal)
                            try:
                                sp = {"model": model, "messages": sm, "max_tokens": 2048,
                                      "timeout": DEFAULT_TIMEOUT}
                                if seed is not None and model in LOCAL_VLLM_MODEL_IDS:
                                    sp["seed"] = seed
                                sr = client.chat.completions.create(**sp)
                                summary = sr.choices[0].message.content or ""
                                summary = truncate_to_words(summary, MAX_SUMMARY_WORDS)
                            except Exception:
                                summary = truncate_to_words(page, 300)
                            all_urls.append(url)
                            total_visits += 1
                            tool_uses.append({"round": round_idx, "kind": "visit",
                                              "url": url, "success": True,
                                              "summary_chars": len(summary)})
                            tool_results.append({"tool_call_id": tool_call.id, "role": "tool",
                                                 "content": f"Content from {url}:\n\n{summary}"})
                    else:
                        tool_results.append({"tool_call_id": tool_call.id, "role": "tool",
                                             "content": f"[Error] Unknown tool: {func_name}"})
                        tool_uses.append({"round": round_idx, "kind": "refused",
                                          "reason": f"unknown_tool:{func_name}"})
                conv_msgs.extend(tool_results)
            else:
                # MAX_ROUNDS exhausted — force final answer.
                conv_msgs.append({"role": "user",
                                  "content": "You've used the maximum tool-call rounds. "
                                             "Provide your final answer now based on the results you have."})
                force_resp = _call_model(conv_msgs, allow_tools=False, first_round=False)
                force_choice = force_resp.choices[0]
                r2 = _capture_reasoning(force_choice.message)
                if r2:
                    reasoning_trace = (
                        (reasoning_trace + "\n\n---\n\n" + r2) if reasoning_trace else r2
                    )
                final_content = force_choice.message.content or final_content

            unique_urls = sorted(set(all_urls))
            return WebSearchResponse(
                content=final_content, urls_cited=unique_urls, num_urls=len(unique_urls),
                tool_called=(rounds_executed > 0), tool_uses=tool_uses,
                n_rounds=rounds_executed, reasoning_trace=reasoning_trace,
            )
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(_jittered_backoff(backoff))
            backoff = min(backoff * 2, MAX_BACKOFF_S)
    return WebSearchResponse(content="", urls_cited=[], num_urls=0)
