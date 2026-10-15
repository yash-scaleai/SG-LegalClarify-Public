"""Model routing config for llm_client.py.

Two blocks:
  * WEB_SEARCH_MODEL_METHOD — models with provider-hosted native web search
  * CUSTOM_WEB_SEARCH_MODELS — models that use our own Serper + Jina tool loop

Covers the paper's model matrix. Any model not in WEB_SEARCH_MODEL_METHOD uses the
custom path; CUSTOM_WEB_SEARCH_MODELS lists the paper's custom-path models for
reference. The Bedrock Gemma-3 id cannot search (run_eval.WEB_SEARCH_BLOCKED_MODELS);
the locally served `gemma-3-27b-it` runs on the custom path via the tool_code parser
in this folder.
"""

from typing import Dict, List

# Models with NATIVE web search support, and the request method to use.
# - openai_responses: use Responses API with tools=[{"type":"web_search"}]
# - chat_web_search_options: use Chat Completions with extra_body={"web_search_options": {...}}
WEB_SEARCH_MODEL_METHOD: Dict[str, str] = {
    # OpenAI (Responses API tools)
    "openai/gpt-5.6-luna": "openai_responses",
    "openai/gpt-5.6-sol": "openai_responses",
    "openai/gpt-5.6-terra": "openai_responses",
    # Anthropic (LiteLLM: chat + provider_specific_fields.citations)
    "anthropic/claude-fable-5": "chat_web_search_options",
    "anthropic/claude-opus-4-8": "chat_web_search_options",
    "anthropic/claude-sonnet-5": "chat_web_search_options",
    # Gemini (LiteLLM: chat + vertex_ai_* grounding metadata)
    "gemini/gemini-3.1-pro-preview": "chat_web_search_options",
    "gemini/gemini-3.5-flash": "chat_web_search_options",
}

# Models that use CUSTOM tool-calling web search (Serper + Jina) -- no native web search support.
CUSTOM_WEB_SEARCH_MODELS: List[str] = [
    "fireworks_ai/deepseek-v4-pro",
    "fireworks_ai/deepseek-v4-pro-0813",
    "fireworks_ai/qwen3p7-plus",
    "fireworks_ai/kimi-k2p6",
    "fireworks_ai/kimi-k3",
    "fireworks_ai/deepseek-v4-flash-0731",
    "mistral/mistral-large-latest",
    "fireworks_ai/deployed/t5cvm6h4",                      # Qwen3.6 27B (Fireworks-deployed)
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "bedrock/us.meta.llama4-scout-17b-instruct-v1:0",
    # Local vLLM models. Qwen-SEA-LION: vLLM's built-in qwen3_xml parser. Gemma-SEA-LION and
    # stock Gemma-3: our custom `tool_code` parser + chat template (web_search/). All three
    # validated end-to-end through the Serper+Jina loop.
    "sea-lion-v4.5-27b",
    "gemma-sea-lion-v4-27b",
    "gemma-3-27b-it",
]

# Custom web search settings (SINGLE round)
CUSTOM_WEB_SEARCH_ENABLE_VISIT: bool = True

# Custom web search settings (MULTI round). The bounded loop terminates when EITHER the
# model stops calling tools OR any of these caps is hit. Cumulative across all rounds
# within one assistant turn.
CUSTOM_WEB_SEARCH_MAX_ROUNDS: int = 999
CUSTOM_WEB_SEARCH_MAX_SEARCHES_TOTAL: int = 9999
CUSTOM_WEB_SEARCH_MAX_VISITS_TOTAL: int = 9999

# Native path: cap Anthropic's server-side web_search invocations per API call. Bumped to
# 100 for the unconstrained full-dataset run — Anthropic's actual behavior tops out well
# below this in the natural-behavior measurement (median 4, max 5), so 100 is effectively
# no cap for them. OpenAI Responses API and Gemini grounding do not expose an analogous
# parameter — their server-side loops run under provider-managed caps.
NATIVE_ANTHROPIC_MAX_USES: int = 100

# Forced-tool-call support per custom-path model, verified empirically. "native": tool_choice
# works as passed. "bypass": needs allowed_openai_params passthrough to get past LiteLLM's own
# param allowlist. Unlisted models fall back to wording-only forcing (e.g. Bedrock Llama models,
# which reject tool_choice at the API level).
FORCE_TOOL_CHOICE_SUPPORT: Dict[str, str] = {
    "fireworks_ai/deepseek-v4-pro": "native",
    "fireworks_ai/deepseek-v4-pro-0813": "native",
    "fireworks_ai/qwen3p7-plus":       "native",
    "fireworks_ai/kimi-k2p6":          "native",
    "fireworks_ai/kimi-k3":            "native",
    "fireworks_ai/deepseek-v4-flash-0731":  "native",
    "mistral/mistral-large-latest":    "native",
    "fireworks_ai/deployed/t5cvm6h4":  "bypass",
    # Bedrock Llama models reject tool_choice at the API level → wording-only
}

NATIVE_NO_TOOL_FORCE: set = set()
