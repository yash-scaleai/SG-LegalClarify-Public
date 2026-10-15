import argparse
import json
import os
import re
import threading as _threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from openai import OpenAI
from excel_scenarios import load_scenarios_from_excel
from user_simulator import (
    classify_assistant_message,
    simulate_user_response,
)


# Configuration

SYSTEM_PROMPTS = {
    "unified_sufficiency_gate_v1": {
        "label": "Unified Sufficiency Gate (V1)",
        "prompt": """You are an expert legal AI assistant advising a user on Singapore law. Your objective is to provide accurate, definitive legal conclusions based on the facts provided by the user.

Decision protocol:
1) First determine whether the facts given are legally sufficient to reach a decisive conclusion on the question asked. Facts are legally sufficient if they allow for a decisive legal conclusion, and any missing facts would not be outcome-determinative.
2) If sufficient, output the exact phrase SUFFICIENT_FACTS. Or if insufficient, output the exact phrase INSUFFICIENT_FACTS.""",
    },
    "unified_sufficiency_gate_v1_websearch": {
        "label": "Unified Sufficiency Gate (V1) + Web Search",
        # Identical to unified_sufficiency_gate_v1 at the gate call. The web-search
        # capability is enabled only for the post-gate answer turns and is exposed
        # via tools at the API layer; the prompt itself is unchanged so the gate's
        # sufficient/insufficient decision stays comparable to the closed-book run.
        "prompt": """You are an expert legal AI assistant advising a user on Singapore law. Your objective is to provide accurate, definitive legal conclusions based on the facts provided by the user.

Decision protocol:
1) First determine whether the facts given are legally sufficient to reach a decisive conclusion on the question asked. Facts are legally sufficient if they allow for a decisive legal conclusion, and any missing facts would not be outcome-determinative.
2) If sufficient, output the exact phrase SUFFICIENT_FACTS. Or if insufficient, output the exact phrase INSUFFICIENT_FACTS.""",
    },
}

# Unified prompt routing texts.
# The gate prompt asks the model under test to emit SUFFICIENT_FACTS / INSUFFICIENT_FACTS.
# Based on that decision, an external actor (this harness) routes the rest of the
# conversation to exactly one of the two standalone path prompts below, so the model
# only ever sees the instructions relevant to its chosen path.
UNIFIED_GATE_PROMPT = SYSTEM_PROMPTS["unified_sufficiency_gate_v1"]["prompt"]

UNIFIED_SUFFICIENT_PROMPT = """You are an expert legal AI assistant advising a user on Singapore law. Your objective is to provide accurate, definitive legal conclusions based on the facts provided by the user. 

General rules:
- Provide a direct legal analysis and a decisive conclusion based only on provided facts. Do not ask any clarifying questions.
- Do not hedge solely for caution or risk-avoidance.
- Identify and keep reasoning focused on legally relevant variables and applicable Singapore law.
- Verbalize the legally relevant variables in your response."""

UNIFIED_INSUFFICIENT_PROMPT = """You are an expert legal AI assistant advising a user on Singapore law. Your objective is to provide accurate, definitive legal conclusions based on the facts provided by the user. 

You are engaging in a multi-turn conversation.
PHASE 1: FACTS GATHERING
Act responsibly and ask clarifying questions for outcome-determinative variables that are missing from the user's initial query. Do not ask redundant or low-impact questions.
PHASE 2: LEGAL ASSESSMENT
If you have gathered sufficient facts from the user, you must provide the most decisive conclusion possible.
OR 
If the facts are still legally insufficient, DO NOT guess or hallucinate hidden assumptions. You must output the exact phrase INSUFFICIENT_FACTS_GATHERED and state what is still missing.

General rules:
- Do not hedge solely for caution or risk-avoidance.
- Identify and keep reasoning focused on legally relevant variables and applicable Singapore law.
- Verbalize the legally relevant variables in your response."""

MODELS = {
    # Anthropic
    "anthropic/claude-fable-5": {"label": "Fable 5"},
    "anthropic/claude-opus-4-8": {"label": "Opus 4.8"},
    "anthropic/claude-sonnet-5": {"label": "Sonnet 5"},
    # OpenAI GPT-5.6 family
    "openai/gpt-5.6-sol": {"label": "GPT-5.6 Sol"},
    "openai/gpt-5.6-terra": {"label": "GPT-5.6 Terra"},
    "openai/gpt-5.6-luna": {"label": "GPT-5.6 Luna"},
    # Google
    "gemini/gemini-3.1-pro-preview": {"label": "Gemini 3.1 Pro"},
    "gemini/gemini-3.5-flash": {"label": "Gemini 3.5 Flash"},
    "fireworks_ai/qwen3p7-plus": {"label": "Qwen3.7 Plus"},
    "fireworks_ai/kimi-k2p6": {"label": "Kimi K2.6"},
    "fireworks_ai/kimi-k3":   {"label": "Kimi K3"},
    "fireworks_ai/deepseek-v4-pro": {"label": "DeepSeek V4 Pro", "default": False},
    "fireworks_ai/deepseek-v4-pro-0813": {"label": "DeepSeek V4 Pro"},
    "fireworks_ai/deepseek-v4-flash": {"label": "DeepSeek V4 Flash", "default": False},
    "fireworks_ai/deepseek-v4-flash-0731": {"label": "DeepSeek V4 Flash"},
    "mistral/mistral-large-latest": {"label": "Mistral 3 Large"},
    "fireworks_ai/deployed/t5cvm6h4": {"label": "Qwen3.6 27B"},
    "bedrock/google.gemma-3-27b-it": {"label": "Gemma 3 27B"},
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": {"label": "Llama 4 Scout", "default": False},
    "bedrock/us.meta.llama4-scout-17b-instruct-v1:0": {"label": "Llama 4 Scout"},
    "sea-lion-v4.5-27b": {"label": "Qwen-SEA-LION v4.5 27B"},
    "gemma-sea-lion-v4-27b": {"label": "Gemma-SEA-LION v4 27B"},
    "gemma-3-27b-it": {"label": "Gemma 3 27B (local)"},
}

DEFAULT_MODELS = [m for m, info in MODELS.items() if info.get("default", True)]

UNIFIED_PROMPT_KEY = "unified_sufficiency_gate_v1"
# Prompt keys that share the unified-gate routing protocol (gate → decisive vs
# facts_gathering downstream prompt). Add new variants here (e.g. the
# web-search-enabled variant `unified_sufficiency_gate_v1_websearch`).
UNIFIED_PROMPT_KEYS = {"unified_sufficiency_gate_v1", "unified_sufficiency_gate_v1_websearch"}

# The LiteLLM proxy requires /v1 in the OpenAI-compatible base URL.
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "")

# Locally-served open-weights models via vLLM (OpenAI-compatible).
# See docs/reproducing.md for the SEA-LION setup — the same recipe
# applies to Gemma-SEA-LION and Gemma-3-27B-IT (qwen3-style reasoning parser
# for SEA-LION variants; Gemma-3 has no CoT emission and is a plain-answer
# instruct model). These endpoints are NOT reachable through the LiteLLM proxy;
# each gets its own client keyed by base URL.
SEALION_MODEL_ID = os.getenv("SEALION_MODEL_ID", "sea-lion-v4.5-27b")
SEALION_BASE_URL = os.getenv("SEALION_BASE_URL", "http://localhost:8001/v1")
SEALION_API_KEY = os.getenv("SEALION_API_KEY", "dummy")  # local vLLM ignores the key

GEMMA_SEALION_MODEL_ID = os.getenv("GEMMA_SEALION_MODEL_ID", "gemma-sea-lion-v4-27b")
GEMMA_SEALION_BASE_URL = os.getenv("GEMMA_SEALION_BASE_URL", "http://localhost:8002/v1")
GEMMA_SEALION_API_KEY = os.getenv("GEMMA_SEALION_API_KEY", "dummy")

GEMMA3_MODEL_ID = os.getenv("GEMMA3_MODEL_ID", "gemma-3-27b-it")
GEMMA3_BASE_URL = os.getenv("GEMMA3_BASE_URL", "http://localhost:8003/v1")
GEMMA3_API_KEY = os.getenv("GEMMA3_API_KEY", "dummy")

# Optional integer seed for the primary-model call (attached to every
# reasoning-enabled request via kwargs["seed"]). None means "do not send seed".
# Set via CLI --seed or PRIMARY_SEED env var and consumed by _build_llm_request.
# The simulator/classifier/judge do NOT use this — they call
# call_llm_with_reasoning with enable_reasoning=False and are intentionally
# seed-agnostic so the same simulated user is presented across all seeded runs.
try:
    _PRIMARY_SEED_ENV = os.getenv("PRIMARY_SEED")
    PRIMARY_SEED: Optional[int] = int(_PRIMARY_SEED_ENV) if _PRIMARY_SEED_ENV else None
except ValueError:
    PRIMARY_SEED = None
# Which run index (1-based) this process represents, for stamping in JSON.
try:
    RUN_ID: Optional[int] = int(os.getenv("RUN_ID")) if os.getenv("RUN_ID") else None
except ValueError:
    RUN_ID = None

# Reasoning is opt-in PER CALL (enable_reasoning=True), never keyed purely on
# model id: the simulator/classifier/router reuse the Opus id and must keep their
# plain, non-thinking behavior.
OPUS_MODEL_ID = "anthropic/claude-opus-4-8"

# Standardized per-turn output ceiling for reasoning-enabled generation (a CEILING,
# not a target; mainly guards against truncation). For SEA-LION this must fit inside
# the local server's --max-model-len alongside the accumulated prompt.
REASONING_MAX_TOKENS = int(os.getenv("REASONING_MAX_TOKENS", "16384"))
# Default per-turn output ceiling for non-reasoning calls (simulator, classifier,
# judge, and non-reasoning-capable models). Bumped from 2048 to 16384 so long
# rubric-judge JSON arrays and long primary answers do not truncate.
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "16384"))
# Reasoning effort for the API models (Anthropic adaptive-thinking output_config.effort
# and DeepSeek reasoning_effort).
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "high")
# Per-provider thinking-mode sampling (published recommendations).
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.6"))
# Qwen-SEA-LION v4.5 thinking-mode sampling (matches the model's generation_config:
# temperature=1.0, top_p=0.95, top_k=20).
SEALION_TEMPERATURE = float(os.getenv("SEALION_TEMPERATURE", "1.0"))
SEALION_TOP_P = float(os.getenv("SEALION_TOP_P", "0.95"))
# Gemma-SEA-LION v4 (Gemma 3) sampling — plain instruct model, NO chain-of-thought.
# Uses Gemma 3's recommended sampling (temperature=1.0, top_p=0.95, top_k=64), NOT
# the qwen thinking-mode params. Do not send enable_thinking (Gemma has no such mode).
GEMMA_SEALION_TEMPERATURE = float(os.getenv("GEMMA_SEALION_TEMPERATURE", "1.0"))
GEMMA_SEALION_TOP_P = float(os.getenv("GEMMA_SEALION_TOP_P", "0.95"))
GEMMA_SEALION_TOP_K = int(os.getenv("GEMMA_SEALION_TOP_K", "64"))

# Chain-of-thought sidecar for the main experiment pipeline. One JSONL record per
# trial (task × model × prompt × question_type), carrying the answer conversation
# and per-turn reasoning. Configured lazily by main() so importing this module
# from other scripts (calibration script, viewer, tests) does not open a file.
# Path defaults to data/experiment_cot_<UTC>.jsonl so each run gets its own file
# and prior runs are not overwritten.
EXPERIMENT_COT_JSONL_DIR = Path(os.getenv("EXPERIMENT_COT_JSONL_DIR",
                                          str(Path("data"))))

# Scenario source and selection config.
# Guarded so importing this module (e.g. to reuse run_conversation from other
# scripts) does not crash when the default sheet/headers are absent.
try:
    SCENARIOS = load_scenarios_from_excel()
except Exception as _scenario_load_error:  # noqa: BLE001
    print(f"[warn] could not preload scenarios from Excel: {_scenario_load_error}")
    SCENARIOS = {}

# Top-level results config: generated JSON/HTML go under this root by scenario.
RESULTS_ROOT_DIR = str(Path("results"))


# Data Classes

@dataclass
class ConversationTurn:
    role: str  # "user" or "assistant"
    content: str

@dataclass
class TrialResult:
    model: str
    model_label: str
    prompt_key: str
    prompt_label: str
    scenario: str
    conversation: list = field(default_factory=list)
    strategy: str = ""  # "clarified" or "enumerated"
    variables_asked: dict = field(default_factory=dict)
    num_vars_asked: int = 0
    rubric_results: list = field(default_factory=list)
    mandatory_passed: int = 0
    mandatory_total: int = 0
    score: float = 0.0
    selected_mode: str = ""  # "decisive" or "facts_gathering"
    simulator_selections: list = field(default_factory=list)
    model_thinks_question_type: str = ""  # "fully_specified" or "underspecified"
    error: Optional[str] = None
    duration_seconds: float = 0.0


# LLM Clients

_client: Optional["OpenAI"] = None
_local_vllm_clients: dict[str, "OpenAI"] = {}

# --- Chain-of-thought sidecar writer (main experiment pipeline) ---
# Opened lazily by main() via configure_cot_writer(). run_trial() calls
# _write_cot_record(...) after every conversation; it is a no-op when the
# writer has not been configured (e.g. when run_eval is imported as a library).
_cot_lock = _threading.Lock()
_cot_file = None
_cot_path: Optional[Path] = None


def configure_cot_writer(
    path: Optional[str | Path] = None,
    truncate: bool = True,
) -> Optional[Path]:
    """Open the experiment CoT sidecar for writing.

    ``path`` defaults to ``EXPERIMENT_COT_JSONL_DIR/experiment_cot_<UTC>.jsonl``
    (one file per run). ``truncate=True`` opens with mode 'w'; False appends.
    Returns the resolved Path, or None if the caller passed ``path=""`` to
    explicitly disable the sidecar.
    """
    global _cot_file, _cot_path
    if path == "":
        # explicit disable
        return None
    if path is None:
        # Include PID as a tiebreaker so parallel processes launched in the same
        # second do not collide on the same filename.
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = EXPERIMENT_COT_JSONL_DIR / f"experiment_cot_{ts}_pid{os.getpid()}.jsonl"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _cot_lock:
        if _cot_file is not None:
            try:
                _cot_file.close()
            except Exception:
                pass
        _cot_file = open(p, "w" if truncate else "a", encoding="utf-8")
        _cot_path = p
    return p


def close_cot_writer() -> None:
    global _cot_file, _cot_path
    with _cot_lock:
        if _cot_file is not None:
            try:
                _cot_file.flush()
                _cot_file.close()
            except Exception:
                pass
        _cot_file = None
        _cot_path = None


def _write_cot_record(record: dict) -> None:
    """Append one JSONL record to the sidecar (no-op if writer not configured)."""
    with _cot_lock:
        if _cot_file is None:
            return
        _cot_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        _cot_file.flush()


def load_dotenv_if_present(dotenv_path: str = ".env") -> None:
    """
    Load KEY=VALUE pairs from .env into process env if file exists.
    Existing environment variables take precedence.
    """
    if not os.path.isfile(dotenv_path):
        return

    with open(dotenv_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def get_litellm_client() -> "OpenAI":
    """Return a cached OpenAI-compatible client pointed at LiteLLM."""
    global _client

    if OpenAI is None:
        raise ImportError("openai package not installed. Run: pip install openai")

    load_dotenv_if_present()
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise EnvironmentError("API_KEY not set. Set API_KEY to your LiteLLM proxy key.")
    if not LITELLM_BASE_URL:
        raise EnvironmentError("LITELLM_BASE_URL not set. Point it at your LiteLLM proxy (…/v1).")

    if _client is None:
        _client = OpenAI(base_url=LITELLM_BASE_URL, api_key=api_key)

    return _client


def get_local_vllm_client(base_url: str, api_key: str) -> "OpenAI":
    """Return a cached OpenAI-compatible client pointed at a local vLLM server."""
    if OpenAI is None:
        raise ImportError("openai package not installed. Run: pip install openai")
    client = _local_vllm_clients.get(base_url)
    if client is None:
        client = OpenAI(base_url=base_url, api_key=api_key)
        _local_vllm_clients[base_url] = client
    return client


def is_retryable_llm_error(exc: Exception) -> bool:
    """Return True for transient API failures worth retrying (rate limits, overload, etc.)."""
    msg = str(exc).lower()
    markers = (
        "429",
        "rate limit",
        "ratelimit",
        "503",
        "502",
        "504",
        "timeout",
        "timed out",
        "connection error",
        "connection reset",
        "overloaded",
        "temporarily unavailable",
        "service unavailable",
        "too many requests",
    )
    return any(marker in msg for marker in markers)


# Model ids whose LiteLLM route accepts `reasoning_effort`. Anthropic and
# DeepSeek have their own branches in _build_llm_request; this list covers the
# other reasoning-capable providers we've verified. Models NOT covered here
# (e.g. gpt-4.1, mistral-large, llama-4) will fall through to the non-reasoning
# path when enable_reasoning=True, so the primary run doesn't 400.
_REASONING_EFFORT_PATTERNS = [
    re.compile(r"^openai/o\d"),                              # OpenAI o-series (o1, o3, o4, …)
    re.compile(r"^openai/gpt-5"),                            # GPT-5 family (incl. gpt-5.6-sol/terra/luna)
    re.compile(r"^gemini/gemini-[3-9]", re.IGNORECASE),      # Gemini 3+ (LiteLLM maps to thinking_level)
    # Fireworks-hosted reasoning models. Verified 2026-07-16 that the LiteLLM
    # Fireworks shim accepts `reasoning_effort` on all four families below and
    # returns reasoning_content (Qwen3 built-in thinking, Kimi K2 thinking,
    # gpt-oss harmony CoT, and Fireworks-deployed Qwen3-27B).
    re.compile(r"^fireworks_ai/qwen3", re.IGNORECASE),       # Qwen3.x plus (Fireworks)
    re.compile(r"^fireworks_ai/kimi-k[0-9]", re.IGNORECASE), # Kimi K2.x, K3.x, ... (Fireworks)
    re.compile(r"^fireworks_ai/deployed/", re.IGNORECASE),   # Fireworks-deployed private models
    # NOTE: the older bedrock/qwen.qwen3-* and bedrock/moonshotai.kimi-k2.* routes
    # rejected `reasoning_effort` via LiteLLM's Bedrock shim. We no longer use
    # those Bedrock ids — Qwen3 and Kimi are now called via Fireworks.
]

# Per-model output-token ceilings. Some providers refuse requests where
# max_tokens exceeds the model's declared output cap. When a model appears
# here, this value is used everywhere instead of REASONING_MAX_TOKENS /
# DEFAULT_MAX_TOKENS.
_MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {
    # Groq's Llama 4 Scout — 8192 context cap by default. Bumped up to
    # be safe; Groq will 400 anything above the model's declared limit.
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": 8192,
    # Bedrock's Llama 4 Scout — same architectural 8192 cap.
    "bedrock/us.meta.llama4-scout-17b-instruct-v1:0": 8192,
}


def _resolve_max_tokens(model: str, default: int) -> int:
    """Return the per-model output-token ceiling, capped by the model's limit."""
    override = _MODEL_MAX_OUTPUT_TOKENS.get(model)
    if override is None:
        return default
    return min(default, override)


def _model_supports_reasoning_effort(model: str) -> bool:
    """Whether this model accepts LiteLLM's `reasoning_effort` param.

    Anthropic and DeepSeek are handled separately; SEA-LION is served locally.
    Add entries to _REASONING_EFFORT_PATTERNS as new reasoning-capable providers
    are verified.
    """
    return any(p.search(model) for p in _REASONING_EFFORT_PATTERNS)


def _is_o_series(model_basename: str) -> bool:
    """Return True for OpenAI o-series basenames (o1, o3, o4-mini, o3-mini, …)."""
    return bool(re.match(r"^o\d", model_basename))


# Providers known to accept the `seed` parameter through the LiteLLM proxy
# without 400ing. Verified 2026-07-17: Fireworks and Gemini return
# UnsupportedParamsError on this proxy (drop_params is off); Anthropic and
# Bedrock don't accept seed either. For those routes seed is skipped and the
# non-zero sampling temperature is what gives run-to-run variance.
_SEED_SUPPORTED_PATTERNS = [
    re.compile(r"^openai/"),        # OpenAI (GPT-5.x, o-series)
    re.compile(r"^mistral/", re.IGNORECASE),
]


def _model_supports_seed(model: str) -> bool:
    return any(p.search(model) for p in _SEED_SUPPORTED_PATTERNS)


def _maybe_inject_seed(kwargs: dict, model: str) -> None:
    """If PRIMARY_SEED is set and the provider accepts `seed`, add it to kwargs."""
    if PRIMARY_SEED is None:
        return
    if _model_supports_seed(model):
        kwargs["seed"] = PRIMARY_SEED


def _build_llm_request(
    model: str, oai_messages: list[dict], enable_reasoning: bool
) -> tuple["OpenAI", dict]:
    """Pick the client and request kwargs for a model.

    Reasoning is opt-in via `enable_reasoning` so the simulator/classifier/router
    (which reuse the Opus id) keep their plain, non-thinking behavior. SEA-LION is
    always called with its thinking-mode sampling since that is the only correct way
    to drive the locally-served reasoning model.
    """
    # Local vLLM endpoints (SEA-LION variants + Gemma-3-27B-IT).
    #  * Qwen-SEA-LION v4.5 uses qwen3's reasoning parser, so vLLM exposes CoT via
    #    message.reasoning_content and the qwen3 thinking-mode chat template must be
    #    enabled explicitly. It gets qwen thinking-mode sampling (top_k=20).
    #  * Gemma-SEA-LION v4 (Gemma 3) and Gemma-3-27B-IT have no chain-of-thought
    #    mode; they are plain instruct models that emit a direct answer. Gemma-SEA-LION
    #    gets Gemma 3's recommended sampling (top_k=64) and NO enable_thinking.
    if model == GEMMA_SEALION_MODEL_ID:
        kwargs = {
            "model": model,
            "messages": oai_messages,
            "max_tokens": REASONING_MAX_TOKENS,
            "temperature": GEMMA_SEALION_TEMPERATURE,
            "top_p": GEMMA_SEALION_TOP_P,
            "extra_body": {"top_k": GEMMA_SEALION_TOP_K},
        }
        return get_local_vllm_client(GEMMA_SEALION_BASE_URL, GEMMA_SEALION_API_KEY), kwargs
    if model == SEALION_MODEL_ID:
        kwargs = {
            "model": model,
            "messages": oai_messages,
            "max_tokens": REASONING_MAX_TOKENS,
            "temperature": SEALION_TEMPERATURE,
            "top_p": SEALION_TOP_P,
            "presence_penalty": 0.0,
            "extra_body": {
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        }
        return get_local_vllm_client(SEALION_BASE_URL, SEALION_API_KEY), kwargs
    if model == GEMMA3_MODEL_ID:
        # Gemma-3 instruct: no reasoning; deterministic temperature=0 by default.
        # vLLM accepts `seed` on chat.completions; seed the primary call so
        # multi-run sampling is reproducible per seed.
        kwargs = {
            "model": model,
            "messages": oai_messages,
            "max_tokens": REASONING_MAX_TOKENS if enable_reasoning else DEFAULT_MAX_TOKENS,
            "temperature": 0,
        }
        if enable_reasoning and PRIMARY_SEED is not None:
            kwargs["seed"] = PRIMARY_SEED
        return get_local_vllm_client(GEMMA3_BASE_URL, GEMMA3_API_KEY), kwargs

    client = get_litellm_client()
    kwargs: dict = {"model": model, "messages": oai_messages}
    model_basename = model.split("/")[-1]

    if enable_reasoning:
        # Anthropic Claude 4+: adaptive thinking via extra_body.
        if model == OPUS_MODEL_ID or model_basename.startswith("claude"):
            kwargs["max_tokens"] = _resolve_max_tokens(model, REASONING_MAX_TOKENS)
            # Claude Opus 4.8 uses adaptive thinking + output_config.effort (the
            # LiteLLM reasoning_effort shim maps to the older thinking.type.enabled
            # which this model rejects, so pass the native format via extra_body).
            # Anthropic thinking requires temperature=1.
            kwargs["temperature"] = 1
            kwargs["extra_body"] = {
                # display:"summarized" is required to receive the CoT text — Opus 4.8
                # defaults to "omitted" (empty thinking field, encrypted signature only).
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": REASONING_EFFORT},
            }
            # Anthropic rejects the `seed` param, so _maybe_inject_seed no-ops here.
            _maybe_inject_seed(kwargs, model)
            return client, kwargs
        # DeepSeek: reasoning_effort + recommended temperature.
        if "deepseek" in model_basename:
            kwargs["max_tokens"] = _resolve_max_tokens(model, REASONING_MAX_TOKENS)
            kwargs["temperature"] = DEEPSEEK_TEMPERATURE
            kwargs["reasoning_effort"] = REASONING_EFFORT
            _maybe_inject_seed(kwargs, model)
            return client, kwargs
        # Other models known to accept LiteLLM's `reasoning_effort` param.
        # OpenAI o-series (o1, o3, o4, …) use max_completion_tokens instead of
        # max_tokens; everything else in this bucket uses max_tokens.
        if _model_supports_reasoning_effort(model):
            if _is_o_series(model_basename):
                kwargs["max_completion_tokens"] = _resolve_max_tokens(model, REASONING_MAX_TOKENS)
            else:
                kwargs["max_tokens"] = _resolve_max_tokens(model, REASONING_MAX_TOKENS)
                # Provider-recommended thinking-mode sampling. Where the provider
                # does not publish a recommendation we leave temperature unset
                # (server default).
                if model_basename.startswith("gpt-5"):
                    kwargs["temperature"] = 1        # OpenAI GPT-5 rejects temp=0
                elif model.startswith("fireworks_ai/qwen3") or model.startswith("fireworks_ai/deployed/"):
                    kwargs["temperature"] = 0.7      # Alibaba Qwen3 thinking-mode recommendation
                elif re.match(r"^fireworks_ai/kimi-k[0-9]", model, re.IGNORECASE):
                    kwargs["temperature"] = 0.6      # Moonshot Kimi K2/K3 thinking recommendation
            kwargs["reasoning_effort"] = REASONING_EFFORT
            _maybe_inject_seed(kwargs, model)
            return client, kwargs
        # Model does not support reasoning params (e.g. gpt-4.1, mistral, llama).
        # Fall through to the non-reasoning path below — no reasoning is sent,
        # but max_tokens is still bumped to DEFAULT_MAX_TOKENS to prevent
        # truncation. reasoning_sink will simply record `reasoning: None` for
        # these turns, which downstream analysis can filter on.

    # Non-reasoning behavior: used by all helper LLMs (simulator/classifier/judge/
    # router), by primary-model calls when reasoning is not requested,
    # and as the fallback for models that don't accept reasoning params.
    # O-series models use max_completion_tokens and do not support temperature.
    if model_basename.startswith("o"):
        kwargs["max_completion_tokens"] = _resolve_max_tokens(model, DEFAULT_MAX_TOKENS)
    else:
        kwargs["max_tokens"] = _resolve_max_tokens(model, DEFAULT_MAX_TOKENS)
        # Models that reject temperature=0 on their LiteLLM route: GPT-5 family
        # and all Claude 4+ models (Opus, Fable, etc.). They accept only
        # temperature=1. Everyone else gets temperature=0 for reproducibility.
        if model_basename.startswith("gpt-5") or model_basename.startswith("claude"):
            kwargs["temperature"] = 1
        else:
            kwargs["temperature"] = 0
    return client, kwargs


def _extract_reasoning(message: Any) -> Optional[str]:
    """Pull the chain-of-thought out of a chat completion message, if present.

    Different backends surface it as `reasoning_content` (vLLM/qwen3, DeepSeek) or
    `reasoning`, and the OpenAI client may stash unknown fields in `model_extra`.
    """
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(message, attr, None)
        if val:
            return val
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            val = extra.get(key)
            if val:
                return val
    # Anthropic surfaces extended-thinking as thinking_blocks under
    # provider_specific_fields (a list of {"type":"thinking","thinking":...}).
    psf = getattr(message, "provider_specific_fields", None)
    if isinstance(psf, dict):
        blocks = psf.get("thinking_blocks")
        if blocks:
            texts = [
                b.get("thinking", "")
                for b in blocks
                if isinstance(b, dict) and b.get("thinking")
            ]
            if texts:
                return "\n\n".join(texts)
    return None


# --- Web search integration ---------------------------------------------------
# When WEB_SEARCH_ENABLED is True and the model is not in WEB_SEARCH_BLOCKED_MODELS,
# the model-under-test turns run through web_search.llm_client's multi-round
# tool loop (Serper + Jina for custom-path models, provider native for
# OpenAI/Anthropic/Gemini). Gate calls, simulator, classifier, and judge stay
# closed-book regardless.
WEB_SEARCH_ENABLED: bool = False
# Models that cannot participate in web search (kept centrally so the wrapper
# transparently falls back to closed-book for them even when the flag is on).
WEB_SEARCH_BLOCKED_MODELS = {"bedrock/google.gemma-3-27b-it"}


def call_llm_with_reasoning_and_web_search(
    model: str,
    system_prompt: str,
    messages: list[dict],
    *,
    enable_reasoning: bool = False,
    tool_uses_sink: Optional[list] = None,
) -> tuple[str, Optional[str], int]:
    """
    Wrapper over call_llm_with_reasoning that routes through web search when
    WEB_SEARCH_ENABLED is on and the model is not blocked.

    Returns (content, reasoning, n_rounds). `n_rounds` is 0 in closed-book mode
    and >0 when the model actually issued tool calls on the web-search path.
    `tool_uses_sink`, if provided, gets extended with the per-tool-call events
    (dicts with round/kind/query/url/etc.).
    """
    if not WEB_SEARCH_ENABLED or model in WEB_SEARCH_BLOCKED_MODELS:
        content, reasoning = call_llm_with_reasoning(
            model, system_prompt, messages, enable_reasoning=enable_reasoning
        )
        return content, reasoning, 0

    # Add web_search/ to sys.path if not already; import lazily so run_eval
    # stays importable without the web_search dir on non-search runs.
    import sys as _sys
    _ws_dir = str(Path(__file__).resolve().parent / "web_search")
    if _ws_dir not in _sys.path:
        _sys.path.insert(0, _ws_dir)
    # The shared web_search code checks LITELLM_API_KEY at import; our .env
    # uses API_KEY. Alias in-process if not already done.
    if not os.environ.get("LITELLM_API_KEY") and os.environ.get("API_KEY"):
        os.environ["LITELLM_API_KEY"] = os.environ["API_KEY"]
    if not os.environ.get("LITELLM_BASE_URL"):
        os.environ["LITELLM_BASE_URL"] = LITELLM_BASE_URL
    from llm_client import chat_with_web_search_messages   # type: ignore

    oai_messages = [{"role": "system", "content": system_prompt}] + list(messages)
    # Local vLLM models accept `seed`; forward PRIMARY_SEED so a re-run with the same
    # server build reproduces the sampling (2026-09-14; earlier web-search runs were unseeded).
    r = chat_with_web_search_messages(
        model, oai_messages,
        reasoning_mode="medium" if enable_reasoning else None,
        seed=PRIMARY_SEED,
    )
    if tool_uses_sink is not None:
        tool_uses_sink.extend(r.tool_uses or [])
    return r.content, r.reasoning_trace, int(getattr(r, "n_rounds", 0))


def call_llm_with_reasoning(
    model: str,
    system_prompt: str,
    messages: list[dict],
    *,
    enable_reasoning: bool = False,
) -> tuple[str, Optional[str]]:
    """Call a model and return (answer_content, reasoning_content).

    `reasoning_content` is None unless the backend returns a chain-of-thought.
    Routes SEA-LION to the local vLLM server; everything else goes through LiteLLM.
    """
    oai_messages = [{"role": "system", "content": system_prompt}] + messages
    client, kwargs = _build_llm_request(model, oai_messages, enable_reasoning)

    max_attempts = int(os.getenv("LLM_MAX_RETRIES", "10"))
    base_delay = float(os.getenv("LLM_RETRY_BASE_DELAY_SEC", "3"))
    max_delay = float(os.getenv("LLM_RETRY_MAX_DELAY_SEC", "90"))
    # A successful API call that returns empty/whitespace content is a rare
    # provider quirk (observed on Anthropic adaptive-thinking and Gemini —
    # thinking budget consumed with no visible content emitted). Bumped to
    # 10 tries after finding a scenario where Fable 5's per-attempt empty
    # rate was ~53% (0.53^4 ≈ 8% chance of 4-in-a-row); 10 attempts drives
    # the residual empty probability below 0.3%. Backoff: 2,4,8,16,32,64,90,90,90,90.
    # See docs/reproducing.md.
    empty_max_attempts = int(os.getenv("LLM_EMPTY_RETRY_MAX", "10"))
    empty_base_delay = float(os.getenv("LLM_EMPTY_RETRY_BASE_DELAY_SEC", "2"))
    empty_attempts_used = 0

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            # Provider rejected our temperature. Any phrasing counts — "deprecated",
            # "does not support temperature", "only temperature=1 is supported", etc.
            _temp_reject = (
                "temperature" in msg and "temperature" in kwargs and (
                    "deprecated" in msg
                    or "does not support" in msg
                    or "only temperature" in msg
                    or "unsupported" in msg
                )
            )
            if _temp_reject:
                kwargs.pop("temperature", None)
                try:
                    response = client.chat.completions.create(**kwargs)
                except Exception as retry_exc:
                    last_exc = retry_exc
                    if attempt >= max_attempts - 1 or not is_retryable_llm_error(retry_exc):
                        raise
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    print(
                        f"    [LLM retry {attempt + 1}/{max_attempts}] {retry_exc}; "
                        f"waiting {delay:.1f}s...",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
            elif attempt < max_attempts - 1 and is_retryable_llm_error(exc):
                last_exc = exc
                delay = min(max_delay, base_delay * (2 ** attempt))
                print(
                    f"    [LLM retry {attempt + 1}/{max_attempts}] {exc}; "
                    f"waiting {delay:.1f}s...",
                    flush=True,
                )
                time.sleep(delay)
                continue
            else:
                raise

        message = response.choices[0].message
        content = message.content or ""
        reasoning = _extract_reasoning(message)

        # Empty-content retry: the API succeeded but the model returned no
        # visible text (Anthropic adaptive thinking that consumed its budget
        # on thinking blocks, or a Gemini edge case). Retry a few times with
        # a short backoff — content varies run-to-run at temperature > 0 and
        # usually comes back on the very next attempt.
        if not content.strip() and empty_attempts_used < empty_max_attempts:
            empty_attempts_used += 1
            delay = min(max_delay, empty_base_delay * (2 ** (empty_attempts_used - 1)))
            print(
                f"    [LLM empty-response retry {empty_attempts_used}/{empty_max_attempts}] "
                f"model={model!r} returned empty content; waiting {delay:.1f}s...",
                flush=True,
            )
            time.sleep(delay)
            continue

        return content, reasoning

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("call_llm failed without exception")


def call_llm(model: str, system_prompt: str, messages: list[dict]) -> str:
    """Call a model and return only the answer content (no reasoning).

    Thin wrapper over call_llm_with_reasoning so existing callers (gate classifier,
    user simulator, router, judge) are unchanged.
    """
    return call_llm_with_reasoning(model, system_prompt, messages)[0]


def run_unified_gate(
    model: str,
    user_question: str,
    enable_reasoning: bool = False,
    reasoning_sink: Optional[list] = None,
) -> str:
    """
    One-shot fact-sufficiency gate for the unified prompt flow.

    The model under test sees the initial gate prompt and the user's question, and
    emits SUFFICIENT_FACTS or INSUFFICIENT_FACTS. An external actor (this harness)
    then routes the rest of the conversation to the matching path prompt.

    Returns the internal mode: "decisive" (SUFFICIENT_FACTS) or
    "facts_gathering" (INSUFFICIENT_FACTS).

    When `enable_reasoning` is set, the gate call runs in thinking mode and, if a
    `reasoning_sink` list is provided, the gate's chain-of-thought is appended to it.
    """

    def _classify(text: str) -> Optional[str]:
        # "SUFFICIENT_FACTS" is a substring of "INSUFFICIENT_FACTS", so check the
        # insufficient token first to avoid a false positive match.
        if re.search(r"INSUFFICIENT_FACTS", text or "", re.IGNORECASE):
            return "facts_gathering"
        if re.search(r"SUFFICIENT_FACTS", text or "", re.IGNORECASE):
            return "decisive"
        return None

    messages = [{"role": "user", "content": user_question}]
    for _ in range(2):  # one initial attempt plus one retry on ambiguous output
        raw, reasoning = call_llm_with_reasoning(
            model, UNIFIED_GATE_PROMPT, messages, enable_reasoning=enable_reasoning
        )
        if reasoning_sink is not None:
            reasoning_sink.append({"kind": "gate", "reasoning": reasoning})
        mode = _classify(raw)
        if mode is not None:
            return mode

    # Ambiguous output: default conservatively to facts gathering.
    print("    [UNIFIED GATE] ambiguous output; defaulting to facts_gathering.")
    return "facts_gathering"


def select_unified_path_prompt(mode: str) -> str:
    """Return the standalone path system prompt for the gate decision."""
    if mode == "decisive":
        return UNIFIED_SUFFICIENT_PROMPT
    if mode == "facts_gathering":
        return UNIFIED_INSUFFICIENT_PROMPT
    raise ValueError(f"Unsupported unified mode: {mode}")


# Rubric Evaluation (LLM-as-Judge)

JUDGE_PROMPT_VERSION = "v4-2026-08-20"

JUDGE_SYSTEM_PROMPT = """You are an expert legal evaluator assessing the quality of an LLM's response to a Singapore law question, applying the same grading guidelines as human contributors. The goal of this project is to ensure that chatbots provide accurate, Singapore-specific answers when users in Singapore ask questions in English that have legal ramifications.

You will be given:
1. The original question
2. The full conversation (including any clarifying questions and answers)
3. A set of rubric criteria to evaluate, each tagged with its dimension (Issue, Rule Recall, Application, or Conclusion)

For each rubric criterion, decide whether the assistant's responses in the conversation meet it: Met ("passed": true) or Not Met ("passed": false). When a criterion is ambiguous, apply the rules below rather than personal judgment.

Core "Met" bar (applies throughout):
- Grade substance, not phrasing. A criterion is Met if the response conveys the required point in any wording; it need not mirror the criterion's words.
- Met requires the point to be present and correct, not merely gestured at. A vague allusion a layperson couldn't recognize as the point = Not Met.
- Partial = Not Met (binary; no half credit). If a criterion bundles two required elements and only one is present, mark Not Met.
- Judge each criterion independently, only against what it asks. Correct content elsewhere doesn't help it; errors elsewhere don't fail it.
- Match the criterion's specificity. Specifically named points must be specifically met; generally named points must be generally met.
- In general, these guidelines support a strict reading and application of the rubric criteria.

Dimension-specific rules:
- Issue criteria ("Identifies whether ..."): Met only if the response explicitly surfaces the specific variable or scope condition as something the outcome depends on. A response can do this by asking about the variable, conditioning its analysis on it, or stating its legal significance outright. The test: would a layperson reading the response learn that this specific fact matters to the outcome? A passing mention inside general doctrine coverage that does not treat the variable as a fact the outcome turns on is Not Met.
- Rule Recall criteria ("State that under section X of the Y Act, ..."): Met if the legal rule is stated correctly — the exact section number is NOT required (a correct citation is a bonus). A wrong statute/section or misstated rule = Not Met (incorrect law is worse than uncited law). Require the citation only when the provision itself is the point (e.g., "Identify Section 320(g) as the provision defining grievous hurt"). Leniency on the section number does not extend to omitting the rule itself.
- Application criteria ("Applies fact X ..."): Met only if the response uses the fact in its legal reasoning (fact -> rule/outcome). A fact merely restated, or not referenced at all, is Not Met. Any linkage between fact and rule/outcome suffices; the response does not need to use the word "apply".
- Conclusion criteria ("Conclude that ..."): Met if the response commits to a direct answer/position on the user's question (caveats are fine). A hedged answer (e.g., "you are likely to face ...") still counts as a conclusion if it identifies a specific outcome, and earlier clarifying questions do not negate a final committed answer. Not Met if the response only lists considerations, asks for more facts, or never commits. An explicit "I can't answer yet" / INSUFFICIENT_FACTS_GATHERED is not a conclusion = Not Met. Conditional forks ("if dangerous then X; if negligent then Y") with the deciding fact left unresolved do not count as a conclusion.

Cross-cutting rules:
- Criteria with multiple elements ("State A and B"): all required elements must be present to be Met. Conversely, if all are present, mark Met — do not over-require. Grade each element in the context the criterion specifies (a figure stated correctly but attached to a different case than the one the criterion names does not help).
- Criteria with qualifiers ("Outcome is X if Y is met"): grade the qualifier at the criterion's level of specificity. Conditioning only on unnamed, generic qualifications when the criterion names a specific gate is Not Met; stating the outcome as unconditional, with the named qualifier missing entirely, is Not Met.

Respond with a JSON array where each element has:
- "id": the rubric criterion id
- "passed": true (Met) or false (Not Met)
- "reasoning": a brief explanation (1-2 sentences)

Include exactly one element for every rubric criterion, in the order given.
Respond ONLY with the JSON array, no other text."""


def evaluate_rubric_with_judge(
    conversation: list[dict],
    scenario: dict,
    judge_model: str,
) -> list[dict]:
    def _coerce_passed(value):
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "pass", "passed", "1"}:
                return True
            if lowered in {"false", "no", "fail", "failed", "0"}:
                return False
        return None

    """Use an LLM judge to evaluate rubric items against the conversation."""

    rubric_text = "\n".join(
        f"- ID {item['id']} [dimension: {item['dimension']}]: {item['item']}"
        for item in scenario["rubric"]
    )

    conv_text = "\n".join(
        f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
        for turn in conversation
    )

    eval_prompt = f"""Original Question: {scenario['user_question']}

Full Conversation:
{conv_text}

Rubric Items to Evaluate:
{rubric_text}

Evaluate each rubric item. Respond with a JSON array only."""

    try:
        response = call_llm(
            judge_model,
            JUDGE_SYSTEM_PROMPT,
            [{"role": "user", "content": eval_prompt}]
        )

        # Parse JSON from response
        # Handle potential markdown code blocks
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        raw_results = json.loads(cleaned)
        if not isinstance(raw_results, list):
            raise ValueError("Judge response is not a JSON array.")

        normalized_results = []
        for row in raw_results:
            if not isinstance(row, dict):
                continue
            normalized_results.append({
                "id": row.get("id"),
                "passed": _coerce_passed(row.get("passed")),
                "reasoning": str(row.get("reasoning", "")),
            })
        return normalized_results

    except Exception as e:
        print(f"    [Judge Error] {e}")
        # Return all items as unevaluated
        return [
            {"id": item["id"], "passed": None, "reasoning": f"Evaluation error: {e}"}
            for item in scenario["rubric"]
        ]


# Multi-Turn Conversation Runner

def run_conversation(
    model: str,
    prompt_key: str,
    system_prompt: str,
    scenario: dict,
    max_turns: int = 8,
    simulator_model: str = "anthropic/claude-opus-4-8",
    log_conversation: bool = False,
    enable_reasoning: bool = False,
    reasoning_sink: Optional[list] = None,
) -> tuple[list[dict], dict, str, str, str, list[dict]]:
    """
    Run a multi-turn conversation with the model, using an LLM-based
    user simulator (inspired by LHAW) to answer clarifying questions.

    The simulator is an LLM that has access to the ground-truth
    point-instance values and responds naturally as a layperson.

    When `enable_reasoning` is set, ONLY the model-under-test turns (and the gate)
    run in thinking mode; the simulator/classifier/router keep their plain behavior.
    If `reasoning_sink` (a list) is provided, each model-under-test turn's
    chain-of-thought is appended as {"turn": i, "kind": "answer", "reasoning": ...}
    (and the gate as {"kind": "gate", ...}). This is opt-in so the broader run_trial
    pathway is unaffected.

    Returns:
        - conversation: list of message dicts
        - variables_asked: dict mapping variable keys to bool
        - strategy: "clarified" or "enumerated"
        - selected_mode: "facts_gathering" or "decisive"
        - model_thinks_question_type: "underspecified" or "fully_specified"
    """
    conversation = []
    asked_at_least_one = False
    used_variable_keys = set()
    simulator_selected_keys: set[str] = set()
    simulator_selections: list[dict] = []
    previous_simulator_response = ""
    previous_simulator_keys = set()
    awaiting_final_assistant_after_handoff = False
    unified_mode: Optional[str] = None
    runtime_system_prompt = system_prompt

    # Initial user message
    conversation.append({
        "role": "user",
        "content": scenario["user_question"]
    })
    if log_conversation:
        print("    --- Conversation ---")
        print("    >>> USER >>>")
        print(f"    {scenario['user_question']}")
        print("    <<< USER <<<")

    if prompt_key in UNIFIED_PROMPT_KEYS:
        # External-actor routing: the model under test first emits a fact-sufficiency
        # decision, then we run the rest of the conversation with only the matching
        # standalone path prompt (no combined both-paths prompt, no hard locks).
        unified_mode = run_unified_gate(
            model,
            scenario["user_question"],
            enable_reasoning=enable_reasoning,
            reasoning_sink=reasoning_sink,
        )
        runtime_system_prompt = select_unified_path_prompt(unified_mode)
        if log_conversation:
            decision_token = "SUFFICIENT_FACTS" if unified_mode == "decisive" else "INSUFFICIENT_FACTS"
            print(f"    [UNIFIED GATE] {decision_token} -> {unified_mode}")

    for turn in range(max_turns):
        # Call the model being evaluated (may route through web-search when
        # WEB_SEARCH_ENABLED). Per-turn tool_uses events and wall time are
        # captured onto the reasoning_sink so downstream analysis can slice
        # by turn without extra plumbing.
        _turn_tool_uses: list = []
        _turn_t0 = time.time()
        response_text, turn_reasoning, _turn_rounds = call_llm_with_reasoning_and_web_search(
            model, runtime_system_prompt, conversation,
            enable_reasoning=enable_reasoning,
            tool_uses_sink=_turn_tool_uses,
        )
        _turn_dt_s = round(time.time() - _turn_t0, 3)
        if reasoning_sink is not None:
            reasoning_sink.append({
                "turn": turn, "kind": "answer", "reasoning": turn_reasoning,
                "dt_s": _turn_dt_s, "n_rounds": _turn_rounds,
                "tool_uses": _turn_tool_uses,
            })
        conversation.append({
            "role": "assistant",
            "content": response_text
        })
        if log_conversation:
            print(f"    [ASSISTANT turn {turn + 1}]")
            print(f"    {response_text}")

        # If we already injected a decisive handoff in the previous user turn,
        # this assistant response is the final chance to conclude.
        if awaiting_final_assistant_after_handoff:
            break

        # Classify assistant turn as clarifying vs definitive vs neither.
        classification = classify_assistant_message(response_text, call_llm)
        has_question = classification.get("has_clarifying_question", False)
        has_answer = classification.get("has_definitive_answer", False)

        # The SUFFICIENT_FACTS (decisive) path is single-turn by construction: its
        # prompt forbids clarifying questions, so stop after the first response.
        if unified_mode == "decisive":
            break

        # Prioritize concrete clarifying questions over provisional conclusions.
        # If both flags are true, continue clarification instead of stopping.
        if has_question:
            asked_at_least_one = True
            # Final allowed iteration: the model just asked a clarifying question.
            # If we let the simulator answer here, the loop terminates with a
            # trailing user turn the model can never respond to, breaking the
            # "conversation ends on assistant" invariant. Stop now — the log
            # still shows the model's unanswered question.
            if turn == max_turns - 1:
                break
            simulator_response, selected_keys, handoff_injected = simulate_user_response(
                conversation,
                scenario,
                call_llm,
                used_variable_keys=used_variable_keys,
                previous_simulator_response=previous_simulator_response,
                previous_simulator_keys=previous_simulator_keys,
                simulator_model=simulator_model,
            )
            conversation.append({
                "role": "user",
                "content": simulator_response
            })
            if log_conversation:
                print("    >>> USER >>>")
                print(f"    {simulator_response}")
                print("    <<< USER <<<")

            previous_simulator_response = simulator_response
            previous_simulator_keys = set(selected_keys)
            simulator_selected_keys.update(selected_keys)
            simulator_selections.append({
                "assistant_turn": turn + 1,
                "selected_variables": sorted(selected_keys),
                "handoff": handoff_injected,
            })

            if handoff_injected:
                awaiting_final_assistant_after_handoff = True
        elif has_answer:
            # Model gave a definitive answer — stop the conversation.
            break
        else:
            # No concrete question and no definitive answer.
            break

    # Invariant: every recorded conversation must end on an assistant turn.
    # Defensive trim in case any future control-flow change reintroduces a
    # trailing user turn (the main path above already prevents this by breaking
    # before simulating on the final iteration).
    while conversation and conversation[-1].get("role") == "user":
        conversation.pop()

    variables_asked = {
        k: k in simulator_selected_keys for k in (scenario.get("variables") or {})
    }
    strategy = "clarified" if asked_at_least_one else "enumerated"
    if unified_mode is not None:
        selected_mode = unified_mode
        if unified_mode == "facts_gathering":
            model_thinks_question_type = "underspecified"
        else:
            model_thinks_question_type = "fully_specified"
    else:
        if asked_at_least_one:
            selected_mode = "facts_gathering"
            model_thinks_question_type = "underspecified"
        else:
            selected_mode = "decisive"
            model_thinks_question_type = "fully_specified"
    if log_conversation:
        print("    --- End conversation ---")

    return (
        conversation, variables_asked, strategy, selected_mode,
        model_thinks_question_type, simulator_selections,
    )


# Main Experiment Runner

def run_trial(
    model: str,
    model_info: dict,
    prompt_key: str,
    prompt_info: dict,
    scenario: dict,
    max_turns: int = 8,
) -> TrialResult:
    """Run a single trial: one model × one prompt variant."""

    start_time = time.time()
    result = TrialResult(
        model=model,
        model_label=model_info["label"],
        prompt_key=prompt_key,
        prompt_label=prompt_info["label"],
        scenario=scenario["title"],
    )

    max_trial_attempts = int(os.getenv("TRIAL_MAX_RETRIES", "3"))
    reasoning_sink: list = []
    for trial_attempt in range(max_trial_attempts):
        result = TrialResult(
            model=model,
            model_label=model_info["label"],
            prompt_key=prompt_key,
            prompt_label=prompt_info["label"],
            scenario=scenario["title"],
        )
        attempt_start = time.time()
        # Reset the sink for this attempt so retries don't inherit stale reasoning
        # from a prior failed attempt.
        reasoning_sink = []

        try:
            # Run the multi-turn conversation with reasoning enabled for the
            # model under test. The helper LLMs (simulator/classifier) go
            # through call_llm, which uses the non-reasoning path.
            (
                conversation,
                variables_asked,
                strategy,
                selected_mode,
                model_thinks_question_type,
                simulator_selections,
            ) = run_conversation(
                model=model,
                prompt_key=prompt_key,
                system_prompt=prompt_info["prompt"],
                scenario=scenario,
                max_turns=max_turns,
                enable_reasoning=True,
                reasoning_sink=reasoning_sink,
            )

            result.conversation = conversation
            result.variables_asked = variables_asked
            result.num_vars_asked = sum(1 for v in variables_asked.values() if v)
            result.strategy = strategy
            result.selected_mode = selected_mode
            result.model_thinks_question_type = model_thinks_question_type
            result.simulator_selections = simulator_selections

            result.rubric_results = [
                {**item, "passed": None, "reasoning": "Not graded"}
                for item in scenario["rubric"]
            ]

        except Exception as e:
            result.error = str(e)
            print(f"    [Error] {e}")

        result.duration_seconds = round(time.time() - attempt_start, 1)

        if result.error is None:
            break
        if trial_attempt < max_trial_attempts - 1 and is_retryable_llm_error(Exception(result.error)):
            delay = min(90.0, 5.0 * (2 ** trial_attempt))
            print(
                f"    [Trial retry {trial_attempt + 2}/{max_trial_attempts}] "
                f"retryable error; waiting {delay:.1f}s before restarting trial...",
                flush=True,
            )
            time.sleep(delay)
            continue
        break

    result.duration_seconds = round(time.time() - start_time, 1)

    # CoT sidecar: one JSONL record per trial (task × model × prompt ×
    # question_type). No-op if configure_cot_writer() was not called (e.g.
    # when run_eval is imported as a library rather than run via main()).
    # Web-search runs additionally derive tool_uses_by_turn and per_turn_dt_s
    # from the answer-kind entries in reasoning_sink for easy downstream analysis.
    answer_turns = [e for e in (reasoning_sink or []) if e.get("kind") == "answer"]
    per_turn_dt_s = [e.get("dt_s") for e in answer_turns if e.get("dt_s") is not None]
    tool_uses_by_turn = [
        {"turn": e.get("turn"), "n_rounds": e.get("n_rounds", 0),
         "dt_s": e.get("dt_s"), "tool_uses": e.get("tool_uses", []) or []}
        for e in answer_turns
    ]
    _write_cot_record({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_id": scenario.get("task_id"),
        "scenario_key": scenario.get("output_slug") or scenario.get("task_id"),
        "scenario_title": scenario.get("title"),
        "model": model,
        "model_label": model_info["label"],
        "prompt_key": prompt_key,
        "prompt_label": prompt_info["label"],
        "question_type": scenario.get("question_type"),
        "is_converted_fully_specified": scenario.get("is_converted_fully_specified", False),
        "answer_conversation": result.conversation,
        "reasoning_turns": reasoning_sink,
        "per_turn_dt_s": per_turn_dt_s,
        "tool_uses_by_turn": tool_uses_by_turn,
        "variables_asked": result.variables_asked,
        "simulator_selections": result.simulator_selections,
        "web_search_enabled": WEB_SEARCH_ENABLED,
        "error": result.error,
        "duration_seconds": result.duration_seconds,
        "run_id": RUN_ID,
        "seed": PRIMARY_SEED,
    })

    return result


def run_experiment(
    models: list[str],
    prompts: list[str],
    scenario: dict,
    max_turns: int = 8,
) -> list[TrialResult]:
    """Run the full experiment matrix."""

    results = []
    total = len(models) * len(prompts)
    current = 0

    for model in models:
        model_info = MODELS[model]
        for prompt_key in prompts:
            prompt_info = SYSTEM_PROMPTS[prompt_key]
            current += 1

            print(f"\n[{current}/{total}] {model_info['label']} × {prompt_info['label']}")

            result = run_trial(
                model=model,
                model_info=model_info,
                prompt_key=prompt_key,
                prompt_info=prompt_info,
                scenario=scenario,
                max_turns=max_turns,
            )

            print(f"    Strategy: {result.strategy}")
            total_variables = len(scenario.get("variables", {}))
            print(f"    Variables asked: {result.num_vars_asked}/{total_variables}")
            assistant_turns = sum(1 for t in result.conversation if t["role"] == "assistant")
            print(f"    Turns: {assistant_turns} assistant turns ({len(result.conversation)} total messages)")
            print(f"    Duration: {result.duration_seconds}s")
            if result.error:
                print(f"    ERROR: {result.error}")

            results.append(result)

            # Small delay to avoid rate limits
            time.sleep(1)

    return results


def save_results(
    results: list[TrialResult],
    output_path: str,
    scenario: dict,
    merge_existing: bool = False,
):
    """Save results to JSON file.

    When ``merge_existing`` is True and a file already exists at ``output_path``,
    new trials are merged into the existing trials instead of overwriting them.
    A pre-existing trial is replaced when it matches an incoming trial by either
    ``(model, prompt_key)`` or ``(model_label, prompt_key)`` -- this keeps trials
    for other models intact while cleanly superseding stale runs even if the
    model id changed but the display label stayed the same.
    """
    new_trials = [asdict(r) for r in results]
    trials_out = new_trials

    if merge_existing and Path(output_path).exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            existing_trials = existing.get("trials", []) or []
        except (OSError, json.JSONDecodeError):
            existing_trials = []

        incoming_model_keys = {(t.get("model"), t.get("prompt_key")) for t in new_trials}
        incoming_label_keys = {
            (t.get("model_label"), t.get("prompt_key")) for t in new_trials
        }
        kept = [
            t
            for t in existing_trials
            if (t.get("model"), t.get("prompt_key")) not in incoming_model_keys
            and (t.get("model_label"), t.get("prompt_key")) not in incoming_label_keys
        ]
        replaced = len(existing_trials) - len(kept)
        trials_out = kept + new_trials
        print(
            f"  Merge: kept {len(kept)} existing trial(s), "
            f"replaced {replaced}, added {len(new_trials)}"
        )

    output = {
        "timestamp": datetime.now().isoformat(),
        "scenario": scenario["title"],
        "trials": trials_out,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


def slugify_filename(text: str) -> str:
    """Create a filesystem-safe lowercase slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return re.sub(r"_+", "_", slug).strip("_")


# CLI

def _resolve_scenario_subset(subset_path: Path, parser: argparse.ArgumentParser) -> list[str]:
    if not subset_path.exists():
        parser.error(f"--scenario-subset file not found: {subset_path}")
    with subset_path.open() as f:
        raw = json.load(f)
    if isinstance(raw, list):
        wanted = list(raw)
    elif isinstance(raw, dict):
        wanted = list(raw.get("scenario_keys") or raw.get("task_ids") or [])
    else:
        parser.error(f"--scenario-subset must be a JSON list or object with 'scenario_keys'/'task_ids'; got {type(raw).__name__}")
    if not wanted:
        parser.error(f"--scenario-subset {subset_path} is empty")
    slug_to_key = {(sc.get("output_slug") or k): k for k, sc in SCENARIOS.items()}
    task_to_key = {sc.get("task_id"): k for k, sc in SCENARIOS.items() if sc.get("task_id")}
    resolved: list[str] = []
    unresolved: list[str] = []
    for x in wanted:
        key = x if x in SCENARIOS else slug_to_key.get(x) or task_to_key.get(x)
        if key is None:
            unresolved.append(x)
        elif key not in resolved:
            resolved.append(key)
    if unresolved:
        print(f"[--scenario-subset] {len(unresolved)} entries did not resolve, e.g.: {unresolved[:3]}")
    print(f"[--scenario-subset] resolved to {len(resolved)} scenarios (from {len(wanted)} entries in {subset_path.name})")
    return resolved


def main():
    global RESULTS_ROOT_DIR, PRIMARY_SEED, RUN_ID, WEB_SEARCH_ENABLED
    parser = argparse.ArgumentParser(
        description=(
            "Generate multi-turn conversations for the Singapore legal benchmark. "
            "Scores are added afterwards by scripts/run_panel_grading.py."
        )
    )
    parser.add_argument(
        "--results-root", default=RESULTS_ROOT_DIR,
        help=f"Folder for per-scenario result JSON (default: {RESULTS_ROOT_DIR}).",
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS.keys()),
        metavar="MODEL",
        help="Models to evaluate (default: DEFAULT_MODELS; any id in MODELS is accepted).",
    )
    parser.add_argument(
        "--prompts", nargs="+", default=None, choices=list(SYSTEM_PROMPTS.keys()),
        help=(
            f"Prompt key recorded on each trial. Default: {UNIFIED_PROMPT_KEY!r}, or "
            "'unified_sufficiency_gate_v1_websearch' with --web-search."
        ),
    )
    parser.add_argument(
        "--max-turns", type=int, default=8,
        help="Maximum assistant turns per conversation (default: 8).",
    )
    parser.add_argument(
        "--allow-overwrite", action="store_true",
        help="Overwrite existing per-scenario result JSON.",
    )
    parser.add_argument(
        "--merge-existing", action="store_true",
        help=(
            "Merge new trials into existing per-scenario JSON. Trials with the same "
            "(model or label, prompt) are replaced; other models are kept."
        ),
    )
    parser.add_argument(
        "--new-data-only", action="store_true",
        help="Skip scenarios that already have a result JSON under --results-root (resume).",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of shards to split the selected scenarios into (default: 1).",
    )
    parser.add_argument(
        "--shard-index", type=int, default=1,
        help="1-based shard index for this process (default: 1).",
    )
    parser.add_argument(
        "--print-selected-keys", action="store_true",
        help="Print the scenario keys selected for this process before running.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Seed for model-under-test calls, sent only to providers that accept it "
            "(OpenAI, Mistral, local vLLM Gemma-3). The simulator and classifier are "
            "never seeded, so every seed sees the same simulated user. Stored in each "
            "CoT record."
        ),
    )
    parser.add_argument(
        "--run-id", type=int, default=None,
        help="Run index (1-based) stored in every CoT record.",
    )
    parser.add_argument(
        "--web-search", action="store_true",
        help=(
            "Give the model under test web search on its answer turns (provider-native "
            "search for OpenAI/Anthropic/Gemini, Serper + Jina tool loop otherwise). "
            "The gate call and the simulator stay closed-book. Models in "
            "WEB_SEARCH_BLOCKED_MODELS fall back to closed-book."
        ),
    )
    parser.add_argument(
        "--scenario-subset", type=str, default=None,
        help=(
            "JSON file listing scenario keys, output slugs or task ids (a bare list, or an "
            "object with 'scenario_keys' or 'task_ids'). Sharding splits the subset."
        ),
    )
    args = parser.parse_args()

    if args.seed is not None:
        PRIMARY_SEED = args.seed
    if args.run_id is not None:
        RUN_ID = args.run_id
    if PRIMARY_SEED is not None:
        print(f"Primary-model seed: {PRIMARY_SEED} (run_id={RUN_ID})")
    if args.web_search:
        WEB_SEARCH_ENABLED = True
        print("Web search: ENABLED (multi-round tool loop on model-under-test answer turns)")
    prompts = args.prompts or [
        "unified_sufficiency_gate_v1_websearch" if args.web_search else UNIFIED_PROMPT_KEY
    ]

    RESULTS_ROOT_DIR = str(Path(args.results_root))

    load_dotenv_if_present()
    if not os.environ.get("API_KEY"):
        print("Warning: API_KEY not set. LiteLLM proxy requests will fail.")

    # Open the CoT sidecar for this run. One JSONL file per invocation, path
    # includes the UTC start time so prior runs are preserved. Set
    # EXPERIMENT_COT_JSONL_PATH="" to disable, or pass an explicit path.
    _cot_env = os.getenv("EXPERIMENT_COT_JSONL_PATH")
    _cot_target = configure_cot_writer(_cot_env, truncate=True)
    if _cot_target is not None:
        print(f"CoT sidecar: {_cot_target} (per-trial reasoning traces)")
    import atexit as _atexit
    _atexit.register(close_cot_writer)

    scenario_keys = list(SCENARIOS.keys())
    if not scenario_keys:
        parser.error("No scenarios loaded from the dataset (see excel_scenarios.py).")
    if args.num_shards < 1:
        parser.error(f"--num-shards must be >= 1 (got {args.num_shards}).")
    if args.shard_index < 1 or args.shard_index > args.num_shards:
        parser.error(f"--shard-index must be in [1, {args.num_shards}] (got {args.shard_index}).")

    if args.scenario_subset:
        scenario_keys = _resolve_scenario_subset(Path(args.scenario_subset), parser)

    if args.num_shards > 1:
        scenario_keys = [
            key for idx, key in enumerate(scenario_keys)
            if idx % args.num_shards == args.shard_index - 1
        ]
        if not scenario_keys:
            print(f"[Shard Selection] No scenarios for shard {args.shard_index}/{args.num_shards}.")
            return
        print(
            f"[Shard Selection] shard={args.shard_index}/{args.num_shards}, "
            f"selected_scenarios={len(scenario_keys)}"
        )
    if args.print_selected_keys:
        print("[Scenario Selection] Keys for this run:")
        for key in scenario_keys:
            print(f"  - {key}")

    ran = 0
    skipped_existing = 0
    for scenario_key in scenario_keys:
        scenario = SCENARIOS[scenario_key]
        slug = scenario.get("output_slug") or slugify_filename(scenario["title"])
        output_json = Path(RESULTS_ROOT_DIR) / slug / f"{slug}_base.json"

        if args.new_data_only and output_json.exists():
            skipped_existing += 1
            continue
        if output_json.exists() and not args.allow_overwrite and not args.merge_existing:
            parser.error(
                f"Refusing to overwrite existing output file: {output_json}. "
                "Re-run with --allow-overwrite, --merge-existing or --new-data-only."
            )

        print("=" * 60)
        print(f"Scenario key: {scenario_key}")
        print(f"Scenario:     {scenario['title']}")
        print(f"Models:       {', '.join(MODELS[m]['label'] for m in args.models)}")
        print(f"Prompts:      {', '.join(SYSTEM_PROMPTS[p]['label'] for p in prompts)}")
        print(f"Max turns:    {args.max_turns}")
        print(f"Total trials: {len(args.models) * len(prompts)}")
        print("=" * 60)

        results = run_experiment(
            models=args.models,
            prompts=prompts,
            scenario=scenario,
            max_turns=args.max_turns,
        )
        save_results(results, str(output_json), scenario, merge_existing=args.merge_existing)
        ran += 1

    print(f"Done: ran {ran} scenario(s); skipped {skipped_existing} with existing results.")


if __name__ == "__main__":
    main()
