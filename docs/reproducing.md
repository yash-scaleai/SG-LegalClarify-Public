# Reproducing the paper's runs

The paper reports two conditions, closed-book and web search, each run with three
seeds (42, 43, 44) over all scenarios. Every run is two steps: generate the
conversations with `run_eval.py`, then grade them with the judge panel.

1. [Environment](#1-environment)
2. [Local models](#2-local-models)
3. [Models per condition](#3-models-per-condition)
4. [Generate](#4-generate)
5. [Grade](#5-grade)
6. [What will not match exactly](#6-what-will-not-match-exactly)
7. [Model settings](#7-model-settings)
8. [Serving the local models](#8-serving-the-local-models)

## 1. Environment

```bash
uv sync
cp .env.example .env
```

Set `API_KEY` and `LITELLM_BASE_URL` for your LiteLLM proxy. For web search also set
`SERPER_API_KEY` and `JINA_API_KEY`. Model ids in `run_eval.MODELS` are LiteLLM route
names (`provider/model`); configure your proxy so those names resolve. The helper models
need two more names on the proxy:

| Role | Model name the code calls |
|---|---|
| Fact selector for the simulated user, yes/no rewriter (`user_simulator.py`) | `anthropic/claude-opus-4-8` |
| Turn classifier (`user_simulator.py`) | `claude-opus-4-6` |
| Judges (`scripts/run_panel_grading.py`) | `azure_ai/DeepSeek-V4-Flash`, `openai/gpt-5.6-sol`, `fireworks_ai/kimi-k3` |

## 2. Local models

Three models run on local vLLM servers, one GPU each. Start them before a run (setup
and troubleshooting in [section 8](#8-serving-the-local-models)):

```bash
scripts/serve_sea_lion.sh          # Qwen-SEA-LION v4.5 27B, port 8001
scripts/serve_gemma_sea_lion.sh    # Gemma-SEA-LION v4 27B, port 8002
scripts/serve_gemma3_27b.sh        # Gemma 3 27B IT, port 8003
```

For web-search runs, start them with `WEB_SEARCH=1` so tool calling is enabled.

## 3. Models per condition

`run_eval.py` defaults to `run_eval.DEFAULT_MODELS`, which includes both Gemma 3 routes.
The paper used one Gemma 3 route per condition:

- **Closed-book:** `bedrock/google.gemma-3-27b-it`.
- **Web search:** the local `gemma-3-27b-it`. The Bedrock route cannot call tools, so
  in a web-search run it silently answers closed-book. Leave it out of web-search runs.

Some providers withdrew routes during the study. `MODELS` keeps every id that appears
in the released results, and marks retired ones `"default": False`:

| Model | Route used first | Route used later |
|---|---|---|
| DeepSeek V4 Pro | `fireworks_ai/deepseek-v4-pro` | `fireworks_ai/deepseek-v4-pro-0813` |
| DeepSeek V4 Flash | `fireworks_ai/deepseek-v4-flash` | `fireworks_ai/deepseek-v4-flash-0731` |
| Llama 4 Scout | `groq/meta-llama/llama-4-scout-17b-16e-instruct` | `bedrock/us.meta.llama4-scout-17b-instruct-v1:0` |

## 4. Generate

Models shared by both conditions, using the current route for each:

```bash
SHARED_MODELS="anthropic/claude-fable-5 anthropic/claude-opus-4-8 anthropic/claude-sonnet-5
  openai/gpt-5.6-sol openai/gpt-5.6-terra openai/gpt-5.6-luna
  gemini/gemini-3.1-pro-preview gemini/gemini-3.5-flash fireworks_ai/qwen3p7-plus
  fireworks_ai/kimi-k2p6 fireworks_ai/kimi-k3 fireworks_ai/deepseek-v4-pro-0813
  fireworks_ai/deepseek-v4-flash-0731 mistral/mistral-large-latest
  fireworks_ai/deployed/t5cvm6h4 bedrock/us.meta.llama4-scout-17b-instruct-v1:0
  sea-lion-v4.5-27b gemma-sea-lion-v4-27b"
```

Closed-book, three seeds in sequence (each seed runs as `N_SHARDS` parallel processes):

```bash
N_SHARDS=10 scripts/run_paper_3seeds.sh -- \
  --models $SHARED_MODELS bedrock/google.gemma-3-27b-it
```

Results go to `results_paper_run_1/`, `results_paper_run_2/` and `results_paper_run_3/`.

Web search:

```bash
N_SHARDS=10 RUN_BASENAME=paper_run_web_search scripts/run_paper_3seeds.sh -- \
  --web-search --models $SHARED_MODELS gemma-3-27b-it
```

With `--web-search`, trials are recorded under the prompt key
`unified_sufficiency_gate_v1_websearch`. The prompts themselves are identical; only the
answer turns get search tools.

Each shard logs to `logs/` and writes its chain-of-thought sidecar to `data/`. More shards
means more simultaneous requests to the proxy; lower `N_SHARDS` if you see 429 errors. To
resume after an interruption, run again with `-- --new-data-only` and a new `RUN_TAG`, so
the new sidecars don't overwrite the old ones. Scenarios that already have a result file
are skipped.

To add or re-run one model in an existing results folder, use `--merge-existing`. It
replaces that model's trials and keeps the rest.

## 5. Grade

For each results folder:

```bash
uv run python scripts/run_panel_grading.py --results-dir results_paper_run_1 --workers 3
uv run python scripts/run_panel_grading.py --results-dir results_paper_run_1 --status
uv run python scripts/run_panel_grading.py --results-dir results_paper_run_1 --heal
uv run python scripts/run_panel_grading.py --results-dir results_paper_run_1 --apply
```

- The first command asks each of the three judges about every trial. It can be resumed,
  and `--shard i/n` splits it across processes.
- `--heal` re-asks a judge wherever it returned no verdict for some item.
- `--apply` writes the final verdicts into each `_base.json`. An item passes if at least
  two of the three judges pass it. If a vote is missing, the majority of the remaining
  votes decides, and a tie counts as a fail.

Every judge uses the frozen prompt `run_eval.JUDGE_SYSTEM_PROMPT` (version
`v4-2026-08-20`). Its SHA-256 hash is stored with every vote.

## 6. What will not match exactly

- **Sampling:** most providers reject a `seed` parameter, so their answers vary between
  runs at the same seed. Only OpenAI, Mistral and the local Gemma 3 receive the seed.
- **Simulator:** the simulated user and classifier call Claude routes, which require
  temperature 1, so they are not deterministic either. They are never seeded.
- **Web search:** results depend on what Serper and the providers return on the day.
- **Retries:** a trial whose API call fails with a transient error is restarted up to
  three times; an empty answer is retried up to ten times.

## 7. Model settings

What `run_eval.py` sends to each model. The source of truth is `_build_llm_request` in
`run_eval.py`.

### Models under test

| Label | Model id(s) | Served by | Open weights |
|---|---|---|---|
| Fable 5 | `anthropic/claude-fable-5` | LiteLLM | no |
| Opus 4.8 | `anthropic/claude-opus-4-8` | LiteLLM | no |
| Sonnet 5 | `anthropic/claude-sonnet-5` | LiteLLM | no |
| GPT-5.6 Sol | `openai/gpt-5.6-sol` | LiteLLM | no |
| GPT-5.6 Terra | `openai/gpt-5.6-terra` | LiteLLM | no |
| GPT-5.6 Luna | `openai/gpt-5.6-luna` | LiteLLM | no |
| Gemini 3.1 Pro | `gemini/gemini-3.1-pro-preview` | LiteLLM | no |
| Gemini 3.5 Flash | `gemini/gemini-3.5-flash` | LiteLLM | no |
| Qwen3.7 Plus | `fireworks_ai/qwen3p7-plus` | LiteLLM (Fireworks) | no |
| Kimi K2.6 | `fireworks_ai/kimi-k2p6` | LiteLLM (Fireworks) | no |
| Kimi K3 | `fireworks_ai/kimi-k3` | LiteLLM (Fireworks) | no |
| DeepSeek V4 Pro | `fireworks_ai/deepseek-v4-pro`, `…-0813` | LiteLLM (Fireworks) | no |
| DeepSeek V4 Flash | `fireworks_ai/deepseek-v4-flash`, `…-0731` | LiteLLM (Fireworks) | no |
| Mistral 3 Large | `mistral/mistral-large-latest` | LiteLLM | no |
| Qwen3.6 27B | `fireworks_ai/deployed/t5cvm6h4` | LiteLLM (dedicated Fireworks deployment) | yes |
| Llama 4 Scout | `groq/meta-llama/llama-4-scout-17b-16e-instruct`, `bedrock/us.meta.llama4-scout-17b-instruct-v1:0` | LiteLLM | yes |
| Gemma 3 27B | `bedrock/google.gemma-3-27b-it` (closed-book), `gemma-3-27b-it` (local, web search) | LiteLLM / vLLM | yes |
| Qwen-SEA-LION v4.5 27B | `sea-lion-v4.5-27b` | vLLM | yes |
| Gemma-SEA-LION v4 27B | `gemma-sea-lion-v4-27b` | vLLM | yes |

### Closed-book settings

Reasoning is enabled for every model that supports it, at `REASONING_EFFORT=high`.
The per-turn output ceiling is `REASONING_MAX_TOKENS=16384` (8192 for Llama 4 Scout).

| Family | Reasoning parameters | Temperature |
|---|---|---|
| Anthropic | `thinking={type: adaptive, display: summarized}`, `output_config={effort: high}` | 1 |
| OpenAI GPT-5.6 | `reasoning_effort=high` | 1 |
| Gemini 3.x | `reasoning_effort=high` | provider default |
| DeepSeek (Fireworks) | `reasoning_effort=high` | 0.6 |
| Qwen (Fireworks, incl. Qwen3.6 27B deployment) | `reasoning_effort=high` | 0.7 |
| Kimi (Fireworks) | `reasoning_effort=high` | 0.6 |
| Mistral, Llama 4 Scout, Bedrock Gemma 3 | none (no reasoning mode) | 0 |
| Qwen-SEA-LION (local) | `chat_template_kwargs={enable_thinking: true}`; `top_p=0.95`, `top_k=20`, `min_p=0`, `repetition_penalty=1`, `presence_penalty=0` | 1.0 |
| Gemma-SEA-LION (local) | none; `top_p=0.95`, `top_k=64` | 1.0 |
| Gemma 3 27B (local) | none | 0 |

If a provider rejects the temperature, the call is retried without it. Chain-of-thought
is stored when the provider returns it (`reasoning_content`, `reasoning`, or Anthropic
thinking blocks); otherwise `null` is stored.

### Web-search settings

Answer turns go through `web_search/llm_client.py` instead. There, reasoning is requested
at `medium`, `max_tokens=8192`, and no sampling parameters are sent to any model. See
`web_search/README.md`.

### Helper models

The sufficiency gate is a call to the model under test with the closed-book settings
above, reasoning included, in both conditions. The simulator, classifier and judges never
use reasoning and are never seeded, so every seed sees the same simulated user. They use
`DEFAULT_MAX_TOKENS=16384`, with temperature 0, or 1 for GPT-5.x and Claude routes, which
reject 0.

| Role | Model |
|---|---|
| Fact selector, yes/no rewriter | `anthropic/claude-opus-4-8` |
| Turn classifier | `claude-opus-4-6` |
| Judge panel | `azure_ai/DeepSeek-V4-Flash`, `openai/gpt-5.6-sol`, `fireworks_ai/kimi-k3` |

### Seeds

`--seed` is sent only to OpenAI, Mistral and the local Gemma 3 (and, on the web-search
path, to all three local models). Other providers reject it, so run-to-run variation comes
from sampling. The seed and run id are stored in every chain-of-thought record.

### Retries

- API errors that look transient (429, 5xx, timeouts) are retried up to 10 times, with
  exponential backoff capped at 90 s.
- An empty answer is retried up to 10 times.
- A trial that still fails with a transient error is restarted up to 3 times.

Environment variables `LLM_MAX_RETRIES`, `LLM_EMPTY_RETRY_MAX` and `TRIAL_MAX_RETRIES`
override these limits.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `API_KEY` | LiteLLM proxy key | required |
| `LITELLM_BASE_URL` | LiteLLM proxy URL, ending in `/v1` | required |
| `SERPER_API_KEY`, `JINA_API_KEY` | Web search (custom path) | required with `--web-search` |
| `HF_TOKEN` | Download the local models | required for `scripts/serve_*.sh` |
| `REASONING_EFFORT` | Reasoning effort for API models | `high` |
| `REASONING_MAX_TOKENS`, `DEFAULT_MAX_TOKENS` | Output ceilings | `16384` |
| `SEALION_BASE_URL`, `GEMMA_SEALION_BASE_URL`, `GEMMA3_BASE_URL` | Local vLLM servers | `http://localhost:8001/v1`, `:8002`, `:8003` |
| `EXPERIMENT_COT_JSONL_PATH` | Chain-of-thought sidecar path (`""` disables) | `data/experiment_cot_<UTC>_pid<PID>.jsonl` |

## 8. Serving the local models

| Script | Hugging Face model | Served name | Port |
|---|---|---|---|
| `scripts/serve_sea_lion.sh` | `aisingapore/Qwen-SEA-LION-v4.5-27B-IT` | `sea-lion-v4.5-27b` | 8001 |
| `scripts/serve_gemma_sea_lion.sh` | `aisingapore/Gemma-SEA-LION-v4-27B-IT` | `gemma-sea-lion-v4-27b` | 8002 |
| `scripts/serve_gemma3_27b.sh` | `google/gemma-3-27b-it` | `gemma-3-27b-it` | 8003 |

Each needs one 80 GB GPU (the paper used H100s). All three serve with
`--max-model-len 65536`, `--tensor-parallel-size 1` and `--gpu-memory-utilization 0.90`.

### vLLM environment

Install vLLM in its own environment, separate from the one `uv sync` creates. The paper
used:

- `vllm==0.22.1` built for CUDA 12.9 (`+cu129`)
- `torch==2.11.0+cu129`, `torchvision==0.26.0+cu129`, `torchaudio==2.11.0+cu129`
- `transformers==5.11.0`
- an NVIDIA driver supporting CUDA 12.9

The PyPI vLLM wheel is built for CUDA 13. On a CUDA 12.9 driver, install the `+cu129`
wheel from the vLLM GitHub release, and torchvision/torchaudio from
`https://download.pytorch.org/whl/cu129`, so all packages share one CUDA build.

### Running the servers

```bash
scripts/serve_sea_lion.sh                    # picks the first idle GPU
GPU=3 PORT=8011 scripts/serve_gemma3_27b.sh  # choose GPU and port
WEB_SEARCH=1 scripts/serve_gemma_sea_lion.sh # enable tool calling for --web-search
```

If you change a port, point `run_eval.py` at it with `SEALION_BASE_URL`,
`GEMMA_SEALION_BASE_URL` or `GEMMA3_BASE_URL`.

Environment overrides read by `scripts/lib_vllm_serve.sh`:

| Variable | Purpose | Default |
|---|---|---|
| `VLLM_BIN` | vLLM executable | `vllm` on `PATH` |
| `CUDA_HOME` | CUDA toolkit | `/usr/local/cuda-12.9` |
| `GPU` | GPU index | first GPU with under 1 GiB in use |
| `MAX_MODEL_LEN` | Context window | `65536` |
| `GPU_MEM_UTIL` | GPU memory fraction | `0.90` |

### Per-model notes

**Qwen-SEA-LION v4.5** is a hybrid linear-attention model with a thinking mode.
- It is served with `--enforce-eager` (CUDA-graph and multi-GPU paths crashed) and
  `--reasoning-parser qwen3`, which puts the chain-of-thought in `reasoning_content`.
- With `WEB_SEARCH=1` it uses vLLM's built-in `qwen3_xml` tool parser.
- FlashInfer compiles a kernel at startup and must find `libcudart`. The script sets
  `CUDA_HOME`, `PATH` and `LD_LIBRARY_PATH` for this.

**Gemma-SEA-LION v4** and **Gemma 3** have no thinking mode. They write tool calls as a
fenced `tool_code` block, so `WEB_SEARCH=1` loads `web_search/gemma_tool_code_parser.py`
and `web_search/gemma3_tool_chat_template.jinja`.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `The NVIDIA driver on your system is too old` | You installed CUDA 13 wheels; install the `+cu129` builds. |
| `libcudart.so.13: cannot open shared object file` | Same cause; reinstall vLLM from the `+cu129` wheel. |
| `operator torchvision::nms does not exist` | torchvision/torchaudio are on a different CUDA build than torch; reinstall them as `+cu129`. |
| `CUDA driver initialization failed` while `nvidia-smi` works | Another CUDA lib dir on `LD_LIBRARY_PATH` shadows torch's runtime. The script sets it to `$CUDA_HOME/lib64` only. |
| `Address already in use` | Set `PORT=` and the matching `*_BASE_URL`. |
