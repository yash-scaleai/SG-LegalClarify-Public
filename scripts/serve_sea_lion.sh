#!/usr/bin/env bash
# Serve aisingapore/Qwen-SEA-LION-v4.5-27B-IT (Qwen3.5 hybrid: linear-attn + Mamba, reasoning).
# Usage: ./scripts/serve_sea_lion.sh          (auto-picks an idle GPU, port 8001)
#        GPU=6 PORT=8005 ./scripts/serve_sea_lion.sh
#        WEB_SEARCH=1 ./scripts/serve_sea_lion.sh   (enable tool-calling for web search)
# See docs/reproducing.md.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_vllm_serve.sh"

# Qwen3.5 hybrid specifics:
#   --enforce-eager       : TP/CUDA-graph paths segfault for this arch; eager is the safe mode.
#   --reasoning-parser qwen3 : model emits chain-of-thought; this splits it into
#                              message.reasoning_content, leaving message.content clean.
# WEB_SEARCH=1 opt-in: enable OpenAI-style tool calling so the model can drive the
# Serper+Jina search loop (web_search/). qwen3_xml matches Qwen3.5's tool-call format
# (hermes does NOT parse it). Off by default — plain serving is byte-for-byte unchanged.
TOOL_ARGS=()
if [ -n "${WEB_SEARCH:-}" ]; then
  TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser qwen3_xml)
fi

launch_vllm aisingapore/Qwen-SEA-LION-v4.5-27B-IT sea-lion-v4.5-27b "${PORT:-8001}" \
  --enforce-eager \
  --reasoning-parser qwen3 \
  ${TOOL_ARGS[@]+"${TOOL_ARGS[@]}"}
