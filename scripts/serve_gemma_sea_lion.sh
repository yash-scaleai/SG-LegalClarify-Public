#!/usr/bin/env bash
# Serve aisingapore/Gemma-SEA-LION-v4-27B-IT (Gemma 3 — mainstream arch, not a reasoning model).
# Usage: ./scripts/serve_gemma_sea_lion.sh    (auto-picks an idle GPU, port 8002)
#        GPU=7 PORT=8006 ./scripts/serve_gemma_sea_lion.sh
#        WEB_SEARCH=1 ./scripts/serve_gemma_sea_lion.sh   (enable tool-calling for web search)
# See docs/reproducing.md.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib_vllm_serve.sh"

# WEB_SEARCH=1 opt-in: enable OpenAI-style tool calling so the model can drive the
# Serper+Jina search loop (web_search/). Gemma-3 emits calls as ```tool_code``` Python
# blocks, which no built-in vLLM parser matches — so we load a custom parser plugin plus
# a tool-aware chat template. Off by default — plain serving is byte-for-byte unchanged.
TOOL_ARGS=()
if [ -n "${WEB_SEARCH:-}" ]; then
  TOOL_ARGS=(
    --enable-auto-tool-choice
    --tool-parser-plugin "$REPO_ROOT/web_search/gemma_tool_code_parser.py"
    --tool-call-parser tool_code
    --chat-template "$REPO_ROOT/web_search/gemma3_tool_chat_template.jinja"
  )
fi

# Gemma 3 works out of the box: CUDA graphs on (better throughput), no reasoning parser
# needed (answers come back clean in message.content).
launch_vllm aisingapore/Gemma-SEA-LION-v4-27B-IT gemma-sea-lion-v4-27b "${PORT:-8002}" \
  ${TOOL_ARGS[@]+"${TOOL_ARGS[@]}"}
