#!/usr/bin/env bash
# Serve google/gemma-3-27b-it (stock Google Gemma 3 27B IT — the base model that
# aisingapore/Gemma-SEA-LION-v4-27B-IT is fine-tuned from).
# Usage: ./scripts/serve_gemma3_27b.sh          (auto-picks an idle GPU, port 8003)
#        GPU=6 PORT=8013 ./scripts/serve_gemma3_27b.sh
#        WEB_SEARCH=1 ./scripts/serve_gemma3_27b.sh   (enable tool-calling for web search)
# See docs/reproducing.md.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib_vllm_serve.sh"

# WEB_SEARCH=1 opt-in: same story as the Gemma-SEA-LION sibling (this is its base model,
# identical Gemma-3 ```tool_code``` call format), so it reuses the same custom parser plugin
# and tool-aware chat template. Off by default — plain serving is byte-for-byte unchanged.
TOOL_ARGS=()
if [ -n "${WEB_SEARCH:-}" ]; then
  TOOL_ARGS=(
    --enable-auto-tool-choice
    --tool-parser-plugin "$REPO_ROOT/web_search/gemma_tool_code_parser.py"
    --tool-call-parser tool_code
    --chat-template "$REPO_ROOT/web_search/gemma3_tool_chat_template.jinja"
  )
fi

# Same arch as the Gemma-SEA-LION sibling (Gemma3ForConditionalGeneration): works out of
# the box — CUDA graphs on, no reasoning parser (answers come back clean in
# message.content). Gated repo on HF, so HF_TOKEN (set by the helper from .env) is required
# for the initial weight download.
launch_vllm google/gemma-3-27b-it gemma-3-27b-it "${PORT:-8003}" \
  ${TOOL_ARGS[@]+"${TOOL_ARGS[@]}"}
