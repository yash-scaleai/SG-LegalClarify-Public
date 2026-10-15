# Web search client

Gives the model under test live web search on its answer turns when `run_eval.py` runs
with `--web-search`. The entry point is
`llm_client.chat_with_web_search_messages(model, messages, reasoning_mode=None, force=False, seed=None)`,
which takes an OpenAI-style message list and returns a `WebSearchResponse` with the
final `content`, the cited or fetched URLs, the reasoning trace, the number of tool
rounds (`n_rounds`) and a per-call log of every search and page visit (`tool_uses`).

The gate call and the user simulator never search; only answer turns do.

## Two paths

- **Native search** for models listed in `config.WEB_SEARCH_MODEL_METHOD` (OpenAI GPT-5.6,
  Anthropic, Gemini). The provider runs the search server-side. We see only what the
  response exposes: citations, grounding metadata and a flag showing a search happened.
- **Custom search** for every other model, including the three local vLLM models.
  The model gets two tools: `search` (Google results through Serper) and
  `visit` (page fetched through Jina Reader, then summarized by the same model
  against the goal it gave). The loop runs until the model stops calling tools or a
  cap in `config.py` is hit. Every query and URL is logged, so `tool_uses` is complete.

## Settings that differ from closed-book runs

- Reasoning is requested at `reasoning_mode="medium"` (closed-book uses `high`).
- `MAX_TOKENS = 8192` per call; no sampling parameters are sent.
- Serper searches use `gl=us`, `hl=en`, location "United States" (queries with Chinese
  characters switch to `gl=cn`).
- `seed` is forwarded only to the local vLLM models.

## Local Gemma models

Gemma-3 and Gemma-SEA-LION write tool calls as a fenced `tool_code` block, which no
built-in vLLM parser handles. `scripts/serve_gemma_sea_lion.sh` and
`scripts/serve_gemma3_27b.sh` load `gemma_tool_code_parser.py` and
`gemma3_tool_chat_template.jinja` from this folder when started with `WEB_SEARCH=1`.

## Keys

`LITELLM_API_KEY` and `LITELLM_BASE_URL` (run_eval copies them from `API_KEY` and its
own setting if unset), plus `SERPER_API_KEY` and `JINA_API_KEY` for the custom path.

## Caveats

- On the native path, URL counts reflect what the model chose to cite. A model can
  search without citing; use the tool-call flag to tell whether it searched.
- The native path never shows the provider's full retrieval set (every query and
  result considered). The custom path does.
- `force=True` (forcing a search before answering) is supported but not used by
  `run_eval.py`. Anthropic thinking and forced tool choice cannot be combined.
