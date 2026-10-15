# SPDX-License-Identifier: Apache-2.0
"""vLLM tool-call parser for Gemma-3's native ```tool_code``` function-call format.

Gemma-3 (and Gemma-SEA-LION) emit tool calls as a markdown-fenced Python block::

    ```tool_code
    serper_search(query="latest python version")
    ```

None of vLLM 0.22.1's built-in parsers match this (functiongemma uses
<start_function_call>, gemma4 uses <|tool_call>, pythonic uses [f(...)]), so we
add one. Load with:

    --enable-auto-tool-choice \
    --tool-call-parser-plugin web_search/gemma_tool_code_parser.py \
    --tool-call-parser tool_code

Parsing is AST-based (reuses vLLM's handle_single_tool), so quoting/number/kwarg
handling matches the pythonic parser. Streaming is a no-op (the web-search loop
calls the chat endpoint non-streamed).
"""
import ast

import regex as re
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
    ToolParserManager,
)
from vllm.tool_parsers.utils import UnexpectedAstError, handle_single_tool

logger = init_logger(__name__)

# ```tool_code ... ``` block. Tolerant of an optional language-tag newline and
# surrounding prose; DOTALL so multi-line calls are captured.
_FENCE = re.compile(r"```tool_code\s*(.*?)```", re.DOTALL)


@ToolParserManager.register_module("tool_code")
class ToolCodeToolParser(ToolParser):
    """Parse Gemma-3 ```tool_code``` fenced Python calls into OpenAI tool_calls."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        tools: list[Tool] | None = None,
    ):
        super().__init__(tokenizer, tools)

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        m = _FENCE.search(model_output)
        if not m:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        code = m.group(1).strip()
        try:
            module = ast.parse(code)
            calls = []
            for node in module.body:
                val = node.value if isinstance(node, ast.Expr) else getattr(node, "value", None)
                if isinstance(val, ast.Call):
                    calls.append(handle_single_tool(val))
            if not calls:
                raise UnexpectedAstError("no function calls in tool_code block")
            # Any prose before the fence is dropped (the model's visible answer
            # comes on a later, tool-free turn), matching pythonic/hermes behavior.
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=calls, content=None
            )
        except Exception:
            logger.exception("tool_code parse failed; treating as plain text")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(self, *args, **kwargs) -> DeltaMessage | None:
        # The web-search loop uses non-streaming chat completions, so no streaming
        # tool-delta handling is needed.
        return None
