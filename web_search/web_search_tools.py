"""
Provides web search capability for models without native search support
using Serper API (search) and Jina API (page fetching).
"""

import http.client
import json
import os
import time
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import requests
import tiktoken
from dotenv import load_dotenv

load_dotenv()

# API Keys
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# Retry settings
MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0
JITTER_RANGE = 0.3

# Content limits (matching native "medium" search_context_size)
MAX_TOKENS_BEFORE_SUMMARY = 30000
MAX_SUMMARY_WORDS = 500
JINA_TIMEOUT = 50


# -----------------------------------------------------------------------------
# Tool Schemas (OpenAI-compatible format for function calling)
# -----------------------------------------------------------------------------

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for information. Returns top 10 results with titles, URLs, and snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
}

VISIT_TOOL = {
    "type": "function",
    "function": {
        "name": "visit",
        "description": "Visit a webpage and get a summary of its content relevant to your goal.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to visit"
                },
                "goal": {
                    "type": "string",
                    "description": "What information you're looking for"
                }
            },
            "required": ["url", "goal"]
        }
    }
}


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def _jittered_backoff(base_s: float) -> float:
    """Add random jitter to backoff to prevent thundering herd."""
    jitter = random.uniform(1 - JITTER_RANGE, 1 + JITTER_RANGE)
    return base_s * jitter


def truncate_to_tokens(text: str, max_tokens: int = MAX_TOKENS_BEFORE_SUMMARY) -> str:
    """Truncate text to a maximum number of tokens."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def truncate_to_words(text: str, max_words: int = MAX_SUMMARY_WORDS) -> str:
    """Truncate text to a maximum number of words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


# -----------------------------------------------------------------------------
# Serper Search
# -----------------------------------------------------------------------------

def serper_search(query: str) -> str:
    """
    Perform a Google search via Serper API.

    Args:
        query: The search query

    Returns:
        Formatted string with search results (titles, URLs, snippets)
    """
    if not SERPER_API_KEY:
        return "[Search Error] SERPER_API_KEY not configured in .env"

    conn = http.client.HTTPSConnection("google.serper.dev", timeout=30)

    # Detect Chinese characters for locale
    def contains_chinese(text: str) -> bool:
        return any('\u4E00' <= char <= '\u9FFF' for char in text)

    if contains_chinese(query):
        payload = json.dumps({"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"})
    else:
        payload = json.dumps({"q": query, "location": "United States", "gl": "us", "hl": "en"})

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    backoff = INITIAL_BACKOFF_S
    for attempt in range(MAX_RETRIES):
        try:
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = res.read()
            results = json.loads(data.decode("utf-8"))
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return f"[Search Error] Search timed out after {MAX_RETRIES} attempts: {str(e)}"
            time.sleep(_jittered_backoff(backoff))
            backoff = min(backoff * 2, MAX_BACKOFF_S)
            continue

    # Check for API errors
    if "organic" not in results:
        if results.get("message", "") in ["Unauthorized.", "Not enough credits"]:
            return "[Search Error] Serper API key is invalid or insufficient balance."
        return f"[Search Error] No results found for query: '{query}'. Try a different query."

    # Format results
    web_snippets = []
    for idx, page in enumerate(results["organic"][:10], 1):
        parts = [f"{idx}. [{page.get('title', 'No title')}]({page.get('link', '')})"]

        if "date" in page:
            parts.append(f"Date: {page['date']}")
        if "source" in page:
            parts.append(f"Source: {page['source']}")
        if "snippet" in page:
            parts.append(page["snippet"])

        web_snippets.append("\n".join(parts))

    content = f"Search results for '{query}':\n\n" + "\n\n".join(web_snippets)
    return content


def extract_urls_from_search_results(search_results: str) -> List[str]:
    """Extract URLs from formatted search results."""
    import re
    urls = re.findall(r'\]\((https?://[^\)]+)\)', search_results)
    return urls


# -----------------------------------------------------------------------------
# Jina Page Fetching
# -----------------------------------------------------------------------------

def jina_fetch(url: str) -> str:
    """
    Fetch webpage content using Jina Reader API.

    Args:
        url: The URL to fetch

    Returns:
        Raw page content as markdown/text
    """
    if not JINA_API_KEY:
        return "[Fetch Error] JINA_API_KEY not configured in .env"

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
    }

    backoff = INITIAL_BACKOFF_S
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=JINA_TIMEOUT
            )
            if response.status_code == 200:
                return response.text
            elif response.status_code in [401, 402]:
                return "[Fetch Error] Jina API key is invalid or insufficient balance."
            else:
                if attempt == MAX_RETRIES - 1:
                    return f"[Fetch Error] Failed to fetch page (status {response.status_code})"
        except requests.exceptions.Timeout:
            if attempt == MAX_RETRIES - 1:
                return "[Fetch Error] Request timed out"
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return f"[Fetch Error] {str(e)}"

        time.sleep(_jittered_backoff(backoff))
        backoff = min(backoff * 2, MAX_BACKOFF_S)

    return "[Fetch Error] Failed to fetch page after retries"


# -----------------------------------------------------------------------------
# Summarization Prompt
# -----------------------------------------------------------------------------

SUMMARIZATION_PROMPT = """Summarize the following webpage content to extract information relevant to the goal.

## Webpage Content
{content}

## Goal
{goal}

## Instructions
1. Identify the sections most relevant to the goal
2. Extract key facts, data, and information
3. Provide a concise summary (max {max_words} words)
4. Focus on accuracy and relevance

## Summary"""


def create_summarization_messages(content: str, goal: str) -> List[Dict[str, str]]:
    """Create messages for summarization request."""
    truncated_content = truncate_to_tokens(content, MAX_TOKENS_BEFORE_SUMMARY)

    return [
        {
            "role": "user",
            "content": SUMMARIZATION_PROMPT.format(
                content=truncated_content,
                goal=goal,
                max_words=MAX_SUMMARY_WORDS
            )
        }
    ]


# -----------------------------------------------------------------------------
# Tool Result Dataclass
# -----------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_name: str
    tool_input: Dict[str, Any]
    output: str
    urls: List[str]  # URLs found/visited


def get_available_tools(enable_visit: bool = True) -> List[Dict]:
    """Get list of available tool schemas."""
    tools = [SEARCH_TOOL]
    if enable_visit:
        tools.append(VISIT_TOOL)
    return tools


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    "SEARCH_TOOL",
    "VISIT_TOOL",
    "serper_search",
    "jina_fetch",
    "extract_urls_from_search_results",
    "truncate_to_tokens",
    "truncate_to_words",
    "create_summarization_messages",
    "ToolResult",
    "get_available_tools",
    "MAX_SUMMARY_WORDS",
]
