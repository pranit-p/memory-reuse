"""Example: a real LangGraph agent with a math tool AND a web search tool.

This is a fully working ReAct-style agent:

* The LLM (via LiteLLM, so it works with OpenAI / Anthropic / Bedrock / Groq /
  Ollama / etc.) decides when to call the ``calculator`` or ``web_search`` tool.
* ``memory-reuse`` caches both the tool results and the LLM calls, so asking
  the same question twice skips the expensive work entirely.

The web search tool is where caching really pays off: a real web request takes
hundreds of milliseconds and may be rate-limited, so serving repeats from the
cache is a big win.

--------------------------------------------------------------------------
Requirements
--------------------------------------------------------------------------

    pip install memory-reuse[litellm] langgraph langchain-core httpx

This example uses Groq.  Set your Groq API key::

    export API_KEY="gsk_..."

Then run::

    python examples/langgraph_math_agent.py

Web search uses the free DuckDuckGo Instant Answer API — no key required.

To use a different LLM provider, change ``MODEL`` below and set the matching
API key.  Examples:

    MODEL = "groq/openai/gpt-oss-120b"           # Groq          (this example)
    MODEL = "gpt-4o-mini"                        # OpenAI        (OPENAI_API_KEY)
    MODEL = "anthropic/claude-3-5-haiku-latest"  # Anthropic     (ANTHROPIC_API_KEY)
    MODEL = "bedrock/amazon.nova-lite-v1:0"      # AWS Bedrock   (AWS creds)
    MODEL = "ollama/llama3.1"                    # local Ollama  (no API key)
"""

from __future__ import annotations

import asyncio
import json
import logging
import operator
import os
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph

from memory_reuse import CacheConfig, MemoryCache
from memory_reuse.integrations import cached_tool
from memory_reuse.integrations.litellm import cached_litellm_completion

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("agent")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL = "groq/openai/gpt-oss-120b"  # change to any LiteLLM-supported model

# Groq API key from the API_KEY environment variable.
#
# We copy it into GROQ_API_KEY, which LiteLLM reads automatically, rather than
# passing it as a per-call `api_key` kwarg.  This keeps the secret OUT of the
# cache key — passing api_key per call would make it part of the hash, so a
# rotated key would silently miss the cache (and the key would be hashed into
# storage).  Setting the env var is both safer and cache-friendly.
if os.environ.get("API_KEY") and not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.environ["API_KEY"]

# One cache instance shared by the whole agent.
# In-memory backend keeps this example dependency-free; swap to Redis in prod.
cache = MemoryCache(CacheConfig(backend="memory", default_ttl=3600))


# --------------------------------------------------------------------------
# Tool 1 — calculator
# --------------------------------------------------------------------------

_OPS = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}


@cached_tool(cache, scope="global", ttl=3600)
async def calculator(operation: str, a: float, b: float) -> dict:
    """Perform a basic arithmetic operation.

    Cached globally for 1 hour — ``2 + 2`` is always ``4``, so the result is
    safe to share across all users and reuse indefinitely.

    Args:
        operation: One of ``"add"``, ``"subtract"``, ``"multiply"``,
            ``"divide"``.
        a: First operand.
        b: Second operand.

    Returns:
        A dict with the numeric ``result`` (or an ``error`` message).
    """
    logger.info("    [TOOL] calculator(%s, %s, %s) — REAL execution", operation, a, b)
    if operation not in _OPS:
        return {"error": f"unknown operation '{operation}'"}
    if operation == "divide" and b == 0:
        return {"error": "division by zero"}
    return {"result": _OPS[operation](a, b)}


# --------------------------------------------------------------------------
# Tool 2 — web search (free DuckDuckGo Instant Answer API, no key needed)
# --------------------------------------------------------------------------


@cached_tool(cache, scope="global", ttl=1800)
async def web_search(query: str) -> dict:
    """Search the web for a short factual answer.

    Uses the free DuckDuckGo Instant Answer API.  Cached globally for 30
    minutes — this is where caching shines: a real HTTP request is slow and
    rate-limited, so repeats served from cache are a big latency and cost win.

    Args:
        query: The search query.

    Returns:
        A dict with an ``abstract`` (short summary) and ``source`` URL, or an
        ``error`` message on failure.
    """
    logger.info("    [TOOL] web_search(%r) — REAL HTTP request", query)
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001 — surface any failure to the LLM
        return {"error": f"web search failed: {exc}"}

    # DuckDuckGo's free Instant Answer API answers entity/definition queries
    # best (e.g. "Python programming language").  We check several fields and
    # fall back to related topics so the LLM gets something useful.
    abstract = data.get("AbstractText") or data.get("Answer") or data.get("Definition") or ""
    if not abstract:
        related = data.get("RelatedTopics") or []
        for topic in related:
            if isinstance(topic, dict) and topic.get("Text"):
                abstract = topic["Text"]
                break

    return {
        "abstract": abstract or "No direct answer found for this query.",
        "source": data.get("AbstractURL") or "https://duckduckgo.com",
    }


# --------------------------------------------------------------------------
# Tool schemas advertised to the LLM (OpenAI function-calling format)
# --------------------------------------------------------------------------

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic arithmetic: add, subtract, multiply, divide.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current facts, definitions, or general "
                "knowledge. Use this for anything you are unsure about."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query."}},
                "required": ["query"],
            },
        },
    },
]

# Map tool names to their implementations so the tool node can dispatch.
_TOOLS = {
    "calculator": calculator,
    "web_search": web_search,
}


# --------------------------------------------------------------------------
# LangGraph state
# --------------------------------------------------------------------------


class AgentState(TypedDict):
    """The state passed between graph nodes.

    Attributes:
        messages: The running conversation.  ``operator.add`` is used as the
            reducer so returning ``{"messages": [msg]}`` appends rather than
            replaces.
    """

    messages: Annotated[list[BaseMessage], operator.add]


# --------------------------------------------------------------------------
# Helpers to translate between LangChain messages and LiteLLM dicts
# --------------------------------------------------------------------------


def _to_litellm_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain message objects into LiteLLM/OpenAI dict format."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            tool_calls = m.additional_kwargs.get("tool_calls")
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
        elif isinstance(m, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content,
                }
            )
    return out


# --------------------------------------------------------------------------
# Graph nodes
# --------------------------------------------------------------------------


async def call_model(state: AgentState) -> dict:
    """Ask the LLM what to do next — answer, or call a tool.

    The LLM call goes through ``cached_litellm_completion`` so identical
    conversations reuse the cached response instead of paying for tokens
    again.
    """
    litellm_messages = _to_litellm_messages(state["messages"])

    logger.info("    [LLM]  model step (cache miss → real call, cache hit → reused)")
    response = await cached_litellm_completion(
        cache,
        model=MODEL,
        messages=litellm_messages,
        tools=TOOL_SCHEMA,
        tool_choice="auto",
        temperature=0,  # deterministic → better cache hit rate
        ttl=3600,
        scope="global",
        # API key is read by LiteLLM from GROQ_API_KEY (set at the top of this
        # file), so it is NOT passed here and stays out of the cache key.
    )

    # cached_litellm_completion returns the rich object on a miss and a plain
    # dict on a hit — normalise to a dict so the rest of the code is uniform.
    data = response if isinstance(response, dict) else response.model_dump()
    choice = data["choices"][0]["message"]

    ai_message = AIMessage(
        content=choice.get("content") or "",
        additional_kwargs=(
            {"tool_calls": choice["tool_calls"]} if choice.get("tool_calls") else {}
        ),
    )
    return {"messages": [ai_message]}


async def call_tools(state: AgentState) -> dict:
    """Execute any tool calls the LLM requested and return tool results."""
    last_message = state["messages"][-1]
    tool_calls = last_message.additional_kwargs.get("tool_calls", [])

    tool_messages: list[BaseMessage] = []
    for call in tool_calls:
        fn = call["function"]
        name = fn["name"]
        args = json.loads(fn["arguments"])

        tool_fn = _TOOLS.get(name)
        if tool_fn is None:
            result: dict = {"error": f"unknown tool '{name}'"}
        else:
            result = await tool_fn(**args)

        tool_messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    """Route to the tool node if the LLM asked for a tool, else finish."""
    last_message = state["messages"][-1]
    if last_message.additional_kwargs.get("tool_calls"):
        return "tools"
    return END


# --------------------------------------------------------------------------
# Build the graph
# --------------------------------------------------------------------------


def build_agent():
    """Compile the ReAct-style agent graph (calculator + web_search)."""
    builder = StateGraph(AgentState)

    builder.add_node("model", call_model)
    builder.add_node("tools", call_tools)

    builder.set_entry_point("model")
    builder.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "model")  # loop back so LLM can use tool output

    return builder.compile()


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


async def ask(agent, question: str) -> str:
    """Run one question through the agent and return the final answer."""
    logger.info("\n>>> Question: %s", question)
    start = time.perf_counter()

    initial_state: AgentState = {
        "messages": [
            SystemMessage(
                content=(
                    "You are a helpful assistant. Use the calculator tool for "
                    "arithmetic and the web_search tool for facts you are unsure "
                    "about. Give a short, direct final answer."
                )
            ),
            HumanMessage(content=question),
        ]
    }

    final_state = await agent.ainvoke(initial_state)
    answer = final_state["messages"][-1].content
    elapsed = (time.perf_counter() - start) * 1000

    logger.info("<<< Answer: %s  (%.0f ms)", answer, elapsed)
    return answer


async def main() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "No API key found. Set your Groq key first:\n\n" "    export API_KEY='gsk_...'\n"
        )

    agent = build_agent()

    # Two questions: one exercises the calculator, one exercises web search.
    # DuckDuckGo's free API answers entity/topic queries best, so we ask about
    # a well-known topic that returns a real abstract.
    questions = [
        "What is 128 multiplied by 47?",
        "Tell me about the Python programming language.",
    ]

    logger.info("\n=== First run (cold cache — real LLM + real tool calls) ===")
    for q in questions:
        await ask(agent, q)

    questions = [
        "What is 256 multiplied by 1?",
        "what is python prgramming language.",
    ]

    logger.info("\n=== Second run (warm cache — everything reused, 0 cost) ===")
    for q in questions:
        await ask(agent, q)

    stats = cache.stats
    logger.info("\n=== Cache stats ===")
    logger.info("  Hits:       %d", stats.hits)
    logger.info("  Misses:     %d", stats.misses)
    logger.info("  Hit rate:   %.1f%%", stats.hit_rate * 100)
    logger.info(
        "\nNotice: in the second run there are NO '[TOOL] ... REAL' or real LLM\n"
        "calls — every LLM response and tool result was served from cache."
    )

    await cache.close()


if __name__ == "__main__":
    asyncio.run(main())
