"""Example: a real LangGraph agent with math + web search AND a semantic cache.

This builds on ``langgraph_math_agent.py`` but turns on the **semantic cache**,
so a *reworded* question reuses a cached answer instead of paying for a fresh
LLM call.

* The LLM (via LiteLLM — OpenAI / Anthropic / Bedrock / Groq / Ollama / etc.)
  decides when to call the ``calculator`` or ``web_search`` tool.
* The **whole agent run** is semantically cached, keyed on the user's
  question (``cache.lookup`` / ``cache.store`` in ``ask``).  A reworded
  question — "Can you calculate 128 times 47?" vs "What is 128 multiplied by
  47?" — matches by embedding similarity and returns the stored final answer,
  skipping the agent and every LLM/tool call.  (Individual LLM calls inside
  the ReAct loop use the exact cache only: their prompts contain near-unique
  tool payloads, so semantic matching there would never hit.)
* Embeddings are produced by a **local** sentence-transformers model
  (``all-MiniLM-L6-v2``), so no embedding API is called. Swap
  ``embedding_provider`` to ``"openai"`` or ``"litellm"`` for hosted providers.

It also shows two safety features:

* **Per-user scope isolation** — the LLM cache is scoped ``scope="user"``, so
  alice's cached answers are never served to bob (demonstrated by re-running
  the same questions as a different user).
* **``exact_only=True`` on the calculator** — a correctness-sensitive tool opts
  out of semantic matching so a *similar* call is never treated as a match.

--------------------------------------------------------------------------
Requirements
--------------------------------------------------------------------------

    pip install "memory-reuse[litellm,semantic-local]" langgraph langchain-core httpx

On a CPU-only machine, install a CPU torch wheel first to avoid a large GPU
download (sentence-transformers pulls in PyTorch)::

    pip install torch --index-url https://download.pytorch.org/whl/cpu

This example uses Groq for the LLM.  Set your Groq API key::

    export API_KEY="gsk_..."

Then run::

    python examples/semantic_agent.py

Web search uses the free DuckDuckGo Instant Answer API — no key required.
The local embedding model (``all-MiniLM-L6-v2``, ~90 MB) downloads from Hugging
Face on first use and is cached on disk afterwards. This example loads it in
offline mode, so run it once online first to populate that cache (see the
``HF_HUB_OFFLINE`` note near the top of the file).

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

# --- Quiet the model-load noise (this example owns its process) --------------
# These settings belong in an application/script, NOT in the library: they
# change process-global behaviour. The library only quiets its own logging and
# the per-embedding progress bar; the extra suppression below is the demo's
# choice to keep output tidy.
#
# Disable the "Loading weights" / download tqdm progress bars from Hugging Face.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Load the embedding model fully offline from the local cache. This silences
# both Hugging Face's cache-validation HTTP checks and the one-time
# "unauthenticated requests" advisory (which is printed by HF's native download
# layer and can't be filtered from Python).
#
# FIRST RUN: the model must already be cached on disk. If you have never used
# it, download it once WITH THESE LINES DISABLED (comment them out or run
# `HF_HUB_OFFLINE=0 python examples/semantic_agent.py`), then switch back to
# offline. The model (~90 MB) is cached after that first download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph  # noqa: E402

from memory_reuse import CacheConfig, MemoryCache  # noqa: E402
from memory_reuse.integrations import cached_tool  # noqa: E402
from memory_reuse.integrations.litellm import cached_litellm_completion  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("agent")

# Silence LiteLLM's per-call INFO line and the HF/transformers chatter so the
# demo output shows only the agent's own messages. huggingface_hub is raised to
# ERROR to hide its warning-level "unauthenticated requests" advisory (a token
# only speeds up downloads; it is not required for this demo).
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
for _noisy in ("sentence_transformers", "transformers", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL = "groq/openai/gpt-oss-120b"  # change to any LiteLLM-supported model

# Groq API key from the API_KEY environment variable, copied into GROQ_API_KEY
# (which LiteLLM reads automatically) so the secret stays OUT of the cache key.
if os.environ.get("API_KEY") and not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.environ["API_KEY"]

# One semantic-enabled cache shared by the whole agent.
#
# semantic_enabled=True turns on similarity matching; the whole agent run is
# routed through the combined exact-then-semantic flow in ask(). Embeddings are
# produced by a local sentence-transformers model (no embedding API cost).
#
# similarity_threshold (0.0–1.0) controls how close "close enough" is: a higher
# value demands a closer paraphrase. 0.85 comfortably matches the reworded demo
# questions with all-MiniLM-L6-v2 while still rejecting unrelated ones; raise it
# toward the 0.95 default if you see false positives.
#
# Swap backend="memory" for backend="redis" (+ redis_url) to share the cache
# across processes.  Requires the redis extra:  pip install "memory-reuse[redis]"
cache = MemoryCache(
    CacheConfig(
        backend="memory",
        default_ttl=3600,
        semantic_enabled=True,
        embedding_provider="local",
        embedding_model="all-MiniLM-L6-v2",
        similarity_threshold=0.85,
        # Opt-in answer extraction: on a semantic hit with a string answer,
        # return only the sentence(s) that best match the new question instead
        # of the whole stored answer. Purely extractive (no LLM), so it can only
        # return text already present in the cached answer. Try asking
        # "who created python" after "tell me about python" to see it narrow
        # the paragraph down to the "created by Guido van Rossum" sentence.
        extract_answer=True,
        extract_min_similarity=0.45,
    )
)


# cache = MemoryCache(
#     CacheConfig(
#         backend="redis",
#         redis_url="redis://localhost:6379/0",
#         default_ttl=3600,
#         semantic_enabled=True,
#         embedding_provider="local",
#         embedding_model="groq/openai/gpt-oss-120b",
#         similarity_threshold=0.85,
#     )
# )


# --------------------------------------------------------------------------
# Tool 1 — calculator
# --------------------------------------------------------------------------

_OPS = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}


@cached_tool(cache, scope="global", ttl=3600, exact_only=True)
async def calculator(operation: str, a: float, b: float) -> dict:
    """Perform a basic arithmetic operation.

    Cached globally for 1 hour — ``2 + 2`` is always ``4``, so the result is
    safe to share across all users and reuse indefinitely.

    ``exact_only=True`` forces the Phase 1 exact cache and skips semantic
    matching for this tool, even though the cache has semantic enabled.
    Arithmetic correctness depends on the exact operands, so a *similar* call
    ("128 x 47" vs "129 x 47") must never be treated as a match — that is
    exactly the false-positive class semantic caching must avoid for
    correctness-sensitive tools.

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
    minutes — a real HTTP request is slow and rate-limited, so repeats served
    from cache are a big latency and cost win.

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
        user_id: The id of the user this run belongs to.  Threaded into the
            LLM cache lookup so one user's cached answers are never served to
            another (per-user scope isolation).
    """

    messages: Annotated[list[BaseMessage], operator.add]
    user_id: str


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

    The LLM call goes through ``cached_litellm_completion`` as an **exact**
    cache only (no ``semantic=True``).  Inside a ReAct loop each step's prompt
    contains the full running conversation — tool-call IDs, tool JSON output,
    etc. — which is effectively unique, so semantic matching *here* would never
    hit and just waste an embedding on every step.

    Semantic matching is instead applied once per run, around the user's
    question, in :func:`ask` — that is the level at which rewording actually
    repeats.
    """
    litellm_messages = _to_litellm_messages(state["messages"])

    logger.info("    [LLM]  model step (exact hit → reused, miss → real call)")
    response = await cached_litellm_completion(
        cache,
        model=MODEL,
        messages=litellm_messages,
        tools=TOOL_SCHEMA,
        tool_choice="auto",
        temperature=0,  # deterministic → better cache hit rate
        ttl=3600,
        # Scope the LLM cache per user so one user's answers are never served
        # to another.
        scope="user",
        user_id=state["user_id"],
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


async def ask(agent, question: str, user_id: str) -> str:
    """Run one question through the agent, as ``user_id``, and return the answer.

    This is where the **semantic cache** is applied — around the whole run,
    keyed on the user's question:

    1. ``cache.lookup`` tries the exact cache (same question text) first, then
       embeds the question and looks for a *semantically similar* one already
       answered for this user.  A hit returns the stored final answer and skips
       the agent (and every LLM/tool call) entirely.
    2. On a miss the agent runs normally, and ``cache.store`` saves the final
       answer against this question so a reworded repeat can match next time.

    Caching at the question→answer level (rather than per individual LLM call)
    is what makes rewording actually repeat: intermediate ReAct steps carry
    near-unique tool payloads and never line up, but the user's question does.
    """
    logger.info("\n>>> [%s] Question: %s", user_id, question)
    start = time.perf_counter()

    # The exact-cache key includes the user's question text; the semantic
    # query_text is the question itself, so a reworded question can match.
    key_parts = ["agent-run", question]

    cached_answer = await cache.lookup(
        key_parts, query_text=question, scope="user", scope_id=user_id
    )
    if cached_answer is not None:
        elapsed = (time.perf_counter() - start) * 1000
        logger.info("    [CACHE] run served from cache — agent NOT invoked")
        logger.info("<<< Answer: %s  (%.0f ms)", cached_answer, elapsed)
        return cached_answer

    initial_state: AgentState = {
        "user_id": user_id,
        "messages": [
            SystemMessage(
                content=(
                    "You are a helpful assistant. Use the calculator tool for "
                    "arithmetic and the web_search tool for facts you are unsure "
                    "about. Give a short, direct final answer."
                )
            ),
            HumanMessage(content=question),
        ],
    }

    final_state = await agent.ainvoke(initial_state)
    answer = final_state["messages"][-1].content

    # Store the final answer so a reworded repeat of this question (for this
    # user) hits the semantic cache next time.
    await cache.store(key_parts, query_text=question, value=answer, scope="user", scope_id=user_id)

    elapsed = (time.perf_counter() - start) * 1000
    logger.info("<<< Answer: %s  (%.0f ms)", answer, elapsed)
    return answer


async def main() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "No API key found. Set your Groq key first:\n\n" "    export API_KEY='gsk_...'\n"
        )

    logger.info("Using the local sentence-transformers embedder (all-MiniLM-L6-v2).")

    agent = build_agent()

    # Set a default context so any call that doesn't pass an explicit user_id
    # still resolves one.  We also pass user_id explicitly per run below, which
    # takes precedence — this just shows the fallback exists.
    cache.set_context(user_id="alice")

    # First run (as alice): cold cache — real LLM + real tool calls.
    first_run = [
        "What is 128 multiplied by 47?",
        "Tell me about the Python programming language.",
    ]

    # Second run (as alice): the SAME questions reworded.  Exact matching would
    # miss (different wording), but the semantic cache recognises the same
    # intent and serves the stored answer — no real LLM call.
    reworded_run = [
        "128*47",
        "who created python programing",
    ]

    logger.info("\n=== Run 1 — alice, cold cache (real LLM + real tool calls) ===")
    for q in first_run:
        await ask(agent, q, user_id="alice")

    hits_before = cache.stats.semantic_hits

    logger.info("\n=== Run 2 — alice, REWORDED questions (semantic cache serves them) ===")
    for q in reworded_run:
        await ask(agent, q, user_id="alice")

    alice_semantic_hits = cache.stats.semantic_hits - hits_before

    # Third run (as bob): the SAME reworded questions, but a DIFFERENT user.
    # Because the LLM cache is scoped per user, bob does NOT see alice's cached
    # answers — every call is a real LLM call again.  This is the safety
    # guarantee that stops one user's data leaking to another.
    logger.info("\n=== Run 3 — bob, same questions (per-user scope → no cross-user reuse) ===")
    hits_before_bob = cache.stats.semantic_hits
    for q in reworded_run:
        await ask(agent, q, user_id="bob")
    bob_semantic_hits = cache.stats.semantic_hits - hits_before_bob

    stats = cache.stats
    logger.info("\n=== Cache stats ===")
    logger.info("  Hits:          %d", stats.hits)
    logger.info("  Exact hits:    %d", stats.exact_hits)
    logger.info("  Semantic hits: %d", stats.semantic_hits)
    logger.info("  Misses:        %d", stats.misses)
    logger.info("  Hit rate:      %.1f%%", stats.hit_rate * 100)
    logger.info(
        "\nNotice:\n"
        "  • alice's reworded run produced %d semantic hit(s) — same intent,\n"
        "    different wording, answer reused with no fresh LLM call.\n"
        "  • bob's identical run produced %d semantic hit(s) from alice's data —\n"
        "    per-user scoping kept the caches isolated (bob hits the real LLM).",
        alice_semantic_hits,
        bob_semantic_hits,
    )

    await cache.close()


if __name__ == "__main__":
    asyncio.run(main())
