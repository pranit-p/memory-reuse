---
# Vision & design

!!! info "About this document"
    This is the **original design document** behind memory-reuse — the full
    long-term vision. It is kept for context on *why* the project exists and
    where it's headed. Not everything here is shipped yet.

    **Shipped today (v0.2.0):**

    - Exact cache, tool cache, and (opt-in) semantic cache
    - In-memory and Redis backends; in-memory and Redis vector indexes
    - Embedding providers: local (sentence-transformers), OpenAI, LiteLLM
    - Multi-scope isolation (global / user / session)
    - LangGraph (`cached_node`, `cached_tool`) and LiteLLM integrations

    **Planned (see the [roadmap](index.md) — Phase 3+), described below but not
    yet available:**

    - Node-level and graph-level caching (`cache.wrap_graph`, node skipping)
    - Strands / CrewAI integrations
    - SQLite / Postgres / Qdrant backends and an AWS AgentCore backend
    - Cost analytics dashboard and Prometheus / OpenTelemetry export

---

# memory-reuse
### A Framework-Agnostic Python Package to Reduce AI Agent Costs

---

## One-Line Pitch

> **Stop making AI agents repeat work they have already done.**
> One pip install. Three decorators. Your agent stops wasting money on LLM calls and tool calls it has already made — works with LangGraph, Strands, CrewAI, or any Python agent framework.

---

## The Problem in One Picture

```
Without SDK — every request pays full cost:

User A: "What is the return policy?"   → LLM call  ($$$)
User B: "How do I return a product?"   → LLM call  ($$$)  ← same answer
User C: "Can I get a refund?"          → LLM call  ($$$)  ← same answer
User A: "What is the return policy?"   → LLM call  ($$$)  ← asked before

With SDK — pay once, reuse forever:

User A: "What is the return policy?"   → LLM call  ($$$)  ← cache miss, store
User B: "How do I return a product?"   → Cache HIT  (free) ← semantic match
User C: "Can I get a refund?"          → Cache HIT  (free) ← semantic match
User A: "What is the return policy?"   → Cache HIT  (free) ← exact match
```

---

## Why This Exists

Every AI agent framework has tools for **memory** — what the user said, what they prefer, what happened last session. None of them have a tool that simply **stops the agent from doing the same expensive work twice**.

That is the gap this SDK fills.

```
What memory tools do:              What this SDK does:
──────────────────────             ────────────────────────────
"Remember user preferences"        "Skip this LLM call — done it before"
"Store conversation history"       "Skip this API call — result cached"
"Extract user facts"               "Skip this graph node — output unchanged"
"Summarize past sessions"          "Return answer in <1ms — 0 tokens used"
```

They make agents **smarter**. This makes agents **cheaper and faster**. Both matter. Neither replaces the other.

---

## What It Does — Three Cache Levels

The SDK caches at three levels of granularity. You can use one, two, or all three together.

---

### Level 1 — Tool Cache (Most Granular)

Cache the result of individual function calls — API calls, database queries, any external call.

```
Agent execution — without tool cache:
  Every request:
    → fetch_customer_info(id=456)    ← DB call    ($)
    → search_confluence("setup")     ← API call   ($)
    → call_llm(prompt)               ← LLM call   ($$$)

Agent execution — with tool cache:
  Request 1:   calls all three, stores results
  Request 2+:  fetch_customer_info → HIT ✅ (0ms, free)
               search_confluence   → HIT ✅ (0ms, free)
               call_llm            → HIT ✅ (0ms, free)
```

```python
@cached_tool(cache, ttl=300)           # cache for 5 minutes
def fetch_customer_info(customer_id: str) -> dict:
    return db.query(customer_id)

@cached_tool(cache, ttl=1800)          # cache for 30 minutes
def search_confluence(query: str) -> list:
    return confluence.search(query)
```

**Best for:** DB queries, REST APIs, search APIs, MCP tools, internal services.

---

### Level 2 — Node Cache (Mid-Level)

Cache the entire output of a LangGraph node. If the node's input hasn't changed, skip everything inside it.

```
LangGraph Agent — with node cache:

  Node 1: understand_request        → always runs (fast, cheap)
  Node 2: gather_context            → NODE CACHE CHECK
    ├── fetch_customer_info()         Input state hash matches?
    ├── fetch_order_history()         → YES: entire node skipped ✅
    └── fetch_preferences()           → NO:  node runs, result stored
  Node 3: generate_response         → NODE CACHE CHECK
    └── call_llm()                    Semantic match on input state?
                                      → YES: node skipped ✅
  Node 4: return_result             → always runs (no side effects)
```

```python
@cached_node(cache, scope="global", ttl=600)
def gather_context(state: AgentState) -> AgentState:
    # entire node skipped if same input state was seen before
    state["customer"] = fetch_customer_info(state["customer_id"])
    state["history"]  = fetch_order_history(state["customer_id"])
    return state
```

**Best for:** Nodes that combine multiple expensive operations, LLM generation nodes.

---

### Level 3 — Graph Cache (Most Coarse)

Cache the result of an entire agent execution. If the same or semantically similar request was fully executed before, return the final answer — zero nodes run.

```
Graph cache — maximum cost reduction:

Request 1: "How do I set up the dev environment?"
  → Graph cache MISS
  → Full graph executes (Node 1 → 2 → 3 → 4)
  → Final answer stored in graph cache

Request 2: "Steps to set up local development?"
  → Graph cache SEMANTIC HIT ✅
  → 0 nodes executed
  → 0 LLM tokens consumed
  → Answer returned in milliseconds
```

```python
# Wrap your entire compiled graph
cached_graph = cache.wrap_graph(
    graph,
    semantic=True,
    similarity_threshold=0.90,
    ttl=7200    # 2 hours
)

# Invoke exactly as before — SDK handles everything
result = cached_graph.invoke({"question": "How do I set up dev env?"})
```

**Best for:** FAQ agents, support bots, Confluence Q&A — where same question = same answer.

---

### All Three Levels Together

Stack them for maximum coverage:

```
Incoming Request
      │
      ▼
┌─────────────────────────────────────────┐
│  Level 3 — Graph Cache                  │
│  "Have I answered this before?"         │
│  HIT → return final answer (0 work) ✅  │
│  MISS → continue                    ↓   │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Level 2 — Node Cache                   │
│  "Has this node run with same input?"   │
│  Node A: HIT → skip ✅                  │
│  Node B: HIT → skip ✅                  │
│  Node C: MISS → run                 ↓   │
└─────────────────────────────────────────┘
      │  (only Node C executes)
      ▼
┌─────────────────────────────────────────┐
│  Level 1 — Tool Cache                   │
│  "Has this exact call been made?"       │
│  fetch_data(): HIT → skip ✅            │
│  call_llm():   HIT → skip ✅            │
│  send_result(): runs (side effect)      │
└─────────────────────────────────────────┘
      │
      ▼
   Response
```

**Best case:** Graph cache hits → answer in <1ms, 0 tokens, 0 API calls.
**Good case:** Some nodes cached → partial execution, big savings.
**Worst case:** All misses → full run, result stored for next time.

---

## Three Cache Types

Across all three levels, the SDK supports three types of cache lookup:

| Type | How It Works | Best For |
|---|---|---|
| **Exact Cache** | Hash the input, find exact match | Identical repeated calls |
| **Semantic Cache** | Embed the input, find similar match by cosine similarity | Same intent, different wording |
| **Tool Cache** | Hash tool name + args, find exact match with TTL | API/DB calls with expiry |

```
Exact:    "What is order 123 status?" = "What is order 123 status?"
           Same string → instant hash match

Semantic: "What is order 123 status?" ≈ "Where is my order 123?"
           Different words, same intent → embedding similarity match

Tool:     fetch_order(order_id="123") called 5 min ago
           Same function + same args → return cached result (TTL=30min)
```

---

## Framework Support

Works with any Python agent framework. No lock-in.

```python
# LangGraph
from memory_reuse.integrations.langgraph import cached_node, cached_tool

@cached_node(cache, ttl=600)
def my_langgraph_node(state: AgentState) -> AgentState: ...

# Strands
from memory_reuse.integrations.strands import cached_tool

@cached_tool(cache, ttl=300)
def my_strands_tool(input: dict) -> dict: ...

# Any custom agent — plain decorator
@cache.cache_llm_call(ttl=3600)
def call_llm(prompt: str) -> str: ...

@cache.cache_tool_call(ttl=300)
def fetch_data(query: str) -> dict: ...
```

---

## How It Is Different From Everything Else

### The Positioning Map

```
                    WHAT PROBLEM DO THEY SOLVE?

    "Remember what              "Reduce redundant
     users said"                 execution cost"
          │                            │
          ▼                            ▼
  ┌──────────────────┐        ┌─────────────────────┐
  │  LangGraph       │        │                     │
  │  Memory          │        │   This SDK ✅       │
  │                  │        │                     │
  │  AgentCore       │        │  (only tool in      │
  │  Memory          │        │   this category)    │
  │                  │        │                     │
  │  Mem0 / LangMem  │        └─────────────────────┘
  │  Zep / Letta     │
  └──────────────────┘
  Conversational memory          Execution cost reducer
```

Nobody lives in the right column. That is the gap.

---

### vs LangGraph Memory / AgentCore Memory / Mem0 / LangMem

These store **what the user said and what the agent learned**.

```
They store:                        We cache:
─────────────────────              ─────────────────────────────
User preferences                   LLM call results
Conversation history               Tool / API call results
Session summaries                  Graph node outputs
Facts about users                  Execution paths
"User prefers dark mode"           "This prompt → this answer, skip LLM"
```

They make agents **personal and context-aware**.
We make agents **cheap and fast**.

**Complementary, not competing.** Use both together for best results.

---

### vs GPTCache / LangChain Redis Cache

These only cache **LLM responses at the API boundary**. One level, one hop.

```
GPTCache / LangChain Cache:        This SDK:
───────────────────────────        ────────────────────────────────
Your code                          Your code
    ↓                                  ↓
[Cache layer]  ← caches only here  [SDK] → Tool calls  → Tool Cache
    ↓                                     → Node outputs → Node Cache
LLM API                                  → LLM calls    → Semantic Cache
                                         → Full graph   → Graph Cache
```

Also GPTCache:
- Abandoned / not actively maintained
- No LangGraph or Strands integration
- No tool caching
- 3.3% cache hit rate on real agent tasks (per 2026 research) — embedding-only approach fails on agent workloads
- No analytics

---

### The Unique Position

| Capability | GPTCache | LangChain Cache | Mem0 / LangMem | AgentCore Memory | **This SDK** |
|---|---|---|---|---|---|
| Exact LLM cache | ✅ | ✅ | ❌ | ❌ | ✅ |
| Semantic LLM cache | ✅ | ✅ | ❌ | ❌ | ✅ |
| Tool / API result cache | ❌ | ❌ | ❌ | ❌ | ✅ |
| Node-level cache | ❌ | ❌ | ❌ | ❌ | ✅ |
| Graph-level cache | ❌ | ❌ | ❌ | ❌ | ✅ |
| LangGraph native | ❌ | ❌ | ❌ | ✅ | ✅ |
| Strands / CrewAI | ❌ | ❌ | ❌ | ❌ | ✅ |
| Conversational memory | ❌ | ❌ | ✅ | ✅ | ❌ |
| User preference memory | ❌ | ❌ | ✅ | ✅ | ❌ |
| Cost analytics | ❌ | ❌ | ❌ | ❌ | ✅ |
| Open source | ✅ | ✅ | ✅ | ❌ | ✅ |

---

## Real-World Example — Confluence Agent

A Confluence Q&A agent for a 50-person engineering team.

**What happens on every request without the SDK:**

```
Employee: "How do I set up the dev environment?"
    ↓
Agent
    ↓
search_confluence("dev environment setup")  ← Confluence API call
    ↓
fetch_page_content(page_id="DEV-123")       ← Confluence API call
    ↓
call_llm(page_content + question)           ← LLM call  ($$$)
    ↓
Answer returned
```

**What happens with the SDK (after cache warms up):**

```
Employee 1: "How do I set up the dev environment?"
  → All cache MISS → full execution → results cached

Employee 2: "Steps for local dev setup?"
  → Graph cache SEMANTIC HIT ✅
  → 0 API calls, 0 LLM tokens, answer in <50ms

Employee 3: "Local environment setup instructions?"
  → Graph cache SEMANTIC HIT ✅
  → 0 API calls, 0 LLM tokens, answer in <50ms

50 employees, 5 questions/day, 22 days:
  Without SDK: 5,500 LLM calls + 11,000 API calls
  With SDK:    ~1,650 LLM calls + ~2,200 API calls
  Savings:     70% LLM cost reduction, 80% API call reduction
```

---

## User-Based Cache Scoping

The SDK handles multi-user environments safely. Three scope levels prevent data leaking between users.

```
┌─────────────────────────────────────────────┐
│  Global scope  (shared across all users)    │
│  "What is the return policy?"               │
│  Same answer for everyone — safe to share   │
├─────────────────────────────────────────────┤
│  User scope    (isolated per user_id)       │
│  "What are my orders?"                      │
│  Different per user — never shared          │
├─────────────────────────────────────────────┤
│  Session scope (isolated per session_id)    │
│  "Continue from my last question"           │
│  Per conversation — cleared on session end  │
└─────────────────────────────────────────────┘
```

```python
# Public knowledge — global scope
@cached_node(cache, scope="global", ttl=3600)
def get_refund_policy(state): ...

# User-specific data — isolated per user
@cached_tool(cache, scope="user", ttl=600)
def get_user_orders(state):
    # cache key = hash("user:{user_id}:{args}")
    ...

# Session context — isolated per conversation
@cached_node(cache, scope="session", ttl=300)
def get_session_context(state): ...
```

**Safety rule:** The SDK never allows user-scoped results to be served globally. Mixing scope incorrectly raises a configuration error.

---

## Where It Saves Cost — Honest Assessment

### Yes, it saves cost when:

| Use Case | Expected Hit Rate | Cost Reduction |
|---|---|---|
| FAQ / support bot | 50–80% | 50–80% |
| Confluence / docs Q&A agent | 60–80% | 60–80% |
| E-commerce order agent | 40–70% | 40–70% |
| Data pipeline / ETL agent | 60–90% | High |
| Multi-user SaaS product | 40–60% | Significant |

### Minimal savings when:

| Use Case | Why |
|---|---|
| Creative / generative agent | Every output is intentionally unique |
| Research / novel-query agent | Low repetition by design |
| Real-time data (live prices) | Short TTL = frequent misses |
| Highly dynamic prompts | Input varies too much for semantic match |

**The honest rule:** The SDK does not make LLM calls cheaper. It makes you call the LLM less often. Savings are proportional to repetition in your workload.

---

## Running on AgentCore Runtime

If your LangGraph agent runs on Amazon Bedrock AgentCore Runtime, the SDK must use an **external shared cache backend** — not in-memory or SQLite.

**Why:** AgentCore runs each session in an isolated microVM. Each VM starts with empty memory. In-memory cache in VM 1 is completely invisible to VM 2.

```
Without external cache (broken on AgentCore):

User A → microVM 1 → caches result in memory
User B → microVM 2 → empty cache → pays full cost again ❌

With external cache (correct):

User A → microVM 1 → caches result in ElastiCache
User B → microVM 2 → ElastiCache HIT ✅ → free
```

**Recommended AWS stack:**

```
AgentCore Runtime (microVMs)
         │
         ├── Exact + Tool Cache → Amazon ElastiCache (Redis)
         │                        same VPC, <1ms latency
         │
         ├── Semantic Cache     → Amazon OpenSearch Serverless
         │                        vector k-NN search
         │
         └── Long-term Memory   → AgentCore Memory Service
                                  cross-session, managed
```

**SDK setup for AgentCore:**

```python
from amazon_bedrock_agentcore import BedrockAgentCoreApp
from memory_reuse import MemoryCache

app   = BedrockAgentCoreApp()
cache = MemoryCache.from_env()   # reads MEMORY_REUSE_* env vars

@app.entrypoint
def handler(input: dict, context) -> dict:
    cache.set_context(
        user_id    = context.identity.user_id,
        session_id = context.session_id
    )
    return graph.invoke(input)
```

---

## Storage Backend Options

The **Status** column reflects what ships in v0.2.0 vs what is planned (see the
[Backend roadmap](#backend-roadmap-implement-on-demand) above).

| Backend | Exact Cache | Semantic Cache | Latency | Cost | Best For | Status |
|---|---|---|---|---|---|---|
| In-Memory | ✅ | ✅ | <0.1ms | Free | Dev / testing only | ✅ Shipped |
| Redis Stack | ✅ | ✅ | <1ms | ~$25/mo (t4g.small) | Teams already on Redis | ✅ Shipped |
| Upstash Redis | ✅ | ✅ | <5ms | $0.20/100K ops | Small teams, low traffic | ✅ Shipped (Redis-compatible) |
| SQLite | ✅ | ⚠️ brute-force | ~1ms | Free | Single-machine persistence, no server | 🔜 Planned |
| PostgreSQL + pgvector | ✅ | ✅ | 5–15ms | ~$0 if existing | Teams already on Postgres | 🔜 Planned |
| Redis + Qdrant | ✅ | ✅ | <1ms | ~$30–50/mo | High-scale vector search | 🔜 Planned |
| AWS AgentCore Memory | ✅ | ✅ | ~100–300ms | managed | Cross-microVM shared cache on AWS | 🔜 Planned |
| ElastiCache + OpenSearch | ✅ | ✅ | <1ms | ~$90/mo | AgentCore / AWS production | 🔜 Planned |

**Decision rule:**
- Getting started / dev → In-Memory (shipped, zero setup)
- Production, shared across processes → Redis Stack or Upstash Redis (shipped)
- Single machine, want persistence without a server → SQLite (planned)
- Already on Postgres → PostgreSQL + pgvector (planned)
- Deploying on AWS Bedrock AgentCore → AgentCore Memory, optionally with
  ElastiCache as a hot layer (planned)
- Very large vector scale → Redis + Qdrant (planned)

---

## Cost Comparison — Confluence Agent (50-person team)

**Baseline:** 5,500 requests/month, Claude Sonnet 3.5 ($3/M input, $15/M output), 2,500 tokens avg per request.

```
Without caching:
  LLM cost:     $74.25/month
  Runtime:       $4.88/month
  Total:        ~$79/month

With SDK + Upstash Redis (Option C):
  LLM cost:     $22.28/month  (70% hit rate)
  Upstash:       $0.05/month  (near-zero)
  Runtime:       $4.88/month
  Total:        ~$27/month    ← 66% cheaper

With SDK + Upstash + AgentCore Memory (Option D — Recommended):
  LLM cost:     $22.28/month
  Upstash:       $0.05/month
  AgentCore Mem: $4.40/month
  Runtime:       $4.88/month
  Total:        ~$32/month    ← 59% cheaper + long-term memory included
```

| Option | Monthly Cost | LLM Savings | Tool Savings | Infra Complexity |
|---|---|---|---|---|
| No cache | ~$79/mo | 0% | 0% | None |
| AgentCore Memory only | ~$32/mo | ~70% | 0% | None |
| Upstash Redis | ~$27/mo | 70% | 80% | None |
| **Upstash + AgentCore Memory** | **~$32/mo** | **70%** | **80%** | **Minimal** |
| ElastiCache + AgentCore (200 ppl) | ~$175/mo | 70% | 80% | Medium |

---

## Python Package

### Install

!!! note "Aspirational extras"
    Some extras below (`agentcore`) and the `all` set are **planned**, not yet
    shipped — see the [Backend roadmap](#backend-roadmap-implement-on-demand).
    For current install instructions and the extras that exist today, see
    [Install](install.md).

```bash
pip install memory-reuse                  # minimal, in-memory only
pip install memory-reuse[redis]           # + Redis backend
pip install memory-reuse[semantic]        # + semantic cache (API embeddings)
pip install memory-reuse[semantic-local]  # + local embeddings (sentence-transformers)
pip install memory-reuse[agentcore]       # + AWS AgentCore backend (planned)
pip install memory-reuse[all]             # everything
```

### Package Structure

```
memory-reuse/
│
├── memory_reuse/
│   ├── __init__.py                  ← MemoryCache, CacheConfig (public API)
│   │
│   ├── cache/
│   │   ├── exact.py                 ← hash-based exact cache
│   │   ├── semantic.py              ← embedding similarity cache
│   │   ├── tool.py                  ← tool result cache with TTL
│   │   └── graph.py                 ← graph-level execution cache
│   │
│   ├── backends/
│   │   ├── base.py                  ← abstract interface (swap any backend)
│   │   ├── memory.py                ← in-memory (dev)
│   │   ├── sqlite.py                ← SQLite
│   │   ├── redis.py                 ← Redis / Redis Stack
│   │   ├── postgres.py              ← PostgreSQL + pgvector
│   │   ├── qdrant.py                ← Qdrant
│   │   └── agentcore.py             ← AWS Bedrock AgentCore
│   │
│   ├── integrations/
│   │   ├── langgraph.py             ← cached_node, cached_tool decorators
│   │   ├── strands.py               ← Strands agent integration
│   │   └── langchain.py             ← LangChain cache integration
│   │
│   ├── embeddings/
│   │   ├── openai.py                ← OpenAI text-embedding-3-small
│   │   ├── bedrock.py               ← AWS Bedrock Titan embeddings
│   │   └── sentence_transformers.py ← local / offline models
│   │
│   └── analytics/
│       ├── tracker.py               ← hit rate, tokens saved, cost saved
│       └── exporters.py             ← Prometheus, OpenTelemetry
│
├── tests/
├── examples/
├── pyproject.toml
└── README.md
```

### pyproject.toml

```toml
[project]
name = "memory-reuse"
version = "0.1.0"
description = "Execution cache layer for AI agents — reduce LLM and tool call costs"
requires-python = ">=3.10"
dependencies = ["langgraph>=0.2.0", "langchain-core>=0.2.0"]

[project.optional-dependencies]
redis     = ["redis>=5.0.0"]
semantic  = ["sentence-transformers>=3.0.0", "numpy>=1.26.0"]
postgres  = ["psycopg2-binary>=2.9.0", "pgvector>=0.2.0"]
qdrant    = ["qdrant-client>=1.9.0"]
agentcore = ["amazon-bedrock-agentcore>=1.0.0", "boto3>=1.34.0"]
all       = ["memory-reuse[redis,semantic,postgres,qdrant,agentcore]"]
```

---

## Build Phases

### Phase 1 — Exact Cache (MVP) ✅ Shipped (v0.1)
- [x] Cache LLM calls by exact prompt hash
- [x] Cache tool call results by input hash + TTL
- [x] In-memory and Redis backends
- [x] LangGraph decorator integration (`cached_node`, `cached_tool`)
- [x] Cache hit / miss metrics

### Phase 2 — Semantic Cache ✅ Shipped (v0.2)
- [x] Embedding-based similarity matching
- [x] Configurable similarity threshold (config, env var, and per-call override)
- [x] Multiple embedding providers (OpenAI, LiteLLM/Bedrock, local)
- [x] Answer extraction — return only the best-matching sentence(s) (`extract_answer`)
- [ ] Intent canonicalization before embedding — *deferred*: ship raw cosine
  first, measure real hit rates, add canonicalization only if the data justifies it.

### Backend roadmap (implement on demand) 🔜
Storage is abstracted behind `AbstractBackend` (key-value: exact + tool caches)
and `VectorIndex` (nearest-neighbour: semantic cache), so new backends slot in
without touching the caches. Shipped today: **in-memory** and **Redis** (both
key-value and vector). The following are planned and will be built when a real
need appears — each has a specific trigger:

- [ ] **SQLite** — file-based persistence with no server. *Trigger:* single-machine
  apps that want the cache to survive restarts without running Redis. (Lowest
  effort, fully testable — the likely next backend.)
- [ ] **PostgreSQL + pgvector** — one datastore for both key-value and indexed
  vector search. *Trigger:* teams already running Postgres.
- [ ] **Qdrant** — dedicated vector database for large-scale KNN. *Trigger:*
  millions of embeddings where vector search is the bottleneck.
- [ ] **AWS AgentCore Memory** — managed, cross-microVM backend. *Trigger:*
  deploying on Amazon Bedrock AgentCore (see Phase 4).

### Phase 3 — Graph-Level Cache + Partial Reuse 🔜 Planned
- [ ] Graph-level execution cache (`cache.wrap_graph`)
- [ ] Node-level output cache
- [ ] Detect which nodes can be skipped on similar requests
- [ ] Node-level cache invalidation

### Phase 4 — Analytics + Integrations 🔜 Planned
- [ ] Real-time dashboard: hit rate, tokens saved, cost saved, latency saved
- [ ] Prometheus + OpenTelemetry export
- [ ] Strands Agents integration
- [ ] CrewAI integration
- [ ] AgentCore backend (managed AWS option)

---

## Success Metrics

| Metric | Target |
|---|---|
| Cache hit rate | > 40% in production workloads |
| LLM calls avoided | 40–70% reduction |
| Tool calls avoided | 50–80% reduction |
| Latency on cache hit | < 5ms |
| Code changes to existing agent | Decorator only — zero restructuring |

---

## Open Questions / Risks

| Risk | Mitigation |
|---|---|
| Semantic false positives ("cancel order" ≈ "delete order") | Safety classifier before serving cached result for mutation-like intents |
| Embedding cost overhead | Use local `sentence-transformers` for zero cost, or batch embedding calls |
| Cache invalidation when source data changes | Tag-based invalidation + webhook support (e.g. Confluence page update) |
| Cold start — cache empty on first run | Expected — document this, show hit rate growth curve |
| Multi-tenant data isolation | Explicit `scope` parameter required — SDK raises error if user data cached globally |

---

## Vision

Make execution caching a **standard layer** in every production AI agent stack — the way Redis became standard for web caching.

```
Today (every agent framework):           Future (with this SDK):

Request → Framework → LLM (pay)         Request → SDK → Cache HIT  (free)
Request → Framework → LLM (pay)                       → Cache HIT  (free)
Request → Framework → LLM (pay)                       → Cache MISS → LLM (pay, store)
Request → Framework → LLM (pay)                       → Cache HIT  (free)

Every request pays.                      Most requests are free.
```

---

## One-Paragraph Explainer (For Anyone)

> Every AI agent — whether built with LangGraph, Strands, CrewAI, or anything else — wastes money calling LLMs and external APIs for work it has already done. A support bot answers the same question 100 times and pays for 100 LLM calls. A data agent queries the same database row 50 times in an hour. A Confluence agent fetches the same page repeatedly. This SDK is a Python package you drop into any agent with three decorators. It caches results at the tool level, the node level, and the graph level — and the next time a similar request comes in, it returns the answer in milliseconds without touching the LLM or the API. For a 50-person team, that is a 60–70% reduction in monthly LLM costs with zero changes to your agent's business logic.

---

## End-to-End Example — Confluence Agent with AgentCore Runtime + AgentCore Memory + memory-reuse

This is the complete picture of how all three work together in a real production agent.

---

### The Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Application                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│              AgentCore Runtime                              │
│  Hosts the agent, manages microVMs, handles scaling,        │
│  session isolation, identity, observability                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       microVM (User A) microVM (User B) microVM (User C)
              │               │               │
              └───────────────┼───────────────┘
                              │  (all share same external cache)
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐  ┌────────────────┐  ┌──────────────────┐
│  Agent Memory   │  │  AgentCore     │  │  ElastiCache     │
│  SDK            │  │  Memory        │  │  (Redis)         │
│                 │  │                │  │                  │
│  Tool Cache     │  │  Short-term:   │  │  Exact cache     │
│  Semantic Cache │  │  conversation  │  │  Tool cache      │
│  Graph Cache    │  │                │  │  Semantic cache  │
│                 │  │  Long-term:    │  │                  │
│  "Skip work     │  │  preferences   │  │  ⚠️ Add when     │
│   already done" │  │  user facts    │  │  team grows > 20 │
└─────────────────┘  └────────────────┘  └──────────────────┘

> 📝 This example uses AgentCore Memory as the backend for the SDK — no Redis required.
> AgentCore Memory stores results and serves them across sessions and microVMs.
> When your team grows beyond ~20 people or response latency becomes a concern,
> add Amazon ElastiCache (Redis) as the hot cache layer in front of AgentCore Memory.
> See the "Scaling Note" at the end of this example.
```

---

### What We Are Building

A Confluence Q&A agent for an engineering team. Employees ask questions about internal documentation. The agent:
- Searches Confluence for relevant pages
- Fetches page content
- Calls an LLM to generate an answer
- Remembers each user's preferences and past context across sessions
- Skips redundant work using the SDK cache

---

### Project Structure

```
confluence-agent/
├── agent.py              ← main agent entrypoint (AgentCore handler)
├── graph.py              ← LangGraph graph definition
├── nodes.py              ← individual graph nodes
├── tools.py              ← Confluence API tools
├── memory.py             ← AgentCore Memory client wrapper
├── requirements.txt
└── agentcore.yaml        ← AgentCore deployment config
```

---

### Step 1 — Define the Confluence Tools with Tool Cache

```python
# tools.py
from memory_reuse import MemoryCache
from memory_reuse.integrations.langgraph import cached_tool

# Backend = agentcore (reads MEMORY_REUSE_* env vars)
# No Redis needed — AgentCore Memory is the shared external cache
cache = MemoryCache.from_env()

@cached_tool(cache, scope="global", ttl=1800)   # 30 min — search results rarely change
def search_confluence(query: str) -> list[dict]:
    """Search Confluence for pages matching the query."""
    response = confluence_client.search(
        cql=f'text ~ "{query}" AND space = "ENG"',
        limit=5
    )
    return response["results"]


@cached_tool(cache, scope="global", ttl=3600)   # 1 hour — page content mostly static
def fetch_page_content(page_id: str) -> str:
    """Fetch the full text content of a Confluence page."""
    page = confluence_client.get_page_by_id(
        page_id,
        expand="body.storage"
    )
    return page["body"]["storage"]["value"]


@cached_tool(cache, scope="user", ttl=600)      # 10 min — user-specific, scoped per user
def get_user_space_permissions(user_id: str) -> list[str]:
    """Check which Confluence spaces this user can access."""
    return confluence_client.get_user_permissions(user_id)
```

**What this does:**
- `search_confluence` — if the same query was searched in the last 30 minutes by anyone, return cached results. No Confluence API call.
- `fetch_page_content` — if the same page was fetched in the last hour, return cached content. No Confluence API call.
- `get_user_space_permissions` — scoped per user so User A's permissions are never served to User B.

---

### Step 2 — Define the LangGraph Nodes with Node Cache

```python
# nodes.py
from typing import TypedDict
from memory_reuse.integrations.langgraph import cached_node
from tools import search_confluence, fetch_page_content

class ConfluenceAgentState(TypedDict):
    question:       str
    user_id:        str
    session_id:     str
    user_context:   str    # injected from AgentCore Memory
    user_prefs:     dict   # injected from AgentCore Memory
    search_results: list
    page_content:   str
    answer:         str


def understand_request(state: ConfluenceAgentState) -> ConfluenceAgentState:
    """Parse and clean the user's question. Fast, no caching needed."""
    state["question"] = state["question"].strip()
    return state


@cached_node(cache, scope="global", ttl=1800)
def retrieve_confluence_content(state: ConfluenceAgentState) -> ConfluenceAgentState:
    """Search and fetch relevant Confluence pages.
    
    Entire node skipped if same question was retrieved in last 30 min.
    Tool-level cache inside also protects individual API calls.
    """
    results = search_confluence(state["question"])
    state["search_results"] = results

    if results:
        top_page_id = results[0]["id"]
        state["page_content"] = fetch_page_content(top_page_id)
    return state


@cached_node(cache, scope="user", ttl=600, semantic=True, similarity_threshold=0.90)
def generate_answer(state: ConfluenceAgentState) -> ConfluenceAgentState:
    """Call LLM to generate an answer.
    
    Semantic cache: if this user asked something similar before (same preference
    profile + similar question), skip LLM and return cached answer.
    User-scoped: different users with different preferences get different answers.
    """
    prompt = f"""
    You are a helpful engineering assistant.
    
    User context from past sessions: {state["user_context"]}
    User preferences: {state["user_prefs"]}
    
    Confluence documentation:
    {state["page_content"]}
    
    Question: {state["question"]}
    
    Answer based on the documentation, tailored to the user's experience level
    and preferences.
    """

    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state


def format_response(state: ConfluenceAgentState) -> ConfluenceAgentState:
    """Format final response. Always runs — fast and side-effect free."""
    return state
```

---

### Step 3 — Build the LangGraph Graph with Graph-Level Cache

```python
# graph.py
from langgraph.graph import StateGraph, END
from memory_reuse import MemoryCache
from nodes import (
    ConfluenceAgentState,
    understand_request,
    retrieve_confluence_content,
    generate_answer,
    format_response
)

# Backend = agentcore — shared across all microVMs, no Redis needed
cache = MemoryCache.from_env()

def build_graph():
    builder = StateGraph(ConfluenceAgentState)

    builder.add_node("understand",  understand_request)
    builder.add_node("retrieve",    retrieve_confluence_content)
    builder.add_node("generate",    generate_answer)
    builder.add_node("format",      format_response)

    builder.set_entry_point("understand")
    builder.add_edge("understand", "retrieve")
    builder.add_edge("retrieve",   "generate")
    builder.add_edge("generate",   "format")
    builder.add_edge("format",     END)

    graph = builder.compile()

    # Wrap entire graph with graph-level semantic cache
    # If a semantically similar question was fully answered before,
    # skip all nodes and return the cached final answer
    cached_graph = cache.wrap_graph(
        graph,
        scope="user",               # per-user graph cache (preferences affect answer)
        semantic=True,
        similarity_threshold=0.88,
        ttl=7200                    # 2 hours
    )

    return cached_graph
```

---

### Step 4 — AgentCore Memory Client

```python
# memory.py
from amazon_bedrock_agentcore.memory import MemoryClient

client = MemoryClient()
MEMORY_ID = "mem-confluence-agent-prod"


def get_user_context(user_id: str, session_id: str, question: str) -> tuple[str, dict]:
    """
    Retrieve two things from AgentCore Memory before execution:
    1. Conversational context — what happened in past sessions
    2. User preferences — how this user likes to receive answers
    """

    # Retrieve relevant past context for this question
    conv_results = client.retrieve_memories(
        memory_id = MEMORY_ID,
        namespace = f"user:{user_id}:conversations",
        query     = question,
        max_results = 3
    )
    conversational_context = " | ".join(
        r["content"] for r in conv_results.get("memoryRecords", [])
    )

    # Retrieve user preferences
    pref_results = client.retrieve_memories(
        memory_id = MEMORY_ID,
        namespace = f"user:{user_id}:preferences",
        query     = "user preferences experience level communication style",
        max_results = 5
    )
    preferences = {}
    for record in pref_results.get("memoryRecords", []):
        # AgentCore extracts structured preferences
        # e.g. {"experience": "senior", "format": "CLI", "language": "Python"}
        preferences.update(record.get("metadata", {}))

    return conversational_context, preferences


def store_conversation_event(user_id: str, session_id: str,
                              question: str, answer: str):
    """
    Store this interaction in AgentCore Memory after execution.
    AgentCore's long-term strategies will extract preferences and
    facts from this conversation automatically.
    """
    client.create_event(
        memory_id  = MEMORY_ID,
        session_id = session_id,
        actor_id   = user_id,
        payload    = [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer}
        ]
    )
```

---

### Step 5 — AgentCore Runtime Entrypoint (Wires Everything Together)

```python
# agent.py
from amazon_bedrock_agentcore import BedrockAgentCoreApp
from memory_reuse import MemoryCache
from graph import build_graph
from memory import get_user_context, store_conversation_event

app   = BedrockAgentCoreApp()
cache = MemoryCache.from_env()   # agentcore backend — no Redis needed
graph = build_graph()


@app.entrypoint
def handler(input: dict, context) -> dict:
    """
    Main agent entrypoint — called by AgentCore Runtime on each request.
    
    Flow:
    1. Get user identity from AgentCore Runtime context
    2. Retrieve user preferences + past context from AgentCore Memory
    3. Set cache context (user_id, session_id) on SDK
    4. Invoke graph — SDK intercepts at graph/node/tool level
       - Cache HIT  → returns immediately, skips all execution
       - Cache MISS → graph runs, result cached for next time
    5. Store conversation event in AgentCore Memory (always, even on cache hit)
    """

    user_id    = context.identity.user_id
    session_id = context.session_id
    question   = input.get("question", "")

    # ── Step 1: Get user context from AgentCore Memory ──────────────────
    user_context, user_prefs = get_user_context(user_id, session_id, question)
    # user_context: "User had issues with Docker setup last session"
    # user_prefs:   {"experience": "senior", "format": "CLI", "lang": "Python"}

    # ── Step 2: Set SDK cache context ───────────────────────────────────
    # SDK uses these to build correctly scoped cache keys
    # user-scoped: key = hash("user:{user_id}:{question}:{prefs}")
    cache.set_context(
        user_id    = user_id,
        session_id = session_id
    )

    # ── Step 3: Invoke graph (SDK handles all caching transparently) ─────
    #
    # What happens at each level:
    #
    # Graph level:  "Has this user (with their preference profile) asked
    #                something similar before?"
    #               YES → return final answer immediately (0 nodes, 0 tokens)
    #               NO  → continue to nodes
    #
    # Node level:   "Has 'retrieve_confluence_content' run with this
    #                question before?"
    #               YES → skip entire node (no API calls)
    #               NO  → run node, cache output
    #
    # Tool level:   "Has search_confluence(query) been called recently?"
    #               YES → skip Confluence API call, return cached results
    #               NO  → call Confluence API, cache result for 30 min
    #
    result = graph.invoke({
        "question":     question,
        "user_id":      user_id,
        "session_id":   session_id,
        "user_context": user_context,   # ← from AgentCore Memory
        "user_prefs":   user_prefs,     # ← from AgentCore Memory
        "search_results": [],
        "page_content": "",
        "answer":       ""
    })

    # ── Step 4: Store in AgentCore Memory (always, even on cache hit) ────
    # The conversation happened regardless of whether we used cached answer.
    # AgentCore Memory will:
    #   - Store raw event in short-term memory (this session)
    #   - Extract long-term insights (preferences, facts) asynchronously
    #     e.g. "User is a senior Python dev who prefers CLI examples"
    store_conversation_event(
        user_id    = user_id,
        session_id = session_id,
        question   = question,
        answer     = result["answer"]
    )

    return {
        "answer":      result["answer"],
        "cache_hit":   cache.last_hit_level,   # "graph" | "node" | "tool" | None
        "tokens_used": cache.last_tokens_used  # 0 on cache hit
    }
```

---

### Step 6 — Environment Variables for AgentCore Deployment

No Redis needed for this setup. The SDK uses AgentCore Memory as its backend — fully managed, zero extra infrastructure.

```yaml
# agentcore.yaml
runtime:
  name: confluence-agent
  entrypoint: agent.handler

environment:
  # memory-reuse — using AgentCore Memory as backend (no Redis required)
  MEMORY_REUSE_BACKEND:              agentcore
  MEMORY_REUSE_AGENTCORE_MEMORY_ID:  mem-confluence-agent-prod
  MEMORY_REUSE_SEMANTIC_ENABLED:     "true"
  MEMORY_REUSE_EMBEDDING_PROVIDER:   bedrock
  MEMORY_REUSE_EMBEDDING_MODEL:      amazon.titan-embed-text-v2
  MEMORY_REUSE_SIMILARITY_THRESHOLD: "0.90"

  # Confluence config
  CONFLUENCE_URL:                    https://your-org.atlassian.net
  CONFLUENCE_API_KEY:                "{{secret:confluence-api-key}}"
```

> **Why no Redis here?**
> AgentCore Memory is an external managed service — it lives outside microVMs and
> is accessible across all sessions. For a small-to-medium team this is enough.
> The tradeoff is retrieval latency (~100–300ms) vs Redis (<1ms).
> See the Scaling Note at the end for when to add Redis.

---

### What Happens on Each Request — Decision Tree

```
Employee: "How do I deploy to production?"
                │
                ▼
AgentCore Memory: retrieve user context
  → "User is senior dev, prefers CLI, had staging issues last week"
                │
                ▼
memory-reuse: graph-level cache check
  key = hash("deploy to production" + "senior+CLI+staging-issues" + user_id)
                │
       ─────────┴──────────
      │                    │
   CACHE HIT            CACHE MISS
   (seen before)        (first time or expired)
      │                    │
      ▼                    ▼
  Return answer       retrieve_confluence_content node
  in <5ms                   │
  0 tokens used             ▼
  0 API calls         Node cache check:
      │               "retrieve" node seen this question before?
      │                     │
      │              ───────┴──────────
      │             │                  │
      │          HIT                  MISS
      │          Node skipped          │
      │          0 API calls           ▼
      │                          Tool cache checks:
      │                          search_confluence() → HIT? skip
      │                          fetch_page_content() → HIT? skip
      │                                │
      │                                ▼
      │                          generate_answer node
      │                          Semantic cache check:
      │                          Similar question in last 2hr?
      │                                │
      │                         ───────┴──────────
      │                        │                  │
      │                     HIT                  MISS
      │                     LLM skipped          LLM called
      │                     0 tokens             tokens used
      │                                          result cached
      │                          │
      └──────────────────────────┘
                │
                ▼
      AgentCore Memory: store_conversation_event()
      (always runs — conversation recorded regardless of cache hit)
                │
                ▼
      Response returned to employee
```

---

### Results After 1 Week of Team Usage

After the cache warms up across 50 employees:

```
Week 1 (cold cache):
  Monday:   mostly cache misses, full LLM calls
  Tuesday:  ~20% hit rate as common questions start repeating
  Wednesday: ~45% hit rate
  Thursday: ~60% hit rate
  Friday:   ~70% hit rate

Steady state (week 2+):
  Cache hit rate:         ~70–80%
  LLM calls avoided:      ~75%
  Confluence API calls:   ~85% avoided (pages rarely change)
  Average response time:  <50ms (vs 2–4 sec without cache)
  Monthly LLM cost:       ~$22 (vs ~$74 without SDK)

AgentCore Memory benefit:
  Each user gets answers tailored to their experience level
  Agent remembers past issues (e.g. Docker problems)
  Preferences captured after session 1, used from session 2 onward
```

---

### The Three Services — Final Summary in Context of This Agent

```
AgentCore Runtime
  Role:    Hosts and scales the agent
  Handles: microVM isolation, auto-scaling, identity, observability
  Impact:  Zero infra to manage, handles 1 to 1000 concurrent users

AgentCore Memory
  Role:    Remembers who each user is across sessions
  Handles: Conversation history, preference extraction, past context
  Impact:  Personalised answers — junior devs get step-by-step,
           senior devs get concise CLI commands

memory-reuse (our package)
  Role:    Stops the agent from doing the same work twice
  Handles: Tool cache, node cache, graph cache, semantic deduplication
  Impact:  70–80% cost reduction, <50ms response on cache hits
```

**None of the three replaces the others. Each owns a distinct job. Together they make a production-grade agent that is fast, cheap, and personalised.**

---

### ⚠️ Scaling Note — When to Add Redis

This example uses **AgentCore Memory as the only external backend** — no Redis, no ElastiCache, zero extra infrastructure.

This works well for small teams. As your team grows, the tradeoffs shift.

```
Team size       Backend recommendation
─────────────────────────────────────────────────────────────
< 10 people     AgentCore Memory only  ← this example
                Free to start, zero infra, ~100–300ms cache lookup

10–50 people    AgentCore Memory only, watch latency
                If response time feels slow, add Upstash Redis

50+ people      Add Amazon ElastiCache (Redis)
                Cache lookups drop from 200ms → <1ms
                Redis instance (~$25/mo) pays for itself immediately

100+ people     ElastiCache + AgentCore Memory (split responsibilities)
                Redis   → hot path cache (exact + tool, <1ms)
                AgentCore Memory → long-term user memory (cross-session)
```

**How to upgrade when ready — one config change:**

```yaml
# agentcore.yaml — upgrade from agentcore backend to Redis + agentcore

environment:
  # Switch SDK to Redis for hot-path cache
  MEMORY_REUSE_BACKEND:              redis
  MEMORY_REUSE_REDIS_URL:            redis://your-elasticache-endpoint:6379
  MEMORY_REUSE_SEMANTIC_ENABLED:     "true"
  MEMORY_REUSE_EMBEDDING_PROVIDER:   bedrock
  MEMORY_REUSE_EMBEDDING_MODEL:      amazon.titan-embed-text-v2

  # AgentCore Memory still used for long-term user memory
  # (handled directly via memory.py — unchanged)
```

No code changes. No graph restructuring. Just a config variable swap.
The SDK swaps the backend transparently — all decorators, all cache levels work identically.
