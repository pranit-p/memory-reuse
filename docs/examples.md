# Examples

Runnable examples live in the
[`examples/`](https://github.com/pranit-p/memory-reuse/tree/main/examples)
directory of the repository.

| Example | What it shows |
|---|---|
| `basic_exact_cache.py` | The cache primitives with no framework. |
| `langgraph_agent_example.py` | Cached nodes and tools in a LangGraph-style flow. |
| `langgraph_math_agent.py` | A real ReAct agent with a **calculator** and a **web-search** tool, calling an LLM via LiteLLM. |
| `semantic_cache_demo.py` | A reworded query hitting the **semantic cache** via the combined `lookup`/`store` flow (offline, no model download). |
| `semantic_agent.py` | A real ReAct agent (**calculator** + **web search**) whose LLM calls run through the **semantic cache** with a local embedding model, plus per-user scoping and answer extraction. |

Running the LLM-backed examples (they use Groq via LiteLLM by default):

```bash
export API_KEY="your-groq-key"
python examples/langgraph_math_agent.py
```

The `semantic_agent.py` example additionally needs the local embedding model:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "memory-reuse[semantic-local,litellm]" langgraph langchain-core httpx
python examples/semantic_agent.py
```
