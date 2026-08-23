# Install

Works with both **pip** and **uv** — pick whichever you use.

=== "pip"
    ```bash
    pip install memory-reuse
    ```

=== "uv"
    ```bash
    uv add memory-reuse
    ```

The core has **zero required dependencies**. Optional features are opt-in via
extras.

## Optional extras

| Extra | What it adds | Install |
|---|---|---|
| `redis` | Redis backend support | `pip install "memory-reuse[redis]"` |
| `litellm` | LiteLLM cached wrappers | `pip install "memory-reuse[litellm]"` |
| `semantic` | Semantic cache with **API** embeddings (OpenAI / LiteLLM) — no torch | `pip install "memory-reuse[semantic]"` |
| `semantic-local` | Semantic cache with **local** embeddings (sentence-transformers, pulls in torch) | `pip install "memory-reuse[semantic-local]"` |
| `all` | Everything above | `pip install "memory-reuse[all]"` |

!!! note "Two commands cover every embedding provider"
    The `semantic` extra bundles the small `openai` and `litellm` clients (plus
    `numpy`) and installs **no torch** — enough for the OpenAI and LiteLLM
    embedding providers. Only the **local** provider needs
    `semantic-local`.

## Local embeddings and PyTorch

`sentence-transformers` (the `local` provider) depends on **PyTorch**. On a
CPU-only machine, install the CPU torch wheel first to avoid a ~2 GB GPU/CUDA
download — the CPU build is ~200 MB:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "memory-reuse[semantic-local]"
```

With a GPU you can skip the first line. Either way, the model weights
(e.g. `all-MiniLM-L6-v2`, ~90 MB) download from Hugging Face on first use and
are then cached on disk for offline reuse.

!!! tip "Quieter / fully offline runs"
    memory-reuse silences the Hugging Face log chatter and the per-embedding
    progress bar. Once the model is cached, you can also skip Hugging Face's
    HTTP cache-validation checks by exporting `HF_HUB_OFFLINE=1` (and
    `TRANSFORMERS_OFFLINE=1`) **in your own application** — never inside a
    shared library, and only after the first (downloading) run.
