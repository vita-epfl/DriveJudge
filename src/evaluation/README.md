# 🧪 src/evaluation/

Everything that runs a VLM over the benchmark and writes a results JSON, split by model access:

- [`open_source/`](open_source/) — locally-hosted models (Qwen3-VL, Qwen3-Omni, Cosmos-Reason,
  InternVL3.5, LLaVA-OneVision) run on the cluster via vLLM / transformers / lmdeploy.
- [`closed_source/`](closed_source/) — API models (**GPT**, **Gemini**, **Claude**) run through
  their providers' Batch APIs.

All paths produce the same downstream shape, consumed by [`../analysis/`](../analysis/). The
tool-using **agent** lives one level up in [`../agentic/`](../agentic/). See the
[root README](../../README.md).
