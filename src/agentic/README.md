# 🤖 src/agentic/

A `qwen-agent` pipeline that wraps **Qwen3-VL** with perception tools and lets it *investigate* a
clip before answering. It talks to a **local vLLM server that must already be running on port 8000**
(the [root README](../../README.md#-drivejudge-the-tool-augmented-vlm-judge) has the exact
`vllm serve` command and the agent's venv). It runs in its own environment, [`requirements-agent.txt`](../../requirements-agent.txt)
(transformers 5 for SAM 3), separate from the vLLM server's. Run it with the agent dir as the
working directory:
`cd src/agentic && python main_multi_tools_v4.py --storage_path <STORAGE>`. `ROOT_DIR` /
`STORAGE_PATH` at the top of the script and `MODEL_PATH` in `utils/sam.py` are hardcoded to our
cluster: edit them first.

**Contents**
- `main_multi_tools_v4.py` — the winning agent (`v4o` / `v4p`). Registers `get_motion_info` (RAFT),
  `get_masks` (SAM), `get_frequency_analysis` (FFT) and `final_answer` (`{evaluation, answer}`).
- `main_multi_tools_timing.py` — latency-instrumented variant (feeds the timing study).
- `utils/raft.py`, `utils/sam.py` — the optical-flow and segmentation tool workers.
- `fft/compute_fft.py` — the FFT spectral tool (the cue that exposes generation artifacts).
- `postprocess_missing.py` — recovers answers from raw output when `final_answer` wasn't called.
- `video_resolve.py` — maps a question's video path to the clip on disk by its path relative to
  `final_dataset/` (not its basename: several clips share a basename across folders).
- `counterfactual/` — the counterfactual-video ablation (see its `HANDOFF.md`).

Flags worth knowing: `--num_workers` (parallel questions), `--temperature` (0.3 in the paper),
`--fft-lattice` (adds the lattice score to the FFT tool output; off = paper behaviour),
`--save-history/--no-save-history`.

This dir is self-contained: every import (`utils.*`, `fft.compute_fft`) resolves with `src/agentic/`
as the working directory. See the [root README](../../README.md).
