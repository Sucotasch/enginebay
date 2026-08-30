# AGENTS.md

## What this is

Windows-only llama.cpp inference server. Runs Qwen3.6-27B (IQ4_XS, 14GB GGUF) on RTX 4070 Ti SUPER (16GB VRAM). Provides OpenAI-compatible API for Hermes CLI client.

## Key files

| File | Purpose |
|------|---------|
| `start-llama.bat` | Start server — Qwen3.6-27B on port 8080 |
| `start-beellama.bat` | Start server — Qwen3.6-27B with BeeLlama KVarN (port 8080) |
| `stop-llama.bat` | Kill server |
| `Launcher.bat` | Open PyQt6 GUI (`launcher.py`) |
| `launch-hermes-llama.bat` | Start server + Hermes with local provider |
| `launcher.py` | GUI: model selection, presets, llama.cpp version management, **engine selection** |
| `launcher_presets.json` | Saved configs: "Qwen3.6-27B" (port 8080), "Agentic AI" (port 8888), "Qwen3.6-27B (Bee KVarN)" |
| `configs/inference.env` | Server parameters |
| `scripts/start_llama_cpp.sh` | Alternative launcher (Git Bash) |
| `scripts/smoke_test.py` | Verify server is responding |

## Critical constraints

- **96K context (98304) is optimal.** 128K kills decode speed (6 tok/s vs 34 tok/s). Do not change `-c` to 131072.
- **Upstream llama.cpp: q4_0 KV cache required.** q8_0 kills speed. Both `--cache-type-k` and `--cache-type-v` must be `q4_0`.
- **BeeLlama engine: use KVarN cache types** — `--cache-type-k kvarn5 --cache-type-v kvarn4 --kv-tail-tokens 1024` (balanced default). Upstream q4_0 also works but forfeits the fork's KV compression.
- **`--n-cpu-moe` is useless** for Qwen3.6-27B — it's a dense model, not MoE.
- **`--reasoning off`** required for fast single-shot inference on Qwen3.6-27B and Gemma. **Exception: Qwen3.8 (qwen35 arch) breaks with `--reasoning off`** — use `--reasoning auto` or the sharp chat template with `--chat-template-kwargs`.
- **`-np 1` on Qwen3.8 presets** — auto parallel (`n_parallel=-1`) creates 4 slots × 96K KV cache, which tanks VRAM and drops speed to ~3.8 tok/s.
- **Two ports**: 8080 (Qwen3.6-27B) and 8888 (Gemma 4 26B). Do not mix presets.

## How to run

```bash
# Start server (Windows CMD, upstream llama.cpp)
start-llama.bat

# Start server (Windows CMD, BeeLlama KVarN)
start-beellama.bat

# Start server + Hermes
launch-hermes-llama.bat

# Open GUI
Launcher.bat

# Smoke test (after activating venv)
source .venv/Scripts/activate
python scripts/smoke_test.py
```

## Alternative engines

The launcher supports multiple llama-server engines. Each engine has its own GitHub release source, version download dir, and asset parser:

| Engine | Repo | Versions dir | Extra cache types |
|--------|------|--------------|-------------------|
| `llama.cpp` (upstream) | `ggml-org/llama.cpp` | `llama.cpp/versions/` | — |
| `beellama.cpp` (fork) | `Anbeeld/beellama.cpp` | `beellama.cpp/versions/` | `kvarn2`-`kvarn8`, `q2_0`-`q6_1`, `--kv-tail-tokens` |
| `ik_llama.cpp` (fork, **manual build**) | `ikawrakow/ik_llama.cpp` | `ik_llama.cpp/versions/` | IQ4_KT/IQ4_KS, IQK quants (ggml types > 49) |

**ik_llama.cpp has NO prebuilt Windows binaries** (only tag `t0002` with 0 assets). It must be built from source. This is the ONLY engine that can read models quantized with ik_llama.cpp's extended types — e.g. `Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf` (ggml type 144, unreadable by beellama which stops at type 49).

To build (see `build-ikllama.bat`):
1. Requires: VS BuildTools (vcvars64.bat), CMake ≥ 3.24, Git, CUDA Toolkit (nvcc + cublas.lib).
2. CUDA 13.1 + MSVC 14.44 need `CUDA_PATH_V13_1` env var set BEFORE vcvars64, or MSBuild fails with "CUDA Toolkit directory '' does not exist".
3. Build command: `cmake -S ik_llama.cpp -B ik_llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_LLAMAFILE=OFF` then `cmake --build ... --config Release --parallel 6`. **Run `build-ikllama.bat` by hand (double-click / own terminal), NOT through the agent sandbox** — the sandbox blocks the inter-process pipes MSBuild uses for parallel worker nodes, so the build silently degrades to ONE process (11-13% CPU, several hours). Outside the sandbox, `--parallel N` (default 6) runs 4-6 parallel cl.exe processes like normal MSVC projects.
4. Runtime CUDA DLLs (cublas64_13.dll, cublasLt64_13.dll, cudart64_13.dll) are NOT in the local CUDA v13.1 `bin` — copy them from `beellama.cpp/versions/preview-v0.4.4-cuda-13.1/`.
5. **Builds take 30-90 min on i7-5820K.** Never interrupt — partial builds must restart from scratch.
6. The launcher's "Check Updates" on this engine shows build instructions (manual_build flag); it does NOT query GitHub releases.

To add a new engine: extend the `ENGINES` dict in `launcher.py` with `label`, `repo`, `api`, `versions_dir`, a `classify(asset_name) -> (kind, cuda)` parser (or `classify: None` + `manual_build: True` for source-built engines), and `default_params`. The GUI dropdown, update checker, and version download all read from this registry.

To use BeeLlama:
1. In the GUI, set the **Engine** dropdown to "BeeLlama.cpp (fork)".
2. **Check Updates** → download a Windows CUDA build (e.g. `v0.4.3-cuda-13.3`).
3. Click **Use** to activate it, then load the "Qwen3.6-27B (Bee KVarN)" preset.

## Model details

Qwen3.6-27B is a **hybrid Dense** model: 48 DeltaNet layers (linear attention, no KV cache) + 16 GQA Attention layers (traditional, needs KV cache). Only 16 layers consume KV cache — that's why a 14GB model fits in 16GB VRAM at 96K context.

## llama-server binary

Auto-discovered from:
1. `LLAMA_SERVER_PATH` env var
2. LM Studio cache: `~/.cache/lm-studio/extensions/backends/`
3. PATH
4. Common install locations
5. Installed engine versions (via the "Use" button in the GUI)

## Hermes integration

Launcher auto-registers `local-llama` provider in `~/.hermes/config.yaml`. Port changes in the GUI auto-sync to the Hermes config.

```yaml
# ~/.hermes/config.yaml (auto-managed)
providers:
  local-llama:
    base_url: http://127.0.0.1:8888/v1
    api_key: not-needed
    models:
    - Local Model
```

## Optimal launch command

```bash
llama-server \
  -m "G:/Ai/Models/Qwen3.6-27B.i1-IQ4/Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf" \
  -c 98304 -ngl 99 -b 2048 -ub 512 \
  --kv-unified --cache-type-k q4_0 --cache-type-v q4_0 \
  -t 5 --flash-attn on --reasoning off --jinja \
  --temp 1.0 --min-p 0.05 --top-p 0.95 --top-k 64 \
  --host 127.0.0.1 --port 8080
```
