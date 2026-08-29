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

To add a new engine: extend the `ENGINES` dict in `launcher.py` with `label`, `repo`, `api`, `versions_dir`, a `classify(asset_name) -> (kind, cuda)` parser, and `default_params`. The GUI dropdown, update checker, and version download all read from this registry.

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
