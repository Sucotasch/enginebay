# AGENTS.md

## What this is

Windows-only llama.cpp inference server (EngineBay). Runs Qwen3.8-27B (IQ4_KT/KS, 14GB GGUF) on RTX 4070 Ti SUPER (16GB VRAM) via ik_llama.cpp. Provides OpenAI-compatible API for Hermes CLI client and DeepSeek Harness GUI.

## Key files

| File | Purpose |
|------|---------|
| `start-llama.bat` | Start server — Qwen3.8-27B on port 8080 |
| `start-beellama.bat` | Start server — Qwen3.8-27B with BeeLlama KVarN (port 8080) |
| `stop-llama.bat` | Kill server |
| `Launcher.bat` | Open PyQt6 GUI (`launcher.py`) |
| `launch-hermes-llama.bat` | Start server + Hermes with local provider |
| `launcher.py` | GUI: model selection, presets, llama.cpp version management, **engine selection** |
| `launcher_presets.json` | Saved configs: "Qwen3.8-27B" (port 8080), "Agentic AI" (port 8888), "Qwen3.8-27B (Bee KVarN)" |
| `configs/inference.env` | Server parameters |
| `scripts/start_llama_cpp.sh` | Alternative launcher (Git Bash) |
| `scripts/smoke_test.py` | Verify server is responding |
| `scripts/update_opencode_models.py` | Refresh OpenCode Free models in `~/.dsh/settings.yaml` (see skill `dsh-providers`) |

## Critical constraints

- **96K context (98304) is optimal.** 128K drops decode speed (32.1 vs 35.9 t/s on ik_llama). Do not change `-c` to 131072.
- **Upstream llama.cpp: q4_0 KV cache required.** q8_0 kills speed. Both `--cache-type-k` and `--cache-type-v` must be `q4_0`.
- **BeeLlama engine: use KVarN cache types** — `--cache-type-k kvarn5 --cache-type-v kvarn4 --kv-tail-tokens 1024` (balanced default). Upstream q4_0 also works but forfeits the fork's KV compression.
- **`--n-cpu-moe` is useless** for Qwen3.8-27B — it's a dense model, not MoE.
- **`--reasoning auto` + `--jinja` required on Qwen3.8 (qwen35)** — `--reasoning off` breaks tools requests. (Upstream Qwen3.6-27B and Gemma use `--reasoning off`.)
- **`-np 1` on Qwen3.8 presets** — auto parallel (`n_parallel=-1`) creates 4 slots × 96K KV cache, which tanks VRAM and drops speed to ~3.8 tok/s.
- **Two ports**: 8080 (Qwen3.6/3.8-27B) and 8888 (Gemma 4 26B / Agentic). Do not mix presets.

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

## Git push from this workspace (working method)

Plain `git push` fails in this sandbox: `credential.helper=manager` tries to
open a GUI prompt and `sh.exe` can't create its signal pipe. The working
method uses the already-authenticated GitHub CLI token via an embedded URL
(push only; remote is restored to the clean URL afterwards):

```powershell
$token = gh auth token          # gh is logged in as Sucotasch
$orig  = git remote get-url origin
git remote set-url origin "https://x-access-token:$token@github.com/Sucotasch/enginebay.git"
git push origin main
git remote set-url origin "$orig"   # always restore!
```

(Tested 2026-09-01: pushed `main`, worked. Don't commit the embedded-token URL.)

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
3. Click **Use** to activate it, then load the "Qwen3.8-27B (Bee KVarN pure)" preset.

## Model details

Qwen3.8-27B (qwen35 arch) is a **dense** model (not MoE). The primary
`Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf` (~14 GB) is only readable by
ik_llama.cpp (trellis quants 144/145). The IQ4_XS variant (~13.5 GB) is readable
by all engines. On 16 GB VRAM, measured 35.9 t/s at 96K context (pure mode);
MTP speculation fits only up to 48K. (The older Qwen3.6-27B is a hybrid dense
model: 48 DeltaNet layers + 16 GQA attention layers.)

## llama-server binary

Auto-discovered from:
1. `LLAMA_SERVER_PATH` env var
2. LM Studio cache: `~/.cache/lm-studio/extensions/backends/`
3. PATH
4. Common install locations
5. Installed engine versions (via the "Use" button in the GUI)

## Hermes + DeepSeek Harness integration

Launcher auto-registers the server as a local provider in BOTH clients, and
port changes in the GUI auto-sync to both configs:

| Client | Config | Provider | Sync mechanism |
|---|---|---|---|
| Hermes CLI | `~/.hermes/config.yaml` | `local-llama` | `_on_port_changed` → regex update of `base_url` |
| DeepSeek Harness GUI | `~/.dsh/settings.yaml` | `local_llama` (under `llm-pi-ai.providers`) | `_on_port_changed` → `_dsh_upsert_local_llama()` block-scoped update of `baseURL` |

The provider is created on first launch if missing. DSH settings.yaml is a
user-owned config — only the `local_llama` block is ever touched, and `_ensure_dsh_provider()`
backs up nothing itself: always keep `~/.dsh/settings.yaml.pre-*` backups if you
hand-edit it. `_dsh_upsert_local_llama(content, host, port)` is a pure module-level
function (unit-testable); it updates the port when the block exists and inserts a
fresh block under `llm-pi-ai: providers:` when it doesn't.

## Optimal launch command (ik_llama, Qwen3.8-27B IQ4_KT/KS, 96K pure, 35.9 t/s)

```bash
llama-server \
  -m "/path/to/Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf" \
  -c 98304 -np 1 -ngl 99 -b 1024 -ub 256 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -t 5 -tb 6 --flash-attn on --jinja --reasoning auto \
  --temp 1.0 --min-p 0.0 --top-p 0.95 --top-k 20 \
  --presence-penalty 0.0 --repeat-penalty 1.0 --no-mmproj-offload \
  --host 127.0.0.1 --port 8080
```

For upstream llama.cpp with Qwen3.6-27B IQ4_XS, use `--reasoning off --kv-unified -b 2048 -ub 512` instead.
