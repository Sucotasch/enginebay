# EngineBay — LLM Inference Server

Local llama.cpp inference server with PyQt6 GUI launcher. Provides OpenAI-compatible API for Hermes CLI client and DeepSeek Harness GUI.

## Features

- **PyQt6 GUI launcher** — model selection, presets, llama.cpp version management
- **Auto-discovery** — finds Hermes config and llama-server binary automatically
- **Version manager** — download, install, and switch between llama.cpp releases from GitHub
- **Multi-engine support** — select between upstream llama.cpp, BeeLlama.cpp (fork), and ik_llama.cpp (fork, source-built) with per-engine version management, download, and activation
- **Engine-safe presets** — presets carry params/host/port only (never model paths); incompatible flags are auto-detected when switching engines or loading presets, with a warning dialog instead of a crash
- **Health monitor** — real-time server status in the GUI
- **Hermes + DeepSeek Harness sync** — auto-registers the running server as a local provider in both `~/.hermes/config.yaml` and `~/.dsh/settings.yaml`, keeping the port in sync when you change it in the GUI

> Engineering notes, measured VRAM budgets, engine flag differences, and build
> gotchas live in **[dev_guide.md](dev_guide.md)**.

## Project Structure

```
llm-inference-server/
├── launcher.py              ← PyQt6 GUI (model selection, presets, engines, versions)
├── launcher_presets.json    ← saved configs (incl. BeeLlama KVarN, ik_llama presets)
├── launcher_config.json     ← last used config (git-ignored)
├── dev_guide.md             ← engineering notes, VRAM budgets, engine flags, gotchas
├── AGENTS.md                ← agent instruction file (engine table, build steps)
├── Launcher.vbs             ← launch GUI (no terminal)
├── Launcher.bat             ← launch GUI (fallback)
├── start-llama.bat          ← start server (Qwen3.6-27B, upstream, port 8080)
├── start-beellama.bat       ← start server (Qwen3.6-27B, BeeLlama KVarN, port 8080)
├── stop-llama.bat           ← stop server
├── launch-hermes-llama.bat  ← start server + Hermes
├── build-ikllama.bat        ← build ik_llama.cpp from source (manual-build engine)
├── configs/
│   └── inference.env        ← server parameters
├── scripts/
│   ├── start_llama_cpp.sh   ← launcher (Git Bash)
│   ├── start_llama_cpp.bat  ← launcher (Windows CMD)
│   ├── smoke_test.py        ← verify server
│   └── probe_hardware.py    ← detect GPU/CPU/RAM
├── benchmarks/
│   ├── bench.sh             ← TTFT/TPOT benchmark
│   └── systematic_bench.py  ← automated config testing
├── llama.cpp/versions/      ← downloaded upstream llama.cpp releases
├── beellama.cpp/            ← BeeLlama source checkout (reference)
│   └── versions/            ← downloaded BeeLlama releases
├── ik_llama.cpp/            ← ik_llama source + build output (git-ignored; manual-build)
│   └── versions/            ← built llama-server.exe + DLLs
└── pyproject.toml
```

## Client integration (Hermes + DeepSeek Harness)

The launcher auto-registers the running server as a local provider in **both**
clients, and port changes in the GUI auto-sync to both configs:

| Client | Config | Provider | Sync mechanism |
|---|---|---|---|
| **Hermes** CLI | `~/.hermes/config.yaml` | `local-llama` | `_on_port_changed` → regex update of `base_url` |
| **DeepSeek Harness** GUI | `~/.dsh/settings.yaml` | `local_llama` (under `llm-pi-ai.providers`) | `_on_port_changed` → `_dsh_upsert_local_llama()` block-scoped update of `baseURL` |

The provider is created on first launch if missing; only the `local_llama` block
in `settings.yaml` is ever touched. See [AGENTS.md](AGENTS.md#hermes--deepseek-harness-integration).

# EngineBay Technical Documentation

Welcome to the technical documentation for the **LLM Inference Server**, a tailored, Windows-optimized `llama.cpp` inference environment equipped with a PyQt6 Graphical User Interface (GUI). This project is specifically engineered to run large hybrid dense models (like Qwen3.6-27B) natively on consumer-grade hardware (such as the RTX 4070 Ti SUPER with 16GB VRAM) while providing a seamless OpenAI-compatible API for CLI clients like Hermes.

---

## 1. Real Capabilities of the Program

The application acts as a comprehensive wrapper and manager for local LLM inference, delivering the following core features:

* **PyQt6 GUI Launcher:** A rich graphical interface for managing model selection, network parameters (host/port), and inference flags.
* **Llama.cpp Version Manager:** Automatically fetches release data from the GitHub API, allowing users to download, install, and hot-swap between different `llama-server` binary versions without leaving the UI.
* **Auto-Discovery Engine:** Intelligently locates essential dependencies without manual configuration:
  * Automatically finds existing `llama-server` binaries (via environment variables like `LLAMA_SERVER_PATH`, system `PATH`, or within LM Studio's cache).
  * Auto-discovers the Hermes CLI configuration directory (`~/.hermes/config.yaml`).
* **Hermes + DeepSeek Harness Integration:** Automatically registers and updates the server as a local provider in both `~/.hermes/config.yaml` (`local-llama`) and `~/.dsh/settings.yaml` (`local_llama`), keeping API ports synchronized. The provider is created on first launch if missing.
* **Console-Free Execution:** Utilizes VBScript (`Launcher.vbs`) to start the server and GUI silently in the background, preventing terminal clutter.
* **Health & Status Monitoring:** Built-in polling to verify if the inference server is responding and healthy.

---

## 2. Algorithm of Operation and Architecture

### System Architecture
The repository relies on a loosely coupled architecture separating the graphical management layer from the underlying execution engine:

1. **Presentation & Management Layer (`launcher.py`):** 
   Built using PyQt6. It acts as the orchestrator. State is preserved across sessions using lightweight JSON storage:
   * `launcher_config.json`: Saves the last-used state (model path, binary path, ports).
   * `launcher_presets.json`: Stores user-defined parameter profiles (e.g., specific ports for specific models).
2. **Execution Engine (`llama-server.exe`):**
   The underlying C++ inference backend. When launched via the GUI, a subprocess is created. The GUI intercepts process state and exposes an OpenAI-compatible REST API (typically on port 8080 or 8888).
3. **Integration Layer:**
   When settings change in the GUI, the application parses and mutates the YAML configuration of external agents (like Hermes) to ensure local endpoints are perfectly aligned with the running server.

### Hardware & Model Optimization (Qwen3.6-27B)
The server is explicitly configured around the unique physical architecture of **Qwen3.6-27B**, which is a **Hybrid Dense** model (not a Mixture of Experts / MoE):
* **Layer Topology:** It consists of 64 total layers—48 DeltaNet layers (linear attention) and 16 GQA Attention layers. 
* **KV Cache Economics:** Because DeltaNet layers do not require a traditional KV Cache, only 16 layers consume VRAM for context. This architectural quirk is what allows a 14GB quantized model to process a massive **96,000-token context window** while fitting entirely inside a 16GB RTX 4070 Ti SUPER.

---

## 3. Installation and Configuration

### Prerequisites
* **OS:** Windows 10/11
* **Hardware:** Modern GPU with at least 16GB VRAM (e.g., RTX 4070 Ti SUPER)
* **Software:** Python 3.x

### Setup Process
1. Clone the repository to your local machine.
2. Install the necessary Python dependencies for the GUI and testing scripts:
   ```bash
   pip install PyQt6 openai psutil
   ```
3. *(Optional but Recommended)* Ensure `llama-server.exe` is either in your system PATH, or download it directly through the Launcher's built-in Version Manager.

### Critical Server Tuning & Constraints
To achieve optimal performance (~34 tokens/sec decoding, ~58 tokens/sec prefilling), you **must** adhere to the following configuration constraints:

* **Optimal Context Limit (`-c 98304`):** 96K is the absolute optimal context. Pushing the context to 128K will catastrophically degrade decoding speeds (dropping from ~34 tok/s to ~6 tok/s).
* **KV Cache Quantization (`--cache-type-k q4_0 --cache-type-v q4_0`):** Both Key and Value caches **must** be quantized to `q4_0` when running upstream llama.cpp. Using higher precision (like `q8_0`) at large contexts will exhaust VRAM and ruin generation speeds.
* **CPU MoE Flag:** Do not use `--n-cpu-moe`. Qwen3.6-27B is a dense model; this flag will have no effect.
* **Reasoning Off (`--reasoning off`):** Required to ensure fast, single-shot inference responses without inner-monologue delays.

### Alternative Engines (BeeLlama.cpp)

The launcher can manage multiple llama-server engines. Besides upstream `llama.cpp`, it supports **BeeLlama.cpp** — a performance-focused fork that adds:

- **KVarN KV-cache quantization** — variance-normalized cache types `kvarn2`–`kvarn8` (e.g. `--cache-type-k kvarn5 --cache-type-v kvarn4`)
- **KV cache precision tail** — keep recent tokens exact in F16/BF16 (`--kv-tail-tokens 1024`)
- **Additional low-bit caches** — `q2_0`–`q6_1`
- **Adaptive DFlash draft control** and **reasoning-loop protection**

To use it:

1. Open `Launcher.bat`
2. Set the **Engine** dropdown to `BeeLlama.cpp (fork)`
3. **Check Updates** → download a Windows CUDA build (e.g. `v0.4.3-cuda-13.3`)
4. Click **Use** to activate the binary
5. Load the `Qwen3.6-27B (Bee KVarN)` preset — it swaps the KV cache to KVarN + 1024-token tail

Each engine keeps its own versions directory (`llama.cpp/versions/` vs `beellama.cpp/versions/`), so you can switch back and forth freely. See [AGENTS.md](AGENTS.md#alternative-engines) for how to register additional engines.

### ik_llama.cpp (fork — trellis quants, manual build)

**ik_llama.cpp** is the **only engine that can read the IQ4_KT/IQ4_KS trellis
quants** (ggml types 144/145, e.g. `Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf`).
Neither beellama (stops at type 49) nor vllm.cpp (decodes only up to IQ4_XS) can
open these files.

- **No prebuilt Windows binaries** (tag `t0002`, 0 assets) — build from source via
  `build-ikllama.bat` (CMake + CUDA, arch 89). See [dev_guide.md](dev_guide.md#5-сборка-ik_llama.cpp-из-исходников) for build details.
- Adds speculative MTP chains: `--spec-type ngram-mod:... --spec-type mtp:...`
  (two-stage), but on 16GB VRAM MTP only fits up to 48K context — **for ≥64K use
  the pure preset** (measured 35.9 t/s @ 96K).
- Does **not** support `--kv-unified`, `kvarn*`, `--kv-tail-tokens`, or
  `draft-mtp` (beellama flags). The launcher detects these and warns.

### Engine-safe presets

Presets store **params/host/port only — never model paths** (the model is picked
in the GUI). Because each engine uses different flag names for the same concept
(KV cache, speculative decoding, reasoning), the launcher:

1. Resets params to the engine's defaults whenever you switch engines;
2. Detects stale/foreign flags on config load and resets to safe defaults;
3. **Warns with a dialog when you load a preset whose flags the current engine
   cannot parse** (offers "Load engine defaults" or "Apply preset anyway").

---

## 4. Usage Examples

### 1. Launching the GUI (Recommended)
To open the graphical interface without spawning an annoying terminal window, double-click:
```text
Launcher.vbs
```
*Alternatively, you can run `Launcher.bat` or execute `python launcher.py` from your terminal.*

### 2. One-Click Launch (Server + Client)
If you want to immediately boot the inference server, wait for the model to load into VRAM, and simultaneously start the Hermes CLI client attached to it, run:
```bat
launch-hermes-llama.bat
```

### 3. CLI: The Optimal Launch Command
If you are bypassing the GUI and scripting your own environment, use this optimized launch string for Qwen3.6-27B (IQ4_XS):

```bash
llama-server \
  -m "G:/Ai/Models/Qwen3.6-27B.i1-IQ4/Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf" \
  -c 98304 -ngl 99 -b 2048 -ub 512 \
  --kv-unified --cache-type-k q4_0 --cache-type-v q4_0 \
  -t 5 --flash-attn on --reasoning off --jinja \
  --temp 1.0 --min-p 0.05 --top-p 0.95 --top-k 64 \
  --host 127.0.0.1 --port 8080
```

### 4. Running the Smoke Test
To verify that your server is running, healthy, and successfully exposing the OpenAI-compatible API, execute the built-in Python diagnostic tool:

```bash
# Ensure your virtual environment is active if applicable
python scripts/smoke_test.py
```
*Output Expectation:* The script will connect to `http://127.0.0.1:8080/v1` and list the available loaded models, confirming API health.

### 5. Probing Hardware Capabilities
To double-check how much VRAM/RAM your system currently has available before loading massive 96K contexts, use the hardware probe script:
```bash
python scripts/probe_hardware.py
```
