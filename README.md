# LLM Inference Server

Local llama.cpp inference server with PyQt6 GUI launcher. Provides OpenAI-compatible API for Hermes CLI client.

## Features

- **PyQt6 GUI launcher** — model selection, presets, llama.cpp version management
- **Auto-discovery** — finds Hermes config and llama-server binary automatically
- **Version manager** — download, install, and switch between llama.cpp releases from GitHub
- **Health monitor** — real-time server status in the GUI

## Project Structure

```
llm-inference-server/
├── launcher.py              ← PyQt6 GUI (model selection, presets, versions)
├── launcher_presets.json    ← saved configs
├── launcher_config.json     ← last used config
├── Launcher.vbs             ← launch GUI (no terminal)
├── Launcher.bat             ← launch GUI (fallback)
├── start-llama.bat          ← start server (Qwen3.6-27B, port 8080)
├── stop-llama.bat           ← stop server
├── launch-hermes-llama.bat  ← start server + Hermes
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
├── llama.cpp/versions/      ← downloaded llama.cpp releases
└── pyproject.toml
```

# LLM Inference Server Technical Documentation

Welcome to the technical documentation for the **LLM Inference Server**, a tailored, Windows-optimized `llama.cpp` inference environment equipped with a PyQt6 Graphical User Interface (GUI). This project is specifically engineered to run large hybrid dense models (like Qwen3.6-27B) natively on consumer-grade hardware (such as the RTX 4070 Ti SUPER with 16GB VRAM) while providing a seamless OpenAI-compatible API for CLI clients like Hermes.

---

## 1. Real Capabilities of the Program

The application acts as a comprehensive wrapper and manager for local LLM inference, delivering the following core features:

* **PyQt6 GUI Launcher:** A rich graphical interface for managing model selection, network parameters (host/port), and inference flags.
* **Llama.cpp Version Manager:** Automatically fetches release data from the GitHub API, allowing users to download, install, and hot-swap between different `llama-server` binary versions without leaving the UI.
* **Auto-Discovery Engine:** Intelligently locates essential dependencies without manual configuration:
  * Automatically finds existing `llama-server` binaries (via environment variables like `LLAMA_SERVER_PATH`, system `PATH`, or within LM Studio's cache).
  * Auto-discovers the Hermes CLI configuration directory (`~/.hermes/config.yaml`).
* **Hermes CLI Integration:** Automatically registers and updates the server as a local provider (`local-llama` or `local-qwen`) in the Hermes configuration, keeping API ports synchronized.
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
* **KV Cache Quantization (`--cache-type-k q4_0 --cache-type-v q4_0`):** Both Key and Value caches **must** be quantized to `q4_0`. Using higher precision (like `q8_0`) at large contexts will exhaust VRAM and ruin generation speeds.
* **CPU MoE Flag:** Do not use `--n-cpu-moe`. Qwen3.6-27B is a dense model; this flag will have no effect.
* **Reasoning Off (`--reasoning off`):** Required to ensure fast, single-shot inference responses without inner-monologue delays.

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
