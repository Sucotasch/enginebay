# LLM Inference Server

Local llama.cpp inference server with PyQt6 GUI launcher. Runs Qwen3.6-27B (IQ4_XS, 14GB GGUF) on RTX 4070 Ti SUPER (16GB VRAM). Provides OpenAI-compatible API for Hermes CLI client.

## Features

- **PyQt6 GUI launcher** — model selection, presets, llama.cpp version management
- **Console-free launch** — `Launcher.vbs` starts without terminal window
- **Auto-discovery** — finds Hermes config and llama-server binary automatically
- **Version manager** — download, install, and switch between llama.cpp releases from GitHub
- **Health monitor** — real-time server status in the GUI

## Hardware

| Component | Value |
|-----------|-------|
| CPU | Intel Haswell Xeon, 12 cores |
| RAM | 64 GB |
| GPU | RTX 4070 Ti SUPER, 16 GB VRAM |
| CUDA | 13.2, Driver 596.49, CC 8.9 |
| OS | Windows 10 |

## Quick Start

### Option 1: GUI launcher (recommended)

Double-click `Launcher.vbs` — opens the GUI without a terminal window.

### Option 2: One-click server + Hermes

```bat
launch-hermes-llama.bat
```

### Option 3: Server only

```bat
start-llama.bat
```

## Model

| Property | Value |
|----------|-------|
| Model | Qwen3.6-27B (**DENSE**, not MoE) |
| Architecture | Hybrid: Gated DeltaNet + GQA Attention |
| Layers | 64 total (48 DeltaNet + 16 GQA Attention) |
| KV Cache | Only 16 layers need traditional KV cache |
| Quantization | IQ4_XS (~4.5 bits/weight) |
| File size | 14 GB |
| Format | GGUF |
| Context | 96K tokens (98304) — OPTIMAL |

## Performance (verified 2026-06-28)

| Context | KV Cache | Batch | Decode Speed | Prefill | VRAM |
|---------|----------|-------|-------------|---------|------|
| **96K** | **q4_0** | **2048** | **34.1 tok/s** | **58.1 tok/s** | **15937 MB** |
| 64K | q4_0 | 2048 | 34.3 tok/s | 62.6 tok/s | 15507 MB |
| 128K | q4_0 | 512 | 6.3 tok/s | 18.1 tok/s | 15592 MB |

**96K is optimal** — 5x faster decode than 128K, 50% more context than 64K.

## Critical Constraints

- **96K context (`-c 98304`) is optimal.** 128K kills decode speed (6 tok/s vs 34 tok/s).
- **q4_0 KV cache required.** `q8_0` at 64K+ kills performance.
- **`--n-cpu-moe` is useless** — Qwen3.6-27B is dense, not MoE.
- **`--reasoning off`** required for fast single-shot inference.

## Optimal Launch Command

```bash
llama-server \
  -m "G:/Ai/Models/Qwen3.6-27B.i1-IQ4/Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf" \
  -c 98304 -ngl 99 -b 2048 -ub 512 \
  --kv-unified --cache-type-k q4_0 --cache-type-v q4_0 \
  -t 5 --flash-attn on --reasoning off --jinja \
  --temp 1.0 --min-p 0.05 --top-p 0.95 --top-k 64 \
  --host 127.0.0.1 --port 8080
```

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

## Smoke Test

```bash
source .venv/Scripts/activate
python scripts/smoke_test.py
```

## License

MIT
