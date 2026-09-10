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
| `launcher.py` | GUI: model selection, presets, llama.cpp version management, **engine selection**, **VRAM guard + DLL diagnostics**, **Model Library** |
| `launcher_presets.json` | Saved configs: 10 dense (Qwen3.8-27B/Agentic/Gemma4/KVarN/MTP) + "Qwen3.6-35B-A3B (Bee MoE)" + "Gemma-4-26B-A4B (Bee MoE)" (port 8888, `--n-cpu-moe 21/6`) |
| `vram_records.json` | Measured VRAM footprints per model+engine+ctx (auto-written after each model load; feeds the guard AND Library fit dots). Git-ignored, machine-local |
| `scripts/_research_moe/RESEARCH.md` | MoE campaign fact base: sources, benchmarks (server_96k.jsonl etc), VRAM-edge law, measured negatives |
| `configs/inference.env` | Server parameters |
| `scripts/start_llama_cpp.sh` | Alternative launcher (Git Bash) |
| `scripts/smoke_test.py` | Verify server is responding |
| `scripts/update_opencode_models.py` | Refresh OpenCode Free models in `~/.dsh/settings.yaml` (see skill `dsh-providers`) |
| `scripts/engine_diag.py` | CLI: `check <exe>` (missing-DLL report via PE import walk), `vram` (PDH/DXGI free VRAM + per-process holders), `gguf <file>` (GGUF header metadata: arch/ctx/blocks/MoE). Ported from Quartermaster (MIT) |
| `scripts/test_job_tree.py`, `test_gguf_meta.py`, `test_gguf_moe.py`, `test_library_e2e.py` | Live/e2e tests: job-tree orphan kill, GGUF parser, MoE detect, Library dialog |

## Critical constraints

- **96K context (98304) is optimal.** 128K drops decode speed (32.1 vs 35.9 t/s on ik_llama). Do not change `-c` to 131072.
- **Upstream llama.cpp: q4_0 KV cache required.** q8_0 kills speed. Both `--cache-type-k` and `--cache-type-v` must be `q4_0`.
- **BeeLlama engine: use KVarN cache types** — `--cache-type-k kvarn5 --cache-type-v kvarn4 --kv-tail-tokens 1024` (balanced default). Upstream q4_0 also works but forfeits the fork's KV compression.
- **`--n-cpu-moe` is useless** for Qwen3.8-27B — it's a dense model, not MoE.
- **`--reasoning auto` + `--jinja` required on Qwen3.8 (qwen35)** — `--reasoning off` breaks tools requests. (Upstream Qwen3.6-27B and Gemma use `--reasoning off`.)
- **`-np 1` on Qwen3.8 presets** — auto parallel (`n_parallel=-1`) creates 4 slots × 96K KV cache, which tanks VRAM and drops speed to ~3.8 tok/s.
- **Two ports**: 8080 (Qwen3.6/3.8-27B) and 8888 (Gemma 4 26B / Agentic). Do not mix presets.

## MoE models — measured optimal configs (2026-09-10 campaign)

Full research: `scripts/_research_moe/RESEARCH.md` (sources + benchmark logs: `server_96k.jsonl`, `upstream_96k.jsonl`, `qwen_results.txt`, `gemma_edge.txt` etc). All numbers live-measured on this box (RTX 4070 Ti SUPER, i7-5820K, 64 GB RAM, beellama v0.4.5-cuda-13.3), 96K ctx, port 8099 test protocol.

**The VRAM-edge law (most important finding):** fewer CPU-expert layers is NOT faster. Once the GPU part exceeds VRAM, the driver silently spills to WDDM shared memory and tg craters (33 → 6 t/s). The optimum is the largest GPU expert share that still fits with ~600+ MB headroom. The cliff is one layer wide: qw ncmoe20 fits (blade-edge), 18 spills; gm 4 fits (blade-edge), 2 spills.

**Qwen3.6-35B-A3B Q6_K (qwen35moe)** — preset "Qwen3.6-35B-A3B (Bee MoE)": `--n-cpu-moe 21` (of 40 blocks), KV q4_0, `-b 2048 -ub 512 -t 5 -tb 6` → **~29.6 t/s tg / ~420 t/s pp**, 650 MB free. vs dense Qwen3.8-27B's 35.9 t/s — slower but 35B-total/A3B with 26.6 GB Q6_K quality.
**Gemma-4-26B-A4B heretic Q4_K_M (gemma4)** — preset "Gemma-4-26B-A4B (Bee MoE)": `--n-cpu-moe 6` (of 30 blocks), same base → **~47.5 t/s tg / ~1530 t/s pp**, 660 MB free. Faster than every dense 27B we run.

Measured negatives (do not retry blindly):
- **MTP draft on the A3B** (mtp-Q8_0 +1.9 GB): 24-27.5 t/s in ALL tested combos (ncmoe 22/24/26) — never beats pure. The 3B-active MoE leaves no VRAM for the draft and acceptance doesn't cover the cost.
- **KVarN cache on MoE archs**: qw 28.0 vs 33.0 q4_0; gm 34.0 vs 54.3 q4_0. KVarN is a dense-qwen35 optimization — on qwen35moe/gemma4 it costs VRAM AND speed. (Existing kvarn presets for DENSE Qwen3.8-27B stay valid.)
- **ub > 512 at the edge**: ub1024 on qw ncmoe20 → 10.9 t/s (spill). DS-guide's `-b/-ub 4096` advice targets huge-CPU-weight rigs, inverts at our VRAM edge.
- **threads 6**: 25.2/40.1 vs 29.0/47.2 at t5 — the 6th thread starves the GPU driver. Keep `-t 5 -tb 6`.
- **Blade-edge configs are real but not preset material**: qw ncmoe20 (33 t/s, 150 MB free) and gm ncmoe4 (54 t/s, 160 MB free) run faster until any dwm/browser spike pushes them into WDDM spill — on a desktop system they flake (rep runs dropped to 26/44 t/s).
- **Upstream llama.cpp b10712 loses to beellama on both MoE** (qw 26.8 vs 29.5; gm 42.7 vs 47.5) — beellama v0.4.5 is THE MoE engine here.
- **ik_llama 15dddc6 also loses on both MoE** (qw 26.5; gm 24.0-27.1; pp 130-160 t/s vs beellama's 420-1530) and uses MORE VRAM at the same ncmoe. Its fused-MoE/`-t 1` tuning targets big-EPYC+300GB-RAM rigs: `-t 1` on our 6-core dropped tg to 12.1 t/s. `--no-fmoe` crashed the server on qw. Keep ik_llama for its DENSE Qwen3.8 IQ4_KT/KS niche (trellis quants) — not for MoE.

Engine note: beellama's `--n-cpu-moe N` counts from the FIRST layers (help text + logs), upstream counts from the highest — a cross-engine preset is NOT portable without flipping N.

## VRAM guard + DLL diagnostics (launcher)

Ported from Quartermaster (github.com/Quartermaster-Labs/Quartermaster, MIT — attribution kept in `scripts/engine_diag.py`).

**DLL diagnostics** (`scripts/engine_diag.py check <exe>`): walks the PE import table (graph, app-dir→System32→SysWOW64→SystemRoot→PATH, api-ms-*/ext-ms-* skipped) and names missing DLLs with actionable advice (CUDA runtime / NVIDIA driver / VC++ / HIP). Wired into the launcher: checked before launch (advisory Yes/No) and re-run automatically when the server dies before "model loaded" (typical 0xC0000135 silent exit).

**VRAM reading** (`scripts/engine_diag.py vram`): vendor-neutral, no nvidia-smi — PDH `\GPU Adapter Memory(*)\Dedicated Usage` (system-wide usage, grouped/summed per LUID) + classic DXGI `IDXGIFactory::EnumAdapters`/`GetDesc` (total; matched to the PDH adapter by LUID). Free = total − used. Also lists per-process VRAM holders via PDH `\GPU Process Memory(*)` (pid from instance names).

**Guard behavior** (in `launcher.py` `_vram_guard`, reserve = 512 MB — below ~100 MB free llama.cpp offloads to RAM instead of crashing):
- 🟢 free ≥ need + 0.5 GB → silent launch
- 🟡 need ≤ free < need + 0.5 GB → log + status bar, launch
- 🔴 free < need **and need is MEASURED** → modal, EVERY launch (nothing remembered): Launch anyway / Smaller context (one-launch `-c` shrink: weights + KV·ratio ≤ budget; preset untouched) / Who holds VRAM (live holder list; own llama-* servers stop freely, other inference apps stop with warning, browsers/games display-only) / Re-measure (2.5 s driver settle)
- 🔴 free < need **but only estimated** → warn only, launch — an estimate NEVER blocks (first launch of a new model must work)
- ⚪ PDH/DXGI unavailable → launch without check

**Need value source**: `vram_records.json`, keyed `engine|model-basename|c<ctx>`. Real measurement = PDH system-wide delta between pre-launch baseline and post-"model loaded" snapshot (works on engines that don't print VRAM lines, e.g. beellama v0.4.5). Fallback estimate = GGUF file size + 2 GB buffer, always labelled "estimate". Presets are NEVER modified by the guard.

## Update downloader (fixed 2026-09-10)

`_download_paired` in `launcher.py`: streamed 256 KB-chunk fetch with **live MB progress** (`X / Y MB` via `_set_progress_signal`), 30 s stall-timeout per read, 3 retries with backoff, truncation/size checks, errors surfaced to BOTH the log and the `dl_progress_label`. Root cause of the old "hangs forever on Downloading...": GitHub API 403 rate-limit (60 req/hr unauth per IP — reproduced live) or a stalled asset connection left the old `resp.read()` silent for up to 300 s with zero feedback.

**Optional proxy** (user-mandated design — never forced): `"proxy": "http://host:port"` in `launcher_config.json` → `urllib.request.build_opener(ProxyHandler({...}))` for API + asset fetches; empty/absent = direct. Works for any proxy system (HTTP CONNECT).

**Release layout (v3.0.0+)**: two assets — the main zip (code, presets, scripts; engines download on demand via the GUI) and a separate `ik_llama-*.zip` with the prebuilt `ik_llama.cpp/versions/15dddc6/` (source-built, no upstream binaries exist — most users don't need it). Machine-local files (`launcher_config.json`, `vram_records.json`, `launcher_history.json`, logs) are never packaged.

## Model Library (launcher "Library" button)

`ModelLibraryDialog` in `launcher.py`: recursive `*.gguf` discovery under a **user-chosen root** (nothing hardcoded — `cfg["model_root"]` in `launcher_config.json`, gitignored; first open asks via native directory dialog, defaults checked: `~/Ai/Models`, `~/models`, `~/.cache/lm-studio/models`). Search-by-substring, per-row fit dot, `[projector]`/`[draft]` tags from filename, arch/size columns.

- **Fit-dot math (fixed 2026-09-10)**: need = measured record (max across engines/ctx, case-insensitive basename match) else size + **calibrated overhead** (`_dense_overhead_mb`: avg of measured footprint−file_size across `vram_records.json`, fallback flat `VRAM_BUF_EST_MB`). Compared against a **ceiling = total−600 MB**, NOT live free: WDDM evicts dwm/chrome when llama.cpp allocates, so live-free dots called 14 GB models red that launch fine (measured: IQ4_KT 15204 MB, Magistry 15237 MB, both fit 16063−600). `library_entry_need_mb` takes the path as an explicit arg — the library cache stores it as the DICT KEY (`entry.get("path")` is empty → records never matched; this bug made every row use the estimate branch).
- **Cache contract** (chosen by user over full-scan/lazy variants): `cfg["library"]` = {path: {size, mtime, meta}}. Opening the dialog = cheap stat validation (new files appear as unindexed, vanished files drop). "Rescan" = full rewalk + GGUF header read into cache. `_ensure_scan` counts indexed entries **under the current root only** — leftover foreign-dir entries (e.g. from an e2e test) must never suppress the first scan.
- **GGUF header reader** `diag.gguf_metadata(path)`: bounded stream (≤8 MB read, early exit once arch+name+ctx known — tokenizer arrays at header tail skipped; ~1 ms warm vs 2.3 s naive full parse on HDD). Returns arch/name/size_label/ctx/blocks/quant/file_type. `general.file_type` is often ABSENT (ik-quant builds) — quant falls back to filename.
- **MoE models (Qwen3.6-35B-A3B, Gemma-4-26B-A4B)**: detected via `size_label` regex `NNB-AxB` (total-active params) or `expert_count` KV — no tensor parsing. In the list: ⚫ dot + `[MoE offload]` tag instead of a false 🔴 (a 26.6 GB MoE file does NOT mean "doesn't fit" — it runs with `--n-cpu-moe`/`-ot exps=CPU` keeping only attention+KV+shared on GPU). At launch (`_vram_guard`), a MoE model without an offload flag in params gets a loud log+statusbar warning (never blocks): experts would spill to shared memory (WDDM) and crawl.
- **Draft tagging rule**: a companion draft is a separate `mtp-*` file (e.g. `mtp-Qwen_Qwen3.6-35B-A3B-Q8_0.gguf`); a main model merely NAMED `*-MTP` (Qwen3.8-27B MTP-tuned, 14 GB) is NOT a draft — never tag by suffix.
- **Tests must not touch `launcher_config.json`**: the dialog persists via `save_config(self.cfg)` — `test_library_e2e.py` monkey-patches `launcher.CONFIG_FILE` to a scratch file (an earlier version leaked `_libtest_tmp` entries into the real config and suppressed the user's first scan). Tests: `test_gguf_meta.py`, `test_gguf_moe.py`, `test_library_e2e.py` (isolated), `test_lib_fit_dots.py` (dots vs records), `test_launcher_smoke.py` (offscreen: 0 red rows, MoE markers, download-path asserts).

## Job Object — children die with the parent (PORTABLE RECIPE)

`engine_diag.setup_job_tree()` (called once in `launcher.py:main()` before QApplication): the process assigns ITSELF to a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | BREAKAWAY_OK` and leaks the handle. Every spawned child inherits the job → when the parent dies BY ANY MEANS (graceful close, crash, kill -9, TerminateProcess), the OS reaps the whole tree. Orphaned llama-server.exe holding VRAM becomes physically impossible.

- **ctypes gotcha (the reason a naive port fails)**: kernel32 function pointers default to `restype=c_int` and TRUNCATE 64-bit handles — `GetCurrentProcess()`'s pseudo-handle arrives as `0x00000000FFFFFFFF` → `ERROR_INVALID_HANDLE (6)`. You MUST set `restype = c_void_p` on `CreateJobObjectW` and `argtypes = [c_void_p, c_void_p]` on `AssignProcessToJobObject` (assign BEFORE SetInformationJobObject — order matters when the target is self).
- Nested jobs are fine on Win8+ (task scheduler / CI parents).
- `BREAKAWAY_OK` keeps self-update relaunch legal (`CREATE_BREAKAWAY_FROM_JOB`), costs nothing.
- On failure return False and keep manual taskkill fallbacks — the job is a safety net, never a requirement.
- Verified live (`scripts/test_job_tree.py`): parent hard-kills itself with TerminateProcess → child dies with it, zero orphans.
- **Reusable in other projects** (e.g. qwengate-deepseek when it spawns long-lived children): copy `setup_job_tree()` from `scripts/engine_diag.py` — it is dependency-free (ctypes only).

## Lessons from Quartermaster (for future work)

- **Admission vs shed ceilings must differ**: admission (can I load?) charges a reserve; a shed/watchdog ceiling (should I evict a RUNNING model?) must NOT — otherwise a model sized within the reserve of the budget unload/reload-loops forever (they hit this exact bug). Relevant the day EngineBay runs two servers (8080+8888) against one GPU.
- **Quant token parsing belongs in ONE place**: model-name → (base, quant) parsing, written as stable quant FAMILIES (IQ*, K*, TQ, MXFP4, NVFP4…) rather than an enumeration, or every new ggml type breaks four consumers at once (their `quant.go` story). Relevant when the model fleet grows beyond Qwen/Gemma.
- **GGUF parser + auto load-planner** (`autogen/gguf.go` + `generate_sizing.go`, ~1.5k lines Go): reads arch/blocks/quant-from-tensors from the GGUF header and computes -ngl/-c/--n-cpu-moe per architecture (dense vs MoE vs recurrent, KV per-token per-arch, MTP-draft overhead). DELIBERATELY DEFERRED, not rejected: the measured-record approach (PDH before/after) is more accurate for already-launched presets, but the parser still covers (a) FIRST launch of a brand-new model with no record, (b) real quant read from tensor types, not filename, (c) cross-arch context tiering when new models arrive. Implement when adding vLLM/Gemma models or when preset hand-tuning hurts.

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
