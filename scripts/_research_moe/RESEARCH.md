# MoE Offload Research — Qwen3.6-35B-A3B & Gemma-4-26B-A4B on RTX 4070 Ti SUPER 16GB

Collected 2026-09-10 via Hermes web-tools bridge. All claims cite their source.
Target hardware: RTX 4070 Ti SUPER 16GB VRAM, i7-5820K (6C/12T, DDR4 quad-channel),
64GB RAM, CUDA 13.x, engines: beellama v0.4.5-cuda-13.3 / llama.cpp upstream / ik_llama manual build.

## Sources (full texts in this dir)

| ID | Source | What it gave |
|----|--------|--------------|
| DS | gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0 ("Guide to optimizing inference performance of large MoE models across CPU+GPU using llama.cpp and its derivatives", by Doctor-Shotgun with Geechan's ik_llama section) — `ds_guide.txt` | Core MoE offload mechanics, -ot syntax, n-cpu-moe semantics, batch tuning, ik_llama flags |
| HF | huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide (HF mirror of the same guide) — `hf_blog_guide.txt` | Same content, published Jan 30 2026; ik_llama threshold formula |
| UB | gist.github.com/ubergarm/0f9663fd56fc181a00ec9f634635eb38 ("Qwen3 235B and 30B MoE Quant Benchmarking Roundup") — `ubergarm_bench.txt` | ik_llama -fmoe flag usage on Qwen3-30B-A3B; llama-sweep-bench methodology; layer importance data |
| QML | quantml.org/guides/gemma-4-gguf ("Gemma 4 GGUF — stack overview") — read via `w_quantml.json` | Gemma-4-26B-A4B architecture numbers (128 experts, 8 active + 1 shared, 3.8B active / 25.2B total), UD-Q4_K_XL ≈ 17.1 GB, KV q4_0 = 75% savings vs f16, --no-mmap for snapshot stability, L40S 50-60 t/s full-VRAM baseline |
| G3060 | github.com/julien9679/Gemma-4-26B-A4B-on-RTX-3060-12GB — read via `w_gemma3060.json` | 12GB-VRAM recipe: IQ2_M 9.97 GB full-fit + turboquant 3-bit KV (fork-only), -c 32768, ~40-55 t/s, 10.5/12 GB used. NOT our path (we have 16GB and no turboquant fork), but proves ctx 32K fits when weights+KV ≤ VRAM |
| ALT1 | aliteq.com/llama-cpp-moe-offloading-run-bigger-models-less-vram-2026 (S1 of w_moe1) | Background only |
| SMELT | smeltcore.com/recipes/qwen3-35b-moe-on-rtx-4070-80-tok-s-local-llm-guide/ (S1 of w_moe2) | RTX 4070 12GB + Qwen3-30B-A3B Q4_K_M via CPU-offload: ~12GB VRAM claim (unvalidated detail) |

## Extracted facts (each tied to source)

### MoE offload mechanics (DS)
1. MoE = Attention + Dense FFN (optional) + Shared expert FFN (optional) + Routed expert FFN. First three are "always active" per token → belong on GPU. Routed experts are the bulk of size but activate fractionally → offload to CPU.
2. Basic config: `-ngl 999 --cpu-moe` (or `-ot "exps=CPU"`). All attention+dense+shared+KV on GPU, routed experts on CPU.
3. Fill leftover VRAM with routed expert layers: `-ot "blk\.([0-9]|[1-2][0-9]|30)\.=CUDA0,exps=CPU"` = layers 0-30 fully on GPU (incl. their experts), rest experts → CPU. First pattern takes priority, comma-separated list.
4. `--n-cpu-moe N` is the built-in equivalent: N layers' routed experts to CPU, **counting from the HIGHEST layer number**. Off-by-few vs explicit -ot when dense-FFN layers sit at the model start.
5. llama.cpp mainline: disable auto-fit with `-fit off` while tuning (clear OOM signal).
6. Prompt processing (PP) with CPU+GPU split: llama.cpp "op offload" copies CPU weights to GPU for a batch; default trigger = 32 tokens (mainline) / `32 * total_experts / active_experts` (ik_llama). Qwen3.6-35B-A3B: 128/8 → threshold ≈ 512 tokens.
7. PP batch tuning: `-b` (logical, default 2048) and `-ub` (physical, default 512); ub ≤ b; **DS recommends -b 4096 -ub 4096** for CPU+GPU MoE. Higher ub = more VRAM for compute buffer → may need fewer GPU expert layers.
8. `GGML_OP_OFFLOAD_MIN_BATCH` env (mainline) / `-cuda offload-batch-size=32` (ik_llama) tune the trigger threshold.
9. ik_llama-specific: `--merge-qkv` (merge Q/K/V attention tensors, no penalty when attention on GPU, needs same quant type on Q/K/V), `-gr` graph reuse (small gain), `-smgs` split-mode graph scheduling (may crash with tensor overrides), `-mla 3`/`-amb 512` (DeepSeek-arch only, not Qwen/Gemma), `-sm graph` (multi-GPU only — not us).
10. `llama-sweep-bench` (ik_llama only) = THE benchmark tool, repeatable, substitute for llama-server. **Use it for our tests.**
11. NUMA advice is Linux-only (numactl etc.) — irrelevant on our single-socket Windows box.

### ik_llama -fmoe flag (UB)
12. ubergarm launches Qwen3-30B-A3B on ik_llama with `-fmoe` + `-fa -ctk f16 -ctv f16 -c 262144 -ngl 99 -t 1`. -fmoe is ik_llama's fused-MoE optimization.

### Gemma-4-26B-A4B specifics (QML)
13. Architecture: 128 experts, 8 active + 1 shared, 3.8B active of 25.2B total. Decode speed ∝ active params.
14. Weights sizes: UD-Q4_K_XL ≈ 17.1 GB (too big for full-fit on 16GB with KV; that's why offload). mmproj-F32 ≈ 2.0 GB (vision, optional).
15. KV cache f16 ≈ 3 GB at moderate ctx on 48GB card (scaled numbers); **q4_0 KV = 75% savings vs f16** (QML "Aggressive KV cache" lever).
16. --no-mmap: "treated as required, not optional" in QML's snapshot setup (restore semantics), DS uses it too for CPUmaxx; on Windows with 64GB RAM it also avoids double-buffering during load. Cost: slower startup, RAM spike while loading.
17. Concurrency levers: --parallel N splits ctx. We are np 1 (single user).

### Gemma-4 on 12GB (G3060)
18. Full-fit alternative: IQ2_M 9.97 GB + 3-bit KV (turboquant fork only) + -c 32768 = 10.5/12 GB, 40-55 t/s. Our Q4_K_M heretic file is 15.7 GB → cannot full-fit; must offload experts. (G3060 chose smaller quant instead of offload; both paths valid, offload keeps Q4 quality.)

### Qwen3.6-35B-A3B specifics (our GGUF header + UB/SMELT cross-ref)
19. Our file: Q6_K, 26.6 GB, arch qwen35moe, ctx 262144, 40 blocks, 733 tensors, 128 experts/8 active (A3B ⇒ ~3.3B active), mtp draft file available (mtp-Qwen_Qwen3.6-35B-A3B-Q8_0.gguf, 1.9 GB).
20. SMELT (RTX 4070 12GB, Qwen3-30B-A3B Q4_K_M 18.6 GB weights): "experts live in RAM, ~12 GB VRAM" — i.e. all routed experts off CPU-side, attention+shared on GPU. 80 t/s claimed in title (not independently verified).
21. MTP draft: llama.cpp speculative decoding with a draft model = -md <draft.gguf> --draft-max N. For Qwen3.6-35B-A3B the mtp-*.gguf is the official draft. Gains depend on acceptance rate; Qwen3-30B-A3B MTP acceptance reported high (~70-80% in Qwen3 Next announcements — TODO verify).

## Derived plan for OUR hardware (not yet benchmarked — to test)

Common base: `-ngl 999 --cpu-moe` (or -ot "exps=CPU") + `-fa on` + KV q4_0 both (AGENTS constraint: q4_0 KV on upstream; beellama → kvarn5/kvarn4 if supported for this arch — check beellama logs) + `-np 1` + `--jinja` + reasoning off (Gemma) / auto (Qwen3.6 per AGENTS: upstream Qwen3.6-27B uses off; 3.6-35B-A3B to be tested) + `-t 5 -tb 6` (our CPU) — DS says threads for CPU experts matter; our 6C/12T needs testing (t 5 leaves 1 for OS/GPU driver).

Qwen3.6-35B-A3B Q6_K (26.6 GB): weights total 26.6 GB, experts ≈ (26.6 × ~90% routed) — plan: attention+shared+first-K-layers' experts on GPU. Start `--n-cpu-moe 24` (of 40) ≈ 60% experts CPU; tune by VRAM. -b 2048 -ub 2048 (DS recommends 4096 but our i7-5820K DDR4 bandwidth is the constraint; sweep 1024/2048/4096).
Then: -md mtp draft test on top of the winner.

Gemma-4-26B-A4B Q4_K_M heretic (15.7 GB): 30 blocks, 128 experts/8+1 shared. `--n-cpu-moe 12` start (of 30) ≈ 40% experts CPU; tune. KV q4_0. Reasoning off. mmproj optional — skip for now (chat only).

Benchmark protocol (per DS #10): llama-sweep-bench where available (ik_llama build), else llama-server + /health + timing a fixed prompt via API with stream:false and reading "tokens per second" from server logs (eval metrics in response usage). Variants per model: n-cpu-moe sweep {baseline-999-all-CPU, n/2, n/4, 0-cpu-moe-if-fits} × ub {1024, 2048, 4096} — 9-12 runs, 3-5 min each on our CPU.

## Open questions to resolve in benchmarks
- beellama: does it accept --n-cpu-moe (it's upstream-based)? kvarn cache for qwen35moe/gemma4 arch? (log check)
- ik_llama build: qwen35moe arch support? (build is old-ish; check)
- Does -fmoe exist in current ik_llama release? (UB used it)
- MTP draft acceptance on Qwen3.6-35B-A3B: real speedup?

---

## BENCHMARK RESULTS (2026-09-10, live, port 8099, beellama v0.4.5-cuda-13.3)

Hardware as stated in header. Server protocol: load → /health → PDH VRAM snapshot →
tg (220-tok story gen) → pp (4.2k-tok summary prompt). Full log: `server_96k.jsonl`,
`upstream_96k.jsonl`, `qwen_results.txt`, `qwen_edge.txt`, `gemma_results.txt`,
`gemma_edge.txt`, `final_ctx_results.txt`.

### llama-bench screening (tiny ctx, pp512/tg128)
Qwen3.6-35B-A3B Q6_K: ncmoe 40→20.6 t/s, **20→35.4**, 10→6.3 (WDDM spill!), 5→4.9 (spill).
Gemma-4-26B Q4_K_M: ncmoe 30→17.0, 15→32.4, 8→48.7, **4→68.9**, 2→21.5 (spill), 0→13.9 (spill).
KEY INSIGHT: fewer CPU experts ≠ faster — once GPU part exceeds VRAM, the driver
silently spills to shared memory and t/s craters (6-9 t/s). The optimum is the LARGEST
GPU expert share that still fits, not the smallest CPU share.

### beellama server @ 96K ctx (the real preset budget)
Qwen (KV q4_0, -b 2048 -ub 512 -t 5 -tb 6):
| ncmoe | tg t/s (runs) | pp t/s | free MB |
|---|---|---|---|
| 20 | 33.0 / 26.1 | 432/413 | 83-150 ← blade edge, flaky |
| **21** | 26.0 / 30.2 / 30.8 | 418-421 | **~650 stable** |
| 22 | 28.4 / 28.8 / 29.3 | 396-398 | ~1280 |
| 24 | 28.6 | 352 | 2535 |
| 18 | 8.66 (kvarn) — SPILL | 34.6 | 102 |
MTP draft (+1.9 GB, ncmoe22/24/26): 24.0-27.5 t/s — NEVER beats pure; rejected.
KVarN (kvarn5/kvarn4): qw ncmoe20→28.0 (vs 33.0 q4_0); gm ncmoe4→34.0 (vs 54.3) — REJECTED for MoE.
ub1024 (ncmoe20): 10.9 t/s, free=91 — DS's big-batch advice inverts at the VRAM edge; keep ub 512.

Gemma (same base):
| ncmoe | tg t/s (runs) | pp t/s | free MB |
|---|---|---|---|
| 4 | 54.3 / 43.8 | 1826/1648 | 158-162 ← blade edge |
| 5 | 48.9 | 1635 | 238 |
| **6** | 48.2 / 46.6 / 47.2* | 1527/1511 | **~700 stable** |
| 8 | 40.3 | 1232 | 1635 |
| 10 | 39.4 | 1028 | 2536 |
(*t6 run 40.1 — threads 6 worse; keep t 5)

### Engine comparison (winner configs)
| config | beellama | upstream b10712 |
|---|---|---|
| Qwen ncmoe21 | ~29.5 t/s | 26.8 t/s |
| Gemma ncmoe6 | ~47 t/s | 42.7 t/s |
beellama v0.4.5 wins both → beellama is the engine for MoE presets.

### ik_llama 15dddc6 (manual build, MSVC 19.44, 2026-08-30) — `ikllama_96k.jsonl`
| config | tg t/s | pp t/s | free MB |
|---|---|---|---|
| qw ncmoe21 | 26.5 | 159 | 614 |
| qw ncmoe20 | 26.6 | 154 | 110 ← less headroom than beellama's same config |
| gm ncmoe6 | 24.0 | 131 | 110 |
| gm ncmoe4 | 27.1 | 138 | 127 |
| qw ncmoe21 + `-t 1` (ubergarm's advice) | 12.1 | 167 | — EPYC-rig advice, harmful on 6-core |
| qw ncmoe21 + `--no-fmoe` | server died rc=1 — flag incompatible with this build/arch combo |

ik_llama LOSES on both MoE models vs beellama (qw 26.5 vs 29.5; gm 24.0-27.1 vs 47.5) and
uses MORE VRAM for the same ncmoe (holder 14962 vs 14930; the -fmoe path adds buffers).
pp is dramatically worse (130-160 vs 420-1530 t/s) — the build's CUDA paths for the new
qwen35moe/gemma4 archs lag beellama's. Its fused-MoE optimizations target DeepSeek-class
rigs (big EPYC + 300 GB RAM), not ours. beellama v0.4.5 = THE MoE engine on this box,
unanimously across all three engines tested.

### FINAL WINNERS (measured, reproducible)
- **Qwen3.6-35B-A3B (agentic MoE)**: `--n-cpu-moe 21` + q4_0 KV + `-b 2048 -ub 512 -t 5 -tb 6` → ~29-31 t/s tg, ~420 t/s pp, 650 MB VRAM headroom, 96K ctx.
- **Gemma-4-26B-A4B (heretic)**: `--n-cpu-moe 6` + q4_0 KV + same base → ~47 t/s tg, ~1500 t/s pp, 700 MB headroom, 96K ctx.
- MTP draft: rejected (slower in all tested combos — 3B-active MoE leaves no VRAM for the draft and acceptance doesn't cover the cost).
- KVarN cache: rejected for MoE archs (it's a dense-qwen35 optimization; on qwen35moe/gemma4 it costs both VRAM and speed).
- Blade-edge configs (qw20 33 t/s, gm4 54 t/s) exist but live with <160 MB free — one dwm spike from WDDM spill; NOT preset material on a desktop system.

### Notes on sources vs measurements
- DS guide's `-b 4096 -ub 4096` recommendation: inverted at our VRAM edge (ub1024 already spilled) — the guide targets 300GB-CPU-weight rigs, not ours.
- DS `--n-cpu-moe counts from highest layers` (upstream): beellama counts from FIRST layers (verified in help text + probe logs: n_slots/n_gpu_layers behaviour identical, speed cliff symmetric).
- QML's q4_0-KV-75%-savings: confirmed directionally (our 96K q4_0 KV fits where f16 wouldn't).
- SMELT's "80 t/s on RTX 4070 Qwen3-30B-A3B Q4_K_M": their file is 18.6 GB vs our Q6_K 26.55 GB — different quant, not comparable; our Q6_K Qwen result (29.5) vs their Q4 (80 t/s claimed) is quant+claim, unmeasured by us.
