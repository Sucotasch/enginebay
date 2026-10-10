# Consolidated repo review — 6 repositories vs EngineBay (2026-09-10)

Scope: what to integrate, what to borrow, what to skip. Evidence-first: every claim
below was verified either in local source/binaries on this machine or via the
authenticated GitHub API. Raw material: `scripts/_research_repos/`.

Machine context: Windows, RTX 4070 Ti SUPER 16 GB, i7-5820K 6C/12T, 64 GB RAM,
models on `G:/Ai`. Current MoE winners: Qwen3.6-35B-A3B Q6_K `--n-cpu-moe 21` →
29.6 t/s tg / ~420 pp; Gemma-4-26B-A4B Q4_K_M `--n-cpu-moe 6` → 47.5 t/s tg / ~1530 pp.

## 0. The decisive local finding — the built-in router already does what llama-swap does

Verified in source AND live on this box (not from the web):

| Capability | upstream b10894 | beellama v0.4.5/v0.4.7 | ik_llama 15dddc6 |
|---|---|---|---|
| `--models-dir` (auto-discover) | yes | yes | **no** |
| `--models-preset` (INI presets) | yes | yes | **no** |
| `--models-max N` (default 4, 0 = unlimited) | yes | yes | **no** |
| `--models-autoload` / `--no-models-autoload` | yes | yes | **no** |
| `--sleep-idle-seconds` (unload an idle model) | yes | yes | **no** |
| `/models/load`, `/models/unload`, `/models/reload`, `/models/sse` | — | yes | **no** |
| LRU eviction + request queueing at `models_max` | yes | yes | no |
| preset `dedup-cache-models` (hide HF-cache duplicates) | yes | yes | no |
| preset-only `load-on-startup`, `stop-timeout` | — | yes | no |

Evidence:
- Flags confirmed by running `--help` on each local `llama-server.exe`:
  `llama.cpp/versions/b10894-cuda-13.3/`, `beellama.cpp/versions/preview-v0.4.5-cuda-13.3/`,
  `beellama.cpp/versions/preview-v0.4.7-cuda-13.3/`, `ik_llama.cpp/versions/15dddc6/`.
  ik_llama printed 484 help lines with **zero** router/idle flags.
- Router mode is compiled in on Windows: the server logs
  `the following feature(s) are enabled: router mode` and
  `starting server in router mode. models will be automatically loaded on-demand`.
  (`common/subproc.cpp` gates it on `LLAMA_SUBPROCESS`; the release build has it.)
- INI schema: `beellama.cpp/docs/preset.md` — `[*]` shared section, one section per
  model named as the router model id, option names WITHOUT leading dashes
  (`ctx-size`, `n-cpu-moe`, `cache-type-k`), CLI args override preset values,
  remote presets via `-hf user/preset-repo`, never put `hf-token` in a preset.
- LRU/eviction/queue implementation: `beellama.cpp/tools/server/server-models.cpp`
  `unload_lru()` (L956), queue-at-limit (L78-113), `ensure_model_ready()` (L1389,
  evicts an idle LRU victim to make room), `last_used` bumped on POST (L1521),
  `stop-timeout` with force-kill (L1129). Upstream master has the same core:
  `unload_lru()` at `tools/server/server-models.cpp:1107`,
  `models_max limit reached, removing LRU` at L1121, `ensure_model_ready()` at L1488,
  and `dedup-cache-models` at L756-787 (verified via the GitHub API against master).
  So LRU swapping and dedup are NOT fork-only — beellama's addition is the
  `/models/*` management API plus the two preset-only keys.
- Sleep semantics (the mechanism behind `--sleep-idle-seconds`):
  `beellama.cpp/tools/server/server-context.cpp:1114-1130` — entering the sleeping
  state calls `destroy()` (model unloaded, VRAM released) and exiting it calls
  `load_model()` again; the queue loop arms it with
  `queue_tasks.start_loop(params.sleep_idle_seconds * 1000)` (L4785) and a request
  wakes the server (`server-queue.cpp:117-127`, "requesting to stop sleeping").
  This is an idle-VRAM-release mechanism built into the server, not a wrapper feature.
- `/v1/models` shape: `{data: [{id, aliases, tags, object, owned_by, created,
  status: {value, args, preset, failed, exit_code}, source, can_remove,
  architecture}]}` (L1983-2054); router `/props` returns
  `{role: "router", max_instances, models_autoload}` (L1893-1915).

**Live proof of automatic swapping** (`scripts/_research_moe/test_router_swap.py`,
router with `--models-max 1` and an INI carrying our real MoE flags):

```
starting server in router mode. models will be automatically loaded on-demand
[11919] model loaded
unload_lru: models_max limit reached, removing LRU name=qwen-moe
unload: stopping model instance name=qwen-moe
ensure_model: model name=gemma-moe is not loaded, loading...
```

Loading `gemma-moe` when `qwen-moe` held the single slot evicted the running model
and loaded the new one — i.e. the built-in router performs llama-swap's core job
(on-demand swap with a concurrency cap) with **no Go binary and no extra process**.
Caveat observed: the router also published an unrelated
`unsloth/gemma-4-E2B-it-GGUF:Q4_K_XL` entry auto-discovered from the local HF cache
(beellama's `dedup-cache-models` preset key exists to hide such duplicates).
Caveat 2: with `--models-max 1` and a 16 GB card, only one model can be resident at
a time — which is exactly the honest fit for this GPU.

## 1. Verdict table

| # | Repo | What it really is | Verdict | Effort |
|---|---|---|---|---|
| 1 | `mostlygeek/llama-swap` | Go swap-router/proxy in front of any OpenAI/Anthropic server; 5912★, MIT | **BORROW-IDEAS** (config/UX contracts); not needed as a dependency | S–M |
| 2 | `thecodacus/local-ai-configs` | 10★ reference stack: llama-swap → llama.cpp router + `models.ini` presets | **BORROW-IDEAS** (best single reference for our target architecture) | S |
| 3 | `GenerelSchwerz/llama.cpp` (`moe-cache` branch) | llama.cpp fork adding a **budgeted CUDA expert cache**: hot experts resident in VRAM, cold in host RAM | **SUPERSEDED by upstream** (PR #29887, merged 2026-10-07) — test upstream first, fork = fallback only | **S** (was M–L) |
| 4 | `thecodacus/llama.cpp` (`perf` branch) | llama.cpp fork with its own MoE expert cache + host-register/prefetch env vars | **BORROW-IDEAS / MONITOR** — issues disabled, no releases; key env var already upstream | S to test, M to adopt |
| 5 | `Edge0-AI/Edge0` | Apple-MLX-first streaming-MoE framework + model release; Windows layer is a thin llama.cpp fork | **MONITOR** | S (ideas) |
| 6 | `Niko1221/Strata` | Standalone C++/CUDA engine for one model family (Qwen3.8-Flash-Next), 20k★ | **BORROW-IDEAS** (supervisor/API contracts) — engine itself **SKIP** | S–M |

## 2. GenerelSchwerz/llama.cpp — the MoE expert cache (highest-value finding)

**§2.0 UPDATE 2026-10-10 (web check before building — this changes the recommendation).**
The expert cache **is now in upstream llama.cpp**; the fork build is NOT needed:

- **PR #29887** *"llama: add a GPU cache for MoE experts kept in host memory"* by
  `am17an` (a ggml maintainer), **merged 2026-10-07** — an explicit **port of the
  qvac-fabric MoE cache** (`tetherto/qvac-fabric-llm.cpp`, Tether's edge-inference
  project, 133★). Mechanism: LRU cache of experts — MUL_MAT_ID ops on host experts
  run on the GPU, only cache misses get uploaded; **small batches only (≤ 32
  tokens**, i.e. the decode path — large batches keep the regular offload path);
  layers with different expert layouts get separate banks; guidance: **~10% of the
  total expert size is a good `--moe-cache-mib` value**.
- **PR #30112** *multi-GPU support*, merged 2026-10-08 (same author).
- **Nightly `b11541` (published 2026-10-10, today) contains it** — verified via the
  GitHub API against the tag: `src/llama-moe-cache.cpp` (25 546 bytes) and
  `--moe-cache-mib` in `common/arg.cpp`. Windows CUDA asset naming is unchanged
  (`llama-b11541-bin-win-cuda-13.4-x64.zip` + `cudart-…`) and our launcher's
  classifier (`_classify_llama_asset` / `_extract_cuda_version`) parses it fine —
  Check Updates → download is the whole integration cost.
- **Upstream's own benchmark** (PR #29887, Qwen3.8-Flash-Next Q4_0, 93.7 GiB model
  with 65.4 GiB of experts, EPYC 7742, SPEED-Bench, no MTP):

  | GPU | config | experts in VRAM | tg t/s | speed-up | hit rate |
  |---|---|---|---:|---:|---:|
  | RTX 4090 | master `--fit` | 16.5 + 0.0 GB | 25.0 | 1.00× | — |
  | RTX 4090 | `--fit --moe-cache-mib 6544` | 10.4 + 6.4 GB | 39.4 | **1.57×** | 72% |
  | RTX 4090 | `-cmoe --moe-cache-mib 11000` | 0 + 10.7 GB | 40.7 | **1.62×** | 77% |
  | RTX 5090 | `-cmoe --moe-cache-mib 19000` | 0 + 18.6 GB | 67.8 | **2.20×** | 89% |

- **Independent field report** (dotnetperls.com, 2026-10-08): RTX 3060,
  **gemma-4-26B q4_0: 48 → 68 tok/s (+29%) with `--moe-cache-mib 7200`**; tips that
  match our preset discipline — the cache needs free VRAM (quantize KV, we already
  run q4_0 KV) and disabling MTP helped (consistent with our MTP-on-A3B negative).
- **No regressions reported so far** — the feature is 3 days old; the issue tracker
  shows only follow-ups: per-device comma values (#30205, #30163), GPU-resident LRU
  as a future improvement (#29949), pipeline parallelism (#29963). An earlier
  attempt of the same idea existed as #24524 (closed).
- **FreeToken** (dedicated MoE serving runtime, Linux/CUDA-focused) measured at
  **parity** with the moe-cache approach on an RTX 5070 Ti 16 GB by the fork's own
  wiki (short: 150.9 vs 148.6 t/s; filled 64 K: 117.9 vs 121.4) — so there is no
  "better engine to switch to", the llama.cpp path is already at the frontier, and
  FreeToken does not serve our Windows box anyway.
- **beellama does NOT have the cache** (Anbeeld/beellama.cpp last pushed
  2026-10-07, before the merge) — so the A/B must run on the upstream engine. Our
  earlier "upstream loses to beellama on MoE by ~10%" (b10712 era) is dwarfed by
  the cache's expected +29…+77 %.

**Revised verdict:** the fork is demoted from "build it" to "reference material".
Its remaining extras (early-router, host-pinning, generic hybrid execution,
profiling) are exactly the parts with open crashes (#151, #112) and were NOT
taken upstream. Test upstream `--moe-cache-mib` first — effort **S** instead of
M–L, zero new code (the launcher already downloads nightly builds).

**What this means for our two MoE presets** (16 GB VRAM, 64 GB RAM): raise
`--n-cpu-moe` to free VRAM and hand it to the cache — e.g. Qwen3.6-35B-A3B from
`ncmoe 21` (29.6 t/s, 650 MB free) to `ncmoe 40` + `--moe-cache-mib ~9000`
(modeled on the PR's best `-cmoe + 10.7 GB cache` row), Gemma from `ncmoe 6` to
`ncmoe 30` + `--moe-cache-mib ~9500`. Decode is the target path (batches ≤ 32
tokens); prefill keeps the regular offload path.

**The original fork assessment follows for reference.**

**What it is** (README + `docs/fork-features.md` + wiki): a maintained llama.cpp fork
whose `moe-cache` branch is the **default branch**, adding an opt-in, CUDA-only
**expert cache**. Frequently routed experts of each layer stay resident in VRAM while
cold experts remain host-backed. Same server, same API. No flag = stock behaviour.

Flags: `--moe-expert-cache-size N` (slots per cached expert tensor per owning CUDA
backend), `--moe-expert-cache-mib MiB[,MiB...]` (byte budget per device),
`--moe-expert-cache-layers N[,N-M]`, `--moe-expert-cache-host-pinned-mb N`
(0 = try full pinning, then auto-register whatever groups CUDA accepts),
`--moe-early-router` (predict next layer's experts, lookahead 1), `--experimental-logs`.
Env aliases `LLAMA_ARG_MOE_EXPERT_CACHE_*`. Eviction is decaying-frequency with LRU
tie-break (`GGML_CUDA_MOE_FREQUENCY=0` forces plain LRU).

**Measured claims on hardware in our class** — RTX 5070 Ti 16 GB, one request at a
time, stock vs fork, stock peak VRAM *higher* in every pair
(`scripts/_research_repos/schwerz-wiki/Short-Prompt-MoE-Comparison-2026-09-26.md`):

| Model | stock tg | fork tg | stock peak VRAM | fork peak VRAM |
|---|---:|---:|---:|---:|
| Qwen3.6 35B Q4_K_M | 64.08 | **146.77** | 11 751 MiB | 11 443 MiB |
| Qwen3.6 28B REAP Q4_K_M | 64.68 | **152.95** | 9 923 | 9 687 |
| Gemma 4 Q4_0 | 56.09 | **151.24** | 9 303 | 9 081 |
| Nemotron 3.5 Lightning | 55.74 | **108.60** | 14 147 | 13 469 |

Geometry: `-c 4096 -b 2048 -ub 512 -np 1 -n 1024 --load-mode none -fa on -ctk f16
-ctv f16 -kvo --fit off --jinja --reasoning off -ngl all --cache-ram 0`, fork with
`--moe-expert-cache-size SLOTS --moe-early-router --ple-prefetch --backend-sampling
--decode-overlap --decode-boundary-overlap --phase-aware-workspace`, plus inherited
`--n-cpu-moe 20` for Qwen 35B / Gemma. Long-prompt run: prefill **442 t/s** at 64 K.
The fork's own showcase table also lists Qwen3.6 35B 42.8 → 111.6 tok/s and Gemma 4
34.2 → 102.2 (RTX 5070 Ti 16 GB, ~62 GiB RAM), explicitly labelled *historical
results, not measurements of the current head*.

**Windows reality (matters most for us):** the fork ships a dedicated wiki page
*Windows WDDM setup* — WDDM refuses to pin a whole expert source, so the guidance is
`--moe-expert-cache-host-pinned-mb 0` = automatic partial pinning (full pinning
attempted, then each complete group CUDA accepts is retained, the rest is staged),
`--load-mode none` as the baseline, and *"Leave ~900 MB of VRAM free beyond the pack"*.
The page is honest that its 24 GB partial-pin benchmarks ran on Linux and do **not**
verify native-Windows speed. `--ple-prefetch` performance is likewise marked
"native Windows remains unverified".

**Risks / open problems** (from its issue tracker):
- `#90` (open): the expert slot pool was pinned to CUDA0 — irrelevant for our single
  GPU, but it shows the multi-GPU path is young; even after the fix a reporter measured
  a 2× regression vs the earlier commit.
- `#151` (open): `--moe-expert-profile` **crashes on the first request** on
  Qwen3.6-35B-A3B (our exact architecture) with
  `op not supported ffn_moe_gate-0 (MUL_MAT_ID) -> unspecified launch failure`
  (single-GPU non-Optional: their test was a merged local branch).
- `#120` (open), `#104` (closed): grouped MoE / MoE-cache failures on
  Qwen3.8-Flash-Next; `#104` was "corrupted output with MoE expert cache enabled on
  Windows 11", closed same day.
- `#112` (open): CUDA illegal memory access in `prepare_residency`.
- **No releases at all** (`releases` = 0 assets, `NONE`) — only tags
  (`moe-cache-b11814-28d73c8`, `moe-cache-b11612-167742d`, ...). The wiki expects you
  to build from source or fetch a "matched fork executable and DLL set" yourself.
- Maintainer is a recent graduate (README "Job Outreach"); repo created 2026-06-09,
  138★, 78 open issues, pushed 2026-10-10 — active but single-maintainer.

**Why it still tops the list:** the claim is exactly our problem (experts that don't
fit VRAM), measured on a 16 GB card, and it beats our current best by ~3–5× in the
single-request scenario while *reducing* peak VRAM. If it reproduces on this box, the
MoE presets get a step change. Verification path is cheap: build the branch
(~30–90 min on this CPU, same manual process as ik_llama) or obtain a matched binary,
then A/B our existing preset with `--n-cpu-moe 20` + a slot count.

**Cross-check on lineage:** GenerelSchwerz also maintains a `beellama/main` branch
combining BeeLlama's KVarN with this fork's memory controls and 16 GB guidance —
but the maintainer states explicitly (`#113`) that KVarN is **not** integrated into
the `moe-cache` expert-cache path and that this is a separate lineage. So: KVarN
(beellama) and the expert cache (moe-cache) do **not** currently combine. For MoE
models KVarN already measured as a loss on this box, so that is not a blocker.

## 3. thecodacus/llama.cpp — second expert cache, different design

`perf` is the default branch (352★, MIT, created 2026-05-06, pushed 2026-10-09).
Its README documents TWO independent fork features:

1. **MoE expert cache** (hot experts VRAM-resident, cold on CPU; output claimed
   bit-identical). Needs a one-time routing profile per model captured with
   `llama-moe-trace` (`tools/moe-trace/moe-trace.cpp` exists, 4886 bytes), then
   `--moe-cache-profile mymodel.csv --moe-cache-slots 112`. Supported architectures
   listed: `qwen35moe` (**ours**), `qwen4exp`, `deepseek2`, `laguna`.
   Measured on RTX 3060 12 GB with `-ncmoe 99`: Qwen3.6-35B-A3B Q4_K_M 42.3 → 51.3
   t/s (+21%) at 124 slots; **+66%** stacked with `--spec-type draft-mtp`; **+78%**
   with async CPU splits. Guidance: leave ~900 MB VRAM free; the pack is
   all-or-nothing (`pack allocation failed - expert cache disabled` falls back to
   baseline); ~25% expert traffic coverage is roughly break-even; wrong-workload
   profile still helps ~+28% worst case.
   Crucially it also works per model inside `--models-preset` INI sections
   (`moe-cache-profile = ...`, `moe-cache-slots = ...`) and via
   `LLAMA_ARG_MOE_CACHE_*` env vars — i.e. it composes with the built-in router.
2. **Fable's prefill optimizations** via env vars:
   `GGML_CUDA_REGISTER_HOST=1` and `GGML_SCHED_PREFETCH_EXPERTS=1`, claimed
   ~1143 → ~1880 t/s prefill (+64%) on Qwen3.6-35B-A3B with `--n-cpu-moe 26`.

**Important correction to that second claim:** `GGML_CUDA_REGISTER_HOST` is **already
in upstream ggml-org master** (`ggml/src/ggml-cuda/ggml-cuda.cu`, verified via the
GitHub API) and **already in our local beellama tree**
(`beellama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4712`, gated by `getenv`, paired with
`cudaHostRegister(..., cudaHostRegisterPortable | cudaHostRegisterReadOnly)`), and it
is exposed as a backend proc at `ggml-cuda.cu:5801`. So the pinning half is not
fork-exclusive — and we **live-tested it: no benefit on our preset** (ABBA, tg −4.6%,
pp −0.3%; see §8/P1). Only `GGML_SCHED_PREFETCH_EXPERTS` is absent from beellama's
source, and that half remains untested here.

**Maturity red flags:** `open_issues_count` is 9 but the issue lists came back
**completely empty** for both open and closed — issues/discussions are not an open
channel for this fork. **Zero releases** (`releases: []`), only tags. 352★ / 103
forks with a single visible maintainer. No public bug backlog means no public
validation of the cache's edge cases — the opposite of GenerelSchwerz, where the
failures are at least documented.

## 4. llama-swap — the reference implementation of what the router now does natively

Go binary, zero dependencies, one YAML config. 5912★, MIT, 100 open issues, active
(2026-10-10). Its value for us is the *feature list and contracts*, not the binary:

- OpenAI **and Anthropic** (`/v1/messages`, `/v1/messages/count_tokens`) on the same
  port, plus `/v1/responses`, embeddings, audio, images, rerank, infill.
- `ttl` (auto-unload after idle), `unloadTimeout`, `aliases`, `env`, `${PORT}`,
  `hooks` (preload at startup), `matrix` DSL for concurrency, `filters`
  (`stripParams`/`setParams`), groups for exclusivity.
- Management surface: `GET /running`, `POST /api/models/unload[/:id]`, `/ui`,
  `/logs/stream[/{model}]`, `/health`, `/metrics`, `/api/mcp`.
- Issue evidence that our design choices are right: **`#100`** "Terminating
  llama-swap doesn't always kill child process" (32 comments) — EngineBay already
  solves this permanently with the Job Object; **`#1004`** (open) requests exactly
  the memory-budget scheduler (`ram_gb` per model + eviction priority) with
  community pushback that a single number is too coarse and that naive LRU can evict
  the wrong model — i.e. our measured-record + admission-reserve approach is
  ahead of the reference implementation here; **`#1127`** asks for `swappable: false`
  to protect heavy models from accidental unloads — a good idea to steal as a preset
  flag ("protected: never auto-unload"); **`#426`** is the community debate
  "New in llama.cpp: Model Management — does llama-swap still make sense?", where
  maintainer and users agree llama-swap retains value mainly for *multi-backend*
  routing (llama.cpp + vLLM + whisper + stable-diffusion) and for exclusivity groups.
  That distinction is the key to our decision: EngineBay is single-backend, so the
  built-in router suffices.

## 5. thecodacus/local-ai-configs — the target architecture, already proven

Tiny repo (10★, MIT, created 2026-10-05, 9 files) but it is precisely the stack we
would build, and it is *documented with measurements*:

- `llama-swap/config.yaml` defines a model `llamacpp` whose `cmd` starts
  **`llama-server --models-preset /models/models.ini --models-max 1`**, with
  `ttl: 0` and `includeAliasesInList: true`, and lists every INI section name as an
  alias so `/v1/models` exposes each preset individually. Comment: *"the router itself
  idles at ~0 VRAM (its presets sleep after 600 s, see models.ini)"*.
- `llama-swap/models.ini` uses `sleep-idle-seconds = 600` in the `[*]` shared
  section — *"unload the model (free VRAM) after 10 idle minutes; the next request
  reloads it (~11 s for the 27B)"*. Other shared keys: `ngl = 99`, `fa = on`,
  `jinja = 1`, `metrics = 1`, `no-mmap = 1`, `parallel = 1`, `cache-reuse = 256`,
  `batch-size = 2048`, `ctk/ctv = q8_0`, and `spec-type = draft-mtp` +
  `spec-draft-n-max = 2`.
- Per-model sections carry the MoE placement directly:
  `[qwen36-35b-a3b-128k] n-cpu-moe = 10`, `[qwen36-35b-a3b-256k] n-cpu-moe = 13`,
  plus a note that *"Two keys may differ on upstream llama.cpp: `no-mmap` (upstream
  uses `load-mode`) and `spec-type = draft-mtp`"* — a portability warning exactly
  like the one we would need to encode.
- Measured swap times on their box: llama.cpp router → Strata ~34 s, Strata → router
  ~15 s.

This is the strongest single argument that the INI-preset router is the right
substrate for EngineBay: a third party already runs the whole fleet — including
Qwen3.6-35B-A3B MoE presets and MTP drafting — through it, on 2× 12 GB cards.

## 6. Edge0 — MONITOR (not applicable on this machine)

Summary of the delegated digest: it is a product + model release + research project,
not a library. Half the tree is a vendored copy of Apple MLX
(`macos/third_party/mlx/`, 567 of 1131 files). The Python framework is
**macOS/Apple-Silicon/MLX-only**; `backends/cuda/` is a reserved slot and the unified
framework is promised for "Q4 2026". Weights are safetensors (GGUF only as an export
path). Windows is a thin Tauri/React shell over a private llama.cpp fork with only
`windows/serve/{mem_budget,prefetch}.cc` of native code — no own GPU kernels.
The prerouter is a *trained* head bound to their checkpoints, and no training code is
in the repo, so it cannot be reused for third-party GGUFs.

The advertised memory figure is also softer than the tagline: README says 2.9 GiB peak
*active* memory, and a code-reading issue (`#106`) establishes that this is the MLX
allocator high-water mark, excluding file-backed expert pages and the MLX buffer
cache — the paper reports process RSS of 4.91/5.61/6.39 GiB against allocator peaks of
2.33/2.60/3.22 GiB. Observed throughput on real user machines: 0.7-0.8 tok/s on an
M1 8 GB (`#17`, "~30x below documented"), 7-13 s TTFT on an M1 Pro 16 GB (`#110`),
~5 tok/s from an external USB SSD (`#16`). On our box the whole 26.6 GB Q6_K MoE
already fits in 64 GB of RAM, so SSD streaming buys nothing and costs latency.

One genuinely portable idea: their patches add **byte-budget flags** (`--pool-mb`,
`--mem-budget-mb`, `edge0_apply_mem_budget`) instead of a layer count — conceptually
closer to our measured "VRAM-edge law" (pick bytes, not layers) and implementable in
`_vram_guard`/`vram_records.json` without any Edge0 code. Verdict: **MONITOR**, S for
the idea, L for any real integration (blocked on their roadmap, not on us).

## 7. Strata — BORROW-IDEAS, engine SKIP

Summary of the delegated digest: it is a real standalone engine (289 .cpp, 183 .hpp,
58 .cu; own CUDA kernels, own expert cache/VMM, own speculative decoder), built for
exactly one model family (Qwen3.8-Flash-Next 125B). llama.cpp is **not** vendored
(`third_party/` holds 3 files). 65% of the tree is `bench/` (1587 of 2422 files), and
there is **no CI** (`.github/` has no `workflows`).

Its pro-side evidence for us: a mature installer flow (hardware detect, size menu with
GB/RAM cost, resume-on-rerun *"run it again: it continues where it left off"*, reuse of
a previous install *"nothing big is downloaded"*, engine pinning in `engine/BUILD.json`,
SHA-256 tables, ModelScope mirror, non-interactive flags), an OpenAI **and Anthropic**
surface (`/v1/messages`, `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`), `/v1/responses`,
`serve/winjob.py` (their own Windows Job Object — same idea as our `setup_job_tree`),
`tools/dll_closure.py` (analogue of our `engine_diag.py check`), and explicit
lifecycle tests (`test_lifecycle.py`, `test_fatal_recovery.py`, `test_restart_waiters.py`).

Its con-side: version 0.1.x with ~1700 issues in ~12 days; the top open issues are
**crashes** (a session that restarts the PC after 25-30 min on gfx1100 `#872`; infinite
loop `#728`; `illegal memory access` in prefill with `std::exit(1)` `#1468`; 72× slower
prefill `#1549`), output degeneration traced to MoE output rows (`#606`, `#879`),
host OOM from expert pinning (`#1250`), and multi-GPU is unsolved (`#932`, `#841`,
`#1352`). On an RTX 5070 Ti 16 GB under Windows it self-calibrates ~13-14.4 tok/s and
delivers 9-10.6 (`#922`).

Borrowable, with effort:
- **Anthropic `/v1/messages` shim** over llama-server — S for basic non-streaming +
  streaming, M for faithful tool-calls/reasoning. This is EngineBay's real functional
  gap (it exposes OpenAI only). The known hard part is documented in their `#804`:
  an output parser that fails to leave the reasoning state when a `tool_call` appears
  inside ` thinking`, yielding silent `tool_calls: []`.
- **`fit_max_tokens`-style behaviour** (`#1736`): when prompt + max_tokens exceeds the
  context, shrink rather than refuse, and say so. Directly useful at our 96 K.
- **Restart supervisor contract** (`test_fatal_recovery.py`, `test_restart_waiters.py`,
  *"the engine stopped unexpectedly (exit code -6); the next request restarts it"*).
- Anti-lesson worth keeping: their `--resident-budget-gib` default made a user's system
  "dead slow" until removed (`#1080`) — confirms EngineBay's rule that an *estimate*
  must never block or auto-modify a preset.

## 8. Prioritised action list

**P0 — verify the MoE cache on this box. REVISED 2026-10-10: effort S, no build.**
The cache is **upstream** (PR #29887, merged 2026-10-07, port of qvac-fabric;
multi-GPU in #30112) and nightly **b11541** (2026-10-10) ships it for Windows CUDA.
Do NOT build the GenerelSchwerz fork — its extras are the buggy part and were not
upstreamed. Plan: Check Updates → download b11541 → A/B our two MoE presets on the
upstream engine with `--n-cpu-moe ALL` + `--moe-cache-mib ~9000-10000` (≈10% of
expert size per the PR guidance, sized to leave ~900 MB free), watching tg only
(cache serves decode batches ≤ 32 tokens) and requiring coherent output. Baselines
to beat: qw 29.6 t/s, gm 47.5 t/s. If the gain reproduces, add new presets (never
modify the existing ones) and record footprints in `vram_records.json`.

**P1 — test the free prefill win already in our binaries. RESULT: NEGATIVE, closed.**
`GGML_CUDA_REGISTER_HOST=1` was measured live on the current beellama MoE preset
(Qwen3.6-35B-A3B Q6_K, `--n-cpu-moe 21`, 96 K, port 8099) with an ABBA design to
control order bias — `scripts/_research_moe/bench_host_register_abba.py`,
log `host_register_abba.jsonl`:

| run | condition | tg t/s | pp t/s |
|---|---|---:|---:|
| 1 | BASE | 30.57 | 421.2 |
| 2 | REGISTER_HOST=1 | 28.42 | 421.3 |
| 3 | REGISTER_HOST=1 | 27.61 | 418.2 |
| 4 | BASE | 28.19 | 420.8 |
| — | **BASE mean** | **29.38** | **421.0** |
| — | **REG mean** | **28.02** | **419.8** |
| — | **delta** | **−4.6%** | **−0.3%** |

An earlier single-run A/B showed 18.57 → 29.59 t/s, which was **run-order/cold-HDD
noise**: the ABBA baseline reproduces our historical 29.6 t/s, and the env var is
neutral-to-slightly-negative. So the thecodacus fork's headline prefill claim does
**not** transfer to this box. Hypothesis (not verified): with `--load-mode none` the
expert source is ordinary allocated RAM rather than an mmap'd mapping, and the
registration path is aimed at mapped buffers — but whatever the reason, the measured
answer for our preset is "no benefit", so no preset or config change is warranted.
Do not re-test without a different load mode.
**P2 — router-mode support in EngineBay.** Since upstream and beellama already ship
`--models-preset`/`--models-max`/`--sleep-idle-seconds` with LRU eviction, EngineBay
can offer **auto-swap** without llama-swap: generate a `models.ini` from our existing
presets (one section per preset, `[*]` for shared params, per-model `n-cpu-moe`),
launch one router server with `--models-max 1` and `--sleep-idle-seconds`, and let the
client ask for any preset by name. Steal from local-ai-configs: expose each section
name as an alias so `/v1/models` lists them individually, and reuse `dedup-cache-models`
to hide HF-cache duplicates (the duplicate entry our live test surfaced). Effort M.
Note: ik_llama has no router mode, so this is a beellama/upstream-only feature — the
dense ik_llama presets must keep the classic single-model launch path.

**P3 — Anthropic `/v1/messages` shim.** EngineBay currently speaks OpenAI only;
llama-swap and Strata both treat Anthropic compatibility as table stakes.
S for the basic bridge, M for correct tool-call/reasoning boundaries.

**P4 — small borrowings, all S:** `swappable: false`-style "protected preset" flag
(llama-swap `#1127`); byte-budget framing of the expert share instead of layer counts
(Edge0 patches + our own VRAM-edge law); `fit_max_tokens` truncation instead of refusal
(Strata `#1736`).

**Explicitly not doing:** adopting llama-swap as a dependency (the built-in router
covers our single-backend case), adopting the Strata or Edge0 engines (different
problem class, unstable 0.1.x, not our platform), and the `GGML_SCHED_PREFETCH_EXPERTS`
half of the thecodacus fork until P0/P1 are settled (issues disabled, zero releases).

## 9. Corrections to our own notes made during this review

- **`--n-cpu-moe` counting direction was documented WRONG** in `AGENTS.md` and is now
  fixed: `N` moves the **first N layers'** experts to CPU in *both* beellama and
  upstream. Evidence: `llm_ffn_block_regex()` builds `blk\.%d` for `i = 0..N-1`
  (`beellama.cpp/common/common.h:1242`, applied at `common/arg.cpp:3046`), and upstream
  b10894 help says "keep the Mixture of Experts (MoE) weights of the first N layers in
  the CPU". There is no opposite direction; the same `N` transfers between engines.
- **`GGML_CUDA_REGISTER_HOST` is not a fork feature** — it is in upstream master and in
  our local beellama build (see §3).