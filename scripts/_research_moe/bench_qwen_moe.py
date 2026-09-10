"""Benchmark Qwen3.6-35B-A3B MoE offload configs with llama-bench (beellama).

llama-bench measures pp (prompt processing) + tg (token generation) n times
per config without running a server. Each run loads the model (30-60 s from
HDD) and runs the workload — total ~10-15 min for the screening matrix.

Screening matrix (Qwen3.6-35B-A3B-Q6_K, 40 blocks, 128 experts / 8 active):
  n-cpu-moe: 999 (all experts CPU), 20, 10, 5  (beellama counts from FIRST
             layers, so N experts on CPU = layers 0..N-1; GPU gets the tail)
  Note: llama-bench has no --cpu-moe (all) — use -ncmoe 40 (block count).

Run: python scripts/_research_moe/bench_qwen_moe.py
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "beellama.cpp" / "versions" / "preview-v0.4.5-cuda-13.3" / "llama-bench.exe"
MODEL = Path(r"G:\Ai\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q6_K.gguf")

# (label, extra args) — edge refinement around the VRAM cliff (ncmoe20 won
# screening at 35.4 t/s; 10/5 spill to WDDM shared memory and crawl).
CONFIGS = [
    ("ncmoe24",          ["-ncmoe", "24", "-fa", "on"]),
    ("ncmoe22",          ["-ncmoe", "22", "-fa", "on"]),
    ("ncmoe20_r3",       ["-ncmoe", "20", "-fa", "on"]),
    ("ncmoe18",          ["-ncmoe", "18", "-fa", "on"]),
    ("ncmoe16",          ["-ncmoe", "16", "-fa", "on"]),
]


def main() -> int:
    assert BENCH.is_file(), f"bench not found: {BENCH}"
    assert MODEL.is_file(), f"model not found: {MODEL}"
    results = []
    for label, extra in CONFIGS:
        cmd = [str(BENCH), "-m", str(MODEL),
               "-p", "512", "-n", "128", "-r", "3", "-t", "6",
               *extra]
        print(f"\n===== {label} =====")
        print(" ".join(cmd))
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=1200)
        dt = time.time() - t0
        out = r.stdout or ""
        print(out[-1500:])
        if r.returncode != 0:
            print(f"!! exit={r.returncode} stderr tail:", (r.stderr or "")[-500:])
        results.append((label, dt, out.strip().splitlines()[-1] if out.strip() else ""))
    print("\n\n===== SUMMARY =====")
    for label, dt, last in results:
        print(f"{label:22s} {dt/60:5.1f} min  {last}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
