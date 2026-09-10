"""Benchmark Gemma-4-26B-A4B MoE offload configs with llama-bench (beellama).

Screening matrix (gemma-4-26b-a4b-it-heretic Q4_K_M, 30 blocks, 128 experts /
8 active + 1 shared). Full GPU needs ~17 GB > our 16, so offload a slice.

Run: python scripts/_research_moe/bench_gemma_moe.py
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "beellama.cpp" / "versions" / "preview-v0.4.5-cuda-13.3" / "llama-bench.exe"
MODEL = Path(r"G:\Ai\Models\Gemma Heretic\gemma-4-26b-a4b-it-heretic.q4_k_m.gguf")

CONFIGS = [
    ("g_ncmoe6",       ["-ncmoe", "6",  "-fa", "on"]),
    ("g_ncmoe4",       ["-ncmoe", "4",  "-fa", "on"]),
    ("g_ncmoe2",       ["-ncmoe", "2",  "-fa", "on"]),
    ("g_ncmoe0_full",  ["-ncmoe", "0",  "-fa", "on"]),
]


def main() -> int:
    assert BENCH.is_file(), f"bench not found: {BENCH}"
    assert MODEL.is_file(), f"model not found: {MODEL}"
    results = []
    for label, extra in CONFIGS:
        cmd = [str(BENCH), "-m", str(MODEL),
               "-p", "512", "-n", "128", "-r", "2", "-t", "6",
               *extra]
        print(f"\n===== {label} =====", flush=True)
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
        print(f"{label:18s} {dt/60:5.1f} min  {last}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
