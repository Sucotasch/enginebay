"""Final matrix: REAL preset ctx (p98304 = 96K) + q4_0 KV, edge configs.

llama-bench sizes the KV buffer from -p/-n, so -p 98304 reproduces our
presets' -c 98304 KV budget. pp98k takes several minutes per config — the
matrix is trimmed to edge configs only.

Qwen edge: 18/20/22 at tiny ctx; 96K KV adds ~0.5-0.8 GB → cliff may move
up (more layers must go CPU). Gemma edge: 4/6/8 — same shift expected.

Run: python scripts/_research_moe/bench_final_ctx.py
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "beellama.cpp" / "versions" / "preview-v0.4.5-cuda-13.3" / "llama-bench.exe"
QWEN = Path(r"G:\Ai\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q6_K.gguf")
GEMMA = Path(r"G:\Ai\Models\Gemma Heretic\gemma-4-26b-a4b-it-heretic.q4_k_m.gguf")

BASE = ["-p", "98304", "-n", "128", "-r", "1", "-t", "6", "-fa", "on",
        "-ctk", "q4_0", "-ctv", "q4_0", "-b", "2048", "-ub", "2048"]

RUNS = [
    ("qw_ncmoe22_32k", QWEN, ["-ncmoe", "22"]),
    ("qw_ncmoe20_32k", QWEN, ["-ncmoe", "20"]),
    ("qw_ncmoe18_32k", QWEN, ["-ncmoe", "18"]),
    ("gm_ncmoe8_32k", GEMMA, ["-ncmoe", "8"]),
    ("gm_ncmoe6_32k", GEMMA, ["-ncmoe", "6"]),
    ("gm_ncmoe4_32k", GEMMA, ["-ncmoe", "4"]),
]


def main() -> int:
    results = []
    for label, model, extra in RUNS:
        cmd = [str(BENCH), "-m", str(model), *BASE, *extra]
        print(f"\n===== {label} =====", flush=True)
        print(" ".join(cmd), flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=2400)
        dt = time.time() - t0
        out = r.stdout or ""
        # keep only the result table lines
        lines = [ln for ln in out.splitlines()
                 if ln.startswith("| qwen") or ln.startswith("| gemma")
                 or ln.startswith("| model") or ln.startswith("| ---")]
        for ln in lines:
            print(ln)
        if r.returncode != 0:
            print(f"!! exit={r.returncode} stderr tail:", (r.stderr or "")[-400:])
        results.append((label, dt, lines))
    print("\n===== SUMMARY (pp32k / tg128, q4_0 KV) =====")
    for label, dt, lines in results:
        vals = [ln.split("|")[-2].strip() for ln in lines
                if ("pp" in ln or "tg" in ln) and ln.startswith("| q") is False
                and ln.startswith("| g") is False]
        # simpler: print last two table rows' test+ts columns
        try:
            rows = [ln for ln in lines if ln.startswith("| ") and "test" not in ln and "---" not in ln]
            test_vals = [(ln.split("|")[-3].strip(), ln.split("|")[-2].strip()) for ln in rows]
        except Exception:
            test_vals = []
        print(f"{label:16s} {dt/60:5.1f} min  {test_vals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
