"""Steady-state test for the upstream MoE cache (post-warmup tg, like PR #29887).

The 220-token single-shot measurement in bench_moe_cache.py includes LRU
cold-start (first tokens = 100% misses). PR #29887 reports post-warmup medians.
Protocol here: warmup generation, then a measured generation on a warmed
cache. If steady-state tg clearly beats our beellama records, the cache
matters for chat workloads; if not, the measured negative stands.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import engine_diag as diag  # noqa: E402

SRV = REPO / "llama.cpp" / "versions" / "b11541-cuda-13.4" / "llama-server.exe"
OUT = REPO / "scripts" / "_research_moe" / "moe_cache_steady.jsonl"
PORT = 8099

RUNS = [
    ("qw_cache10500", bench_model := r"G:\Ai\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q6_K.gguf",
     ["-cmoe", "--moe-cache-mib", "10500"]),
    ("gm_cache9500", r"G:\Ai\Models\Gemma Heretic\gemma-4-26b-a4b-it-heretic.q4_k_m.gguf",
     ["-cmoe", "--moe-cache-mib", "9500"]),
]

BASE = ["-c", "98304", "-np", "1", "-ngl", "99", "-fa", "on", "-t", "5", "-tb", "6",
        "-ctk", "q4_0", "-ctv", "q4_0", "-b", "2048", "-ub", "512",
        "--kv-unified", "--jinja", "--warmup", "--load-mode", "none",
        "--temp", "1.0", "--min-p", "0.05", "--top-p", "0.95", "--top-k", "64"]

PROMPTS = [
    "Write a Python function that merges two sorted lists efficiently, with tests.",
    "Explain how a merge sort works, then write a short story about a lighthouse.",
]


def post(payload: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gen(prompt: str, max_tokens: int) -> float:
    t0 = time.time()
    r = post({"messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens, "temperature": 0.7, "stream": False})
    dt = time.time() - t0
    ct = r.get("usage", {}).get("completion_tokens", 0) or 1
    return ct / dt


def main() -> int:
    for label, model, extra in RUNS:
        args = [str(SRV), "-m", model, *BASE, *extra,
                "--host", "127.0.0.1", "--port", str(PORT)]
        print(f"\n===== {label} =====", flush=True)
        log = open(REPO / "scripts/_research_moe/_srv.log", "w",
                   encoding="utf-8", errors="replace")
        p = subprocess.Popen(args, cwd=str(SRV.parent),
                              stdout=subprocess.DEVNULL, stderr=log)
        try:
            t0 = time.time()
            while time.time() - t0 < 600:
                if p.poll() is not None:
                    print("server died", flush=True)
                    return 1
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                        if json.loads(r.read()).get("status") == "ok":
                            break
                except Exception:
                    pass
                time.sleep(2)
            time.sleep(3)
            snap = diag.vram_snapshot()
            print(f"  loaded | free={snap['free_mb']} MB", flush=True)

            # warmup (fills the LRU), then measured runs on the warm cache
            w = gen(PROMPTS[0], 250)
            print(f"  warmup run: {w:.2f} t/s", flush=True)
            results = []
            for pr in PROMPTS:
                tg = gen(pr, 250)
                results.append(tg)
                print(f"  steady: {tg:.2f} t/s", flush=True)
            steady = sum(results) / len(results)
            r = {"label": label, "warmup_tg": round(w, 2),
                 "steady_tg": round(steady, 2), "runs": [round(x, 2) for x in results],
                 "free_mb": snap.get("free_mb"), "ok": True}
        finally:
            p.terminate()
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
            log.close()
            time.sleep(5)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n===== STEADY-STATE SUMMARY (beellama records: qw 29.6 / gm 47.5) =====",
          flush=True)
    for line in OUT.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        print(f"  {r['label']:<16} warmup={r['warmup_tg']}  steady={r['steady_tg']}  "
              f"runs={r['runs']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())