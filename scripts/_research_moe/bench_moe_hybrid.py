"""Push the cache ceiling + test the PR's hybrid (--fit + cache, keeps some
expert layers GPU-resident so prefill survives).

Runs (steady-state protocol: warmup gen, then 2 measured gens, then a pp shot):
  1. qw -cmoe --moe-cache-mib 11500   (ceiling push; 10500 left ~2 GB free)
  2. gm -cmoe --moe-cache-mib 10500   (ceiling push)
  3. qw --fit --moe-cache-mib 8000    (hybrid: fit picks the layer/cache split)
  4. gm --fit --moe-cache-mib 8000    (hybrid)
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
OUT = REPO / "scripts" / "_research_moe" / "moe_cache_hybrid.jsonl"
PORT = 8099

QWEN = r"G:\Ai\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q6_K.gguf"
GEMMA = r"G:\Ai\Models\Gemma Heretic\gemma-4-26b-a4b-it-heretic.q4_k_m.gguf"

BASE = ["-c", "98304", "-np", "1", "-ngl", "99", "-fa", "on", "-t", "5", "-tb", "6",
        "-ctk", "q4_0", "-ctv", "q4_0", "-b", "2048", "-ub", "512",
        "--kv-unified", "--jinja", "--warmup", "--load-mode", "none",
        "--temp", "1.0", "--min-p", "0.05", "--top-p", "0.95", "--top-k", "64"]

RUNS = [
    ("qw_cmoe_c11500", QWEN, ["-cmoe", "--moe-cache-mib", "11500"]),
    ("gm_cmoe_c10500", GEMMA, ["-cmoe", "--moe-cache-mib", "10500"]),
    ("qw_fit_c8000", QWEN, ["--moe-cache-mib", "8000"]),
    ("gm_fit_c8000", GEMMA, ["--moe-cache-mib", "8000"]),
]

PROMPTS = [
    "Write a Python function that merges two sorted lists efficiently, with tests.",
    "Explain how a merge sort works, then write a short story about a lighthouse.",
]


def post(payload: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gen(prompt: str, max_tokens: int) -> tuple[float, int]:
    t0 = time.time()
    r = post({"messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens, "temperature": 0.7, "stream": False})
    ct = r.get("usage", {}).get("completion_tokens", 0) or 1
    return ct / (time.time() - t0), ct


def pp_shot() -> float:
    long_prompt = ("Please summarize the following text. " +
                   ("The quick brown fox jumps over the lazy dog near the river bank at dawn. " * 260))
    t0 = time.time()
    r = post({"messages": [{"role": "user", "content": long_prompt}],
              "max_tokens": 4, "temperature": 0.0, "stream": False}, timeout=900)
    pt = r.get("usage", {}).get("prompt_tokens", 0) or 1
    return pt / (time.time() - t0)


def main() -> int:
    for label, model, extra in RUNS:
        args = [str(SRV), "-m", model, *BASE, *extra,
                "--host", "127.0.0.1", "--port", str(PORT)]
        print(f"\n===== {label} =====", flush=True)
        log = open(REPO / "scripts/_research_moe/_srv.log", "w",
                   encoding="utf-8", errors="replace")
        p = subprocess.Popen(args, cwd=str(SRV.parent),
                             stdout=subprocess.DEVNULL, stderr=log)
        res: dict = {"label": label}
        try:
            t0 = time.time()
            while time.time() - t0 < 600:
                if p.poll() is not None:
                    print("  SERVER DIED", flush=True)
                    log.close()
                    with open(REPO / "scripts/_research_moe/_srv.log",
                              encoding="utf-8", errors="replace") as lf:
                        print("  " + lf.read()[-500:], flush=True)
                    res["ok"] = False
                    break
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                        if json.loads(r.read()).get("status") == "ok":
                            break
                except Exception:
                    pass
                time.sleep(2)
            else:
                res["ok"] = False
                res["error"] = "health timeout"
            if res.get("ok", True) is False and "error" not in res:
                res["error"] = "died at startup"
            if "error" in res and res["error"] == "died at startup":
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                continue

            time.sleep(3)
            snap = diag.vram_snapshot()
            res["free_mb"] = snap.get("free_mb")
            print(f"  loaded | free={snap['free_mb']} MB", flush=True)

            w, _ = gen(PROMPTS[0], 250)
            print(f"  warmup: {w:.2f} t/s", flush=True)
            steady = []
            for pr in PROMPTS:
                tg, ct = gen(pr, 250)
                steady.append(tg)
                print(f"  steady: {tg:.2f} t/s ({ct} tok)", flush=True)
            pp = pp_shot()
            print(f"  pp: {pp:.1f} t/s", flush=True)
            res.update(ok=True, warmup=round(w, 2),
                       steady=round(sum(steady) / len(steady), 2),
                       runs=[round(x, 2) for x in steady], pp=round(pp, 1))
        finally:
            p.terminate()
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
            log.close()
            time.sleep(5)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("\n===== SUMMARY (beellama: qw 29.6 tg/420 pp; gm 47.5 tg/1530 pp) =====",
          flush=True)
    for line in OUT.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        print(f"  {r['label']:<16} ok={r.get('ok')} steady={r.get('steady')} "
              f"runs={r.get('runs')} pp={r.get('pp')} free={r.get('free_mb')}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())