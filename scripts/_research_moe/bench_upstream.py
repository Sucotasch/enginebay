"""Engine comparison: run the SAME winning configs on upstream llama.cpp.

Differences from beellama runner (bench_server_96k.py):
  - upstream has no --warmup / --kv-unified / --load-mode; uses --no-mmap
  - qwen35moe arch support must be verified live

Run: python scripts/_research_moe/bench_upstream.py
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

SRV = REPO / "llama.cpp" / "versions" / "b10712-cuda-13.3" / "llama-server.exe"
QWEN = Path(r"G:\Ai\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q6_K.gguf")
GEMMA = Path(r"G:\Ai\Models\Gemma Heretic\gemma-4-26b-a4b-it-heretic.q4_k_m.gguf")
PORT = 8099
BASE = ["-c", "98304", "-np", "1", "-ngl", "99", "-fa", "on", "-t", "5", "-tb", "6",
        "-ctk", "q4_0", "-ctv", "q4_0", "-b", "2048", "-ub", "512",
        "--jinja", "--no-mmap", "--reasoning", "off",
        "--temp", "1.0", "--min-p", "0.05", "--top-p", "0.95", "--top-k", "64"]


def http_post(path, payload, timeout):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http_get(path, timeout=5.0):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def run_cfg(label, model, extra):
    args = [str(SRV), "-m", str(model), *BASE, *extra,
            "--host", "127.0.0.1", "--port", str(PORT)]
    print(f"\n===== {label} =====", flush=True)
    print(" ".join(args), flush=True)
    log = open(REPO / "scripts/_research_moe/_srv_up.log", "w",
               encoding="utf-8", errors="replace")
    t0 = time.time()
    p = subprocess.Popen(args, cwd=str(SRV.parent), stdout=subprocess.DEVNULL, stderr=log)
    res = {"label": label, "ok": False}
    try:
        deadline = time.time() + 600
        loaded = False
        while time.time() < deadline:
            if p.poll() is not None:
                raise RuntimeError(f"server died rc={p.returncode}")
            try:
                if http_get("/health").get("status") == "ok":
                    loaded = True
                    break
            except Exception:
                pass
            time.sleep(2)
        if not loaded:
            raise RuntimeError("health timeout")
        load_s = time.time() - t0
        time.sleep(3)
        snap = diag.vram_snapshot()
        holder = next((h["mb"] for h in snap.get("holders", [])
                       if "llama-server" in h.get("name", "")), None)
        print(f"loaded in {load_s:.0f}s | free={snap['free_mb']} MB | holder={holder} MB", flush=True)

        t1 = time.time()
        r = http_post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Tell me a short story about a robot."}],
            "max_tokens": 220, "temperature": 0.7, "stream": False}, timeout=300)
        dt = time.time() - t1
        ct = r.get("usage", {}).get("completion_tokens", 0) or 1
        print(f"tg: {ct} tok / {dt:.1f}s = {ct/dt:.2f} t/s", flush=True)

        long_prompt = ("Please summarize the following text. "
                       + ("The quick brown fox jumps over the lazy dog near the river bank at dawn. " * 260))
        t2 = time.time()
        r2 = http_post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": long_prompt}],
            "max_tokens": 4, "temperature": 0.0, "stream": False}, timeout=600)
        dt2 = time.time() - t2
        pt2 = r2.get("usage", {}).get("prompt_tokens", 0)
        print(f"pp: {pt2} tok in {dt2:.1f}s = {pt2/dt2:.1f} t/s", flush=True)

        res.update(ok=True, load_s=round(load_s), free_mb=snap.get("free_mb"),
                   holder_mb=holder, tg=round(ct / dt, 2), pp=round(pt2 / dt2, 1),
                   tg_tokens=ct, pp_tokens=pt2)
    except Exception as e:
        res["error"] = str(e)[:300]
        print("ERROR:", res["error"], flush=True)
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()
        log.close()
    return res


def main():
    runs = [
        ("up_qw_ncmoe21", QWEN, ["--n-cpu-moe", "21"]),
        ("up_gm_ncmoe6", GEMMA, ["--n-cpu-moe", "6"]),
    ]
    out = REPO / "scripts/_research_moe/upstream_96k.jsonl"
    for label, model, extra in runs:
        r = run_cfg(label, model, extra)
        line = json.dumps(r, ensure_ascii=False)
        print("\nRESULT " + line, flush=True)
        with open(out, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
