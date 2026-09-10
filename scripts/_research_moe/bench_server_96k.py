"""Server-based benchmark at REAL preset ctx (98304) + q4_0 KV.

llama-bench cannot set -c and its pp98k on a spilling config runs for hours
(the 32K matrix already timed out once). A real llama-server on port 8099
with -c 98304 reproduces the actual preset budget exactly, and one
streamed completion gives us tg speed (what the user actually feels).

Protocol per config:
  1. launch server (beellama) with candidate params
  2. poll /health until success (model loaded) — timeout 10 min (HDD load)
  3. PDH VRAM snapshot -> free_mb, llama holder MB
  4. one short streaming completion (max_tokens=200, prompt "Say hello")
     -> tg t/s from usage/elapsed; one long prompt (3.5k tok) -> rough pp t/s
  5. kill server, settle 3 s

Run: python scripts/_research_moe/bench_server_96k.py [--label x --model qw|gm --ncmoe N]
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import engine_diag as diag  # noqa: E402

SRV = REPO / "beellama.cpp" / "versions" / "preview-v0.4.5-cuda-13.3" / "llama-server.exe"
QWEN = Path(r"G:\Ai\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q6_K.gguf")
GEMMA = Path(r"G:\Ai\Models\Gemma Heretic\gemma-4-26b-a4b-it-heretic.q4_k_m.gguf")
PORT = 8099
BASE = ["-c", "98304", "-np", "1", "-ngl", "99", "-fa", "on", "-t", "5", "-tb", "6",
        "-ctk", "q4_0", "-ctv", "q4_0", "-b", "2048", "-ub", "512",
        "--kv-unified", "--jinja", "--warmup", "--load-mode", "none",
        "--temp", "1.0", "--min-p", "0.05", "--top-p", "0.95", "--top-k", "64"]


def http_post(path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http_get(path: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def run_cfg(label: str, model: Path, extra: list[str]) -> dict:
    args = [str(SRV), "-m", str(model), *BASE, *extra,
            "--host", "127.0.0.1", "--port", str(PORT)]
    print(f"\n===== {label} =====", flush=True)
    print(" ".join(args), flush=True)
    log = open(REPO / "scripts/_research_moe/_srv.log", "w", encoding="utf-8", errors="replace")
    t0 = time.time()
    p = subprocess.Popen(args, cwd=str(SRV.parent), stdout=subprocess.DEVNULL,
                         stderr=log)
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
        time.sleep(3)  # driver settle
        snap = diag.vram_snapshot()
        holder = next((h["mb"] for h in snap.get("holders", [])
                       if "llama-server" in h.get("name", "")), None)
        print(f"loaded in {load_s:.0f}s | free={snap['free_mb']} MB | llama holder={holder} MB", flush=True)

        # tg test: short gen
        t1 = time.time()
        r = http_post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Tell me a short story about a robot."}],
            "max_tokens": 220, "temperature": 0.7, "stream": False,
        }, timeout=300)
        dt = time.time() - t1
        u = r.get("usage", {})
        ct = u.get("completion_tokens", 0) or 1
        pt = u.get("prompt_tokens", 0)
        tg = ct / dt
        print(f"tg: {ct} tok / {dt:.1f}s = {tg:.2f} t/s (prompt {pt} tok)", flush=True)

        # pp test: long prompt, tiny gen
        long_prompt = ("Please summarize the following text. " + ("The quick brown fox jumps over the lazy dog near the river bank at dawn. " * 260))
        t2 = time.time()
        r2 = http_post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": long_prompt}],
            "max_tokens": 4, "temperature": 0.0, "stream": False,
        }, timeout=600)
        dt2 = time.time() - t2
        u2 = r2.get("usage", {})
        pt2 = u2.get("prompt_tokens", 0)
        pp = pt2 / dt2
        print(f"pp: {pt2} tok in {dt2:.1f}s = {pp:.1f} t/s (incl. 4 gen tok)", flush=True)

        res.update(ok=True, load_s=round(load_s), free_mb=snap.get("free_mb"),
                   holder_mb=holder, tg=round(tg, 2), pp=round(pp, 1),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qw", "gm"], required=True)
    ap.add_argument("--ncmoe", type=int, required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra", default="",
                    help="extra args, semicolon-separated (e.g. '--cache-type-k;kvarn5')")
    a = ap.parse_args()
    model = QWEN if a.model == "qw" else GEMMA
    extra = ["--n-cpu-moe", str(a.ncmoe)] + [s for s in a.extra.split(";") if s]
    label = a.label or f"{a.model}_ncmoe{a.ncmoe}_96k"
    r = run_cfg(label, model, extra)
    line = json.dumps(r, ensure_ascii=False)
    print("\nRESULT " + line, flush=True)
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
