"""Smoke-launch the two new (MoE cache) presets EXACTLY as the GUI would:
shlex-split the preset params string + the model path, launch, health, one
generation, kill. Guards against a bad flag combo in the shipped preset.
"""
import json
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRV = REPO / "llama.cpp" / "versions" / "b11541-cuda-13.4" / "llama-server.exe"
PORT = 8099
MODELS = {
    "Qwen3.6-35B-A3B (MoE cache)":
        r"G:\Ai\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q6_K.gguf",
    "Gemma-4-26B-A4B (MoE cache)":
        r"G:\Ai\Models\Gemma Heretic\gemma-4-26b-a4b-it-heretic.q4_k_m.gguf",
}


def main() -> int:
    presets = json.loads((REPO / "launcher_presets.json").read_text(encoding="utf-8"))
    failures = []
    for name, model in MODELS.items():
        params = shlex.split(presets[name]["params"])
        args = [str(SRV), "-m", model, *params,
                "--host", "127.0.0.1", "--port", str(PORT)]
        print(f"\n===== {name} =====", flush=True)
        log = open(REPO / "scripts/_research_moe/_srv.log", "w",
                   encoding="utf-8", errors="replace")
        p = subprocess.Popen(args, cwd=str(SRV.parent),
                             stdout=subprocess.DEVNULL, stderr=log)
        ok = False
        try:
            t0 = time.time()
            while time.time() - t0 < 420:
                if p.poll() is not None:
                    print(f"  SERVER DIED rc={p.returncode}", flush=True)
                    with open(REPO / "scripts/_research_moe/_srv.log",
                              encoding="utf-8", errors="replace") as lf:
                        print("  " + lf.read()[-400:], flush=True)
                    break
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                        if json.loads(r.read()).get("status") == "ok":
                            ok = True
                            break
                except Exception:
                    pass
                time.sleep(2)
            if not ok:
                failures.append(f"{name}: no healthy start")
                continue
            print(f"  healthy after {time.time()-t0:.0f}s", flush=True)
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                data=json.dumps({"messages": [{"role": "user", "content": "Say hi."}],
                                 "max_tokens": 30, "stream": False}).encode(),
                headers={"Content-Type": "application/json"})
            t1 = time.time()
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            ct = resp["usage"]["completion_tokens"]
            text = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
            print(f"  gen ok: {ct} tok in {time.time()-t1:.1f}s — {text[:40]!r}",
                  flush=True)
            if not text.strip():
                failures.append(f"{name}: empty completion")
        finally:
            p.terminate()
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
            log.close()
            time.sleep(5)
    print("\n" + ("FAIL: " + "; ".join(failures) if failures else "BOTH PRESETS SMOKE-OK"),
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())