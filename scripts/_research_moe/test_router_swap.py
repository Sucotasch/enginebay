"""Live test: beellama router mode as an alternative to llama-swap.

Proves (or disproves) the integration EngineBay would rely on:
  1. INI preset carries our per-model MoE flags (--n-cpu-moe, KV type, ctx)
  2. /v1/models lists router models with status
  3. On-demand load via POST /models/load
  4. models-max=1 forces LRU eviction of the running model when the
     second one is requested -> automatic swapping on a 16GB card
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRV = REPO / "beellama.cpp" / "versions" / "preview-v0.4.5-cuda-13.3" / "llama-server.exe"
PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"

QWEN = r"G:\Ai\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q6_K.gguf"
GEMMA = r"G:\Ai\unsloth\gemma-4-26b-a4b-it-heretic.q4_k_m.gguf"
INI = REPO / "scripts" / "_research_moe" / "_router_test.ini"
LOG = REPO / "scripts" / "_research_moe" / "_router_test.log"


def http(path, method="GET", body=None, timeout=30):
    req = urllib.request.Request(f"{BASE}{path}", method=method)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    try:
        return json.loads(raw)
    except Exception:
        return raw[:400]


def main():
    ini = f"""[*]
ctx-size = 98304
parallel = 1
n-gpu-layers = 99
flash-attn = on
threads = 5
threads-batch = 6
batch-size = 2048
ubatch-size = 512
cache-type-k = q4_0
cache-type-v = q4_0
kv-unified = 1
jinja = 1

[qwen-moe]
model = {QWEN}
n-cpu-moe = 21

[gemma-moe]
model = {GEMMA}
n-cpu-moe = 6
"""
    INI.write_text(ini, encoding="utf-8")
    print("=== INI written ===")
    print(ini)

    log = open(LOG, "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen([str(SRV), "--models-preset", str(INI), "--models-max", "1",
                          "--host", "127.0.0.1", "--port", str(PORT)],
                         cwd=str(SRV.parent), stdout=log, stderr=log)
    try:
        # wait for the router to listen
        for _ in range(60):
            try:
                http("/props", timeout=3)
                break
            except Exception:
                if p.poll() is not None:
                    print("router died rc=", p.returncode)
                    return 1
                time.sleep(1)
        else:
            print("router never came up")
            return 1

        props = http("/props")
        print("=== /props ===")
        print(json.dumps({k: v for k, v in props.items()
                          if k in ("role", "max_instances", "models_autoload")}, indent=2))

        models = http("/v1/models")
        print("=== /v1/models (before any load) ===")
        for m in models.get("data", []):
            meta = m.get("meta") or {}
            print(f"  {m.get('id'):<12} status={meta.get('value')}")

        # load model A
        print("=== POST /models/load qwen-moe ===")
        try:
            print(http("/models/load", "POST", {"model": "qwen-moe"}))
        except Exception as e:
            print("load call:", e)
        for _ in range(180):
            models = http("/v1/models")
            st = {m["id"]: (m.get("meta") or {}).get("value") for m in models.get("data", [])}
            if st.get("qwen-moe") in ("loaded", "running"):
                break
            if st.get("qwen-moe") == "failed":
                print("qwen-moe FAILED to load")
                break
            time.sleep(2)
        print("status now:", st)

        # request model B -> models-max=1 must evict A (LRU)
        print("=== POST /models/load gemma-moe (expect LRU eviction of qwen) ===")
        try:
            print(http("/models/load", "POST", {"model": "gemma-moe"}))
        except Exception as e:
            print("load call:", e)
        for _ in range(180):
            models = http("/v1/models")
            st = {m["id"]: (m.get("meta") or {}).get("value") for m in models.get("data", [])}
            if st.get("gemma-moe") in ("loaded", "running"):
                break
            if st.get("gemma-moe") == "failed":
                print("gemma-moe FAILED to load")
                break
            time.sleep(2)
        print("status now:", st)

        # access the loaded model through the OpenAI surface (model alias routing)
        print("=== POST /v1/chat/completions model=gemma-moe ===")
        try:
            r = http("/v1/chat/completions", "POST",
                     {"model": "gemma-moe", "messages": [{"role": "user", "content": "Hi"}],
                      "max_tokens": 8, "stream": False}, timeout=180)
            ch = r.get("choices", [{}])[0].get("message", {}).get("content", "")
            print("  response ok:", repr(ch[:60]))
        except Exception as e:
            print("  chat failed:", e)

        print("=== router log: eviction / swap lines ===")
        log.flush()
        txt = LOG.read_text(encoding="utf-8", errors="replace")
        for line in txt.splitlines():
            if any(k in line for k in ("LRU", "models_max", "unload", "evict",
                                       "router mode", "loaded", "child")):
                print("  " + line.strip()[:160])
        return 0
    finally:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                       capture_output=True)
        log.close()
        time.sleep(2)


if __name__ == "__main__":
    sys.exit(main())