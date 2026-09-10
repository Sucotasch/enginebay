"""Probe2: replicate bench_server_96k Popen EXACTLY, wait 60 s, dump log."""
import subprocess
import time
from pathlib import Path

REPO = Path(r"D:\Arx\Software Downloads\Hermes copy\llm-inference-server")
SRV = REPO / "beellama.cpp" / "versions" / "preview-v0.4.5-cuda-13.3" / "llama-server.exe"
QWEN = Path(r"G:\Ai\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q6_K.gguf")
LOG = REPO / "scripts/_research_moe/_probe2.log"

BASE = ["-c", "98304", "-np", "1", "-ngl", "99", "-fa", "on", "-t", "5", "-tb", "6",
        "-ctk", "q4_0", "-ctv", "q4_0", "-b", "2048", "-ub", "512",
        "--kv-unified", "--jinja", "--warmup", "--load-mode", "none",
        "--temp", "1.0", "--min-p", "0.05", "--top-p", "0.95", "--top-k", "64"]

args = [str(SRV), "-m", str(QWEN), *BASE, "--n-cpu-moe", "22",
        "--host", "127.0.0.1", "--port", "8099"]
print(" ".join(args))
log = open(LOG, "w", encoding="utf-8", errors="replace")
p = subprocess.Popen(args, cwd=str(SRV.parent), stdout=subprocess.DEVNULL, stderr=log)
for i in range(12):
    time.sleep(5)
    rc = p.poll()
    print(f"t={(i+1)*5}s rc={rc}", flush=True)
    if rc is not None:
        break
if p.poll() is None:
    p.terminate()
    try:
        p.wait(timeout=10)
    except Exception:
        p.kill()
log.close()
print("--- log content ---")
print(LOG.read_text(encoding="utf-8", errors="replace")[:2000])
