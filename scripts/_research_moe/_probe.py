"""Probe: launch beellama server via subprocess list-args and capture stderr."""
import subprocess
import time

SRV = r"D:\Arx\Software Downloads\Hermes copy\llm-inference-server\beellama.cpp\versions\preview-v0.4.5-cuda-13.3\llama-server.exe"
QWEN = r"G:\Ai\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q6_K.gguf"

args = [SRV, "-m", QWEN, "-c", "98304", "-np", "1", "-ngl", "99", "-fa", "on", "-t", "5", "-tb", "6",
        "-ctk", "q4_0", "-ctv", "q4_0", "-b", "2048", "-ub", "512",
        "--kv-unified", "--jinja", "--warmup", "--load-mode", "none",
        "--temp", "1.0", "--min-p", "0.05", "--top-p", "0.95", "--top-k", "64",
        "--n-cpu-moe", "22", "--host", "127.0.0.1", "--port", "8099"]
print(" ".join(args))
p = subprocess.Popen(args, cwd="\\".join(SRV.split("\\")[:-1]),
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     text=True, encoding="utf-8", errors="replace")
time.sleep(5)
rc = p.poll()
print("alive:", rc is None)
if rc is not None:
    print("STDERR:", (p.stderr.read() or "")[-600:])
else:
    p.terminate()
