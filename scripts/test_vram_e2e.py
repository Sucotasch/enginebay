"""E2E test for the VRAM guard: real launch → measure → record → re-check.

Runs headless (QT_QPA_PLATFORM=offscreen) with the real beellama binary and
the real IQ4_XS model on a scratch port (8099), never touching 8080/8888.
"""
import sys
import time
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import launcher  # noqa: E402
import engine_diag as diag  # noqa: E402

app = launcher.QApplication([])
w = launcher.LLMLauncher()

MODEL = "G:/Ai/Models/qwen3.8-27b-IQ4_XS/qwen3.8-27b-IQ4_XS-pure.gguf"
BINARY = "beellama.cpp/versions/preview-v0.4.5-cuda-13.3/llama-server.exe"
PARAMS = ("-c 98304 -np 1 -ngl 99 -b 1024 -ub 256 --kv-unified "
          "--cache-type-k kvarn5 --cache-type-v kvarn4 --kv-tail-tokens 1024 "
          "-t 5 -tb 6 --flash-attn on --jinja --reasoning auto --temp 1.0 "
          "--min-p 0.0 --top-p 0.95 --top-k 20 --no-mmproj-offload")

w.model_entry.setText(MODEL)
w.bin_entry.setText(BINARY)
w.params_text.setPlainText(PARAMS)
w.host_entry.setText("127.0.0.1")
w.port_entry.setText("8099")

print("--- launching via _start_server (guard runs first) ---")
w._start_server()
assert w.process is not None, "process failed to start"
print("pid:", w.process.pid)

deadline = time.time() + 180
while time.time() < deadline and not w._server_ready:
    app.processEvents()
    time.sleep(0.2)
print("ready:", w._server_ready)
assert w._server_ready, "server never became ready in 180s"

app.processEvents()
time.sleep(1.5)   # let the record land
app.processEvents()

key = launcher.vram_need_key(MODEL, w._current_engine_id(), PARAMS)
recs = launcher.load_vram_records()
print("record key:", key)
print("record:", recs.get(key))
assert key in recs and recs[key].get("mb", 0) > 1000, "no VRAM record written!"

need_mb, measured = w._vram_need_mb(MODEL, w._current_engine_id(), PARAMS)
print(f"second-launch need: {need_mb} MB, measured={measured}")
assert measured and need_mb == recs[key]["mb"]

snap = diag.vram_snapshot()
free_mb = snap["free_mb"]
level = ("green" if free_mb >= need_mb + 512
         else "yellow" if free_mb >= need_mb else "red")
print(f"free={free_mb}, need={need_mb} -> level={level} "
      f"(server running: second launch should be red/yellow)")

w._stop_server()
time.sleep(3)
snap2 = diag.vram_snapshot()
free2 = snap2["free_mb"]
print(f"after stop: free={free2} (VRAM released)")
assert free2 > free_mb, "VRAM did not free after stop"

print("E2E: ALL OK")
