"""Measure REAL footprints of the two 'red-dot' dense models, so their
library dots become MEASURED instead of estimated (the campaign rule:
no assumptions — measure).

Protocol = the guard's: PDH snapshot before launch, /health wait, PDH
snapshot after, delta = footprint. Params mirror the user's dense presets
(96K ctx, q4_0 KV, -ngl 99).
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import engine_diag as diag  # noqa: E402

SRV_DEFAULT = REPO / "beellama.cpp" / "versions" / "preview-v0.4.5-cuda-13.3" / "llama-server.exe"
SRV_IK = REPO / "ik_llama.cpp" / "versions" / "15dddc6" / "llama-server.exe"
CFG = REPO / "launcher_config.json"
LIB = json.loads(CFG.read_text(encoding="utf-8")).get("library", {})
MODELS = [
    # (name, engine srv, params extra)
    ("Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf", SRV_IK,
     ["-c", "98304", "-np", "1", "-ngl", "99", "-b", "1024", "-ub", "256",
      "-ctk", "q4_0", "-ctv", "q4_0", "-t", "5", "-tb", "6", "-fa", "on",
      "--jinja", "--reasoning", "auto", "--no-mmap",
      "--temp", "1.0", "--min-p", "0.0", "--top-p", "0.95", "--top-k", "20",
      "--presence-penalty", "0.0", "--repeat-penalty", "1.0", "--no-mmproj-offload"]),
    ("sophosympatheia_Magistry-24B-v1.1-Q4_K_L.gguf", SRV_DEFAULT,
     ["-c", "98304", "-np", "1", "-ngl", "99", "-fa", "on", "-t", "5", "-tb", "6",
      "-ctk", "q4_0", "-ctv", "q4_0", "-b", "2048", "-ub", "512",
      "--kv-unified", "--jinja", "--reasoning", "off", "--load-mode", "none"]),
]
PORT = 8099
HOST_ARGS = ["--host", "127.0.0.1", "--port", str(PORT)]


def http_get(path, timeout=5.0):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return json.loads(r.read())


results = {}
eng_label = ""
for name, srv, extra in MODELS:
    m = Path(name)
    if not m.is_file():
        # fall back to searching the library cache for the basename
        for p in LIB:
            if Path(p).name.lower() == m.name.lower():
                m = Path(p)
                break
    eng_label = "ik_llama.cpp" if "ik_llama" in str(srv) else "beellama.cpp"
    print(f"\n=== {m.name} on {eng_label} ===")
    if not m.is_file():
        print("NOT FOUND, skip")
        continue
    base_snap = diag.vram_snapshot()
    base_free = base_snap.get("free_mb")
    log = open(REPO / "scripts/_research_moe/_foot.log", "a", encoding="utf-8", errors="replace")
    p = subprocess.Popen([str(srv), "-m", str(m), *extra, *HOST_ARGS], cwd=str(srv.parent),
                         stdout=subprocess.DEVNULL, stderr=log)
    try:
        deadline = time.time() + 600
        ok = False
        while time.time() < deadline:
            if p.poll() is not None:
                raise RuntimeError(f"died rc={p.returncode}")
            try:
                if http_get("/health").get("status") == "ok":
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(2)
        if not ok:
            raise RuntimeError("health timeout")
        time.sleep(3)
        snap = diag.vram_snapshot()
        holder = next((h["mb"] for h in snap.get("holders", [])
                       if "llama-server" in h.get("name", "")), None)
        delta = (base_free or 0) - (snap.get("free_mb") or 0)
        mb = holder if holder else delta
        print(f"base_free={base_free} now_free={snap.get('free_mb')} "
              f"holder={holder} delta={delta}")
        print(f"FOOTPRINT: {mb} MB")
        results[f"{eng_label}|{m.name}|c98304"] = {"mb": mb}
        # quick sanity gen so the footprint includes warmup compute buffers
        t1 = time.time()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "Hi"}],
                             "max_tokens": 8, "stream": False}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            json.loads(r.read())
        print(f"gen ok in {time.time()-t1:.1f}s")
        snap2 = diag.vram_snapshot()
        holder2 = next((h["mb"] for h in snap2.get("holders", [])
                        if "llama-server" in h.get("name", "")), None)
        if holder2 and holder2 != mb:
            print(f"post-gen holder: {holder2} MB (was {mb})")
            results[f"{eng_label}|{m.name}|c98304"]["mb"] = max(mb, holder2)
    except Exception as e:
        print("ERROR:", e)
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()
        log.close()
        time.sleep(3)

print("\n=== RESULTS ===")
print(json.dumps(results, indent=2))
out = REPO / "scripts/_research_moe/footprints_measured.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("saved to", out)
