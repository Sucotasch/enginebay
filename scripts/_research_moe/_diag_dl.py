"""Diagnose update download: reproduce the launcher's exact network path.

1. Query the GitHub API like _check_updates does (this works for the user).
2. Take the newest tag's cudart + bins asset URLs.
3. Try to fetch them with the same urllib stack as _download_paired,
   with a hard cap and per-read progress, to see WHERE it stalls:
   - connect? TLS? first byte? streaming? total size mismatch?
"""
import json
import socket
import ssl
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"

t0 = time.time()
req = urllib.request.Request(API, headers={"User-Agent": "LLM-Launcher"})
with urllib.request.urlopen(req, timeout=15) as resp:
    releases = json.loads(resp.read())
print(f"API ok in {time.time()-t0:.1f}s, releases: {len(releases)}")

# replicate _classify_llama_asset
def classify(name: str):
    if not name.endswith("-x64.zip"):
        return None
    kind = "cudart" if name.startswith("cudart-") else "bin"
    if "-bin-win-cuda-" not in name:
        return None
    cuda = name.rsplit("-bin-win-cuda-", 1)[1].replace("-x64.zip", "")
    return kind, cuda

groups = {}
for r in releases[:10]:
    for a in r.get("assets", []):
        parsed = classify(a["name"])
        if not parsed:
            continue
        kind, cuda = parsed
        groups.setdefault((r["tag_name"], cuda), {})[kind] = a

newest = None
for (tag, cuda), assets in groups.items():
    if "cudart" in assets and "bin" in assets:
        if newest is None:
            newest = (tag, cuda, assets)
        print(f"group: {tag} cuda-{cuda}: cudart={assets['cudart']['name']} ({assets['cudart']['size']//1024//1024}MB) bin={assets['bin']['name']} ({assets['bin']['size']//1024//1024}MB)")

tag, cuda, assets = newest
url = assets["cudart"]["browser_download_url"]
print(f"\n=== fetching first asset: {url}")

t1 = time.time()
try:
    req = urllib.request.Request(url, headers={"User-Agent": "LLM-Launcher"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        print(f"connected: status={resp.status} in {time.time()-t1:.1f}s")
        print(f"headers: content-length={resp.headers.get('Content-Length')}")
        total = 0
        chunks = 0
        while True:
            buf = resp.read(64 * 1024)
            if not buf:
                break
            total += len(buf)
            chunks += 1
            if chunks % 200 == 0:  # every ~12 MB
                print(f"  {total//1024//1024} MB read ({time.time()-t1:.0f}s)", flush=True)
            if total > 30 * 1024 * 1024:  # cap at 30 MB — proof enough
                print(f"  capped at {total//1024//1024} MB — download WORKS")
                break
    print(f"read done: {total//1024//1024} MB in {time.time()-t1:.0f}s")
except Exception as e:
    print(f"DOWNLOAD FAILED after {time.time()-t1:.0f}s: {type(e).__name__}: {e}")

print("\n=== socket-level facts ===")
print("default proxies:", urllib.request.getproxies())
print("ssl:", ssl.OPENSSL_VERSION)
try:
    s = socket.create_connection(("api.github.com", 443), timeout=10)
    print("tcp api.github.com:443 ok, peer:", s.getpeername())
    s.close()
except Exception as e:
    print("tcp api.github.com:443 FAIL:", e)
try:
    s = socket.create_connection(("objects.githubusercontent.com", 443), timeout=10)
    print("tcp objects.githubusercontent.com:443 ok, peer:", s.getpeername())
    s.close()
except Exception as e:
    print("tcp objects.githubusercontent.com:443 FAIL:", e)
