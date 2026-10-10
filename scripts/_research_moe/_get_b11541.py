"""Download upstream llama.cpp b11541 (first build with the MoE expert cache)
into llama.cpp/versions/b11541-cuda-13.4/, mirroring the existing b10894 layout.

curl.exe fails in this sandbox (schannel SEC_E_NO_CREDENTIALS) — python urllib
works (proven by the launcher's own downloader). Chunked 256 KB reads, progress,
stall timeout, 3 retries, truncation check.
"""
import sys
import time
import urllib.request
from pathlib import Path
import zipfile

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "llama.cpp" / "versions" / "b11541-cuda-13.4"
BASE = "https://github.com/ggml-org/llama.cpp/releases/download/b11541"
ASSETS = ["llama-b11541-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"]
CHUNK = 256 * 1024
STALL_S = 30.0


def fetch(url: str, dst: Path, expected_min: int = 1024 * 1024) -> bool:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=STALL_S) as r:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                t_last = time.time()
                with open(dst, "wb") as f:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        if time.time() - t_last > 15:
                            print(f"  {dst.name}: {got/1024/1024:.0f} MB"
                                  + (f" / {total/1024/1024:.0f} MB" if total else ""),
                                  flush=True)
                            t_last = time.time()
            if got >= expected_min:
                print(f"  fetched {dst.name}: {got} bytes"
                      + (f" (expected {total})" if total and total != got else ""),
                      flush=True)
                if total and total != got:
                    print("  SIZE MISMATCH — will retry", flush=True)
                    continue
                return True
            print(f"  too small ({got} bytes) — retry", flush=True)
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    return False


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    tmp = DEST.parent / "_dl_b11541"
    tmp.mkdir(exist_ok=True)
    for name in ASSETS:
        z = tmp / name
        if z.exists() and z.stat().st_size > 1024 * 1024:
            print(f"  {name}: already downloaded", flush=True)
        elif not fetch(f"{BASE}/{name}", z):
            print(f"FAILED: {name}", flush=True)
            return 1
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DEST)
        print(f"  extracted {name}", flush=True)
        z.unlink()
    try:
        tmp.rmdir()
    except OSError:
        pass
    files = sorted(p.name for p in DEST.iterdir())
    print(f"installed {len(files)} files into {DEST.name}", flush=True)
    missing = [f for f in ("llama-server.exe", "ggml-base.dll", "ggml-cuda.dll")
               if f not in files]
    if missing:
        print(f"MISSING: {missing}", flush=True)
        return 1
    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())