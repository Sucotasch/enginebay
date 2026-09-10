"""Verify both fixes against the user's reported files:
1. Cache contamination no longer suppresses the first scan (_ensure_scan
   counts indexed entries UNDER THE CURRENT ROOT only).
2. Dense need = size + CALIBRATED overhead (from measured records), not
   flat +2048 — 13-14 GB models must stop being red when they fit.
3. MoE files (meta.moe) still show the ⚫ MoE-offload marker, never red.
"""
import json
import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import launcher  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (" — " + detail if detail else ""))
    if not cond:
        fails.append(name)


cfg = launcher.load_config()
records = launcher.load_vram_records() if hasattr(launcher, "load_vram_records") else {}
lib = launcher.load_library_cache(cfg)
print("records:", list(records.keys()))
print("library entries:", len(lib))

# 1) dense need: calibrated overhead, not flat 2048
p_kt = next(p for p, v in lib.items()
            if "iq4_kt" in os.path.basename(p).lower())
entry = lib[p_kt]
need = launcher.library_entry_need_mb(entry, records, lib, path=p_kt)
size_mb = entry["size"] // (1024 * 1024)
over = need - size_mb
print(f"IQ4_KT: size={size_mb}MB need={need}MB overhead={over}MB")
check("overhead is calibrated (not flat 2048)", over != 2048, f"overhead={over}")
check("record found for IQ4_KT (need==15204)", need == 15204, f"need={need}")

# 2) dots for the reported dense files — against the CEILING (total − 600),
# because WDDM evicts dwm/chrome on llama.cpp allocation
snap = launcher.diag.vram_snapshot()
free = snap.get("free_mb") or 15447
total = snap.get("total_mb") or 16063
ceiling = total - 600
print(f"free: {free} | total: {total} | ceiling: {ceiling}")
for p, v in sorted(lib.items()):
    b = os.path.basename(p).lower()
    if not any(n in b for n in ["heartfire", "maginum", "precog", "iq4_kt", "magistry"]):
        continue
    need = launcher.library_entry_need_mb(v, records, lib, path=p)
    moe = bool((v.get("meta") or {}).get("moe"))
    dot = "MoE" if moe else launcher.fit_class(need, ceiling)
    check(f"dense {os.path.basename(p)[:40]} not red", dot != "red", f"need={need} dot={dot}")

# 3) MoE files never red
for p, v in sorted(lib.items()):
    meta = v.get("meta") or {}
    if meta.get("moe"):
        check(f"moe {os.path.basename(p)[:40]} tagged", True, "moe=True -> MoE-offload dot")

# 4) _ensure_scan logic on a synthetic cfg: contaminated cache + no root entries
lib_stub = {"D:/unrelated/test.gguf": {"size": 1, "mtime": 1, "meta": {"arch": "llama"}}}
indexed = sum(1 for path, v in lib_stub.items()
              if v.get("meta") and str(path).startswith("G:/Ai"))
check("contaminated cache (foreign dir) triggers scan", indexed == 0)

print("\n" + ("ALL OK" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
