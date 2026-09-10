"""Reproduce the user's exact scenario: open library with root G:/Ai,
let it scan, then inspect what dots WOULD be shown for the reported files.

Offscreen Qt — no GUI needed.
"""
import json
import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import launcher  # noqa: E402

cfg = launcher.load_config()
print("before: model_root =", cfg.get("model_root"), "| library =", len(cfg.get("library", {})))

# force the scan of the real root
cfg["model_root"] = "G:/Ai"
launcher.save_config(cfg)

entries = launcher.scan_library(Path("G:/Ai"))
print("scan found", len(entries), "gguf files under G:/Ai")

lib = {}
for e in entries:
    meta = launcher.diag.gguf_metadata(e["path"])
    meta = {k: meta[k] for k in ("arch", "name", "ctx", "blocks", "quant", "moe")
            if meta.get(k) is not None} or None
    lib[e["path"]] = {"size": e["size"], "mtime": e["mtime"], "meta": meta}

cfg["library"] = lib
launcher.save_config(cfg)
print("after: library =", len(lib), "entries saved")

# Now the dots for the reported files
records = launcher.load_vram_records() if hasattr(launcher, "load_vram_records") else {}
MB = 1024 * 1024
needles = ["gemma-4-26b", "qwen3.6-35b", "heartfire", "maginum", "iq4_kt", "magistry", "precog"]
snap = launcher.diag.vram_snapshot()
free = snap.get("free_mb")
print(f"live free: {free} MB")
for p, v in sorted(lib.items()):
    b = os.path.basename(p).lower()
    if not any(n in b for n in needles):
        continue
    meta = v.get("meta") or {}
    moe = bool(meta.get("moe"))
    need = launcher.library_entry_need_mb(v, records)
    fc = launcher.fit_class(need, free)
    print(f"{os.path.basename(p)[:48]:48s} moe={moe!s:5s} need={need:6d}MB "
          f"dot={'MoE-offload' if moe else fc} (free={free})")
