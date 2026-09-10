"""Bisect: call library_entry_need_mb step by step on the real entry."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import launcher  # noqa: E402

records = launcher.load_vram_records()
lib = launcher.load_library_cache(launcher.load_config())

p = next(k for k in lib if "magistry" in k.lower())
entry = lib[p]
print("key:", p)
base = os.path.basename(entry.get("path", ""))
print("base:", repr(base))
best = 0
for key, rec in records.items():
    parts = key.split("|")
    if len(parts) == 3 and parts[1].lower() == base.lower():
        mb = rec.get("mb", 0) if isinstance(rec, dict) else 0
        print(f"  MATCH {key} mb={mb}")
        if mb > best:
            best = mb
print("best:", best)
print("final need:", launcher.library_entry_need_mb(entry, records, lib))
