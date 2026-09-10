"""Isolate: why does Magistry get need=15483 when its record says 15237?"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import launcher  # noqa: E402

records = launcher.load_vram_records()
cfg = launcher.load_config()
lib = launcher.load_library_cache(cfg)

for p, v in lib.items():
    b = os.path.basename(p)
    if "magistry" not in b.lower():
        continue
    print("entry:", p)
    print("size MB:", v["size"] // (1024 * 1024))
    need = launcher.library_entry_need_mb(v, records, lib)
    print("need:", need)
    # manual trace of the record match
    base = b
    for key, rec in records.items():
        parts = key.split("|")
        print(f"  key={key!r} parts[1]={parts[1]!r} match={parts[1].lower() == base.lower()} mb={rec.get('mb')}")
    # and the overhead path
    over = launcher._dense_overhead_mb(records, lib)
    print("calibrated overhead:", over)
