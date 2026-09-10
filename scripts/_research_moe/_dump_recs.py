"""Dump vram_records.json after the user's real Qwen MoE launch."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
recs = json.loads((REPO / "vram_records.json").read_text(encoding="utf-8"))
for k, v in recs.items():
    print(f"{k}: {v['mb']} MB at {v['at']}")
