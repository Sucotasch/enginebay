"""Inspect cached library meta for the files the user reported as red."""
import json
import os

cfg = json.load(open(r"D:\Arx\Software Downloads\Hermes copy\llm-inference-server\launcher_config.json", encoding="utf-8"))
lib = cfg.get("library", {})
print("root:", cfg.get("model_root"), "| library entries:", len(lib))
needles = ["gemma-4-26b", "qwen3.6-35b", "heartfire", "maginum", "iq4_kt", "magistry", "precog"]
MB = 1024 * 1024
for p, v in lib.items():
    b = os.path.basename(p).lower()
    if any(n in b for n in needles):
        meta = v.get("meta") or {}
        mk = list(meta.keys()) if meta else None
        print(f"{os.path.basename(p)[:52]:52s} size={v.get('size', 0) // MB:6d}MB "
              f"meta_keys={mk} moe={meta.get('moe') if meta else None} arch={meta.get('arch') if meta else None}")
