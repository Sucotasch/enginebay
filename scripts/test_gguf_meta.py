"""Smoke: GGUF header parser speed + fields across the G:/Ai zoo.

Run: python scripts/test_gguf_meta.py   (no Qt needed)
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine_diag as diag  # noqa: E402

FILES = [
    r"G:\Ai\Models\qwen3.8-27b-IQ4_XS\qwen3.8-27b-IQ4_XS-pure.gguf",
    r"G:\Ai\Models\Qwen3.8-27B-ZB4.00-MIN-v5.1-IQ4_XS\Qwen3.8-27B-ZB4.00-MIN-v5.1-IQ4_XS.gguf",
    r"G:\Ai\Models\Gemma Heretic\gemma-4-26b-a4b-it-heretic.q4_k_m.gguf",
    r"G:\Ai\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q6_K.gguf",
    r"G:\Ai\Models\mmproj\mmproj-google_gemma-3-12b-it-bf16.gguf",
    r"G:\Ai\Models\Magistry-24B-v1.1\Magistry-24B-v1.1-Q4_K_M.gguf",
]


def main() -> int:
    fails = 0
    for f in FILES:
        if not Path(f).exists():
            print(f"SKIP (missing): {f}")
            continue
        t0 = time.time()
        m = diag.gguf_metadata(f)
        dt = (time.time() - t0) * 1000
        print(f"{dt:7.1f} ms  arch={str(m.get('arch'))[:12]:12s} "
              f"name={str(m.get('name'))[:26]:26s} ctx={m.get('ctx')} "
              f"blocks={m.get('blocks')} quant={m.get('quant')}")
        if not m.get("ok"):
            print(f"  !! not parsed as GGUF: {f}")
            fails += 1
    # Non-GGUF must return {} quietly (not raise)
    m = diag.gguf_metadata(__file__)
    assert m == {}, f"non-GGUF returned {m}"
    print("non-GGUF rejected quietly: OK")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
