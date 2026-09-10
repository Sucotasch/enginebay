"""Smoke: MoE detection across the zoo (dense must stay False).

Run: python scripts/test_gguf_meta.py  (same style, separate concern)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine_diag as diag  # noqa: E402

CASES = [
    # (path, expect_moe)
    (r"G:\Ai\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q6_K.gguf", True),
    (r"G:\Ai\Models\Gemma Heretic\gemma-4-26b-a4b-it-heretic.q4_k_m.gguf", True),
    (r"G:\Ai\Models\qwen3.8-27b-IQ4_XS\qwen3.8-27b-IQ4_XS-pure.gguf", False),
]


def main() -> int:
    fails = 0
    for path, expect in CASES:
        p = Path(path)
        if not p.exists():
            print(f"SKIP (missing): {path}")
            continue
        m = diag.gguf_metadata(path)
        got = bool(m.get("moe"))
        mark = "OK " if got == expect else "FAIL"
        print(f"{mark} moe={got!s:5s} (expect {expect!s:5s}) "
              f"size_label={m.get('size_label')} arch={m.get('arch')} — {p.name}")
        if got != expect:
            fails += 1
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
