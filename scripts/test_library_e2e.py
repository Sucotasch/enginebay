"""E2E: ModelLibraryDialog against a temp library root (offscreen Qt).

Does NOT touch the real launcher_config.json: the dialog gets a throwaway
cfg dict, a temp root with real GGUFs (copied headers-only via 8 KB prefix
of two small zoo files), and every assertion runs on the dialog's behavior:
scan, cache write, fit dots, search filter, pick → selected_path.

Run: $env:QT_QPA_PLATFORM="offscreen"; .venv\\Scripts\\python.exe scripts/test_library_e2e.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from PyQt6.QtWidgets import QApplication  # noqa: E402

import launcher  # noqa: E402

# Real small GGUFs from the zoo: headers are at the file start, so an 8 KB
# prefix is a valid GGUF for the parser (magic + a few KVs) and copies fast.
ZOO = [
    Path(r"G:\Ai\Models\mmproj\mmproj-google_gemma-3-12b-it-bf16.gguf"),
    Path(r"G:\Ai\lmstudio-community\Qwen3.6-35B-A3B-GGUF\mmproj-Qwen3.6-35B-A3B-BF16.gguf"),
]
PREFIX_BYTES = 8 * 1024


def main() -> int:
    # Sandbox note: only pre-created dirs are writable here, so the temp
    # ROOT is a fixed dir under scripts/ (wiped at start and end).
    root = HERE / "_libtest_tmp"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        copied = []
        for src in ZOO:
            if not src.exists():
                print(f"SKIP: {src} not present")
                continue
            dst = root / src.name
            with open(src, "rb") as f, open(dst, "wb") as o:
                o.write(f.read(PREFIX_BYTES))
            copied.append(dst)
        assert copied, "no zoo GGUFs available on this machine"
        # A decoy: plain text with .gguf extension must be filtered by parsing
        (root / "decoy.gguf").write_bytes(b"not a gguf at all" * 64)

        app = QApplication([])

        # ISOLATION: the dialog's _rescan persists via save_config(self.cfg)
        # → without this, the e2e test overwrites the REAL launcher_config.json
        # with the temp-root library (this is how test cache contaminated the
        # user's config once). Point CONFIG_FILE at a scratch file instead.
        real_config_file = launcher.CONFIG_FILE
        scratch_cfg = HERE / "_libtest_tmp_config.json"
        launcher.CONFIG_FILE = scratch_cfg
        try:
            cfg = {"model_root": str(root), "library": {}}
            records = {"beellama.cpp|qwen3.8-27b-IQ4_XS-pure.gguf|c98304": {"mb": 15420}}

            dlg = launcher.ModelLibraryDialog(cfg, records)

            # 1) scan happened on first open (empty cache → _ensure_scan).
            # All *.gguf files under root are cached; the decoy (not a real
            # GGUF) is cached with meta: None and shows as "(unindexed)"-style.
            lib = cfg.get("library", {})
            assert lib, "library cache was not written on first open"
            assert len(lib) == len(copied) + 1, f"expected {len(copied) + 1} cached, got {len(lib)}"
            decoy_key = str(root / "decoy.gguf")
            assert decoy_key in lib, "decoy missing from cache"
            assert lib[decoy_key].get("meta") is None, "decoy parsed as GGUF?!"
            for c in copied:
                k = str(c)
                assert k in lib and lib[k].get("meta"), f"real GGUF not indexed: {k}"
            names = {os.path.basename(p) for p in lib}
            print("cached:", sorted(names))

            # 2) list filled, entries visible
            assert dlg.list_w.count() >= 1, "list is empty"
            row_texts = [dlg.list_w.item(i).text() for i in range(dlg.list_w.count())]
            for t in row_texts:
                print("row:", t.encode("ascii", "backslashreplace").decode())

            # 3) projector tag applied to mmproj files
            tagged = [t for t in row_texts if "[projector]" in t]
            assert tagged, "no [projector] tag on mmproj rows"

            # 4) fit dot present on every row
            for t in row_texts:
                assert any(d in t for d in ("\U0001F7E2", "\U0001F7E1", "\U0001F534", "\u26AA")), t

            # 5) search filter narrows
            dlg.search_ed.setText("qwen")
            rows_qwen = dlg.list_w.count()
            dlg.search_ed.setText("")
            assert rows_qwen >= 1 and rows_qwen <= len(dlg._entries), "filter broken"

            # 6) pick → selected_path
            dlg.list_w.setCurrentRow(0)
            dlg._pick()
            assert dlg.selected_path, "nothing selected"
            assert Path(dlg.selected_path).exists(), "selected file vanished"
            print("selected:", dlg.selected_path)

            # 7) cache persistence contract: cfg["library"] paths exist on disk
            for p in lib:
                assert Path(p).exists(), f"cached path missing: {p}"

            print("ALL OK")
            return 0
        finally:
            launcher.CONFIG_FILE = real_config_file
            scratch_cfg.unlink(missing_ok=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
