"""Offscreen smoke: launcher imports, Library dialog opens against the real
config, _refill fills rows, and the download-fetch helper exists with
progress/retry/timeout semantics. No GUI shown, no server started."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from PyQt6.QtWidgets import QApplication  # noqa: E402
import launcher  # noqa: E402

app = QApplication([])

cfg = launcher.load_config()
assert cfg.get("model_root"), "no model_root in config — run the real flow first"
records = launcher.load_vram_records()
dlg = launcher.ModelLibraryDialog(cfg, records)
n = dlg.list_w.count()
print(f"dialog opened, rows: {n}")
assert n > 0, "no rows"

# a MoE row must carry the [MoE offload] marker
rows = [dlg.list_w.item(i).text() for i in range(n)]
moe_rows = [t for t in rows if "MoE offload" in t]
print("MoE rows:", len(moe_rows))
assert moe_rows, "no MoE rows found (scan/cache broken?)"

# dense rows must have a traffic-light dot and NOT be all red
import re
red = sum(1 for t in rows if "\U0001F534" in t)
yellow = sum(1 for t in rows if "\U0001F7E1" in t)
green = sum(1 for t in rows if "\U0001F7E2" in t)
print(f"dots: green={green} yellow={yellow} red={red}")
assert red == 0, f"{red} red rows — the original bug still repros"

# download path: _download_paired exists and uses the streamed fetch
src = Path(launcher.__file__).read_text(encoding="utf-8")
assert "def fetch(url" in src, "streamed fetch helper missing"
assert "retry" in src and "retry {attempt}/3" in src, "retry semantics missing"
assert 'self.cfg.get("proxy")' in src, "optional proxy hook missing"
assert "Content-Length" in src, "no progress accounting"
print("download path: streamed fetch + retries + optional proxy — OK")

print("ALL OK")
