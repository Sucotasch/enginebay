"""Preset usage notes (`desc`) + tooltip plumbing.

Contract:
  1. Every preset in launcher_presets.json carries a non-empty `desc` — the
     hover tooltip that says WHICH model to pair, WHICH engine it needs, and
     WHEN to use it.
  2. The two (MoE cache) presets must mention b11541 — the engine requirement
     is the one thing that hard-crashes a launch if missed (beellama rejects
     --moe-cache-mib and dies at startup).
  3. params/host/port are untouched shape; no model path leaks into params
     (presets are model-agnostic by design).
  4. launcher.py wires the descs into the GUI: per-item tooltips on the combo,
     a synced tooltip on the closed combo when a preset is selected, and
     _save_preset PRESERVES the existing desc on overwrite (editable prompt)
     instead of writing a fresh {params, host, port} dict.
  5. Live offscreen check: the real LLMLauncher window's combo carries a
     tooltip on every preset row.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

failures = []

# ── 1-3: the data contract ─────────────────────────────────────────────
presets = json.loads((REPO / "launcher_presets.json").read_text(encoding="utf-8"))
for name, p in presets.items():
    desc = (p.get("desc") or "").strip()
    if len(desc) < 40:
        failures.append(f"{name}: desc too short/missing ({len(desc)} chars)")
    if not {"params", "host", "port"} <= set(p):
        failures.append(f"{name}: missing params/host/port")
    low = p.get("params", "").lower()
    if "g:\\" in low or "g:/" in low:
        failures.append(f"{name}: model path leaked into params")

for name in ("Qwen3.6-35B-A3B (MoE cache)", "Gemma-4-26B-A4B (MoE cache)"):
    if "b11541" not in (presets.get(name, {}).get("desc") or ""):
        failures.append(f"{name}: desc must state the upstream >= b11541 requirement")
    if "--moe-cache-mib" not in presets.get(name, {}).get("params", ""):
        failures.append(f"{name}: params lost the --moe-cache-mib flag?")

# ── 4: the source wiring ───────────────────────────────────────────────
src = (REPO / "launcher.py").read_text(encoding="utf-8")
if "Qt.ItemDataRole.ToolTipRole" not in src:
    failures.append("combo items carry no ToolTipRole data")
if "_preset_tooltip" not in src:
    failures.append("_preset_tooltip helper missing")
if 'existing.get("desc")' not in src:
    failures.append("_save_preset does not preserve the existing desc on overwrite")
if "getMultiLineText" not in src:
    failures.append("_save_preset offers no desc editing prompt")
if 'self.preset_combo.setToolTip(self._preset_tooltip(name))' not in src:
    failures.append("selected-preset tooltip not synced to the closed combo")

# ── 5: live offscreen window ───────────────────────────────────────────
from PyQt6.QtWidgets import QApplication  # noqa: E402
import launcher  # noqa: E402

app = QApplication.instance() or QApplication([])
win = launcher.LLMLauncher()
n = win.preset_combo.count()
# +1 for the "(none)" item
if n != len(presets) + 1:
    failures.append(f"combo has {n} items, expected {len(presets) + 1}")
empty = [win.preset_combo.itemText(i)
         for i in range(1, n)
         if not (win.preset_combo.itemData(i, launcher.Qt.ItemDataRole.ToolTipRole) or "").strip()]
if empty:
    failures.append(f"rows without tooltip: {empty}")
# select a real preset -> the closed combo must show its note too
win.preset_combo.setCurrentText("Qwen3.8-27B (ik_llama pure)")
if "35.9" not in (win.preset_combo.toolTip() or ""):
    failures.append("closed-combo tooltip missing after selection")
win.close()

if failures:
    print("FAIL:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"OK: {len(presets)} presets annotated, tooltips wired "
      f"({n - 1} rows + closed-combo sync), desc survives re-save")
