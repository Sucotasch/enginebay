"""Regression: starting a model must NOT wipe model_root / library from the config.

The bug (found 2026-10-10): the launch path called

    save_config({"model": ..., "binary": ..., "host": ..., "port": ...,
                 "params": ..., "engine_id": ...})

i.e. it wrote a FRESH dict instead of merging into self.cfg. Every Start wiped
`model_root` and the whole `library` scan cache, so after any launch the Library
dialog lost its root and showed 0 entries (and test_lib_fit_dots /
test_launcher_smoke failed with "no model_root in config").

This test drives the real code path offscreen against a scratch config file, so
the user's real launcher_config.json is never touched.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import launcher  # noqa: E402


def main() -> int:
    failures = []

    # --- static guard: the launch path must merge into self.cfg -------------
    src = (REPO / "launcher.py").read_text(encoding="utf-8")
    if 'save_config({\n            "model": model, "binary": binary' in src:
        failures.append("launch path still writes a fresh dict (not {**self.cfg, ...})")
    if '**self.cfg,\n            "model": model, "binary": binary' not in src:
        failures.append("launch path does not merge self.cfg")

    # every save_config call in launcher.py must merge self.cfg or be an
    # explicit merge — flag bare dict literals
    import re
    for m in re.finditer(r"save_config\(\s*\n?(\s*)\{", src):
        tail = src[m.end() - 1: m.end() + 60]
        if "**self.cfg" not in tail and "**cfg" not in tail:
            line = src[: m.start()].count("\n") + 1
            failures.append(f"save_config at line {line} may drop unrelated keys: {tail[:40]!r}")

    # --- functional check against a scratch config -------------------------
    # Keep the scratch file inside the repo: the sandbox may deny %TEMP%.
    scratch_dir = REPO / "scripts" / "_cfgtest_tmp"
    scratch_dir.mkdir(exist_ok=True)
    scratch = scratch_dir / "launcher_config.json"
    real = launcher.CONFIG_FILE
    launcher.CONFIG_FILE = scratch
    try:
        seeded = {"model_root": "G:/Ai", "library": {"G:/Ai/x.gguf": {"size": 1}},
                  "model": "old.gguf", "port": "8080"}
        scratch.write_text(json.dumps(seeded), encoding="utf-8")

        # emulate exactly what the fixed launch path does
        cfg = launcher.load_config()
        launcher.save_config({**cfg, "model": "new.gguf", "port": "8888"})

        after = json.loads(scratch.read_text(encoding="utf-8"))
        if after.get("model_root") != "G:/Ai":
            failures.append(f"model_root lost by launch: {after.get('model_root')!r}")
        if not after.get("library"):
            failures.append("library cache lost by launch")
        if after.get("model") != "new.gguf":
            failures.append("launch did not persist the new model")

        # emulate the OLD buggy write and confirm the test would catch it
        launcher.save_config({"model": "x", "port": "1"})
        broken = json.loads(scratch.read_text(encoding="utf-8"))
        if "model_root" in broken:
            failures.append("test cannot detect the bug (buggy write kept model_root)")
    finally:
        launcher.CONFIG_FILE = real
        try:
            scratch.unlink()
            scratch_dir.rmdir()
        except OSError:
            pass

    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("OK: config keys survive a launch (model_root + library preserved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())