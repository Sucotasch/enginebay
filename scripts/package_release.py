"""Package EngineBay release: deterministic zips.

Default — the main zip only: git-tracked files (code, presets, scripts,
docs). Engines (upstream/BeeLlama) download on demand via the GUI.

--ik — additionally package the OPTIONAL ik_llama asset (source-built
ik_llama.cpp/versions/15dddc6/ binaries). Policy (AGENTS.md): re-upload
ONLY when ik_llama moves to a new commit or a different CUDA arch;
otherwise releases just link the existing asset URL from README.

Usage: python scripts/package_release.py [--ver 3.0.0] [--ik] [--allow-dirty]
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def git(args: list[str]) -> str:
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"git {args}: {r.stderr.strip()}")
    return r.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", default=None, help="release version (default: git describe)")
    ap.add_argument("--ik", action="store_true",
                   help="also package the optional ik_llama binaries asset "
                        "(only when ik_llama is rebuilt — see AGENTS.md release policy)")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    status = git(["status", "--porcelain"])
    if status and not args.allow_dirty:
        print("REFUSE: working tree is dirty:")
        print(status)
        print("commit first (or --allow-dirty)")
        return 1

    ver = args.ver or git(["describe", "--tags", "--always"])
    ver = ver.lstrip("v")

    # ── asset 1: main archive (git-tracked files) ──
    files = [f for f in git(["ls-files"]).splitlines() if f.strip()]
    main_zip = REPO / f"enginebay-v{ver}-win64.zip"
    with zipfile.ZipFile(main_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(files):
            z.write(REPO / f, f"enginebay/{f}")
    print(f"main: {main_zip.name}  ({main_zip.stat().st_size / 1e6:.1f} MB, {len(files)} files)")

    if not args.ik:
        print("\nDefault: ik_llama asset NOT repackaged (one-time asset policy).")
        print("If ik_llama was rebuilt on a new commit/arch, re-run with --ik.")
        return 0

    # ── optional asset: ik_llama binaries (rebuild-only) ──
    ik_dir = REPO / "ik_llama.cpp" / "versions" / "15dddc6"
    if not ik_dir.is_dir():
        print(f"SKIP: {ik_dir} not found (ik_llama not built)")
        return 0
    ik_zip = REPO / f"ik_llama-v{ver}-win64.zip"
    with zipfile.ZipFile(ik_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(ik_dir.rglob("*")):
            if f.is_file() and f.name != "llama.log":
                z.write(f, f"ik_llama.cpp/versions/15dddc6/{f.relative_to(ik_dir)}")
    print(f"ik_llama: {ik_zip.name}  ({ik_zip.stat().st_size / 1e6:.1f} MB)")

    print(f"\nUpload the main zip (plus the ik zip) as release assets; update the")
    print(f"README ik_llama download link to the new asset URL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
