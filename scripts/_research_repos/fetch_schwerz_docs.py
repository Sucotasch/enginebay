"""Fetch GenerelSchwerz/llama.cpp fork docs + wiki pages (the MoE-cache lineage).

The fork's docs live in-tree; the guides that matter most for a 16GB Windows
box live in the GitHub wiki, which is a separate git repo.
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

SLUG = "GenerelSchwerz/llama.cpp"
BRANCH = "moe-cache"

DOCS = [
    "docs/fork-features.md",
    "docs/moe-hybrid.md",
    "docs/moe-grouped-multigpu.md",
    "docs/build.md",
]


def gh(args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else ""


def fetch_file(path, branch=BRANCH):
    b64 = gh(["api", f"repos/{SLUG}/contents/{path}?ref={branch}", "--jq", ".content"])
    if not b64.strip():
        return None
    try:
        return base64.b64decode(b64.replace("\n", "")).decode("utf-8", "replace")
    except Exception as e:
        return f"<<decode failed: {e}>>"


def main():
    d = OUT / "schwerz-docs"
    d.mkdir(exist_ok=True)

    for p in DOCS:
        txt = fetch_file(p)
        if txt is None:
            print(f"  MISSING: {p}")
            continue
        dst = d / p.replace("/", "__")
        dst.write_text(txt, encoding="utf-8")
        print(f"  {p}: {len(txt)} chars -> {dst.name}")

    # wiki page list
    w = gh(["api", "repos/GenerelSchwerz/llama.cpp/wiki/pages"])
    print("\n  wiki api:", "ok" if w else "unavailable via api")
    if w:
        (d / "_wiki_pages.json").write_text(w, encoding="utf-8")

    # try fetching the key wiki pages as markdown from the wiki git repo
    wiki_pages = [
        "Windows-WDDM-Partial-Pinning",
        "MoE-Cache",
        "Benchmark-Comparison-Showcase",
        "BeeLlama-Main-Flags-and-16GB-Setup",
    ]
    wd = OUT / "schwerz-wiki"
    wd.mkdir(exist_ok=True)
    for pg in wiki_pages:
        url = f"https://raw.githubusercontent.com/wiki/{SLUG}/{pg}.md"
        r = subprocess.run(["curl", "-sSL", "--max-time", "40", url],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        body = r.stdout or ""
        if body.strip() and "404" not in body[:200]:
            (wd / f"{pg}.md").write_text(body, encoding="utf-8")
            print(f"  wiki {pg}: {len(body)} chars")
        else:
            print(f"  wiki {pg}: not fetched (rc={r.returncode})")

    print("\ndone ->", OUT)


if __name__ == "__main__":
    sys.exit(main())