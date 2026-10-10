"""Dump full text of specific issues (body + comments) for the repo review."""
import json
import subprocess
import sys
from pathlib import Path

TARGETS = [
    ("mostlygeek/llama-swap", 426),   # New in llama.cpp: Model Management
    ("mostlygeek/llama-swap", 1004),  # memory-constrained scheduling
    ("mostlygeek/llama-swap", 1127),  # swappable: false
    ("GenerelSchwerz/llama.cpp", 90),  # MoE cache pinned to GPU0
    ("GenerelSchwerz/llama.cpp", 151),  # moe-expert-profile crash on Qwen3.6-35B-A3B
    ("GenerelSchwerz/llama.cpp", 104),  # corrupted output MoE cache Windows
    ("GenerelSchwerz/llama.cpp", 113),  # implement KVarN
]


def gh(args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else ""


def main():
    lines: list[str] = []

    def out(s: str = "") -> None:
        lines.append(s)
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode("ascii", "replace").decode("ascii"))

    for slug, num in TARGETS:
        out("=" * 100)
        out(f"### {slug}#{num}")
        raw = gh(["api", f"repos/{slug}/issues/{num}"])
        if not raw:
            out("  <<could not fetch>>")
            continue
        it = json.loads(raw)
        out(f"TITLE: {it['title']}")
        out(f"STATE: {it['state']}  comments={it.get('comments')}  "
            f"created={it['created_at'][:10]}  closed={str(it.get('closed_at'))[:10]}")
        out(f"LABELS: {[l['name'] for l in it.get('labels', [])]}")
        out("--- BODY ---")
        out((it.get("body") or "")[:3000])
        craw = gh(["api", f"repos/{slug}/issues/{num}/comments?per_page=100"])
        if craw:
            try:
                cs = json.loads(craw)
            except Exception:
                cs = []
            for c in cs:
                out(f"--- COMMENT by {c['user']['login']} ({c['created_at'][:10]}) ---")
                out((c.get("body") or "")[:2000])
        out()

    dst = Path(__file__).resolve().parent / "_key_issues.txt"
    dst.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    sys.exit(main())