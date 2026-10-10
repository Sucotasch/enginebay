"""Fetch repo intel (README, key files, issues) for the integration review.

Uses the authenticated gh CLI (5000 req/h) instead of anonymous scraping.
Writes raw material to scripts/_research_repos/ so it can be read selectively.
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

REPOS = {
    "llama-swap": "mostlygeek/llama-swap",
    "local-ai-configs": "thecodacus/local-ai-configs",
    "edge0": "Edge0-AI/Edge0",
    "strata": "Niko1221/Strata",
    "codacus-llamacpp": "thecodacus/llama.cpp",
    "schwerz-llamacpp": "GenerelSchwerz/llama.cpp",
}


def gh(args: list[str]) -> str:
    r = subprocess.run(["gh", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return ""
    return r.stdout


def readme(slug: str) -> str:
    b64 = gh(["api", f"repos/{slug}/readme", "--jq", ".content"]).strip()
    if not b64:
        return ""
    try:
        return base64.b64decode(b64).decode("utf-8", "replace")
    except Exception as e:
        return f"<<decode failed: {e}>>"


def issues(slug: str, state: str, n: int = 40) -> list[dict]:
    raw = gh(["api",
              f"repos/{slug}/issues?state={state}&sort=comments&direction=desc&per_page={n}"])
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except Exception:
        return []
    out = []
    for it in items:
        if "pull_request" in it:
            continue
        out.append({
            "number": it["number"],
            "title": it["title"],
            "comments": it.get("comments", 0),
            "state": it["state"],
            "created": it["created_at"][:10],
            "labels": [l["name"] for l in it.get("labels", [])],
            "body": (it.get("body") or "")[:1200],
        })
    return out


def tree(slug: str, branch: str) -> list[str]:
    raw = gh(["api", f"repos/{slug}/git/trees/{branch}?recursive=1"])
    if not raw:
        return []
    try:
        t = json.loads(raw)
    except Exception:
        return []
    return [e["path"] for e in t.get("tree", []) if e["type"] == "blob"]


def file_content(slug: str, path: str) -> str:
    b64 = gh(["api", f"repos/{slug}/contents/{path}", "--jq", ".content"]).strip()
    if not b64:
        return ""
    try:
        return base64.b64decode(b64.replace("\n", "")).decode("utf-8", "replace")
    except Exception as e:
        return f"<<decode failed: {e}>>"


def main():
    summary = {}
    for name, slug in REPOS.items():
        print(f"--- {name} ({slug})")
        d = OUT / name
        d.mkdir(exist_ok=True)

        rm = readme(slug)
        (d / "README.md").write_text(rm, encoding="utf-8")
        print(f"    README: {len(rm)} chars")

        # default branch + tree (first 400 paths)
        branch = gh(["api", f"repos/{slug}", "--jq", ".default_branch"]).strip() or "main"
        tr = tree(slug, branch)
        (d / "tree.txt").write_text("\n".join(tr), encoding="utf-8")
        print(f"    branch={branch}  files={len(tr)}")

        for state in ("open", "closed"):
            iss = issues(slug, state)
            (d / f"issues_{state}.json").write_text(
                json.dumps(iss, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"    issues {state}: {len(iss)}")

        summary[name] = {"slug": slug, "branch": branch, "readme_chars": len(rm),
                         "files": len(tr)}

    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\ndone ->", OUT)


if __name__ == "__main__":
    sys.exit(main())