"""Fetch the concrete artifacts of the local-ai-configs stack + thecodacus metadata.

local-ai-configs is the reference "llama-swap in front of llama.cpp router"
recipe; its config.yaml and models.ini are the actual working files.
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
D = OUT / "local-ai-configs"


def gh(args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else ""


def get(slug, path):
    b64 = gh(["api", f"repos/{slug}/contents/{path}", "--jq", ".content"])
    if not b64.strip():
        return None
    try:
        return base64.b64decode(b64.replace("\n", "")).decode("utf-8", "replace")
    except Exception as e:
        return f"<<decode failed: {e}>>"


def main():
    D.mkdir(exist_ok=True)
    slug = "thecodacus/local-ai-configs"
    for p in ["llama-swap/config.yaml", "llama-swap/models.ini",
              "llama-swap/README.md", "llama-swap/Dockerfile",
              "llama-swap/strata/strata.sh"]:
        txt = get(slug, p)
        if txt is None:
            print(f"  MISSING {p}")
            continue
        fname = p.replace("/", "__")
        (D / fname).write_text(txt, encoding="utf-8")
        print(f"  {p}: {len(txt)} chars -> {fname}")

    print("\n=== thecodacus/llama.cpp metadata ===")
    meta = gh(["api", "repos/thecodacus/llama.cpp",
               "--jq", "{stars: .stargazers_count, forks: .forks_count, "
                       "open: .open_issues_count, created: .created_at, "
                       "pushed: .pushed_at, branch: .default_branch, "
                       "homepage: .homepage, license: .license.spdx_id}"])
    print(meta)
    rel = gh(["api", "repos/thecodacus/llama.cpp/releases"])
    print("releases:", rel.strip()[:200] or "NONE")
    wf = gh(["api", "repos/thecodacus/llama.cpp/contents/.github/workflows?ref=perf",
             "--jq", "[.[].name] | join(\", \")"])
    print("workflows:", wf.strip() or "(none)")
    # the trace tool + cache source files existence
    for p in ["src/llama-moe-cache.cpp", "tools/moe-trace/moe-trace.cpp",
              "src/llama-moe-trace.cpp"]:
        r = gh(["api", f"repos/thecodacus/llama.cpp/contents/{p}?ref=perf",
                "--jq", ".size"])
        print(f"  {p}: {r.strip() or 'absent'}")
    print("\ndone ->", D)


if __name__ == "__main__":
    sys.exit(main())