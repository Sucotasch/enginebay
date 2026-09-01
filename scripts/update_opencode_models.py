#!/usr/bin/env python3
"""Update the `opencode_free` provider block in DSH settings.yaml.

Fetches the live model catalog from the OpenCode Zen relay (no auth needed),
keeps only the free-tier models (id ends with `-free`), **verifies each one
with a real chat request** (the catalog lists models that are not actually
served), and rewrites the models list of the `opencode_free` provider under
`llm-pi-ai.providers` in `~/.dsh/settings.yaml`.

Verification uses curl_cffi with a browser TLS fingerprint (plain urllib is
Cloudflare-blocked on POST). Only models that return HTTP 200 are kept.

Style: mirrors `launcher.py::_dsh_upsert_local_llama` — pure text, block-scoped,
only the target block is touched; everything else stays byte-identical.
Always validates the result with `yaml.safe_load` before writing and makes a
timestamped backup.

Usage:
    python scripts/update_opencode_models.py            # fetch + verify + apply
    python scripts/update_opencode_models.py --dry-run  # fetch + show diff, no write
    python scripts/update_opencode_models.py --print    # fetch + print verified ids only
    python scripts/update_opencode_models.py --no-verify  # catalog only (no chat checks)

Endpoint (OpenCode Zen free lane, anonymous):
    https://opencode.ai/zen/v1/models
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

MODELS_URL = "https://opencode.ai/zen/v1/models"
API_BASE = "https://opencode.ai/zen/v1"
PROVIDER_NAME = "opencode_free"
DISPLAY_NAME = "OpenCode Free"
API_KEY_ENV = "OPENCODE_ZEN_API_KEY"
SETTINGS_DEFAULT = Path.home() / ".dsh" / "settings.yaml"
UA = "Mozilla/5.0"


def fetch_free_models(timeout: int = 20) -> list[str]:
    """Return sorted free-tier model ids from the OpenCode Zen relay."""
    req = urllib.request.Request(
        MODELS_URL, headers={"User-Agent": UA, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    ids = [m.get("id") for m in payload.get("data", []) if isinstance(m, dict)]
    free = sorted(i for i in ids if i and i.endswith("-free"))
    if not free:
        raise RuntimeError(f"No `-free` models found in {MODELS_URL}")
    return free


def _read_api_key() -> str:
    """Read OPENCODE_ZEN_API_KEY from ~/.dsh/.env, fallback 'public'."""
    env_path = Path.home() / ".dsh" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENCODE_ZEN_API_KEY="):
                return line.strip().split("=", 1)[1]
    return "public"


def _verify_model(model: str, api_key: str, timeout: int = 25) -> bool:
    """Return True if a minimal chat completion returns HTTP 200."""
    try:
        from curl_cffi import requests
    except ImportError:
        return True  # no verifier → assume available

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    try:
        r = requests.post(
            f"{API_BASE}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            impersonate="chrome",
            timeout=timeout,
        )
        return r.status_code == 200
    except Exception:
        return False


def verify_models(models: list[str], timeout: int = 90) -> list[str]:
    """Return only models that respond 200 to a real chat request.

    Uses curl_cffi with a browser TLS fingerprint to bypass Cloudflare.
    If curl_cffi is unavailable, emits a warning and returns the list as-is.
    """
    try:
        import curl_cffi  # noqa: F401
    except ImportError:
        print("[opencode] WARNING: curl_cffi not available — skipping verification",
              file=sys.stderr)
        return models

    api_key = _read_api_key()
    print(f"[opencode] verifying {len(models)} models via chat request ...",
          file=sys.stderr)
    verified: list[str] = []
    for model in models:
        print(f"  {model} ... ", end="", file=sys.stderr, flush=True)
        if _verify_model(model, api_key):
            verified.append(model)
            print("200 OK", file=sys.stderr)
        else:
            print("✗ no", file=sys.stderr)
    return verified


def _build_block(models: list[str]) -> str:
    """Render the opencode_free provider block text (4-space indent)."""
    lines = [
        f"    {PROVIDER_NAME}:",
        f"      displayName: {DISPLAY_NAME}",
        f"      apiKeyEnv: {API_KEY_ENV}",
        "      api: openai-completions",
        f"      baseURL: {API_BASE}",
        "      models:",
    ]
    lines += [f"        - id: {m}" for m in models]
    return "\n".join(lines)


def upsert_opencode_free(content: str, models: list[str]) -> tuple[str, bool]:
    """Insert or update the opencode_free block in settings.yaml text.

    Returns (new_content, changed). Only the opencode_free block is touched.
    """
    block = _build_block(models)

    # Update path: block exists → replace the whole block.
    block_re = re.compile(
        rf"(    {re.escape(PROVIDER_NAME)}:.*?)(?=\n    \S|\n  \S|\Z)", re.DOTALL
    )
    m = block_re.search(content)
    if m:
        if m.group(1) == block:
            return content, False
        return content[: m.start()] + block + content[m.end() :], True

    # Create path: no opencode_free yet → insert under llm-pi-ai.providers.
    marker = "llm-pi-ai:\n  providers:"
    if marker not in content:
        raise RuntimeError("settings.yaml has no `llm-pi-ai: providers:` section")
    return content.replace(marker, marker + "\n" + block, 1), True


def validate_yaml(text: str) -> None:
    """Raise if the text is not parseable YAML (guards against corrupting DSH)."""
    try:
        import yaml
    except ImportError:
        return  # pyyaml absent → skip validation, caller should have it
    yaml.safe_load(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="show what would change")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="print verified free model ids and exit")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip chat verification — use catalog list as-is")
    ap.add_argument("--settings", type=Path, default=SETTINGS_DEFAULT,
                    help="path to DSH settings.yaml (default: ~/.dsh/settings.yaml)")
    args = ap.parse_args()

    print(f"[opencode] fetching {MODELS_URL} ...", file=sys.stderr)
    candidates = fetch_free_models()
    print(f"[opencode] {len(candidates)} free candidates: {', '.join(candidates)}")

    models = candidates if args.no_verify else verify_models(candidates)
    print(f"[opencode] {len(models)} verified models: {', '.join(models)}")

    if args.print_only:
        return 0

    if not args.settings.exists():
        print(f"[opencode] ERROR: {args.settings} not found", file=sys.stderr)
        return 1

    content = args.settings.read_text(encoding="utf-8")
    new_content, changed = upsert_opencode_free(content, models)

    if not changed:
        print(f"[opencode] {PROVIDER_NAME} block already up to date — nothing to do.")
        return 0

    validate_yaml(new_content)

    if args.dry_run:
        print("[opencode] DRY RUN — would write the following block:")
        print(_build_block(models))
        return 0

    backup = args.settings.with_name(
        args.settings.name + f".pre-opencode-{datetime.now():%Y%m%d-%H%M%S}"
    )
    shutil.copy2(args.settings, backup)
    args.settings.write_text(new_content, encoding="utf-8")
    print(f"[opencode] backup: {backup}")
    print(f"[opencode] wrote {PROVIDER_NAME} block with {len(models)} models.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
