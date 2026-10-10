"""A/B the new upstream MoE expert cache (--moe-cache-mib, PR #29887) on our
two MoE presets, using llama.cpp b11541 — the first nightly with the cache.

Ladder per model:
  baseline  : our current placement (qw ncmoe 21 / gm ncmoe 6), no cache
  cmoe+cache: ALL experts to CPU (-cmoe) + --moe-cache-mib N, N grown while
              >1.5 GB stays free after load.

Compare against the standing beellama records (qw 29.6 t/s, gm 47.5 t/s).
Cache serves decode batches only (<=32 tokens) — pp uses the regular path
and is expected to DROP with -cmoe; tg is the target metric.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "_research_moe"))
sys.path.insert(0, str(REPO / "scripts"))
import bench_server_96k as bench  # noqa: E402

SRV = REPO / "llama.cpp" / "versions" / "b11541-cuda-13.4" / "llama-server.exe"
OUT = REPO / "scripts" / "_research_moe" / "moe_cache_b11541.jsonl"

MODELS = {
    "qw": dict(model=bench.QWEN, base_ncmoe=21, cache_ladder=[9000, 10500]),
    "gm": dict(model=bench.GEMMA, base_ncmoe=6, cache_ladder=[8500, 9500]),
}


def check_flags() -> list[str]:
    """Verify b11541 accepts our base args + the cache flag. Returns problems."""
    h = subprocess.run([str(SRV), "--help"], capture_output=True, text=True,
                      encoding="utf-8", errors="replace").stdout
    need = ["--moe-cache-mib", "--load-mode", "--kv-unified", "--n-cpu-moe",
            "--cpu-moe", "--cache-type-k"]
    problems = [f for f in need if f not in h]
    if "--warmup" not in h:
        # strip unsupported flag from the shared BASE
        bench.BASE = [a for a in bench.BASE if a != "--warmup"]
    return problems


def run(label: str, model: Path, extra: list[str]) -> dict:
    bench.SRV = SRV  # point the harness at b11541
    r = bench.run_cfg(label, model, extra)
    r["engine"] = "b11541-cuda-13.4"
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    time.sleep(5)  # port + VRAM settle
    return r


def main() -> int:
    if not SRV.exists():
        print(f"binary missing: {SRV}")
        return 1
    problems = check_flags()
    if problems:
        print("FLAG PROBLEMS:", problems)
        return 1
    print("flag check OK (--moe-cache-mib present)", flush=True)

    for tag, spec in MODELS.items():
        model, base_ncmoe, ladder = spec["model"], spec["base_ncmoe"], spec["cache_ladder"]

        # baseline: current placement, no cache
        r0 = run(f"{tag}_ncmoe{base_ncmoe}_nocache", model, ["--n-cpu-moe", str(base_ncmoe)])
        print(f"  baseline: tg={r0.get('tg')} pp={r0.get('pp')} free={r0.get('free_mb')}",
              flush=True)

        # cache ladder: -cmoe + growing cache
        prev_free = None
        for mib in ladder:
            label = f"{tag}_cmoe_cache{mib}"
            r = run(label, model, ["-cmoe", "--moe-cache-mib", str(mib)])
            print(f"  {label}: tg={r.get('tg')} pp={r.get('pp')} "
                  f"free={r.get('free_mb')} err={r.get('error','')}", flush=True)
            if not r.get("ok"):
                print("  run failed — stop growing the cache", flush=True)
                break
            free = r.get("free_mb") or 0
            if free < 1500:
                print(f"  only {free} MB free — do not grow further", flush=True)
                break
            prev_free = free

    print("\n===== FINAL SUMMARY (beellama records: qw 29.6 / gm 47.5 t/s) =====",
          flush=True)
    for line in OUT.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        print(f"  {r['label']:<28} ok={r.get('ok')!s:5} tg={r.get('tg')} "
              f"pp={r.get('pp')} free={r.get('free_mb')} load={r.get('load_s')}s",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())