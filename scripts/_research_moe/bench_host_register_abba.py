"""ABBA A/B for GGML_CUDA_REGISTER_HOST on our MoE (order-bias controlled).

The first run of the previous A/B gave baseline 18.57 t/s while the historical
value WITHOUT the env var is 29.6 t/s, so run order / cold HDD cache is a
confound. This runs an ABBA sequence (REG, base, base, REG) so each condition is
sampled once early and once late.

Usage: python bench_host_register_abba.py [--model qw|gm] [--ncmoe N] [--reps 2]
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "_research_moe"))
sys.path.insert(0, str(REPO / "scripts"))
import bench_server_96k as bench  # noqa: E402

OUT = REPO / "scripts" / "_research_moe" / "host_register_abba.jsonl"
VAR = "GGML_CUDA_REGISTER_HOST"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qw", "gm"], default="qw")
    ap.add_argument("--ncmoe", type=int, default=21)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()

    model = bench.QWEN if a.model == "qw" else bench.GEMMA
    extra = ["--n-cpu-moe", str(a.ncmoe)]

    # ABBA: control order bias — each condition appears once early and once late
    if a.reps == 1:
        seq = [False, True]
    else:
        half = [False, True] * (a.reps // 2)
        seq = half + list(reversed(half))
        if a.reps % 2:
            seq.append(True)

    results = []
    for i, enabled in enumerate(seq, start=1):
        if enabled:
            os.environ[VAR] = "1"
        else:
            os.environ.pop(VAR, None)
        tag = "REG" if enabled else "BASE"
        label = f"{a.model}_ncmoe{a.ncmoe}_r{i}_{tag}"
        print(f"\n########## run {i}/{len(seq)}: {tag} (env={os.environ.get(VAR, '<unset>')}) ##########",
              flush=True)
        r = bench.run_cfg(label, model, extra)
        r["env_register_host"] = enabled
        r["order"] = i
        results.append(r)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n===== SUMMARY =====", flush=True)
    for r in results:
        print(f"  #{r['order']} {r['label']:<34} ok={r.get('ok')} "
              f"tg={r.get('tg')} pp={r.get('pp')} load={r.get('load_s')}s "
              f"err={r.get('error', '')}", flush=True)

    base = [r["tg"] for r in results if not r["env_register_host"] and r.get("tg")]
    reg = [r["tg"] for r in results if r["env_register_host"] and r.get("tg")]
    base_pp = [r["pp"] for r in results if not r["env_register_host"] and r.get("pp")]
    reg_pp = [r["pp"] for r in results if r["env_register_host"] and r.get("pp")]
    if base and reg:
        mb, mr = sum(base) / len(base), sum(reg) / len(reg)
        print(f"\n  tg  BASE mean={mb:.2f} {base}  REG mean={mr:.2f} {reg}  "
              f"delta={(mr - mb) / mb * 100:+.1f}%", flush=True)
    if base_pp and reg_pp:
        pb, pr = sum(base_pp) / len(base_pp), sum(reg_pp) / len(reg_pp)
        print(f"  pp  BASE mean={pb:.1f} {base_pp}  REG mean={pr:.1f} {reg_pp}  "
              f"delta={(pr - pb) / pb * 100:+.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())