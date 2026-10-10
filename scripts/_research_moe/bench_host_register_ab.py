"""A/B: GGML_CUDA_REGISTER_HOST=1 (host-pinned CPU expert weights) on our MoE.

thecodacus/llama.cpp presented this as its fork's headline optimization, but
the env var is ALREADY in upstream ggml-cuda.cu and in beellama. If it helps
our Qwen MoE prefill, we get the win with zero code and zero new engine.

The child process inherits the parent's environment, so setting os.environ
here is enough.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "_research_moe"))
sys.path.insert(0, str(REPO / "scripts"))
import bench_server_96k as bench  # noqa: E402

OUT = REPO / "scripts" / "_research_moe" / "host_register_ab.jsonl"
VAR = "GGML_CUDA_REGISTER_HOST"


def main() -> int:
    model = bench.QWEN
    ncmoe = 21
    extra = ["--n-cpu-moe", str(ncmoe)]
    results = []

    for label, enabled in (("baseline", False), ("REGISTER_HOST=1", True)):
        if enabled:
            os.environ[VAR] = "1"
        else:
            os.environ.pop(VAR, None)
        print(f"\n########## {label} (env {VAR}={os.environ.get(VAR, '<unset>')}) ##########",
              flush=True)
        r = bench.run_cfg(f"qw_ncmoe{ncmoe}_96k_{label}", model, extra)
        r["env_register_host"] = enabled
        results.append(r)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n===== SUMMARY =====", flush=True)
    for r in results:
        print(f"  {r['label']:<32} ok={r.get('ok')} "
              f"tg={r.get('tg')} pp={r.get('pp')} free={r.get('free_mb')} "
              f"err={r.get('error', '')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())