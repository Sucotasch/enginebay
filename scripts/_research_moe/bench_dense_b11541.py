"""Dense Qwen3.8-27B on b11541 (has the new GDN warp kernel #30087, merged
2026-10-08) vs our current beellama v0.4.7 — same model file (IQ4_XS-pure,
readable by both engines), same params (the "Qwen3.8-27B (llama.cpp upstream)"
preset), to answer: did upstream drift + the GDN kernel make the upstream
engine faster for the dense qwen35 arch on THIS box?

Reference points: ik_llama + IQ4_KT = 35.9 t/s @96K (different quant, quality
reference); the fair engine-vs-engine number is beellama vs b11541 below.
"""
import json
import shlex
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "_research_moe"))
sys.path.insert(0, str(REPO / "scripts"))
import bench_server_96k as bench  # noqa: E402

OUT = REPO / "scripts" / "_research_moe" / "dense_b11541.jsonl"
MODEL = Path(r"G:\Ai\Models\qwen3.8-27b-IQ4_XS\qwen3.8-27b-IQ4_XS-pure.gguf")

# Exact params of the "Qwen3.8-27B (llama.cpp upstream)" preset
PARAMS = ('-c 98304 -np 1 -ngl 99 -b 2048 -ub 512 --kv-unified '
          '--cache-type-k q4_0 --cache-type-v q4_0 -t 5 -tb 6 --flash-attn on '
          '--jinja --reasoning auto --load-mode none '
          '--override-tensor "blk\\.(6|14|22|30|38|46|54|62)\\.ffn_.*=CPU" '
          '--temp 1.0 --min-p 0.0 --top-p 0.95 --top-k 20 '
          '--presence-penalty 0.0 --repeat-penalty 1.0 --no-mmproj-offload')

ENGINES = {
    "beellama_v047": REPO / "beellama.cpp" / "versions" / "preview-v0.4.7-cuda-13.3" / "llama-server.exe",
    "b11541": REPO / "llama.cpp" / "versions" / "b11541-cuda-13.4" / "llama-server.exe",
}


def run(engine_tag: str, srv: Path, extra: list[str]) -> dict:
    bench.SRV = srv
    r = bench.run_cfg(f"dense_IQ4XS_{engine_tag}", MODEL, extra)
    r["engine"] = engine_tag
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    time.sleep(5)
    return r


def main() -> int:
    if not MODEL.exists():
        print(f"model missing: {MODEL}")
        return 1
    # Use ONLY the preset params: neutralize the harness BASE
    bench.BASE = []
    extra = shlex.split(PARAMS)

    for tag, srv in ENGINES.items():
        if not srv.exists():
            print(f"engine missing: {srv}")
            return 1
        r = run(tag, srv, extra)
        print(f"  {tag}: ok={r.get('ok')} tg={r.get('tg')} pp={r.get('pp')} "
              f"free={r.get('free_mb')} err={r.get('error', '')}", flush=True)

    print("\n===== DENSE SUMMARY (ik_llama IQ4_KT reference: 35.9 t/s) =====", flush=True)
    for line in OUT.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        print(f"  {r['label']:<28} tg={r.get('tg')} pp={r.get('pp')} "
              f"free={r.get('free_mb')} load={r.get('load_s')}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())