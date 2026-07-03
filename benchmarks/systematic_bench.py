#!/usr/bin/env python3
"""
Systematic benchmark for Qwen3.6-27B IQ4_XS on RTX 4070 Ti SUPER.
Tests different configurations to find optimal speed/quality tradeoff.
"""
import subprocess
import time
import json
import sys
import os
import signal

LLAMA_SERVER = r"C:\Users\sucot\.cache\lm-studio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.20.1\llama-server.exe"
MODEL = r"G:\Ai\Models\Qwen3.6-27B.i1-IQ4\Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf"
PORT = 8099  # Use different port to avoid conflicts

TEST_PROMPT = "Write a detailed 500-word essay about the history of artificial intelligence."
TEST_MAX_TOKENS = 1024


def start_server(extra_args: str) -> subprocess.Popen:
    """Start llama-server with given extra arguments."""
    cmd = [LLAMA_SERVER, "-m", MODEL, "--port", str(PORT), "--host", "127.0.0.1"] + extra_args.split()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )
    return proc


def wait_for_server(timeout=60) -> bool:
    """Wait until server is healthy."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except:
            pass
        time.sleep(1)
    return False


def run_test() -> dict:
    """Run a single benchmark test and return results."""
    import urllib.request
    payload = json.dumps({
        "model": "Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf",
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": TEST_MAX_TOKENS,
        "temperature": 1.0,
    }).encode()
    
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    elapsed = time.time() - t0
    
    timings = data.get("timings", {})
    content = data["choices"][0]["message"].get("content", "")
    
    return {
        "prefill_tok_s": timings.get("prompt_per_second", 0),
        "decode_tok_s": timings.get("predicted_per_second", 0),
        "latency_ms": timings.get("predicted_per_token_ms", 0),
        "prompt_tokens": timings.get("prompt_n", 0),
        "generated_tokens": timings.get("predicted_n", 0),
        "total_time_s": elapsed,
        "content_len": len(content),
    }


def get_vram() -> str:
    """Get VRAM usage."""
    r = subprocess.run(
        "nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader",
        capture_output=True, text=True, shell=True
    )
    return r.stdout.strip()


def stop_server(proc: subprocess.Popen):
    """Stop the server process."""
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT)
        proc.wait(timeout=10)
    except:
        try:
            proc.kill()
        except:
            pass
    time.sleep(3)


def run_config(name: str, args: str) -> dict:
    """Run a full test with given config."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Args: {args}")
    print(f"{'='*60}")
    
    proc = start_server(args)
    
    if not wait_for_server(timeout=90):
        print(f"  FAILED: Server did not start")
        stop_server(proc)
        return {"name": name, "status": "FAILED"}
    
    vram = get_vram()
    print(f"  VRAM: {vram}")
    
    result = run_test()
    result["name"] = name
    result["status"] = "OK"
    result["vram"] = vram
    
    print(f"  Prefill:  {result['prefill_tok_s']:.1f} tok/s")
    print(f"  Decode:   {result['decode_tok_s']:.1f} tok/s")
    print(f"  Latency:  {result['latency_ms']:.1f} ms/token")
    print(f"  Generated: {result['generated_tokens']} tokens in {result['total_time_s']:.1f}s")
    
    stop_server(proc)
    return result


def main():
    if not os.path.exists(MODEL):
        print(f"CRITICAL ERROR: Model not found at {MODEL}")
        sys.exit(1)

    configs = [
        # Config 1: Baseline (user's command without --n-cpu-moe)
        ("baseline_no_moe", 
         "-c 131072 -np 1 -fa on -t 5 -ctk q4_0 -ctv q4_0 -b 512 -ub 512 --kv-unified -ngl 99 --jinja --reasoning off"),
        
        # Config 2: q8_0 KV cache (better quality, tiny KV anyway)
        ("kv_q8_0",
         "-c 131072 -np 1 -fa on -t 5 -ctk q8_0 -ctv q8_0 -b 512 -ub 512 --kv-unified -ngl 99 --jinja --reasoning off"),
        
        # Config 3: Larger batch for faster prefill
        ("batch_2048",
         "-c 131072 -np 1 -fa on -t 5 -ctk q4_0 -ctv q4_0 -b 2048 -ub 512 --kv-unified -ngl 99 --jinja --reasoning off"),
        
        # Config 4: 64K context (minimum requirement)
        ("ctx_64k",
         "-c 65536 -np 1 -fa on -t 5 -ctk q4_0 -ctv q4_0 -b 512 -ub 512 --kv-unified -ngl 99 --jinja --reasoning off"),
        
        # Config 5: More threads (8 instead of 5)
        ("threads_8",
         "-c 131072 -np 1 -fa on -t 8 -ctk q4_0 -ctv q4_0 -b 512 -ub 512 --kv-unified -ngl 99 --jinja --reasoning off"),
        
        # Config 6: 64K + q8_0 KV + batch 2048 (optimal combo?)
        ("ctx_64k_kv_q8_batch",
         "-c 65536 -np 1 -fa on -t 5 -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 --kv-unified -ngl 99 --jinja --reasoning off"),
        
        # Config 7: 128K + q8_0 KV (best quality at 128K)
        ("ctx_128k_kv_q8",
         "-c 131072 -np 1 -fa on -t 5 -ctk q8_0 -ctv q8_0 -b 512 -ub 512 --kv-unified -ngl 99 --jinja --reasoning off"),
    ]
    
    results = []
    for name, args in configs:
        result = run_config(name, args)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'Status':<8} {'Decode':<10} {'Prefill':<10} {'VRAM'}")
    print("-" * 80)
    for r in results:
        if r["status"] == "OK":
            print(f"{r['name']:<30} {r['status']:<8} {r['decode_tok_s']:<10.1f} {r['prefill_tok_s']:<10.1f} {r['vram']}")
        else:
            print(f"{r['name']:<30} {r['status']:<8}")
    
    # Find best
    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        best_decode = max(ok_results, key=lambda x: x["decode_tok_s"])
        print(f"\nBest decode speed: {best_decode['name']} — {best_decode['decode_tok_s']:.1f} tok/s")
        
        # Best with 64K+ context
        ctx_64k_plus = [r for r in ok_results if "64k" in r["name"] or "128k" in r["name"]]
        if ctx_64k_plus:
            best_64k = max(ctx_64k_plus, key=lambda x: x["decode_tok_s"])
            print(f"Best 64K+ decode:  {best_64k['name']} — {best_64k['decode_tok_s']:.1f} tok/s")


if __name__ == "__main__":
    main()
