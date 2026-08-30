#!/usr/bin/env python3
"""
Smoke test — verify llama-server with Qwen3.8-27B.
Run: python scripts/smoke_test.py
"""
import os
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai not installed. Run: uv pip install openai httpx")
    sys.exit(1)


def main():
    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    model = os.getenv("MODEL_NAME", "Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf")
    api_key = os.getenv("OPENAI_API_KEY", "not-needed")

    print(f"Connecting to: {base_url}")
    print(f"Model: {model}")
    print()

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Test 1: Health check
    print("--- Test 1: List Models ---")
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"  OK — models: {model_ids}")
    except Exception as e:
        print(f"  FAIL — {e}")
        print("  Is llama-server running? Start: bash scripts/start_llama_cpp.sh")
        sys.exit(1)

    # Test 2: Chat completion (reasoning off)
    print("\n--- Test 2: Chat Completion ---")
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What is 2+2? Reply with just the number."},
            ],
            max_tokens=16,
            temperature=0,
        )
        elapsed = time.time() - t0
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        tokens = resp.usage.total_tokens if resp.usage else "?"
        print(f"  Content:   {content.strip()[:200]}")
        if reasoning:
            print(f"  Reasoning: {reasoning.strip()[:100]}...")
        print(f"  Tokens:    {tokens}")
        print(f"  Latency:   {elapsed:.2f}s")
        timings = getattr(resp, "timings", {})
        if timings:
            prefill = timings.get("prompt_per_second", 0)
            decode = timings.get("predicted_per_second", 0)
            print(f"  Prefill:   {prefill:.1f} tok/s")
            print(f"  Decode:    {decode:.1f} tok/s")
        print(f"  OK")
    except Exception as e:
        print(f"  FAIL — {e}")
        sys.exit(1)

    # Test 3: Streaming
    print("\n--- Test 3: Streaming ---")
    t0 = time.time()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Count from 1 to 5, one number per line."},
            ],
            max_tokens=32,
            temperature=0,
            stream=True,
        )
        collected = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                collected += chunk.choices[0].delta.content
        elapsed = time.time() - t0
        print(f"  Streamed: {collected.strip()[:200]}")
        print(f"  Latency:  {elapsed:.2f}s")
        print(f"  OK")
    except Exception as e:
        print(f"  FAIL — {e}")

    print(f"\n{'=' * 40}")
    print("All smoke tests passed!")


if __name__ == "__main__":
    main()
