#!/usr/bin/env python3
"""
Hardware probe — detect GPU, CPU, RAM, CUDA for inference planning.
Run: python scripts/probe_hardware.py
"""
import os
import platform
import shutil
import subprocess
import sys


def run(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        return r.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


def main():
    print("=" * 60)
    print("HARDWARE PROBE")
    print("=" * 60)

    # OS
    print(f"\n--- OS ---")
    print(f"Platform: {platform.platform()}")
    print(f"Machine:  {platform.machine()}")
    print(f"Python:   {sys.version}")

    # CPU
    print(f"\n--- CPU ---")
    cpu_info = run("wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors /format:list")
    if cpu_info and "ERROR" not in cpu_info:
        for line in cpu_info.split("\n"):
            if line.strip():
                print(f"  {line.strip()}")
    else:
        print(f"  Processor: {platform.processor()}")
        print(f"  Cores: {os.cpu_count()}")

    # RAM
    print(f"\n--- RAM ---")
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"  Total:     {vm.total / 1024**3:.1f} GB")
        print(f"  Available: {vm.available / 1024**3:.1f} GB")
        print(f"  Used:      {vm.used / 1024**3:.1f} GB ({vm.percent}%)")
    except ImportError:
        print(f"  (psutil not installed — run: uv pip install psutil)")

    # GPU
    print(f"\n--- GPU (nvidia-smi) ---")
    smi = run("nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader")
    if smi and "ERROR" not in smi:
        for line in smi.split("\n"):
            if line.strip():
                parts = [p.strip() for p in line.split(",")]
                print(f"  Name:         {parts[0]}")
                print(f"  VRAM Total:   {parts[1]}")
                print(f"  VRAM Free:    {parts[2]}")
                print(f"  Driver:       {parts[3]}")
                print(f"  Compute Cap:  {parts[4]}")
    else:
        print(f"  nvidia-smi not available or no NVIDIA GPU")

    # CUDA / torch
    print(f"\n--- PyTorch CUDA ---")
    try:
        import torch
        print(f"  torch:    {torch.__version__}")
        print(f"  CUDA:     {torch.version.cuda}")
        print(f"  available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {p.name}, {p.total_memory / 1024**3:.1f} GB, CC {p.major}.{p.minor}")
    except ImportError:
        print(f"  torch not installed")
    except Exception as e:
        print(f"  torch probe failed: {e}")

    # Disk
    print(f"\n--- Disk ---")
    total, used, free = shutil.disk_usage("D:/")
    print(f"  D: Total: {total / 1024**3:.0f} GB, Free: {free / 1024**3:.0f} GB")

    # llama-server
    print(f"\n--- llama-server ---")
    llama = run("where llama-server 2>/dev/null || echo NOT_FOUND")
    print(f"  {llama}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
