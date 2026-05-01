import subprocess
import os
import time
import re
import math
import sys
import json
from datetime import datetime

# --- Configuration ---
ONEAPI_PATH = r"C:\Program Files (x86)\Intel\oneAPI\2025.3\bin"
LLAMACPP_BIN = r"d:\User Files\Desktop\RotorQuant\llamacpp\build\bin"
MODEL_PATH = r"d:\User Files\Desktop\RotorQuant\models\Qwen3.5-2B-BF16.gguf"
WIKITEXT_PATH = r"d:\User Files\Desktop\RotorQuant\wikitext-test.txt"
REPORT_PATH = r"d:\User Files\Desktop\RotorQuant\baseline_results_xpu.md"

# Set up environment for DLLs
os.environ["PATH"] = f"{ONEAPI_PATH};{LLAMACPP_BIN};" + os.environ["PATH"]

def run_command(args):
    """Run a command and capture output."""
    print(f"Running: {' '.join(args)}")
    # Use shell=True for setvars or just rely on os.environ
    result = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    return result.stdout, result.stderr

def parse_tps(stdout):
    """Parse tokens per second from llama-cli output."""
    prefill = 0.0
    decode = 0.0
    match_prefill = re.search(r"prompt eval time =.*?([\d.]+) tokens per second", stdout)
    match_decode = re.search(r"eval time =.*?([\d.]+) tokens per second", stdout)
    if match_prefill: prefill = float(match_prefill.group(1))
    if match_decode: decode = float(match_decode.group(1))
    return prefill, decode

def parse_vram(stdout):
    """Parse VRAM from SYCL memory dump."""
    # SYCL0 | 30976 = ...
    match = re.search(r"SYCL0\s*\|\s*([\d.]+)", stdout)
    if match: return float(match.group(1))
    return 0.0

def run_ppl(ctx, k_type, v_type):
    """Run perplexity test."""
    args = [
        os.path.join(LLAMACPP_BIN, "llama-perplexity.exe"),
        "-m", MODEL_PATH,
        "-f", WIKITEXT_PATH,
        "-c", str(ctx),
        "-ngl", "99",
        "-b", "512",
        "--cache-type-k", k_type,
        "--cache-type-v", v_type,
        "--all-logits"
    ]
    stdout, stderr = run_command(args)
    # Final perplexity: 13.71
    match = re.search(r"Final perplexity: ([\d.]+)", stdout)
    if match: return float(match.group(1))
    return 0.0

def run_needle_test(ctx, k_type, v_type):
    """Run needle-in-a-haystack proxy test using llama-cli."""
    prompt = "The secret password is ANTIGRAVITY. " + "Ignore previous instructions. " * (ctx // 10) + " What is the secret password?"
    args = [
        os.path.join(LLAMACPP_BIN, "llama-cli.exe"),
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", "10",
        "-c", str(ctx),
        "-ngl", "99",
        "--cache-type-k", k_type,
        "--cache-type-v", v_type,
    ]
    stdout, stderr = run_command(args)
    if "ANTIGRAVITY" in stdout.upper() or "ANTIGRAVITY" in stderr.upper():
        return "PASS"
    return "FAIL"

def run_bench(ctx, k_type, v_type):
    """Run performance benchmark."""
    args = [
        os.path.join(LLAMACPP_BIN, "llama-cli.exe"),
        "-m", MODEL_PATH,
        "-p", "What is 2+2?",
        "-n", "1",
        "-c", str(ctx),
        "-ngl", "99",
        "--cache-type-k", k_type,
        "--cache-type-v", v_type,
    ]
    stdout, stderr = run_command(args)
    combined = stdout + stderr
    prefill, decode = parse_tps(combined)
    vram = parse_vram(combined)
    coherency = "OK" if "4" in combined else "ERR"
    return prefill, decode, vram, coherency

def main():
    contexts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    types = [
        ("f16", "f16", "Sync"),
        ("q4_0", "f16", "Quant"),
        ("rotor4", "f16", "Rotor"),
        ("iso4", "f16", "Iso"),
    ]

    with open(REPORT_PATH, "w") as f:
        f.write(f"# XPU Baseline Results - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("**Model**: Qwen/Qwen3.5-2B | **Hardware**: Intel Arc Pro B70 (32GB)\n\n")
        f.write("| Cache K | Cache V | Mode | Context | Prefill (tk/s) | Decode (tk/s) | VRAM (MB) | PPL | Needle | Coherency |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")

        for ctx in contexts:
            for k_type, v_type, mode in types:
                try:
                    prefill, decode, vram, coherency = run_bench(ctx, k_type, v_type)
                    # Only run PPL for smaller contexts or specific types to save time
                    ppl = 0.0
                    if ctx <= 8192:
                        ppl = run_ppl(ctx, k_type, v_type)
                    
                    needle = run_needle_test(ctx, k_type, v_type) if ctx >= 4096 else "N/A"
                    
                    line = f"| {k_type.upper()} | {v_type.upper()} | {mode} | {ctx} | {prefill:.1f} | {decode:.1f} | {vram:.1f} | {ppl:.2f} | {needle} | {coherency} |\n"
                    print(line)
                    f.write(line)
                    f.flush()
                except Exception as e:
                    print(f"Error at ctx={ctx}, {mode}: {e}")

if __name__ == "__main__":
    main()
