import subprocess
import os
import time
import re
from datetime import datetime

# --- Environment Setup (Option B) ---
ONEAPI_PATH = r"C:\Program Files (x86)\Intel\oneAPI\2025.3\bin"
LLAMACPP_BIN = r"d:\User Files\Desktop\RotorQuant\llamacpp\build\bin"
MODEL_PATH = r"d:\User Files\Desktop\RotorQuant\models\Qwen3.5-2B-BF16.gguf"
REPORT_PATH = r"d:\User Files\Desktop\RotorQuant\baseline_results_xpu.md"

os.environ["PATH"] = f"{ONEAPI_PATH};{LLAMACPP_BIN};" + os.environ["PATH"]

def run_bench(ctx):
    print(f">> Benchmarking context {ctx} in llama.cpp...")
    prompt_text = "The quick brown fox " * (ctx // 5)
    prompt_file = f"prompt_{ctx}.txt"
    
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt_text)

    cmd = [
        os.path.join(LLAMACPP_BIN, "llama-completion.exe"),
        "-m", MODEL_PATH,
        "-f", prompt_file,
        "-n", "64",
        "-ngl", "99",
        "-b", "512",
        "-ub", "512",
        "--no-display-prompt",
        "-no-cnv"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8", errors="ignore")
        output = result.stdout + result.stderr
        
        prefill_match = re.search(r"prompt eval time\s*=\s*.*?\(\s*([\d.]+)\s*tokens per second\)", output)
        decode_match = re.search(r"(?<!prompt )eval time\s*=\s*.*?\(\s*([\d.]+)\s*tokens per second\)", output)
        vram_match = re.search(r"SYCL0.*?\|\s*\d+\s*=\s*\d+\s*\+\s*\(\d+\s*=\s*\d+\s*\+\s*([\d.]+)\s*\+", output)
        
        prefill = prefill_match.group(1) if prefill_match else "0.0"
        decode = decode_match.group(1) if decode_match else "0.0"
        vram = vram_match.group(1) if vram_match else "0.0"

        if os.path.exists(prompt_file):
            os.remove(prompt_file)
        return prefill, decode, vram
    except Exception as e:
        print(f"Error at context {ctx}: {e}")
        if os.path.exists(prompt_file):
            os.remove(prompt_file)
        return "0.0", "0.0", "0.0"

def get_ppl(ctx):
    print(f">> Calculating PPL for context {ctx}...")
    cmd = [
        os.path.join(LLAMACPP_BIN, "llama-perplexity.exe"),
        "-m", MODEL_PATH,
        "-f", r"d:\User Files\Desktop\RotorQuant\README.md",
        "-ctx", str(ctx),
        "-ngl", "99",
        "-n", "128" 
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8", errors="ignore")
        ppl_match = re.search(r"Final estimate: PPL = ([\d.]+)", result.stdout + result.stderr)
        return ppl_match.group(1) if ppl_match else "ERR_REGEX"
    except Exception as e:
        return f"ERR_RUN"

def get_needle(ctx):
    print(f">> Running Needle test for context {ctx}...")
    needle = "The secret password is: ANTIGRAVITY_XPU_2026"
    haystack = "The sky is blue. " * (ctx // 5)
    # Insert needle at 75% depth
    pos = int(len(haystack) * 0.75)
    test_content = haystack[:pos] + needle + haystack[pos:]
    test_file = f"needle_{ctx}.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content + "\nWhat is the secret password?")

    cmd = [
        os.path.join(LLAMACPP_BIN, "llama-completion.exe"),
        "-m", MODEL_PATH,
        "-f", test_file,
        "-n", "16",
        "-ngl", "99",
        "--no-display-prompt",
        "-no-cnv"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8", errors="ignore")
        if "ANTIGRAVITY_XPU_2026" in result.stdout:
            res = "PASS"
        else:
            res = "FAIL"
        if os.path.exists(test_file):
            os.remove(test_file)
        return res
    except Exception as e:
        if os.path.exists(test_file):
            os.remove(test_file)
        return "ERR"

def main():
    # Report setup
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"# XPU Performance Matrix - RotorQuant vs. Native Baseline\n\n")
        f.write(f"**Model**: Qwen/Qwen3.5-2B | **Hardware**: Intel Arc Pro B70 (32GB)\n\n")
        f.write("| Cache K | Cache V | Mode | Context | Prefill (TPS) | Decode (TPS) | VRAM (MB) | PPL | Needle | Status |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")

    for ctx in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        prefill, decode, vram = run_bench(ctx)
        ppl = get_ppl(ctx)
        needle = get_needle(ctx)
        
        with open(REPORT_PATH, "a", encoding="utf-8") as f:
            f.write(f"| FP16 (Native) | FP16 (Native) | Sync | {ctx} | {prefill} | {decode} | {vram} | {ppl} | {needle} | OK |\n")
        print(f"Done {ctx}: {prefill} TPS / {decode} TPS / {vram} MB / PPL: {ppl} / Needle: {needle}")

if __name__ == "__main__":
    main()
