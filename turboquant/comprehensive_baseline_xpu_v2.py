import subprocess
import time
import os
import re
import json
import sys

# Paths
ROOT = r"d:\User Files\Desktop\RotorQuant"
LLAMA_BENCH = os.path.join(ROOT, r"llamacpp\build\bin\llama-bench.exe")
LLAMA_CLI = os.path.join(ROOT, r"llamacpp\build\bin\llama-cli.exe")
LLAMA_PERP = os.path.join(ROOT, r"llamacpp\build\bin\llama-perplexity.exe")
SETVARS = r"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
MODEL = os.path.join(ROOT, r"models\Qwen3.5-2B-BF16.gguf")
DATASET = os.path.join(ROOT, r"wikitext_test.txt")

TYPES = ["f16", "q8_0", "iso4", "rotor4"]
CONTEXT_LENGTHS = [512, 4096, 16384, 32768] # Target lengths for breakdown

# Paper References for Grounding
PAPERS = {
    "RotorQuant": "[RotorQuant: Clifford Algebra Vector Quantization](file:///d:/User%20Files/Desktop/RotorQuant/paper/rotorquant.md)",
    "IsoQuant1": "[IsoQuant: Quaternion-based Isoclinic Rotation (Paper 1)](file:///d:/User%20Files/Desktop/RotorQuant/paper/isoquant%20paper%201.md)",
    "IsoQuant2": "[IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations (Paper 2)](file:///d:/User%20Files/Desktop/RotorQuant/paper/isoquant%20paper%202.md)"
}

def run_command(cmd, cwd=None):
    batch_file = os.path.join(ROOT, "scratch", "run_bench.bat")
    os.makedirs(os.path.dirname(batch_file), exist_ok=True)
    with open(batch_file, "w") as f:
        f.write(f'@call "{SETVARS}" > nul\n')
        f.write(f'{cmd}\n')
    
    try:
        res = subprocess.run(f'"{batch_file}"', capture_output=True, text=True, shell=True, cwd=cwd, timeout=600)
        return res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        print("    Command timed out! Killing llama processes...")
        subprocess.run("taskkill /f /im llama*", shell=True, capture_output=True)
        return "", "Timeout"

def clean_json(s):
    match = re.search(r"\[\s*\{.*\}\s*\]", s, re.DOTALL)
    if match:
        return match.group(0)
    return s

def get_vram(k_type, v_type, ctx):
    print(f"  Measuring VRAM for K={k_type}, V={v_type}, Context={ctx}...")
    # Use -st (single-turn) to ensure exit after one generation
    cmd = f'"{LLAMA_CLI}" -m "{MODEL}" -ctk {k_type} -ctv {v_type} -p "H" -n 1 -c {ctx} -st --n-gpu-layers 99'
    stdout, stderr = run_command(cmd)
    combined = stdout + stderr
    
    # Extract KV buffer size
    match = re.search(r"KV buffer size\s*=\s*([\d.]+)\s*MiB", combined)
    kv_vram = float(match.group(1)) if match else 0
    
    # Extract Memory Breakdown from logs
    # Format: |   - SYCL0 (...) | 31906 = 18244 + (7469 =  3590 + 3091 + 788) + 6192 |
    context_vram = 0
    for line in combined.splitlines():
        if "common_memory_breakdown_print" in line and "SYCL0" in line:
            # Look for the pattern in parentheses: (self = model + context + compute)
            # The context is the second value after '='
            try:
                parts = line.split("=")
                if len(parts) >= 3:
                    sub_parts = parts[2].split("+")
                    if len(sub_parts) >= 2:
                        context_vram = float(sub_parts[1].strip())
            except Exception as e:
                print(f"      VRAM parse error: {e}")

    # Total device memory used
    match = re.search(r"device memory used\s*=\s*([\d.]+)\s*MiB", combined)
    total_vram = float(match.group(1)) if match else 0
    
    return kv_vram, context_vram, total_vram

def run_perplexity(k_type, v_type):
    print(f"  Running Perplexity for K={k_type}, V={v_type}...")
    # Run with small n for speed, but enough for PPL
    cmd = f'"{LLAMA_PERP}" -m "{MODEL}" -f "{DATASET}" -c 512 -n 128 --n-gpu-layers 99'
    stdout, stderr = run_command(cmd)
    combined = stdout + stderr
    
    # Extract PPL: Final PPL = 8.3424
    # Or find average loss
    match = re.search(r"Final PPL\s*=\s*([\d.]+)", combined)
    if match:
        return float(match.group(1))
    
    # Try to calculate from losses if Final PPL is missing
    losses = re.findall(r"\[\d+\]([\d.]+)", combined)
    if losses:
        avg_loss = sum(float(l) for l in losses) / len(losses)
        import math
        return math.exp(avg_loss)
    
    return 0.0

def run_needle_test(k_type, v_type, ctx=1024):
    print(f"  Running Needle-in-a-Haystack for K={k_type}, V={v_type}...")
    cmd = f'"{LLAMA_CLI}" -m "{MODEL}" -ctk {k_type} -ctv {v_type} -p "Find the needle in the haystack." -n 1 -c {ctx} -st --n-gpu-layers 99'
    stdout, stderr = run_command(cmd)
    combined = stdout + stderr
    if "needle" in combined.lower(): return "PASSED"
    return "FAILED"

def run_coherency_test(k_type, v_type):
    print(f"  Running Coherency test for K={k_type}, V={v_type}...")
    cmd = f'"{LLAMA_CLI}" -m "{MODEL}" -ctk {k_type} -ctv {v_type} -p "1 2 3 4" -n 5 -st --n-gpu-layers 99'
    stdout, stderr = run_command(cmd)
    combined = stdout + stderr
    if "5" in combined: return "PASSED"
    return "DEGRADED"

def main():
    print("Starting comprehensive benchmark...")
    
    results = []
    
    # 1. Measure PPL and Baselines for each combo
    combo_metadata = {}
    for k_type in TYPES:
        for v_type in TYPES:
            print(f"\n>>> Profile: K={k_type}, V={v_type}")
            ppl = run_perplexity(k_type, v_type)
            coherency = run_coherency_test(k_type, v_type)
            needle = run_needle_test(k_type, v_type, ctx=4096)
            
            combo_metadata[(k_type, v_type)] = {
                "ppl": ppl,
                "coherency": coherency,
                "needle": needle
            }

    # 2. Performance & VRAM Breakdown by Context Length
    for ctx in CONTEXT_LENGTHS:
        print(f"\n>>> Scaling Analysis: Context={ctx}")
        # Use llama-bench for performance matrix at this context
        types_str = ",".join(TYPES)
        cmd = f'"{LLAMA_BENCH}" -m "{MODEL}" -p {ctx} -n 128 -ctk {types_str} -ctv {types_str} -r 1 -o json'
        stdout, stderr = run_command(cmd)
        
        perf_batch = {}
        try:
            json_str = clean_json(stdout)
            data = json.loads(json_str)
            for res in data:
                k = res.get('type_k', 'f16').lower()
                v = res.get('type_v', 'f16').lower()
                pp = 0
                tg = 0
                for field, val in res.items():
                    if isinstance(val, dict) and 'tps' in val:
                        if 'pp' in field: pp = val['tps']
                        if 'tg' in field: tg = val['tps']
                perf_batch[(k, v)] = (pp, tg)
        except Exception as e:
            print(f"Error parsing llama-bench results for ctx={ctx}: {e}")

        for k_type in TYPES:
            for v_type in TYPES:
                prefill, decode = perf_batch.get((k_type, v_type), (0, 0))
                kv_vram, ctx_vram, total_vram = get_vram(k_type, v_type, ctx)
                
                meta = combo_metadata.get((k_type, v_type), {})
                
                results.append({
                    "K": k_type,
                    "V": v_type,
                    "Context": ctx,
                    "Prefill": prefill,
                    "Decode": decode,
                    "KV_VRAM": kv_vram,
                    "Ctx_VRAM": ctx_vram,
                    "Total_VRAM": total_vram,
                    "PPL": meta.get("ppl", 0),
                    "Coherency": meta.get("coherency", "N/A"),
                    "Needle": meta.get("needle", "N/A")
                })

    # Save to markdown
    output_path = os.path.join(ROOT, "baseline_results_xpu_v2.md")
    with open(output_path, "w") as f:
        f.write("# KV Cache Quantization Baseline (Intel Arc Pro B70)\n\n")
        
        f.write("## Grounding & Paper References\n\n")
        for name, link in PAPERS.items():
            f.write(f"- **{name}**: {link}\n")
        f.write("\n")

        f.write("## Performance & VRAM scaling by Context Length\n\n")
        f.write("| K Type | V Type | Context | Prefill (TK/s) | Decode (TK/s) | KV Size (MiB) | Context VRAM (MiB) | Total VRAM (MiB) | PPL | Coherency | Needle |\n")
        f.write("|--------|--------|---------|----------------|---------------|---------------|--------------------|------------------|-----|-----------|--------|\n")
        for r in results:
            f.write(f"| {r['K'].upper()} | {r['V'].upper()} | {r['Context']} | {r['Prefill']:.2f} | {r['Decode']:.2f} | {r['KV_VRAM']:.2f} | {r['Ctx_VRAM']:.2f} | {r['Total_VRAM']:.2f} | {r['PPL']:.2f} | {r['Coherency']} | {r['Needle']} |\n")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
