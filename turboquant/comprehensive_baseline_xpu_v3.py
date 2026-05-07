import subprocess
import time
import os
import re
import json
import sys

# Paths
ROOT = r"d:\User Files\Desktop\RotorQuant"
BINARY_DIR = os.path.join(ROOT, r"llamacpp\build\bin")
LLAMA_BENCH = os.path.join(BINARY_DIR, "llama-bench.exe")
LLAMA_CLI = os.path.join(BINARY_DIR, "llama-cli.exe")
LLAMA_PERP = os.path.join(BINARY_DIR, "llama-perplexity.exe")
SETVARS = r"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
MODEL = os.path.join(ROOT, r"models\Qwen3.5-2B-BF16.gguf")
DATASET = os.path.join(ROOT, r"wikitext-test.txt") # Use the 1.3MB file

TYPES = ["f16", "q8_0", "q4_0", "q4_1", "iso4", "rotor4"]
CONTEXT_LENGTHS = [512, 1024, 2048]

# Paper References for Grounding
PAPERS = {
    "RotorQuant": "[RotorQuant: Clifford Algebra Vector Quantization](file:///d:/User%20Files/Desktop/RotorQuant/paper/rotorquant.md)",
    "IsoQuant1": "[IsoQuant: Quaternion-based Isoclinic Rotation (Paper 1)](file:///d:/User%20Files/Desktop/RotorQuant/paper/isoquant%20paper%201.md)",
    "IsoQuant2": "[IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations (Paper 2)](file:///d:/User%20Files/Desktop/RotorQuant/paper/isoquant%20paper%202.md)"
}

def run_command(cmd, cwd=None, env_debug=False):
    # Workflow Rule: Ensure ONLY ONE instance of llama processes is active
    subprocess.run("taskkill /f /im llama*", shell=True, capture_output=True)
    time.sleep(0.5)
    
    batch_file = os.path.join(ROOT, "scratch", "run_bench.bat")
    os.makedirs(os.path.dirname(batch_file), exist_ok=True)
    with open(batch_file, "w") as f:
        f.write(f'@call "{SETVARS}" > nul\n')
        if env_debug:
            f.write("set GGML_SYCL_DEBUG=1\n")
        f.write(f'{cmd}\n')
    
    try:
        # Increased timeout for large context / PPL
        res = subprocess.run(f'"{batch_file}"', capture_output=True, text=True, shell=True, cwd=cwd, timeout=1200, encoding='utf-8', errors='replace')
        return res.stdout or "", res.stderr or ""
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
    # Use --single-turn to ensure exit after one generation
    cmd = f'"{LLAMA_CLI}" -m "{MODEL}" -ctk {k_type} -ctv {v_type} -p "H" -n 1 -c {ctx} --single-turn --n-gpu-layers 99'
    stdout, stderr = run_command(cmd, env_debug=True)
    combined = stdout + stderr
    
    # Extract KV buffer size
    match = re.search(r"KV buffer size\s*=\s*([\d.]+)\s*MiB", combined)
    kv_vram = float(match.group(1)) if match else 0
    
    # Extract Memory Breakdown from logs
    # Format: |   - SYCL0 (...) | 31906 = 18244 + (7469 =  3590 + 3091 + 788) + 6192 |
    # The value in parentheses after model weight is the context memory.
    context_vram = 0
    total_vram = 0
    for line in combined.splitlines():
        if "SYCL0" in line and "|" in line and "=" in line:
            try:
                # Look for the pattern: total = model + (context = ...) + compute
                parts = line.split("|")
                if len(parts) >= 3:
                    numbers_part = parts[2].split("=")
                    if len(numbers_part) >= 2:
                        total_vram = float(numbers_part[0].strip())
                        # The next part is model + (context) + compute
                        # We use regex to find the value inside the first parenthesis after the first '+'
                        rem = "=".join(numbers_part[1:])
                        # Format: model + (context = ...) + compute
                        ctx_match = re.search(r"\+\s*\(([\d.]+)\s*=", rem)
                        if ctx_match:
                            context_vram = float(ctx_match.group(1))
                        else:
                            # Fallback if no nested parenthesis
                            sub_parts = rem.split("+")
                            if len(sub_parts) >= 2:
                                context_vram = float(sub_parts[1].strip().split()[0])
            except Exception as e:
                pass

    # Total device memory used (from llama.cpp summary)
    match = re.search(r"device memory used\s*=\s*([\d.]+)\s*MiB", combined)
    if match:
        total_vram = float(match.group(1))
    
    return kv_vram, context_vram, total_vram

def run_perplexity(k_type, v_type):
    print(f"  Running Perplexity for K={k_type}, V={v_type}...")
    # Using stride 512 as per workflow rules. Limit chunks for speed.
    cmd = f'"{LLAMA_PERP}" -m "{MODEL}" -f "{DATASET}" -c 2048 -b 512 --ppl-stride 512 --chunks 32 --n-gpu-layers 99'
    stdout, stderr = run_command(cmd)
    combined = stdout + stderr
    
    match = re.search(r"Final PPL\s*=\s*([\d.]+)", combined)
    if match:
        return float(match.group(1))
    
    # Fallback to parsing the last loss block
    losses = re.findall(r"\[\d+\]([\d.]+)", combined)
    if losses:
        avg_loss = sum(float(l) for l in losses) / len(losses)
        import math
        return math.exp(avg_loss)
    
    return 0.0

def run_needle_test(k_type, v_type, ctx=4096):
    print(f"  Running Needle-in-a-Haystack for K={k_type}, V={v_type}, ctx={ctx}...")
    secret = "The secret code name is: ALBATROSS-9988."
    filler = "The quarterly financial review meeting covered budget allocations and departmental spending reports. Infrastructure upgrades are planned for the western regional offices. "
    
    # Approx 4 bytes per token for filler
    target_bytes = ctx * 4
    n_reps = target_bytes // len(filler)
    
    parts = [filler] * n_reps
    # Insert needle at 75% depth
    pos = int(n_reps * 0.75)
    parts.insert(pos, f"\n{secret}\n")
    
    haystack = "".join(parts)
    prompt = f"Background Documents:\n{haystack}\n\nQuestion: What is the secret code name mentioned in the documents? Answer with the code only.\nAssistant:"
    
    prompt_file = os.path.join(ROOT, "scratch", "needle_prompt.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)
    
    cmd = f'"{LLAMA_CLI}" -m "{MODEL}" -ctk {k_type} -ctv {v_type} -f "{prompt_file}" -n 24 -c {ctx + 512} --single-turn --n-gpu-layers 99 --temp 0'
    stdout, stderr = run_command(cmd)
    combined = stdout + stderr
    
    if "ALBATROSS-9988" in combined:
        return "PASSED"
    return "FAILED"

def run_coherency_test(k_type, v_type):
    print(f"  Running Coherency test for K={k_type}, V={v_type}...")
    questions = [
        ("What is the capital city of France?", "Paris"),
        ("What is the chemical formula for water?", "H2O"),
        ("Which planet is known as the Red Planet?", "Mars")
    ]
    
    passed = 0
    prompt_file = os.path.join(ROOT, "scratch", "coherency_prompt.txt")
    for q, a in questions:
        prompt = f"User: {q}\nAssistant:"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)
            
        cmd = f'"{LLAMA_CLI}" -m "{MODEL}" -ctk {k_type} -ctv {v_type} -f "{prompt_file}" --single-turn -n 128 --n-gpu-layers 99 --temp 0'
        stdout, stderr = run_command(cmd)
        if a.lower() in (stdout + stderr).lower():
            passed += 1
            
    if passed == len(questions): return "PASSED"
    if passed > 0: return f"PARTIAL ({passed}/{len(questions)})"
    return "FAILED"

def main():
    print("Starting improved comprehensive benchmark v3...")
    
    # 0. Clean up old results
    old_results = os.path.join(ROOT, "baseline_results_xpu_v2.md")
    if os.path.exists(old_results):
        os.remove(old_results)
        print(f"Deleted old results: {old_results}")

    results = []
    
    # 1. Measure PPL and Baselines for each combo (at fixed context for PPL)
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
        types_str = ",".join(TYPES)
        # We run llama-bench for speed on the matrix
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
    output_path = os.path.join(ROOT, "baseline_results_xpu_v2.md") # Keep same name as requested
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
