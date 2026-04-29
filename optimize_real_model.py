import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DynamicCache
from turboquant.ppl_evaluator import RotorQuantCache, evaluate_ppl
from turboquant import xpu_backend
import time

# Using a smaller model for faster iteration
model_id = "Qwen/Qwen2.5-0.5B-Instruct" 
device = "xpu"

print(f"Loading model {model_id}...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

from turboquant.calibrate_xpu import calibrate_xpu

# Optimization Loop
results = []

for bits in [4, 6, 8]:
    print(f"\n--- Testing {bits}-bit RotorQuant ---")
    
    # 1. Calibrate
    rotors_dict, codebooks_dict = calibrate_xpu(model, tokenizer, bits=bits)
    
    # 2. Measure PPL
    start_time = time.time()
    ppl = evaluate_ppl(
        model, tokenizer, 
        n_samples=5, seq_len=128, 
        cache_class=RotorQuantCache, 
        n_levels=2**bits,
        rotors_dict=rotors_dict,
        codebooks_dict=codebooks_dict
    )
    # Inject codebooks into the cache (RotorQuantCache needs to support this)
    # For now, let's just make sure RotorQuantCache uses the global codebooks_dict or similar.
    duration = time.time() - start_time
    
    print(f"PPL: {ppl:.4f} (Time: {duration:.2f}s)")
    
    results.append({
        'bits': bits,
        'ppl': ppl,
        'time': duration
    })

# Compare with Baseline (No Quantization)
print(f"\n--- Testing FP16 Baseline ---")
start_time = time.time()
base_ppl = evaluate_ppl(
    model, tokenizer, 
    n_samples=5, seq_len=128, 
    cache_class=DynamicCache
)
duration = time.time() - start_time
print(f"Base PPL: {base_ppl:.4f} (Time: {duration:.2f}s)")

# Summary
print("\n=== Final Results Summary ===")
for r in results:
    deg = (r['ppl'] - base_ppl) / base_ppl * 100
    print(f"{r['bits']}-bit: PPL={r['ppl']:.4f} (+{deg:.2f}% degradation)")

# Goal: < 5% degradation at 3x compression (4-bit)
