import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant.ppl_evaluator import evaluate_ppl, IsoQuantCache
from turboquant.calibrate_xpu import calibrate_xpu
import math

def run_iso_ppl():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.half, device_map="xpu")
    model.eval()
    
    # Calibrate
    res = calibrate_xpu(model, tokenizer, bits=4, mode='iso')
    centroids = res['iso_centroids']
    
    print("\n--- Testing 4-bit Calibrated IsoQuant (XPU Fused) ---")
    ppl_iso = evaluate_ppl(model, tokenizer, cache_class=IsoQuantCache, n_levels=16, 
                          model_config=model.config, centroids=centroids)
    print(f"IsoQuant PPL: {ppl_iso:.4f}")

    print("\n--- Testing FP16 Baseline ---")
    ppl_fp16 = evaluate_ppl(model, tokenizer)
    print(f"FP16 PPL:    {ppl_fp16:.4f}")

    deg = (ppl_iso - ppl_fp16) / ppl_fp16 * 100
    print(f"\nDegradation: {deg:.2f}%")

if __name__ == "__main__":
    run_iso_ppl()
