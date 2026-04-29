import torch
import torch.nn as nn
import math
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from turboquant import xpu_backend, isoquant, rotorquant, calibrate_xpu
from turboquant.ppl_evaluator import evaluate_ppl

class AsymmetricKVCache(DynamicCache):
    """
    Asymmetric KV Cache:
    - Keys: Q8 (8-bit linear quantization)
    - Values: RotorQuant / IsoQuant (4-bit Clifford/Quaternion rotation)
    """
    def __init__(self, model_config, v_bits=4, mode='rotor', v_centroids=None, v_rotors=None):
        super().__init__()
        self.head_dim = getattr(model_config, "head_dim", model_config.hidden_size // model_config.num_attention_heads)
        self.v_bits = v_bits
        self.mode = mode
        self.v_centroids = v_centroids
        self.v_rotors = v_rotors
        self.device = torch.device("xpu")
        
    def _quantize_q8(self, x):
        """Simple Q8 quantization for Keys."""
        # Note: In a real kernel this would be int8 + scale. 
        # Here we simulate the effect with float-simulated int8.
        eps = 1e-8
        max_val = x.abs().max(dim=-1, keepdim=True).values.clamp(min=eps)
        scale = max_val / 127.0
        x_q = torch.round(x / scale).clamp(-127, 127)
        return x_q * scale

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # 1. Key Quantization (Q8)
        k_q = self._quantize_q8(key_states)
        
        # 2. Value Quantization (Rotor/Iso)
        if self.mode == 'rotor':
            r = self.v_rotors[layer_idx]
            cb = self.v_centroids[layer_idx]
            v_q = xpu_backend.rotor_full_fused(
                value_states, r, 
                cb['scalar'], cb['vector'], cb['bivector'], cb['trivector']
            )
        else: # iso
            qL = self.v_rotors[layer_idx]
            cb = self.v_centroids[layer_idx]
            v_q = xpu_backend.isoquant_fused(value_states, qL, cb)

        return super().update(k_q, v_q, layer_idx, cache_kwargs)

@torch.no_grad()
def benchmark_real_model(model_id="Qwen/Qwen2-0.5B-Instruct"):
    print(f"\n=== Benchmarking Real Model: {model_id} ===")
    device = torch.device("xpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    model.eval()

    # 1. Calibration
    # We calibrate for both Rotor and Iso to compare
    rotors_r, codebooks_r = calibrate_xpu.calibrate_xpu(model, tokenizer, bits=4, mode='rotor')
    codebooks_iso = calibrate_xpu.calibrate_xpu(model, tokenizer, bits=4, mode='iso')
    # For iso, calibrate_xpu currently returns {'iso_centroids': ...}. 
    # We need to adapt it slightly or mock rotors for simplicity.
    iso_qL = {i: isoquant.make_random_unit_quaternion(((model.config.hidden_size // model.config.num_attention_heads + 3)//4,), device=device) for i in range(model.config.num_hidden_layers)}
    
    # 2. Benchmark Configurations
    configs = [
        {"name": "FP16 (Baseline)", "cache_class": DynamicCache, "kwargs": {}},
        {"name": "Asymmetric (Q8 Keys / Rotor 4b Values)", 
         "cache_class": AsymmetricKVCache, 
         "kwargs": {"v_bits": 4, "mode": 'rotor', "v_centroids": codebooks_r, "v_rotors": rotors_r}},
        {"name": "Asymmetric (Q8 Keys / Iso 4b Values)", 
         "cache_class": AsymmetricKVCache, 
         "kwargs": {"v_bits": 4, "mode": 'iso', "v_centroids": {i: codebooks_iso['iso_centroids'] for i in range(model.config.num_hidden_layers)}, "v_rotors": iso_qL}},
    ]

    results = []
    
    # Testing Context Sizes
    # Testing Context Sizes - Capped for stability on Arc Pro B70
    context_sizes = [1024, 4096, 16384, 32768]
    
    for cfg in configs:
        print(f"Testing Config: {cfg['name']}", flush=True)
        # Baseline DynamicCache doesn't take model_config, but our custom ones do.
        kwargs = cfg['kwargs'].copy()
        if cfg['cache_class'] != DynamicCache:
            kwargs['model_config'] = model.config
        
        print(f"  Evaluating PPL...", flush=True)
        ppl = evaluate_ppl(model, tokenizer, cache_class=cfg['cache_class'], **kwargs)
        print(f"  PPL: {ppl:.4f}", flush=True)
        
        for ctx in context_sizes:
            # Skip excessive context for baseline to avoid hang
            if cfg['name'] == "FP16 (Baseline)" and ctx > 32768:
                print(f"  Ctx={ctx:>7d}: SKIPPED (Baseline VRAM limit)", flush=True)
                continue
            
            print(f"  Testing Context Size: {ctx}...", flush=True)
            # We skip 256k if VRAM is tight, but attempting for benchmark
            try:
                torch.xpu.empty_cache()
                # Prefill Speed
                input_ids = torch.randint(0, model.config.vocab_size, (1, ctx), device=device)
                torch.xpu.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    if cfg['cache_class'] == DynamicCache:
                        cache = cfg['cache_class']()
                    else:
                        cache = cfg['cache_class'](model.config, **cfg['kwargs'])
                    model(input_ids, past_key_values=cache, use_cache=True)
                torch.xpu.synchronize()
                prefill_time = time.perf_counter() - t0
                
                # Decode Speed (next 5 tokens) - Reduced for speed
                decode_times = []
                curr_ids = torch.randint(0, model.config.vocab_size, (1, 1), device=device)
                for _ in range(5):
                    torch.xpu.empty_cache()
                    t1 = time.perf_counter()
                    with torch.no_grad():
                        model(curr_ids, past_key_values=cache, use_cache=True)
                    torch.xpu.synchronize()
                    decode_times.append(time.perf_counter() - t1)
                decode_time_avg = sum(decode_times) / len(decode_times)
                
                print(f"  Ctx={ctx:>7d}: Prefill={prefill_time:7.3f}s | Decode={decode_time_avg*1000:7.2f}ms", flush=True)
                
                # Needle in a Haystack (Simplified: can we recover the last token hidden in context)
                # For brevity in benchmark, we just note the success of the forward pass.
            except Exception as e:
                print(f"  Ctx={ctx:>7d}: FAILED (likely OOM or size limit) - {str(e)}")
                break

if __name__ == "__main__":
    benchmark_real_model()
