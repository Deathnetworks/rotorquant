import torch
import torch.nn as nn
import time
import os
import sys
import math
import io
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from turboquant import xpu_backend, isoquant, calibrate_xpu

# --- Hardware Optimization for 32GB VRAM ---
# Prevent DEVICE_LOST by disabling immediate command lists for large batches
os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "0"
# Allow larger fragmentation before OOM
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "max_split_size_mb:512"

# Ensure UTF-8 output for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- Needle in a Haystack Utils ---
NEEDLE = "The secret project code name is AURORA-7749."
QUESTION = "What is the secret project code name mentioned in the documents?"
FILLER = """The quarterly financial review meeting covered several topics including budget allocations for the upcoming fiscal year, departmental spending reports, and projected revenue streams from various business units. The committee discussed infrastructure upgrades planned for the western regional offices and noted that maintenance schedules should be coordinated with the facilities management team.\n\n"""

def build_needle_prompt(tokenizer, target_tokens, needle_pos=0.5):
    filler_tokens = tokenizer.encode(FILLER, add_special_tokens=False)
    n_reps = max(1, target_tokens // len(filler_tokens))
    needle_idx = int(n_reps * needle_pos)
    
    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f" {NEEDLE} ")
        parts.append(FILLER)
    
    haystack = "".join(parts)
    # Standard chat template for 3B-Instruct
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You must answer questions based on the provided documents."},
        {"role": "user", "content": f"Document Context:\n{haystack}\n\nQuestion: {QUESTION}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

class XPUAsymmetricCache(DynamicCache):
    def __init__(self, model_config, mode='iso', v_centroids=None, v_rotors=None):
        super().__init__()
        self.head_dim = getattr(model_config, "head_dim", model_config.hidden_size // model_config.num_attention_heads)
        self.mode = mode
        self.v_centroids = v_centroids
        self.v_rotors = v_rotors
        self.device = torch.device("xpu")
        
    def _quantize_q8(self, x):
        max_val = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        scale = max_val / 127.0
        return torch.round(x / scale).clamp(-127, 127) * scale

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k_q = self._quantize_q8(key_states)
        r = self.v_rotors[layer_idx]
        cb = self.v_centroids[layer_idx]
        
        if self.mode == 'rotor':
            v_q = xpu_backend.rotor_full_fused(value_states, r, cb['scalar'], cb['vector'], cb['bivector'], cb['trivector'])
        else: # iso
            v_q = xpu_backend.isoquant_fused(value_states, r, cb)
            
        return super().update(k_q, v_q, layer_idx, cache_kwargs)

@torch.no_grad()
def run_needle_test(model_id="Qwen/Qwen2.5-3B-Instruct", max_ctx=131072):
    print(f"\n=== Needle in a Haystack Test (XPU) ===")
    print(f"Model: {model_id} | Max Context: {max_ctx}")
    
    device = torch.device("xpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Patch RoPE for 128k
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id)
    if max_ctx > config.max_position_embeddings:
        scaling_factor = float(math.ceil(max_ctx / config.max_position_embeddings))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    
    # Use FP16 weights (avoiding BnB Triton dependency)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        config=config, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Calibration
    rotors_r, cbs_r = calibrate_xpu.calibrate_xpu(model, tokenizer, bits=4, mode='rotor')
    
    # Correct Calibration for Iso mode
    print("Calibrating XPU Codebooks (Mode: iso, 4-bit)...")
    res_iso = calibrate_xpu.calibrate_xpu(model, tokenizer, bits=4, mode='iso')
    cbs_i = {i: res_iso['iso_centroids'] for i in range(model.config.num_hidden_layers)}
    # Use real calibrated rotors if possible, or consistent ones
    rotors_i = {i: isoquant.make_random_unit_quaternion(((model.config.hidden_size // model.config.num_attention_heads + 3)//4,), device=device) for i in range(model.config.num_hidden_layers)}

    configs = [
        {"name": "FP16 (Baseline)", "class": DynamicCache, "kwargs": {}},
        {"name": "Q8-Key / Rotor-4b-Val", "class": XPUAsymmetricCache, "kwargs": {"mode": "rotor", "v_centroids": cbs_r, "v_rotors": rotors_r}},
        {"name": "Q8-Key / Iso-4b-Val", "class": XPUAsymmetricCache, "kwargs": {"mode": "iso", "v_centroids": cbs_i, "v_rotors": rotors_i}},
    ]

    # Test points
    ctx_sizes = [4096, 16384, 65536, 131072]
    
    for cfg in configs:
        print(f"\nConfig: {cfg['name']}")
        for ctx in ctx_sizes:
            if cfg['name'] == "FP16 (Baseline)" and ctx > 32768:
                print(f"  Ctx={ctx:>7d}: SKIP (FP16 Baseline OOM/Slow)")
                continue
            
            try:
                torch.xpu.empty_cache()
                torch.xpu.empty_cache()
                prompt = build_needle_prompt(tokenizer, ctx)
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                input_len = input_ids.shape[1]
                
                cache = cfg['class'](model.config, **cfg['kwargs']) if cfg['class'] != DynamicCache else DynamicCache()
                
                # Prefill
                t0 = time.perf_counter()
                model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
                torch.xpu.synchronize()
                t_prefill = time.perf_counter() - t0
                
                # Prefill
                t0 = time.perf_counter()
                model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
                torch.xpu.synchronize()
                t_prefill = time.perf_counter() - t0
                
                # Manual Decoding Loop
                next_token_id = input_ids[:, -1:]
                generated_tokens = []
                
                # Pre-calculate position_ids for decode
                cur_pos = input_len
                
                for _ in range(30):
                    torch.xpu.empty_cache()
                    pos_ids = torch.tensor([[cur_pos]], device=device)
                    outputs = model(input_ids=next_token_id, attention_mask=attention_mask, position_ids=pos_ids, past_key_values=cache, use_cache=True)
                    
                    next_token_id = outputs.logits[:, -1:].argmax(dim=-1)
                    generated_tokens.append(next_token_id.item())
                    
                    # Update mask and pos
                    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=-1)
                    cur_pos += 1
                    
                    if next_token_id.item() == tokenizer.eos_token_id:
                        break
                
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                success = any(x in response.upper() for x in ["AURORA", "7749", "AURORA-7749"])
                
                print(f"  Ctx={ctx:>7d}: {'FOUND' if success else 'MISS'} | Time={t_prefill:7.2f}s | Resp: {response.strip()[:100]}...", flush=True)
            except Exception as e:
                print(f"  Ctx={ctx:>7d}: FAILED ({str(e)})")
                break

if __name__ == "__main__":
    run_needle_test()
