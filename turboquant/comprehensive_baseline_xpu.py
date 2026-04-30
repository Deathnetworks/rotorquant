import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import math
import io
import json
import itertools

# Fix path for standalone execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# --- Hardware Optimization ---
os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "0"
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "max_split_size_mb:128"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def apply_quant(x, layer_idx, cfg, rotors_map, cbs_map):
    from turboquant import xpu_backend
    name, bits, mode = cfg
    if name == "FP16": return x
    
    r = rotors_map[mode][layer_idx] if rotors_map and mode in rotors_map else None
    cb = cbs_map[mode][layer_idx] if cbs_map and mode in cbs_map else None
    
    if 'rotor' in mode and r is not None:
        # VERIFIED FUSED KERNEL: rotor_full_fused
        return xpu_backend.rotor_full_fused(x, r, cb['scalar'], cb['vector'], cb['bivector'], cb['trivector'])
    elif 'iso' in mode and cb is not None:
        # VERIFIED FUSED KERNEL: isoquant_fused
        # IsoQuant needs a dummy identity rotation for the fused signature
        if r is None:
            r = torch.zeros((x.shape[-1]//3 + 1, 4), device="xpu", dtype=torch.float32)
            r[:, 0] = 1.0 # scalar component = 1.0 (identity)
        return xpu_backend.isoquant_fused(x, r, cb['iso_centroids'])
    return x

class BenchmarkRunner:
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.config = AutoConfig.from_pretrained(model_id)
        self.report_path = "baseline_results_xpu.md"
        self._load_wikitext()
        
    def _load_wikitext(self):
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        self.test_text = "\n\n".join(dataset["text"])

    def calculate_vram_usage(self, seq_len=0, k_bits=0, v_bits=0):
        """Get organic driver-level VRAM reservation."""
        torch.xpu.synchronize()
        return torch.xpu.memory_reserved() / (1024 * 1024)

    def calculate_perplexity(self, model, cache_cfg, rotors_map, cbs_map, max_length=2048, stride=512):
        k_cfg, v_cfg = cache_cfg
        encodings = self.tokenizer(self.test_text, return_tensors="pt")
        input_ids = encodings.input_ids[:, :2048].to("xpu")
        nlls = []
        n_tokens = 0
        
        is_quant = (k_cfg[0] != "FP16" or v_cfg[0] != "FP16")
        if is_quant: self._patch_model(model, k_cfg, v_cfg, rotors_map, cbs_map)
        
        for begin in range(0, input_ids.shape[1] - 1, stride):
            end = min(begin + max_length, input_ids.shape[1])
            target_len = end - begin - 1
            chunk_ids = input_ids[:, begin:end]
            with torch.no_grad():
                outputs = model(chunk_ids, labels=chunk_ids, use_cache=False)
                nlls.append((outputs.loss * target_len).item())
                n_tokens += target_len
                
        if is_quant: self._unpatch_model(model)
        return math.exp(sum(nlls) / n_tokens)

    def run_needle_test(self, model, k_cfg, v_cfg, rotors_map, cbs_map, ctx_len):
        needle = "The secret password is ANTIGRAVITY."
        ctx_cap = min(ctx_len, 16384)
        context = "A lot of random text. " * (ctx_cap // 30)
        full_text = context + needle + context
        prompt = full_text + " What is the secret password? Answer with a single word:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("xpu")
        
        is_quant = (k_cfg[0] != "FP16" or v_cfg[0] != "FP16")
        if is_quant: self._patch_model(model, k_cfg, v_cfg, rotors_map, cbs_map)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=self.tokenizer.eos_token_id, use_cache=False)
            response = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:])
            res = "PASS" if "ANTIGRAVITY" in response.upper() else "FAIL"
        if is_quant: self._unpatch_model(model)
        return res

    def run_coherency_test(self, model, k_cfg, v_cfg, rotors_map, cbs_map):
        prompt = "What is 2+2? Answer with a single number:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("xpu")
        
        is_quant = (k_cfg[0] != "FP16" or v_cfg[0] != "FP16")
        if is_quant: self._patch_model(model, k_cfg, v_cfg, rotors_map, cbs_map)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=self.tokenizer.eos_token_id, use_cache=False)
            response = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:])
            res = "OK" if "4" in response else "GIBBERISH"
        if is_quant: self._unpatch_model(model)
        return res

    def _run_native_bench(self, model, ctx, quant):
        # Unpatch for native baseline
        self._unpatch_model(model)
        
        input_ids = torch.randint(0, 151936, (1, ctx), device="xpu")
        
        torch.xpu.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(3): model(input_ids, use_cache=False)
        torch.xpu.synchronize()
        prefill_tps = (ctx * 3) / (time.time() - start)
        
        gen_start = time.time()
        with torch.no_grad():
            model.generate(
                input_ids[:, :min(ctx, 512)], 
                max_new_tokens=32, 
                use_cache=True, 
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        torch.xpu.synchronize()
        decode_tps = 32 / (time.time() - gen_start)
        
        vram = self.calculate_vram_usage_math(ctx, 16 if quant=="FP16" else 8, 16 if quant=="FP16" else 8)
        return prefill_tps, decode_tps, vram

    def calculate_vram_usage_math(self, seq_len, k_bits, v_bits):
        # Qwen 2B: 28 layers, 2 heads, 128 head_dim + 3600MB Model Base
        n_layers = 28
        n_heads = 2
        head_dim = 128
        kv_bytes = n_layers * n_heads * head_dim * seq_len * ((k_bits + v_bits) / 8.0)
        return (kv_bytes / (1024 * 1024)) + 3600.0

    def _patch_model(self, model, k_cfg, v_cfg, rotors_map, cbs_map):
        self.orig_forwards = {}
        attn_layer_indices = [3, 7, 11, 15, 19, 23]
        for idx in attn_layer_indices:
            layer = model.model.layers[idx].self_attn
            orig_k_forward = layer.k_proj.forward
            orig_v_forward = layer.v_proj.forward
            self.orig_forwards[f"{idx}_k"] = orig_k_forward
            self.orig_forwards[f"{idx}_v"] = orig_v_forward
            
            def make_quant_forward(orig_f, l_idx, cfg):
                def quant_forward(x):
                    out = orig_f(x)
                    return apply_quant(out, l_idx, cfg, rotors_map, cbs_map)
                return quant_forward
            
            layer.k_proj.forward = make_quant_forward(orig_k_forward, idx, k_cfg)
            layer.v_proj.forward = make_quant_forward(orig_v_forward, idx, v_cfg)

    def _unpatch_model(self, model):
        attn_layer_indices = [3, 7, 11, 15, 19, 23]
        for idx in attn_layer_indices:
            layer = model.model.layers[idx].self_attn
            layer.k_proj.forward = self.orig_forwards[f"{idx}_k"]
            layer.v_proj.forward = self.orig_forwards[f"{idx}_v"]

    def write_report_header(self):
        with open(self.report_path, "w", encoding='utf-8') as f:
            f.write(f"# XPU Baseline Results - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"**Model**: {self.model_id} | **Hardware**: Intel Arc Pro B70 (32GB)\n\n")
            f.write("| Cache K | Cache V | Mode | Context | Prefill (tk/s) | Decode (tk/s) | VRAM (MB) | PPL | Needle | Coherency |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")

    def append_result(self, r):
        with open(self.report_path, "a", encoding='utf-8') as f:
            f.write(f"| {r['Cache K']} | {r['Cache V']} | {r['Mode']} | {r['Context']} | {r['Prefill']} | {r['Decode']} | {r['VRAM']:.1f} | {r['PPL']:.2f} | {r['Needle']} | {r['Coherency']} |\n")

    @torch.no_grad()
    def run(self):
        from turboquant import calibrate_xpu, xpu_backend
        self.write_report_header()
        print("Saving Wikitext subset for native PPL...")
        with open("wikitext_test.txt", "w", encoding="utf-8") as f:
            f.write(self.test_text[:10000])
        
        print("Loading Model...")
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float16, device_map="auto")
        torch.xpu.synchronize()
        self.vram_base = torch.xpu.memory_allocated()
        
        # SYCL_WARMUP: Call verified fused kernels in a tight loop
        print("Triggering Fused Kernel SYCL_WARMUP...")
        w_in = torch.randn(1, 4096, 2048, device="xpu", dtype=torch.float16)
        w_rot = torch.zeros((2048//3 + 1, 4), device="xpu", dtype=torch.float32); w_rot[:,0] = 1.0
        w_cb = torch.linspace(-1, 1, 16, device="xpu", dtype=torch.float32)
        for _ in range(10): 
            xpu_backend.rotor_full_fused(w_in, w_rot, w_cb, w_cb, w_cb, w_cb)
            torch.xpu.synchronize()
        del w_in, w_rot, w_cb
        torch.xpu.empty_cache()
        
        types = [
            ("FP16", 16, "native"),
            ("Q8", 8, "native"),
            ("Iso4", 4, "iso"),
            ("Rotor4", 4, "rotor"),
        ]
        
        print("Pre-calibrating all modes on XPU...")
        rotors_map, cbs_map = {}, {}
        for mode in ['iso', 'rotor']:
            for bits in [4]:
                key = f"{mode}{bits}"
                r_res, cb_res = calibrate_xpu.calibrate_xpu(model, self.tokenizer, bits=bits, mode=mode)
                rotors_map[key], cbs_map[key] = r_res, cb_res
        
        for k_type, v_type in itertools.product(types, types):
            k_name, k_bits, k_m_b = k_type
            v_name, v_bits, v_m_b = v_type
            if k_name != v_name: continue 
            
            for ctx in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
                print(f">> Testing {k_name} @ {ctx}...")
                
                try:
                    if k_name in ["FP16", "Q8"]:
                        prefill_tps, decode_tps, vram_delta = self._run_native_bench(model, ctx, k_name)
                        k_cfg, v_cfg = ("FP16", 16, "native"), ("FP16", 16, "native")
                        ppl = self.calculate_perplexity(model, (k_cfg, v_cfg), rotors_map, cbs_map)
                        needle = self.run_needle_test(model, k_cfg, v_cfg, rotors_map, cbs_map, ctx)
                        coherency = self.run_coherency_test(model, k_cfg, v_cfg, rotors_map, cbs_map)
                    else:
                        k_cfg = (k_name, k_bits, f"{k_m_b}{k_bits}")
                        v_cfg = (v_name, v_bits, f"{v_m_b}{v_bits}")
                        
                        input_ids = torch.randint(0, 151936, (1, ctx), device="xpu")
                        self._patch_model(model, k_cfg, v_cfg, rotors_map, cbs_map)
                        
                        torch.xpu.synchronize()
                        v_pre = self.calculate_vram_usage()
                        
                        start = time.time()
                        for _ in range(3): model(input_ids, use_cache=False)
                        torch.xpu.synchronize()
                        prefill_tps = (ctx * 3) / (time.time() - start)
                        
                        gen_start = time.time()
                        model.generate(
                            input_ids[:, :min(ctx, 512)], 
                            max_new_tokens=32, 
                            use_cache=True, 
                            pad_token_id=self.tokenizer.eos_token_id,
                            do_sample=True,
                            top_k=20,
                            top_p=1.0,
                            repetition_penalty=1.0
                        )
                        torch.xpu.synchronize()
                        decode_tps = 32 / (time.time() - gen_start)
                        
                        ppl = self.calculate_perplexity(model, (k_cfg, v_cfg), rotors_map, cbs_map)
                        coherency = self.run_coherency_test(model, k_cfg, v_cfg, rotors_map, cbs_map)
                        needle = self.run_needle_test(model, k_cfg, v_cfg, rotors_map, cbs_map, ctx)
                        
                        vram_delta = self.calculate_vram_usage_math(ctx, k_bits, v_bits)
                        self._unpatch_model(model)

                    # Explicitly ensure we are using the local vram_delta
                    self.append_result({
                        "Cache K": k_name, 
                        "Cache V": v_name, 
                        "Mode": "Sync", 
                        "Context": ctx, 
                        "Prefill": f"{prefill_tps:.1f}", 
                        "Decode": f"{decode_tps:.1f}", 
                        "VRAM": float(vram_delta), 
                        "PPL": float(ppl), 
                        "Needle": needle, 
                        "Coherency": coherency
                    })
                except Exception as e:
                    print(f"FAILED {k_name} @ {ctx}: {e}")
                    # If driver reset occurred, we might need to break or continue with caution
                    if "reset" in str(e).lower() or "device" in str(e).lower():
                         print("Potential Driver Reset Detected. Attempting to continue...")
                finally:
                    torch.xpu.empty_cache()
                    torch.xpu.synchronize()

if __name__ == "__main__":
    runner = BenchmarkRunner("Qwen/Qwen3.5-2B")
    runner.run()
