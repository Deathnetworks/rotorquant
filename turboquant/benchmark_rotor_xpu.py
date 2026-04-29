import torch
import turboquant.xpu_backend as xpu_backend
import time

def benchmark_rotor_latency():
    print("\n=== RotorQuant XPU Latency Benchmark (Prefill vs Decode) ===")
    device = torch.device('xpu')
    
    # Shapes
    batch = 1
    prefill_seq = 1024
    decode_seq = 1
    head_dim = 128
    n_heads = 32
    
    n_groups = (head_dim + 2) // 3
    rotors = torch.randn(n_groups, 4, device=device, dtype=torch.float32)
    rotors = rotors / torch.norm(rotors, dim=-1, keepdim=True)
    
    n_levels = 256 # 8-bit VQ
    c_scalar = torch.randn(n_levels, device=device, dtype=torch.float32)
    c_vector = torch.randn(n_levels * 3, device=device, dtype=torch.float32)
    c_bivector = torch.randn(n_levels * 3, device=device, dtype=torch.float32)
    c_trivector = torch.randn(n_levels, device=device, dtype=torch.float32)

    def run_bench(inp, name):
        # Warmup
        for _ in range(5):
            _ = xpu_backend.rotor_full_fused(inp, rotors, c_scalar, c_vector, c_bivector, c_trivector)
            _ = xpu_backend.rotor_compress(inp, rotors, c_scalar, c_vector, c_bivector, c_trivector)
        torch.xpu.synchronize()
        
        iters = 50
        # 1. Full Fused (Roundtrip)
        start = time.perf_counter()
        for _ in range(iters):
            _ = xpu_backend.rotor_full_fused(inp, rotors, c_scalar, c_vector, c_bivector, c_trivector)
        torch.xpu.synchronize()
        ms_full = (time.perf_counter() - start) * 1000 / iters

        # 2. Compress Only
        start = time.perf_counter()
        for _ in range(iters):
            _ = xpu_backend.rotor_compress(inp, rotors, c_scalar, c_vector, c_bivector, c_trivector)
        torch.xpu.synchronize()
        ms_comp = (time.perf_counter() - start) * 1000 / iters
        
        # 3. Baseline (FP16 Copy)
        start = time.perf_counter()
        for _ in range(iters):
            _ = inp.clone()
        torch.xpu.synchronize()
        ms_base = (time.perf_counter() - start) * 1000 / iters
        
        print(f"{name:10} | Full: {ms_full:7.4f} ms | Comp: {ms_comp:7.4f} ms | FP16: {ms_base:7.4f} ms | Gain: {ms_base/ms_comp:5.2f}x")

    # Prefill (Large sequence)
    prefill_inp = torch.randn(batch, n_heads, prefill_seq, head_dim, device=device, dtype=torch.half)
    run_bench(prefill_inp, "Prefill")

    # Decode (Single token)
    decode_inp = torch.randn(batch, n_heads, decode_seq, head_dim, device=device, dtype=torch.half)
    run_bench(decode_inp, "Decode")

if __name__ == "__main__":
    benchmark_rotor_latency()
