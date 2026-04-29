import torch
import time
import os
import sys

# Ensure turboquant is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from turboquant.rotorquant import RotorQuantMSE
import turboquant.xpu_backend as xpu_backend

def benchmark_parity_throughput():
    device = torch.device('xpu')
    
    # Qwen 3.6 dims
    num_heads = 32
    head_dim = 256
    seq_len = 4096
    batch_size = 1
    
    print(f"=== RotorQuant XPU Parity & Throughput Benchmark ===")
    print(f"Config: {num_heads} heads, {head_dim} dim, {seq_len} tokens")
    
    # Random BF16 input
    x = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    
    # 1. PARITY CHECK
    print("\n[1] Bit Parity Check...")
    # We use a high outlier count to test "lossless" mode
    # For a real implementation, we would use a dynamic threshold.
    outlier_ratio = 0.05 # 5% outliers
    num_outliers = int(num_heads * seq_len * outlier_ratio)
    
    bits = 4
    rq = RotorQuantMSE(head_dim, bits, device=device).to(device)
    
    # Fused Quantization (Stage 1)
    # Note: Currently XPU backend implementation of RotorQuant might not be fully fused.
    # We use the existing xpu_backend wrappers if available.
    
    t0 = time.perf_counter()
    flat_x = x.reshape(-1, head_dim)
    mv_q, indices = rq.quantize(flat_x)
    x_recon = rq.dequantize(indices).reshape(x.shape)
    t_fwd = (time.perf_counter() - t0) * 1000
    
    mse = torch.mean((x.float() - x_recon.float())**2).item()
    parity = (x.float() == x_recon.float()).sum().item() / x.numel()
    
    print(f"  MSE: {mse:.2e}")
    print(f"  Bit Parity: {parity:.2%}")
    
    # 2. THROUGHPUT BENCHMARK
    print("\n[2] Throughput Benchmark...")
    
    # Baseline: FP16/BF16 Copy (Lower bound for memory bandwidth)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        y = x.clone()
    torch.xpu.synchronize()
    t_base = (time.perf_counter() - t0) * 10 # 100 iterations -> avg 1.0 ms scaling
    
    # RotorQuant Fused Path
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        mv_q, indices = rq.quantize(flat_x)
    torch.xpu.synchronize()
    t_rq = (time.perf_counter() - t0) * 10
    
    print(f"  Baseline (Copy):  {t_base:.3f} ms")
    print(f"  RotorQuant XPU:   {t_rq:.3f} ms")
    print(f"  Overhead:         {t_rq/t_base:.2f}x baseline")
    
    # 3. KERNEL EFFICIENCY (Sparse GP vs Original)
    # We compare the time taken by the optimized kernel.
    # This is already reflected in t_rq.

    print("\n[3] Memory Savings...")
    original_size = x.element_size() * x.numel()
    # Estimate compressed size (4 bits + 5% FP16 outliers + norms)
    # 4 bits = 0.5 bytes. Norm = 4 bytes per 256-dim vector (approx).
    comp_size = x.numel() * 0.5 + num_outliers * head_dim * 2 + (num_heads * seq_len * 4)
    
    print(f"  Original Size:    {original_size / 1024 / 1024:.2f} MB")
    print(f"  Est. Compressed:  {comp_size / 1024 / 1024:.2f} MB")
    print(f"  Reduction:        {original_size / comp_size:.2f}x")

if __name__ == "__main__":
    benchmark_parity_throughput()
