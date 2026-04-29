import torch
import time
import numpy as np
from turboquant import xpu_backend

if __name__ == "__main__":
    if not torch.xpu.is_available():
        print("XPU not available")
        exit(1)
        
    print("TurboQuant XPU Optimization Benchmark (Stabilized)")
    print("=" * 70)
    print(f"{'Seq Len':>10} | {'PyTorch FP16 (ms)':>18} | {'XPU Opt (ms)':>15} | {'Speedup':>10}")
    print("-" * 70)
    
    # Test up to 128k sequence length
    for seq in [1024, 4096, 8192, 16384, 32768, 65536, 131072]:
        try:
            device = torch.device('xpu')
            group_size = 32
            num_groups = seq // group_size
            d = 128
            batch, head = 1, 32
            
            keys = torch.randn(batch, head, num_groups, group_size, d, device=device).half()
            proj = torch.randn(128, d, device=device).half()
            # Outlier indices: [batch, head, num_groups, d // 8] - matching kernel expectation
            outlier_idx = torch.zeros((batch, head, num_groups, 16), device=device, dtype=torch.uint8)
            
            # Warmup
            for _ in range(5):
                _ = xpu_backend.qjl_quant(keys, outlier_idx, proj, 128)
            torch.xpu.synchronize()
            
            # Benchmark XPU
            iters = 50 if seq < 16384 else 10
            t0 = time.perf_counter()
            for i in range(iters):
                _ = xpu_backend.qjl_quant(keys, outlier_idx, proj, 128)
                if i % 2 == 0: torch.xpu.synchronize() # Frequent sync for stability at 128k
            torch.xpu.synchronize()
            t_xpu = (time.perf_counter() - t0) / iters * 1000
            
            # Benchmark PyTorch FP16 Baseline
            t0 = time.perf_counter()
            for i in range(iters):
                res = torch.matmul(keys.half(), proj.half().t())
                _ = (res > 0).to(torch.uint8)
                if i % 2 == 0: torch.xpu.synchronize()
            torch.xpu.synchronize()
            t_pt = (time.perf_counter() - t0) / iters * 1000
            
            speedup = t_pt / t_xpu if t_xpu > 0 else 0
            print(f"{seq:>10} | {t_pt:>18.3f} | {t_xpu:>15.3f} | {speedup:>9.2f}x")
            
            # Clean up
            del keys, proj, outlier_idx
            torch.xpu.empty_cache()
            
        except Exception as e:
            print(f"{seq:>10} | FAILED: {str(e)}")
            torch.xpu.empty_cache()
    
    print("=" * 70)
    print("Benchmarks Complete")
