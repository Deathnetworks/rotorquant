import torch
import torch.nn as nn
from turboquant import xpu_backend, isoquant, planarquant
import time

def evaluate_variants():
    device = torch.device("xpu")
    head_dim = 128
    n_heads = 32
    seq_len = 1024
    bits = 4
    n_levels = 2**bits
    
    # 1. Setup IsoQuant (4D)
    iso = isoquant.IsoQuantMSE(head_dim, bits, mode='fast', device='cpu')
    qL = iso.q_L.to(device)
    cb_iso = iso.centroids.to(device)
    
    # 2. Setup PlanarQuant (2D)
    planar = planarquant.PlanarQuantMSE(head_dim, bits, device='cpu')
    cs = planar.rot2.to(device)
    cb_planar = planar.centroids.to(device)
    
    x = torch.randn(1, n_heads, seq_len, head_dim, device=device, dtype=torch.half)
    
    print(f"=== Testing Variants (4-bit, {head_dim} dim) ===")
    
    # Warmup
    for _ in range(5):
        _ = xpu_backend.isoquant_fused(x, qL, cb_iso)
        _ = xpu_backend.planar_fused(x, cs, cb_planar)
    torch.xpu.synchronize()
    
    # Bench IsoQuant
    iters = 100
    start = time.perf_counter()
    for _ in range(iters):
        y_iso = xpu_backend.isoquant_fused(x, qL, cb_iso)
    torch.xpu.synchronize()
    ms_iso = (time.perf_counter() - start) * 1000 / iters
    diff_iso = (x - y_iso).abs().mean().item()
    
    # Bench PlanarQuant
    start = time.perf_counter()
    for _ in range(iters):
        y_planar = xpu_backend.planar_fused(x, cs, cb_planar)
    torch.xpu.synchronize()
    ms_planar = (time.perf_counter() - start) * 1000 / iters
    diff_planar = (x - y_planar).abs().mean().item()

    # Bench FP16 Baseline
    start = time.perf_counter()
    for _ in range(iters):
        _ = x.clone()
    torch.xpu.synchronize()
    ms_base = (time.perf_counter() - start) * 1000 / iters

    print(f"IsoQuant (4D)   | Time: {ms_iso:7.4f} ms | L1 Error: {diff_iso:.6f} | Speed: {ms_base/ms_iso:.2f}x")
    print(f"PlanarQuant (2D)| Time: {ms_planar:7.4f} ms | L1 Error: {diff_planar:.6f} | Speed: {ms_base/ms_planar:.2f}x")
    print(f"FP16 Baseline   | Time: {ms_base:7.4f} ms")

if __name__ == "__main__":
    evaluate_variants()
