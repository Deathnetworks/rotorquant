# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
RotorQuant XPU Kernel: Sandwich Roundtrip Parity & Throughput Benchmark

Tests:
1. Bit parity: FP16 input → sandwich → inverse → compare with original
2. Throughput: XPU sandwich roundtrip vs FP16 copy baseline
3. Storage: Compressed size vs original FP16

This tests the XPU kernel directly — no Python-side Clifford math.
"""
import torch
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from turboquant import xpu_backend

def make_compact_rotors(n_groups, device, seed=42):
    """Create compact (n_groups, 4) unit rotors: [s, p12, p13, p23]."""
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    
    # Random bivector components 
    bv = torch.randn(n_groups, 3, generator=gen)
    # Normalize to unit rotor: R = cos(θ/2) + sin(θ/2) * B̂
    angle = torch.rand(n_groups, 1, generator=gen) * 3.14159  # [0, π]
    bv_norm = bv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    bv_hat = bv / bv_norm
    
    s = torch.cos(angle / 2)
    sin_ha = torch.sin(angle / 2)
    p12 = sin_ha * bv_hat[:, 0:1]
    p13 = sin_ha * bv_hat[:, 1:2]
    p23 = sin_ha * bv_hat[:, 2:3]
    
    rotors = torch.cat([s, p12, p13, p23], dim=-1).to(device=device, dtype=torch.float32)
    return rotors.contiguous()

def make_full_rotors(n_groups, device, seed=42):
    """Create full (n_groups, 8) multivector rotors for Python API compatibility."""
    compact = make_compact_rotors(n_groups, device, seed)
    full = torch.zeros(n_groups, 8, device=device, dtype=torch.float32)
    full[:, 0] = compact[:, 0]   # scalar
    full[:, 4] = compact[:, 1]   # e12
    full[:, 5] = compact[:, 2]   # e13
    full[:, 6] = compact[:, 3]   # e23
    return full.contiguous()


def test_sandwich_roundtrip_parity(device, emb_dim=128, batch_size=2048):
    """Test: sandwich → inverse sandwich should be bit-exact for FP16 input."""
    n_groups = (emb_dim + 2) // 3
    
    # Generate deterministic FP16 data
    torch.manual_seed(123)
    x_fp16 = torch.randn(batch_size, emb_dim, device=device, dtype=torch.float16)
    
    # Full rotors (8-component) — xpu_backend._compact_rotors will extract [0,4,5,6]
    rotors = make_full_rotors(n_groups, device)
    
    # Forward sandwich: (batch, emb_dim) → (batch, n_groups, 8)
    mv = xpu_backend.rotor_sandwich(x_fp16, rotors)
    
    # Inverse sandwich: (batch, n_groups, 8) → (batch, emb_dim) 
    x_recon = xpu_backend.rotor_inverse(mv, rotors, emb_dim)
    
    torch.xpu.synchronize()
    
    # Parity check
    x_fp16_f32 = x_fp16.float()
    x_recon_f32 = x_recon.float()
    
    total_elements = x_fp16.numel()
    exact_matches = (x_fp16 == x_recon).sum().item()
    mse = ((x_fp16_f32 - x_recon_f32) ** 2).mean().item()
    max_err = (x_fp16_f32 - x_recon_f32).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        x_fp16_f32.view(1, -1), x_recon_f32.view(1, -1)
    ).item()
    
    parity_pct = exact_matches / total_elements * 100
    
    return {
        'total_elements': total_elements,
        'exact_matches': exact_matches,
        'parity_pct': parity_pct,
        'mse': mse,
        'max_err': max_err,
        'cos_sim': cos_sim,
    }


def test_throughput(device, emb_dim=128, seq_lengths=[1024, 4096, 8192, 16384, 32768, 65536, 131072]):
    """Throughput: XPU sandwich roundtrip vs FP16 copy baseline."""
    n_groups = (emb_dim + 2) // 3
    rotors = make_full_rotors(n_groups, device)
    
    results = []
    for seq in seq_lengths:
        try:
            x = torch.randn(seq, emb_dim, device=device, dtype=torch.float16)
            
            # Warmup
            for _ in range(5):
                mv = xpu_backend.rotor_sandwich(x, rotors)
                _ = xpu_backend.rotor_inverse(mv, rotors, emb_dim)
            torch.xpu.synchronize()
            
            # Benchmark XPU roundtrip
            iters = 50 if seq <= 16384 else 20
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                mv = xpu_backend.rotor_sandwich(x, rotors)
                x_recon = xpu_backend.rotor_inverse(mv, rotors, emb_dim)
            torch.xpu.synchronize()
            t_xpu = (time.perf_counter() - t0) / iters * 1000
            
            # Baseline: simple FP16 copy (memory bandwidth bound)
            y = torch.empty_like(x)
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                y.copy_(x)
            torch.xpu.synchronize()
            t_copy = (time.perf_counter() - t0) / iters * 1000
            
            # Baseline: FP16 matmul (compute bound reference)
            w = torch.randn(emb_dim, emb_dim, device=device, dtype=torch.float16)
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = torch.matmul(x, w)
            torch.xpu.synchronize()
            t_matmul = (time.perf_counter() - t0) / iters * 1000
            
            # Storage analysis
            original_bytes = seq * emb_dim * 2  # FP16 = 2 bytes
            # MV representation: (seq, n_groups, 8) in FP16
            mv_bytes = seq * n_groups * 8 * 2
            
            results.append({
                'seq': seq,
                't_xpu_ms': t_xpu,
                't_copy_ms': t_copy,
                't_matmul_ms': t_matmul,
                'speedup_vs_matmul': t_matmul / t_xpu if t_xpu > 0 else 0,
                'original_mb': original_bytes / (1024**2),
                'mv_mb': mv_bytes / (1024**2),
                'expansion': mv_bytes / original_bytes,
            })
            
            del x, y, w, mv
            torch.xpu.empty_cache()
            
        except Exception as e:
            results.append({'seq': seq, 'error': str(e)})
            torch.xpu.empty_cache()
    
    return results


if __name__ == "__main__":
    if not torch.xpu.is_available():
        print("XPU not available")
        sys.exit(1)
    
    device = torch.device('xpu')
    gpu_name = torch.xpu.get_device_name(0)
    print(f"RotorQuant XPU Kernel Benchmark")
    print(f"GPU: {gpu_name}")
    print(f"XPU kernels: {xpu_backend.is_xpu_available()}")
    print()
    
    # ─── Test 1: Bit Parity ───
    print("=" * 74)
    print("TEST 1: Sandwich Roundtrip Bit Parity (FP16)")
    print("=" * 74)
    
    for dim in [128, 256, 512]:
        result = test_sandwich_roundtrip_parity(device, emb_dim=dim, batch_size=4096)
        status = "PASS" if result['parity_pct'] == 100.0 else "FAIL"
        print(f"  dim={dim:4d}: [{status}]  parity={result['parity_pct']:.4f}%  "
              f"MSE={result['mse']:.2e}  MaxErr={result['max_err']:.2e}  "
              f"CosSim={result['cos_sim']:.8f}")
    
    # Also test BF16
    print()
    for dim in [128, 256]:
        n_groups = (dim + 2) // 3
        x = torch.randn(2048, dim, device=device, dtype=torch.bfloat16)
        rotors = make_full_rotors(n_groups, device)
        mv = xpu_backend.rotor_sandwich(x, rotors)
        x_recon = xpu_backend.rotor_inverse(mv, rotors, dim)
        torch.xpu.synchronize()
        exact = (x == x_recon).sum().item()
        total = x.numel()
        parity = exact / total * 100
        status = "PASS" if parity == 100.0 else "FAIL"
        mse = ((x.float() - x_recon.float()) ** 2).mean().item()
        print(f"  dim={dim:4d} BF16: [{status}]  parity={parity:.4f}%  MSE={mse:.2e}")
    
    # ─── Test 2: Throughput ───
    print()
    print("=" * 74)
    print("TEST 2: Throughput (FP16 Sandwich Roundtrip)")
    print("=" * 74)
    print(f"{'Seq Len':>10} | {'XPU RT (ms)':>12} | {'Copy (ms)':>10} | {'Matmul (ms)':>12} | {'vs Matmul':>10} | {'MV Expn':>8}")
    print("-" * 74)
    
    throughput = test_throughput(device)
    for r in throughput:
        if 'error' in r:
            print(f"{r['seq']:>10} | ERROR: {r['error']}")
        else:
            print(f"{r['seq']:>10} | {r['t_xpu_ms']:>12.3f} | {r['t_copy_ms']:>10.3f} | "
                  f"{r['t_matmul_ms']:>12.3f} | {r['speedup_vs_matmul']:>9.2f}x | "
                  f"{r['expansion']:>7.2f}x")
    
    print("=" * 74)
    print("Benchmarks Complete")
