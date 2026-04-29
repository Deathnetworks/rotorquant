# -*- coding: utf-8 -*-
"""
RotorQuant XPU Autoloop Benchmark: KV Cache Parity + Throughput

Simulates Qwen 3.6 KV cache shapes and tests:
  1. Parity: encode -> decode -> compare (target: Q8 or better)
  2. Storage: compressed vs original size
  3. Prefill throughput (batch encode)
  4. Decode throughput (single-token encode)

Only exercises XPU kernels directly - no Python Clifford math.
"""
import sys, io, os, time, json, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from turboquant import xpu_backend

# ─── Qwen 3.6 Config ────────────────────────────────────────────────────
QWEN36_HEAD_DIM = 256
QWEN36_NUM_KV_HEADS = 2
QWEN36_FULL_ATTN_LAYERS = 10  # layers 3,7,11,...,39
QWEN36_DTYPE = torch.bfloat16

# Test sequence lengths
SEQ_LENGTHS_PREFILL = [1024, 4096, 16384]
SEQ_LENGTHS_FULL = [1024, 4096, 16384, 65536, 131072]


def make_compact_rotors(n_groups, device, seed=42):
    """Create compact (n_groups, 4) unit rotors: [s, p12, p13, p23]."""
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    bv = torch.randn(n_groups, 3, generator=gen)
    angle = torch.rand(n_groups, 1, generator=gen) * math.pi
    bv_norm = bv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    bv_hat = bv / bv_norm
    s = torch.cos(angle / 2)
    sin_ha = torch.sin(angle / 2)
    rotors = torch.cat([s, sin_ha * bv_hat[:, 0:1],
                        sin_ha * bv_hat[:, 1:2],
                        sin_ha * bv_hat[:, 2:3]], dim=-1)
    return rotors.to(device=device, dtype=torch.float32).contiguous()


def make_full_rotors(n_groups, device, seed=42):
    """Create full (n_groups, 8) multivector rotors."""
    compact = make_compact_rotors(n_groups, device, seed)
    full = torch.zeros(n_groups, 8, device=device, dtype=torch.float32)
    full[:, 0] = compact[:, 0]   # scalar
    full[:, 4] = compact[:, 1]   # e12
    full[:, 5] = compact[:, 2]   # e13
    full[:, 6] = compact[:, 3]   # e23
    return full.contiguous()


def q8_reference_error(x):
    """Calculate the error introduced by Q8_0 quantization as reference."""
    # Q8_0: per-block (32 elements) symmetric quantization to INT8
    block_size = 32
    flat = x.float().reshape(-1, block_size)
    scale = flat.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-10)
    quant = (flat / scale).round().clamp(-128, 127)
    dequant = quant * scale
    dequant = dequant.reshape(x.shape).to(x.dtype)
    mse = ((x.float() - dequant.float()) ** 2).mean().item()
    max_err = (x.float() - dequant.float()).abs().max().item()
    parity = (x == dequant).sum().item() / x.numel() * 100
    return {'mse': mse, 'max_err': max_err, 'parity_pct': parity}


# ─── Test 1: Parity ─────────────────────────────────────────────────────

def test_parity(device, head_dim=256, seq_len=4096, dtype=torch.bfloat16):
    """Sandwich roundtrip parity on Qwen 3.6 shaped KV cache."""
    n_groups = (head_dim + 2) // 3
    batch = QWEN36_NUM_KV_HEADS  # 2 KV heads
    total_vectors = batch * seq_len

    torch.manual_seed(42)
    x = torch.randn(total_vectors, head_dim, device=device, dtype=dtype)

    rotors = make_full_rotors(n_groups, device)

    # Fused roundtrip (single kernel launch)
    x_recon = xpu_backend.rotor_roundtrip(x, rotors)
    torch.xpu.synchronize()

    # Parity metrics
    total = x.numel()
    exact = (x == x_recon).sum().item()
    mse = ((x.float() - x_recon.float()) ** 2).mean().item()
    max_err = (x.float() - x_recon.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        x.float().reshape(1, -1), x_recon.float().reshape(1, -1)
    ).item()

    # Q8 reference for comparison
    q8_ref = q8_reference_error(x)

    # Relative error (RMSE / mean abs value)
    mean_abs = x.float().abs().mean().item()
    rmse = math.sqrt(mse)
    rel_err = rmse / mean_abs if mean_abs > 0 else float('inf')

    return {
        'dtype': str(dtype).split('.')[-1],
        'seq_len': seq_len,
        'head_dim': head_dim,
        'parity_pct': exact / total * 100,
        'mse': mse,
        'max_err': max_err,
        'cos_sim': cos_sim,
        'rel_err': rel_err,
        'q8_mse': q8_ref['mse'],
        'q8_max_err': q8_ref['max_err'],
        'q8_parity_pct': q8_ref['parity_pct'],
        'better_than_q8': mse <= q8_ref['mse'],
    }


def test_quantized_parity(device, head_dim=256, seq_len=4096, dtype=torch.bfloat16, n_levels=16):
    """Full fused pipeline parity: embed -> sandwich -> QUANTIZE -> inverse -> extract.
    
    This is the real compression path. Quantization is lossy but should still beat Q8.
    """
    n_groups = (head_dim + 2) // 3
    total_vectors = QWEN36_NUM_KV_HEADS * seq_len

    torch.manual_seed(42)
    x = torch.randn(total_vectors, head_dim, device=device, dtype=dtype)
    rotors = make_full_rotors(n_groups, device)

    # Lloyd-Max centroids (uniform approximation for now)
    # In production these would be learned from calibration data
    centroids = torch.linspace(-3.0, 3.0, n_levels, device=device, dtype=torch.float32)

    # Run the full fused kernel (embed -> sandwich -> quantize -> inverse -> extract)
    x_recon = xpu_backend.rotor_full_fused(
        x, rotors,
        centroids, centroids, centroids, centroids  # same centroids for all grades
    )
    torch.xpu.synchronize()

    total = x.numel()
    exact = (x == x_recon).sum().item()
    mse = ((x.float() - x_recon.float()) ** 2).mean().item()
    max_err = (x.float() - x_recon.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        x.float().reshape(1, -1), x_recon.float().reshape(1, -1)
    ).item()
    q8_ref = q8_reference_error(x)

    # Storage: with N-bit quantization
    import math as _math
    bits_per_level = _math.ceil(_math.log2(n_levels))  # e.g. 16 levels = 4 bits
    # Per token: n_groups * 8 indices * bits_per_level bits + per-group norms
    index_bits_per_token = n_groups * 8 * bits_per_level
    norm_bytes_per_token = n_groups * 4 * 2  # 4 grade norms, FP16 each
    total_compressed_per_token = index_bits_per_token / 8 + norm_bytes_per_token
    original_per_token = head_dim * (2 if dtype != torch.float32 else 4)
    compression_ratio = original_per_token / total_compressed_per_token

    return {
        'dtype': str(dtype).split('.')[-1],
        'seq_len': seq_len,
        'n_levels': n_levels,
        'bits_per_level': bits_per_level,
        'parity_pct': exact / total * 100,
        'mse': mse,
        'max_err': max_err,
        'cos_sim': cos_sim,
        'better_than_q8': mse <= q8_ref['mse'],
        'q8_mse': q8_ref['mse'],
        'original_bytes_per_tok': original_per_token,
        'compressed_bytes_per_tok': total_compressed_per_token,
        'compression_ratio': compression_ratio,
    }


# ─── Test 2: Storage ─────────────────────────────────────────────────────

def test_storage(head_dim=256, seq_len=4096, dtype=torch.bfloat16):
    """Storage comparison: original vs compressed."""
    n_groups = (head_dim + 2) // 3
    elem_size = 2  # BF16/FP16 = 2 bytes

    # Original: [num_kv_heads, seq_len, head_dim] per layer
    original_per_layer = QWEN36_NUM_KV_HEADS * seq_len * head_dim * elem_size * 2  # K+V
    original_total = original_per_layer * QWEN36_FULL_ATTN_LAYERS

    # Compressed sandwich representation: [num_kv_heads, seq_len, n_groups, 8] in same dtype
    # Plus rotors: [n_groups, 4] in float32 (fixed per layer, amortized)
    mv_per_layer = QWEN36_NUM_KV_HEADS * seq_len * n_groups * 8 * elem_size * 2  # K+V
    rotors_per_layer = n_groups * 4 * 4  # float32
    compressed_per_layer = mv_per_layer + rotors_per_layer

    # Expansion factor for raw MV (before quantization)
    # head_dim=256 -> n_groups=86 -> 86*8=688 components vs 256 original
    mv_expansion = (n_groups * 8) / head_dim

    return {
        'seq_len': seq_len,
        'original_per_layer_kb': original_per_layer / 1024,
        'original_total_mb': original_total / (1024**2),
        'mv_per_layer_kb': compressed_per_layer / 1024,
        'mv_expansion': mv_expansion,
        'rotor_overhead_bytes': rotors_per_layer,
        # Note: actual compression comes from quantizing the MV to indices
        # The raw MV is LARGER than original. Compression only happens after
        # Lloyd-Max quantization of the MV components to N-bit indices.
    }


# ─── Test 3: Prefill Throughput ──────────────────────────────────────────

def test_prefill(device, head_dim=256, dtype=torch.bfloat16):
    """Prefill throughput: batch encode of entire prompt."""
    n_groups = (head_dim + 2) // 3
    rotors = make_full_rotors(n_groups, device)
    results = []

    for seq_len in SEQ_LENGTHS_PREFILL:
        try:
            total_vectors = QWEN36_NUM_KV_HEADS * seq_len
            x = torch.randn(total_vectors, head_dim, device=device, dtype=dtype)

            # Warmup
            for _ in range(3):
                _ = xpu_backend.rotor_roundtrip(x, rotors)
            torch.xpu.synchronize()

            # Benchmark: fused roundtrip (single kernel launch)
            iters = 50 if seq_len <= 4096 else 10
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                x_recon = xpu_backend.rotor_roundtrip(x, rotors)
            torch.xpu.synchronize()
            t_rt = (time.perf_counter() - t0) / iters * 1000

            # Benchmark: separate sandwich+inverse (double launch, for comparison)
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                mv = xpu_backend.rotor_sandwich(x, rotors)
                x_recon = xpu_backend.rotor_inverse(mv, rotors, head_dim)
            torch.xpu.synchronize()
            t_2launch = (time.perf_counter() - t0) / iters * 1000

            # Baseline: FP16/BF16 copy (memory bandwidth reference)
            y = torch.empty_like(x)
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                y.copy_(x)
            torch.xpu.synchronize()
            t_copy = (time.perf_counter() - t0) / iters * 1000

            # Baseline: matmul (compute reference)
            w = torch.randn(head_dim, head_dim, device=device, dtype=dtype)
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = torch.matmul(x, w)
            torch.xpu.synchronize()
            t_matmul = (time.perf_counter() - t0) / iters * 1000

            results.append({
                'seq_len': seq_len,
                'roundtrip_ms': t_rt,
                '2launch_ms': t_2launch,
                'copy_ms': t_copy,
                'matmul_ms': t_matmul,
                'vs_copy': t_copy / t_rt if t_rt > 0 else 0,
                'vs_matmul': t_matmul / t_rt if t_rt > 0 else 0,
                'fuse_speedup': t_2launch / t_rt if t_rt > 0 else 0,
                'toks_per_sec': seq_len / (t_rt / 1000),
            })
            del x, y, w
            torch.xpu.empty_cache()
        except Exception as e:
            results.append({'seq_len': seq_len, 'error': str(e)})
            torch.xpu.empty_cache()

    return results


# ─── Test 4: Decode Throughput ───────────────────────────────────────────

def test_decode(device, head_dim=256, dtype=torch.bfloat16):
    """Decode throughput: single-token encode latency."""
    n_groups = (head_dim + 2) // 3
    rotors = make_full_rotors(n_groups, device)

    # Single token: [num_kv_heads, head_dim]
    x = torch.randn(QWEN36_NUM_KV_HEADS, head_dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(20):
        _ = xpu_backend.rotor_roundtrip(x, rotors)
    torch.xpu.synchronize()

    # Benchmark: fused roundtrip
    iters = 500
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        x_recon = xpu_backend.rotor_roundtrip(x, rotors)
    torch.xpu.synchronize()
    t_rt = (time.perf_counter() - t0) / iters * 1e6  # microseconds

    # Copy baseline
    y = torch.empty_like(x)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    torch.xpu.synchronize()
    t_copy = (time.perf_counter() - t0) / iters * 1e6

    return {
        'decode_rt_us': t_rt,
        'decode_copy_us': t_copy,
        'vs_copy': t_copy / t_rt if t_rt > 0 else 0,
        'toks_per_sec': 1e6 / t_rt if t_rt > 0 else 0,
    }


# ─── Main ────────────────────────────────────────────────────────────────

def run_all(device):
    gpu_name = torch.xpu.get_device_name(0)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    print(f"RotorQuant XPU KV Cache Benchmark")
    print(f"GPU: {gpu_name}")
    print(f"Timestamp: {timestamp}")
    print(f"XPU kernels: {xpu_backend.is_xpu_available()}")
    print()

    all_results = {'gpu': gpu_name, 'timestamp': timestamp}

    # ─── Test 1: Parity ───
    print("=" * 76)
    print("TEST 1: Sandwich Roundtrip Parity (Qwen 3.6 KV shapes)")
    print("=" * 76)
    print(f"  Target: MSE <= Q8_0 reference")
    print()

    parity_results = []
    for dtype in [torch.bfloat16, torch.float16]:
        dtype_name = 'BF16' if dtype == torch.bfloat16 else 'FP16'
        for seq in [1024, 4096]:
            r = test_parity(device, head_dim=256, seq_len=seq, dtype=dtype)
            verdict = "PASS (>= Q8)" if r['better_than_q8'] else "FAIL (< Q8)"
            print(f"  {dtype_name} seq={seq:5d}: [{verdict}]  "
                  f"MSE={r['mse']:.2e} (Q8={r['q8_mse']:.2e})  "
                  f"MaxErr={r['max_err']:.2e}  CosSim={r['cos_sim']:.8f}")
            parity_results.append(r)
    all_results['parity'] = parity_results

    # ─── Test 1b: Quantized Parity ───
    print()
    print("=" * 76)
    print("TEST 1b: Quantized Fused Pipeline Parity (embed->sandwich->quant->inv->extract)")
    print("=" * 76)

    quant_results = []
    for n_levels in [16, 32, 64, 256]:
        for dtype in [torch.bfloat16]:
            r = test_quantized_parity(device, head_dim=256, seq_len=4096, dtype=dtype, n_levels=n_levels)
            verdict = "PASS" if r['better_than_q8'] else "FAIL"
            bits = r['bits_per_level']
            print(f"  {bits}-bit ({n_levels} lvl): [{verdict}]  MSE={r['mse']:.2e} (Q8={r['q8_mse']:.2e})  "
                  f"CosSim={r['cos_sim']:.8f}  Compress={r['compression_ratio']:.2f}x  "
                  f"({r['compressed_bytes_per_tok']:.0f} vs {r['original_bytes_per_tok']} B/tok)")
            quant_results.append(r)
    all_results['quantized_parity'] = quant_results

    # ─── Test 2: Storage ───
    print()
    print("=" * 76)
    print("TEST 2: Storage Analysis (Qwen 3.6, 10 full-attention layers)")
    print("=" * 76)

    storage_results = []
    for seq in [1024, 4096, 16384, 131072]:
        s = test_storage(head_dim=256, seq_len=seq)
        print(f"  seq={seq:6d}: Original={s['original_total_mb']:.1f}MB  "
              f"MV_expansion={s['mv_expansion']:.2f}x  "
              f"Rotor overhead={s['rotor_overhead_bytes']}B/layer")
        storage_results.append(s)
    all_results['storage'] = storage_results

    print()
    print("  NOTE: Raw MV is larger than original (expansion). Compression comes")
    print("  from Lloyd-Max quantization of MV components to N-bit indices.")

    # ─── Test 3: Prefill ───
    print()
    print("=" * 76)
    print("TEST 3: Prefill Throughput (batch encode)")
    print("=" * 76)
    print(f"{'Seq':>8} | {'Fused (ms)':>10} | {'2-Launch':>10} | {'Fuse/2L':>8} | {'Matmul':>10} | {'vs Matmul':>10} | {'tok/s':>10}")
    print("-" * 80)

    for dtype in [torch.bfloat16]:
        prefill_results = test_prefill(device, dtype=dtype)
        for r in prefill_results:
            if 'error' in r:
                print(f"{r['seq_len']:>8} | ERROR: {r['error']}")
            else:
                print(f"{r['seq_len']:>8} | {r['roundtrip_ms']:>10.3f} | {r['2launch_ms']:>10.3f} | "
                      f"{r['fuse_speedup']:>7.2f}x | {r['matmul_ms']:>10.3f} | "
                      f"{r['vs_matmul']:>9.2f}x | {r['toks_per_sec']:>10.0f}")
        all_results['prefill'] = prefill_results

    # ─── Test 4: Decode ───
    print()
    print("=" * 76)
    print("TEST 4: Decode Throughput (single token)")
    print("=" * 76)

    for dtype in [torch.bfloat16, torch.float16]:
        dtype_name = 'BF16' if dtype == torch.bfloat16 else 'FP16'
        d = test_decode(device, dtype=dtype)
        print(f"  {dtype_name}: RT={d['decode_rt_us']:.1f}us  Copy={d['decode_copy_us']:.1f}us  "
              f"vs_copy={d['vs_copy']:.3f}x  ({d['toks_per_sec']:.0f} tok/s)")
    all_results['decode_bf16'] = d

    print()
    print("=" * 76)
    print("BENCHMARK COMPLETE")
    print("=" * 76)

    return all_results


if __name__ == "__main__":
    if not torch.xpu.is_available():
        print("XPU not available")
        sys.exit(1)
    device = torch.device('xpu')
    results = run_all(device)
