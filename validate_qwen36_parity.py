import torch
import torch.nn as nn
import math
import time
import os
import sys

# Ensure turboquant is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from turboquant.rotorquant import RotorQuantMSE
import turboquant.xpu_backend as xpu_backend

def validate_qwen36_parity():
    device = torch.device('xpu')
    
    # Qwen 3.6 / 3.5 MoE Dimensions
    num_kv_heads = 2
    head_dim = 256
    seq_len = 1024 # Test with 1k first
    batch_size = 1
    
    print(f"--- Qwen 3.6 Parity Validation ---")
    print(f"Config: heads={num_kv_heads}, dim={head_dim}, seq={seq_len}")
    
    # Generate random BF16 keys (matches Qwen 3.6 config)
    # Shape: [batch, heads, seq, dim]
    keys = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    
    # 1. Implementation Parity (XPU vs CPU)
    print("\n1. Implementation Parity Check (XPU vs CPU Reference)...")
    
    # RotorQuant MSE (Stage 1)
    # Using 4 bits as a starting point
    bits = 4
    rq = RotorQuantMSE(head_dim, bits, device=device).to(device)
    
    # Quantize on XPU
    # Note: RotorQuantMSE quantize method expects (..., d)
    flat_keys = keys.reshape(-1, head_dim)
    mv_q, indices = rq.quantize(flat_keys)
    
    # Dequantize
    keys_recon = rq.dequantize(indices)
    keys_recon = keys_recon.reshape(keys.shape)
    
    # Calculate MSE
    mse = torch.mean((keys.float() - keys_recon.float())**2).item()
    print(f"  MSE (Original vs Decoded): {mse:.2e}")
    
    # Bit-exact check (Implementation)
    # We compare the indices generated on XPU with a CPU reference
    keys_cpu = keys.cpu()
    rq_cpu = RotorQuantMSE(head_dim, bits, device='cpu')
    rq_cpu.load_state_dict(rq.state_dict()) # Match rotors and centroids
    
    _, indices_cpu = rq_cpu.quantize(flat_keys.cpu())
    
    matches = True
    for k in indices:
        if not k.startswith('_'): # Skip norms
            xpu_idx = indices[k].cpu()
            cpu_idx = indices_cpu[k]
            mismatches = (xpu_idx != cpu_idx).sum().item()
            if mismatches > 0:
                print(f"  [FAIL] Mismatch in {k} indices: {mismatches} / {xpu_idx.numel()}")
                matches = False
            else:
                print(f"  [PASS] {k} indices match 100%")
                
    if matches:
        print("  SUCCESS: XPU matches CPU reference bit-for-bit.")
    
    # 2. Lossless Roundtrip Investigation
    print("\n2. Lossless Roundtrip Investigation...")
    
    # We find the minimum number of outliers needed for 100% bit parity
    flat_keys_recon = keys_recon.reshape(-1, head_dim)
    diff = (flat_keys.float() - flat_keys_recon.float()).abs()
    # Any vector that isn't bit-exact is a candidate for an outlier
    error_mask = (diff > 0).any(dim=-1)
    num_error_vectors = error_mask.sum().item()
    print(f"  Vectors with any error: {num_error_vectors} / {flat_keys.shape[0]}")
    
    if num_error_vectors > 0:
        print(f"  Targeting 100% Bit Parity via Outliers...")
        # Simulating outlier storage: replace error vectors with original BF16
        keys_lossless = flat_keys_recon.to(flat_keys.dtype).clone()
        keys_lossless[error_mask] = flat_keys[error_mask]
        
        lossless_matches = (flat_keys.float() == keys_lossless.float()).sum().item()
        print(f"  Parity with Outliers: {lossless_matches} / {flat_keys.numel()} (100.00%)")
        
        # Calculate storage for this lossless mode
        # Compressed size + Outlier size (FP16)
        outlier_size = num_error_vectors * head_dim * 2 # 2 bytes per BF16
        total_compressed = sum(v.element_size() * v.numel() for k, v in indices.items() if not k.startswith('_')) + indices['_norms'].element_size() * indices['_norms'].numel()
        total_lossless_size = total_compressed + outlier_size
        print(f"  Lossless Storage: {total_lossless_size / 1024:.1f} KB")
        print(f"  Lossless Reduction: {(keys.element_size() * keys.numel()) / total_lossless_size:.2f}x")
    else:
        print("  SUCCESS: Already bit-exact!")
    
    # 3. Storage Analysis
    print("\n3. Storage Size Analysis (per layer)...")
    
    original_size = keys.element_size() * keys.numel()
    
    # Compressed size calculation
    # bits + norms (float32) + rotors (float32, but fixed per layer)
    idx_size = sum(v.element_size() * v.numel() for k, v in indices.items() if not k.startswith('_'))
    norm_size = indices['_norms'].element_size() * indices['_norms'].numel()
    
    total_compressed = idx_size + norm_size
    
    print(f"  Original (BF16): {original_size / 1024:.1f} KB")
    print(f"  Compressed:      {total_compressed / 1024:.1f} KB")
    print(f"  Reduction:       {original_size / total_compressed:.2f}x")

if __name__ == "__main__":
    validate_qwen36_parity()
