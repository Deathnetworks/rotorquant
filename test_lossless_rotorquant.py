import torch
import torch.nn as nn
import os
import sys

# Ensure turboquant is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from turboquant.rotorquant import RotorQuantMSE
import turboquant.xpu_backend as xpu_backend

def test_lossless_rotorquant():
    device = torch.device('xpu')
    
    # Qwen 3.6 dims
    head_dim = 256
    seq_len = 128
    batch_size = 1
    
    print(f"--- Lossless RotorQuant Check ---")
    
    # Random BF16 keys
    keys = torch.randn(batch_size, 1, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    
    # We want to see how many outliers we need for bit-parity.
    # Note: If the user says "mathematically designed to preserve FP16", 
    # maybe they mean the MSE stage is enough? (No, Lloyd-Max is lossy).
    
    bits = 4
    rq = RotorQuantMSE(head_dim, bits, device=device).to(device)
    
    flat_keys = keys.reshape(-1, head_dim)
    mv_q, indices = rq.quantize(flat_keys)
    keys_recon = rq.dequantize(indices).reshape(keys.shape)
    
    diff = (keys.float() - keys_recon.float()).abs()
    mismatches = (diff > 0).sum().item()
    total = keys.numel()
    
    print(f"  Standard 4-bit Mismatches: {mismatches} / {total} ({mismatches/total:.2%})")
    
    # Identify outliers (those with any error)
    error_mask = (diff > 0).any(dim=-1) # [batch, heads, seq]
    
    # If we treat EVERY vector with error as an outlier...
    num_outliers = error_mask.sum().item()
    print(f"  Vectors needing outliers for 100% parity: {num_outliers} / {seq_len}")
    
    # Wait! If "bit parity" is required, and we want "size reduction", 
    # then the error_mask must be sparse.
    
    # Let's check if the error is small enough to be ignored?
    # No, user said "must match the value after the encode/decode cycle".
    
    # What if we use MORE bits?
    print("\n  Testing scaling bits...")
    for b in [4, 8]:
        rq_b = RotorQuantMSE(head_dim, b, device=device).to(device)
        _, ind_b = rq_b.quantize(flat_keys)
        rec_b = rq_b.dequantize(ind_b)
        err = (flat_keys.float() - rec_b.float()).abs().max().item()
        match_count = (flat_keys.float() == rec_b.float()).sum().item()
        print(f"    {b}-bit: Max Error={err:.2e}, Exact Matches={match_count}/{flat_keys.numel()}")

    # Conclusion: Even at 8 bits, exact matches are rare.
    # Therefore, 100% bit parity MUST come from the OUTLIER path.
    
    # Optimization Goal:
    # 1. Faster outlier identification and storage.
    # 2. Minimum outliers for a given "acceptable" threshold, or 
    #    storing residuals if the user wants true lossless.
    
    print("\n  Hypothesis: User wants 'Outlier-Corrected' lossless reconstruction.")
    print("  We will focus on optimizing the Fused XPU Quantization kernel to handle this.")

if __name__ == "__main__":
    test_lossless_rotorquant()
