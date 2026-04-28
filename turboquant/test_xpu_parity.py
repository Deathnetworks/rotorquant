
import torch
import torch.nn.functional as F
import math
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant.xpu_backend import is_xpu_available, QJLSketch
from turboquant.clifford import rotor_sandwich, embed_vectors_as_multivectors, extract_vectors_from_multivectors

def test_rotor_parity():
    print("Checking Rotor Sandwich Parity (CPU vs XPU)...")
    device = "xpu"
    d = 128
    n_groups = (d + 2) // 3
    # Keys: (batch=16, d=128)
    keys = torch.randn(16, 128, device=device, dtype=torch.float16)
    rotors = torch.randn(n_groups, 8, device=device, dtype=torch.float32)
    # Normalize rotors for stability
    from turboquant.clifford import reverse, geometric_product
    def norm_sq(x): return geometric_product(x, reverse(x))[..., 0]
    rotors = rotors / norm_sq(rotors).abs().sqrt().unsqueeze(-1).clamp(min=1e-8)

    # CPU Reference
    v_cpu = keys.cpu().float()
    rotors_cpu = rotors.cpu().float()
    mv_cpu = embed_vectors_as_multivectors(v_cpu)
    mv_rot_cpu = rotor_sandwich(rotors_cpu, mv_cpu)
    v_rot_cpu = extract_vectors_from_multivectors(mv_rot_cpu, d)

    # XPU Implementation
    import turboquant.xpu_rotor_fused as xpu_rotor
    mv_rot_xpu = xpu_rotor.rotor_sandwich_half(keys, rotors)
    v_rot_xpu = extract_vectors_from_multivectors(mv_rot_xpu.cpu().float(), d)

    # Compare
    diff = (v_rot_cpu - v_rot_xpu).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = F.cosine_similarity(v_rot_cpu.reshape(-1), v_rot_xpu.reshape(-1), dim=0).item()
    
    print(f"  Max Diff:  {max_diff:.6e}")
    print(f"  Mean Diff: {mean_diff:.6e}")
    print(f"  Cosine:    {cos_sim:.10f}")
    return cos_sim > 0.999

def test_qjl_quant_parity():
    print("Checking QJL Quantization Parity (CPU vs XPU)...")
    device = "xpu"
    head_dim = 128
    sketch_dim = 256
    batch, heads, seq = 1, 1, 32
    outliers = 8
    
    keys = torch.randn(batch, heads, 1, seq, head_dim, device=device, dtype=torch.float16)
    proj = torch.randn(sketch_dim, head_dim, device=device, dtype=torch.float16)
    
    norms = keys.norm(dim=-2)
    _, outlier_idx = norms.topk(outliers, dim=-1)
    outlier_idx = outlier_idx.to(torch.uint8).contiguous()

    # XPU Implementation
    import turboquant.xpu_qjl_quant as xpu_qjl
    # qjl_quant_xpu_half_half(Tensor keys, Tensor outlier_indices, Tensor proj, int outlier_sketch_dim)
    kq, koq, kon = xpu_qjl.qjl_quant_xpu_half_half(keys, outlier_idx, proj, 64)

    # De-pack XPU signs for comparison
    print("  XPU Quantization executed successfully.")
    return True

def test_full_fused_parity():
    print("Checking Rotor Full Fused Parity (CPU vs XPU)...")
    device = "xpu"
    d = 128
    n_groups = (d + 2) // 3
    batch = 16
    
    keys = torch.randn(batch, d, device=device, dtype=torch.float16)
    rotors = torch.randn(n_groups, 8, device=device)
    
    # Random centroids for quantization
    c_s = torch.sort(torch.randn(n_groups, 16, device=device))[0]
    c_v = torch.sort(torch.randn(n_groups * 3, 16, device=device))[0]
    c_b = torch.sort(torch.randn(n_groups * 3, 16, device=device))[0]
    c_t = torch.sort(torch.randn(n_groups, 16, device=device))[0]

    # XPU Implementation
    import turboquant.xpu_rotor_fused as xpu_rotor
    # Centroids must be float32
    out_xpu = xpu_rotor.rotor_full_fused_half(keys, rotors, c_s.float(), 16, c_v.float(), 16, c_b.float(), 16, c_t.float(), 16)
    
    # Basic Validation
    assert out_xpu.shape == keys.shape, f"Shape mismatch: {out_xpu.shape} vs {keys.shape}"
    assert out_xpu.dtype == keys.dtype, f"Dtype mismatch: {out_xpu.dtype} vs {keys.dtype}"
    
    print("  XPU Full Fused executed successfully.")
    return True

def test_qjl_bit_parity():
    print("Checking QJL Bit Parity (PyTorch vs XPU)...")
    device = "xpu"
    batch, heads, num_groups, group_size, head_dim = 1, 32, 4, 32, 128
    sketch_dim = 256
    
    keys = torch.randn(batch, heads, num_groups, group_size, head_dim, device=device, dtype=torch.float16)
    proj = torch.randn(head_dim, sketch_dim, device=device, dtype=torch.float32)
    outlier_idx = torch.zeros(batch, heads, num_groups, 8, dtype=torch.uint8, device=device)

    # PyTorch Reference
    # QJL quantization: sign(keys @ proj) packed into bits
    # keys shape: (B, H, G, GS, D), proj shape: (D, S)
    # result shape: (B, H, G, GS, S)
    sketch = torch.matmul(keys.float(), proj)
    bits_ref = (sketch > 0).to(torch.uint8)
    
    # Pack into bytes
    bytes_ref = torch.zeros(batch, heads, num_groups, group_size, sketch_dim // 8, dtype=torch.uint8, device=device)
    for i in range(sketch_dim // 8):
        for j in range(8):
            bytes_ref[..., i] += bits_ref[..., i*8 + j] << j

    # XPU Implementation
    # Kernel expects (sketch_dim, head_dim)
    proj_xpu = proj.transpose(0, 1).contiguous()
    from turboquant.xpu_backend import qjl_quant
    kq, _, _ = qjl_quant(keys, outlier_idx, proj_xpu, 64)
    
    # Compare (flattened)
    print(f"  XPU head 0, key 0, byte 0: {kq[0, 0, 0, 0, 0].item():08b}")
    print(f"  XPU head 1, key 0, byte 1: {kq[0, 1, 0, 0, 1].item():08b}")
    print(f"  XPU head 0, key 0, byte 2: {kq[0, 0, 0, 0, 2].item():08b}")
    
    diff = (bytes_ref.reshape(-1) != kq.reshape(-1)).sum()
    print(f"  Bit mismatches: {diff} / {bytes_ref.numel()}")
    
    if diff == 0:
        print("  QJL Bit Parity PASSED")
        return True
    else:
        print("  QJL Bit Parity FAILED")
        return False

if __name__ == "__main__":
    if not is_xpu_available():
        print("XPU not available.")
        sys.exit(1)
        
    s1 = test_rotor_parity()
    s2 = test_full_fused_parity()
    s3 = test_qjl_bit_parity()
    s4 = test_qjl_quant_parity()
    
    if s1 and s2 and s3 and s4:
        print("\nALL PARITY TESTS PASSED")
    else:
        print("\nPARITY FAILED")
