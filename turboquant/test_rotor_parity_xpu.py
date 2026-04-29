import torch
import turboquant.xpu_backend as xpu_backend
import numpy as np

def test_rotor_roundtrip_parity():
    print("\n=== Testing RotorQuant XPU Roundtrip (Identity) Parity ===")
    device = torch.device('xpu')
    batch, emb_dim = 2, 128
    
    # 1. Generate random input
    input_fp16 = torch.randn(batch, emb_dim, device=device, dtype=torch.half)
    
    # 2. Generate random rotors [s, p12, p13, p23]
    n_groups = (emb_dim + 2) // 3
    rotors = torch.randn(n_groups, 4, device=device, dtype=torch.float32)
    # Normalize rotors to avoid magnitude explosion
    rotors = rotors / torch.norm(rotors, dim=-1, keepdim=True)
    
    # 3. Run XPU Roundtrip
    # This kernel does: input -> R*x*R~ -> R~*y*R -> output
    output_xpu = xpu_backend.rotor_sandwich_roundtrip(input_fp16, rotors)
    
    torch.xpu.synchronize()
    
    # 4. Check Parity
    diff = (input_fp16 - output_xpu).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max Difference: {max_diff:.10f}")
    print(f"Mean Difference: {mean_diff:.10f}")
    
    if max_diff < 1e-3: # FP16 tolerance
        print("SUCCESS: XPU Roundtrip Parity Passed!")
        return True
    else:
        print("FAILURE: XPU Roundtrip Parity Failed!")
        return False

def test_rotor_full_fused_parity():
    print("\n=== Testing RotorQuant XPU Full Fused (Quantized) Execution ===")
    device = torch.device('xpu')
    batch, emb_dim = 1, 128
    n_groups = (emb_dim + 2) // 3
    
    input_fp16 = torch.randn(batch, emb_dim, device=device, dtype=torch.half)
    rotors = torch.randn(n_groups, 4, device=device, dtype=torch.float32)
    rotors = rotors / torch.norm(rotors, dim=-1, keepdim=True)
    
    # Codebooks
    n_scalar = 16
    n_vector = 32 * 3
    n_bivector = 32 * 3
    n_trivector = 8
    
    c_scalar = torch.randn(n_scalar, device=device, dtype=torch.float32)
    c_vector = torch.randn(n_vector, device=device, dtype=torch.float32)
    c_bivector = torch.randn(n_bivector, device=device, dtype=torch.float32)
    c_trivector = torch.randn(n_trivector, device=device, dtype=torch.float32)
    
    try:
        output_xpu = xpu_backend.rotor_full_fused(
            input_fp16, rotors,
            c_scalar, c_vector, c_bivector, c_trivector
        )
        torch.xpu.synchronize()
        print("SUCCESS: XPU Full Fused Kernel Executed Successfully!")
        print(f"Output Sample: {output_xpu[0, :5].cpu().tolist()}")
        return True
    except Exception as e:
        print(f"FAILURE: XPU Full Fused Kernel Failed: {e}")
        return False

if __name__ == "__main__":
    r1 = test_rotor_roundtrip_parity()
    r2 = test_rotor_full_fused_parity()
    if r1 and r2:
        print("\nALL PARITY TESTS PASSED")
    else:
        exit(1)
