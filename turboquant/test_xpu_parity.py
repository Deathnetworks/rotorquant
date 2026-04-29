import torch
import turboquant.xpu_backend as xpu_backend
from turboquant.xpu_backend import xpu_qjl_quant as xpu_qjl
import time

def test_qjl_bit_parity():
    device = torch.device('xpu')
    batch, head, n, gs, d = 1, 1, 1, 32, 128
    sketch_dim = 128
    
    # Force highest precision
    # torch.xpu.set_32bit_math_precision("highest")
    
    keys = torch.randn(batch, head, n, gs, d, device=device).half()
    outlier_idx = torch.zeros((batch, head, n, d // 8), device=device, dtype=torch.uint8)
    proj = torch.randn(sketch_dim, d, device=device).half()
    
    # Reference
    # key (1, 32, 128) @ proj.T (128, 128) -> (1, 32, 128)
    ref_dot = torch.matmul(keys[0,0,0].float(), proj.float().t())
    ref_bits = (ref_dot > 0).to(torch.uint8)
    
    # XPU
    kq, koq, on = xpu_backend.qjl_quant(keys, outlier_idx, proj, 64)
    
    torch.xpu.synchronize()
    xpu_kq = kq.cpu()[0,0,0]
    
    # Pack ref bits into bytes
    ref_packed = torch.zeros((gs, sketch_dim // 8), dtype=torch.uint8)
    for i in range(gs):
        for j in range(sketch_dim // 8):
            byte_val = 0
            for bit in range(8):
                if ref_bits[i, j*8 + bit]:
                    byte_val |= (1 << bit)
            ref_packed[i, j] = byte_val
            
    print(f"  XPU KQ[0, 2]: {xpu_kq[0, 2].item()}")
    print(f"  REF PQ[0, 2]: {ref_packed[0, 2].item()}")
    
    # Debug: Check some bits for row 0, byte 2 (bits 16-23)
    byte_idx = 2
    xpu_byte = xpu_kq[0, byte_idx].item()
    ref_byte = ref_packed[0, byte_idx].item()
    
    if xpu_byte != ref_byte:
        print(f"  Mismatch in byte {byte_idx}: XPU={xpu_byte:08b}, REF={ref_byte:08b}")
        for bit in range(8):
            xpu_bit = (xpu_byte >> bit) & 1
            ref_bit = (ref_byte >> bit) & 1
            if xpu_bit != ref_bit:
                p_idx = byte_idx * 8 + bit
                val = ref_dot[0, p_idx].item()
                print(f"    Bit {bit} (Proj {p_idx}) mismatch! Ref Dot Val: {val:.10f}")

    mismatches = (ref_packed != xpu_kq).sum().item()
    total = ref_packed.numel()
    
    print("Checking QJL Bit Parity (PyTorch vs XPU)...")
    print(f"  Bit mismatches: {mismatches} / {total}")
    
    return mismatches == 0

if __name__ == "__main__":
    test_qjl_bit_parity()
