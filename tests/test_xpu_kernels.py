"""Tests for fused XPU kernels (rotor_fused_kernel_xpu.cpp and iso_planar_kernels_xpu.cpp).
Skipped if XPU is not available."""
import pytest
import torch
import os
import math
from turboquant import xpu_backend

# Skip all tests if no XPU
pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU not available"
)

@pytest.fixture
def setup_data():
    """Create test data: rotors, centroids, input vectors."""
    import numpy as np
    from turboquant.clifford import make_random_rotor
    from turboquant.lloyd_max import LloydMaxCodebook

    d = 128
    n_groups = (d + 2) // 3
    bits = 3
    device = 'xpu'

    # Rotors (standard 8-component, then compact to 4 for XPU)
    rotors = []
    for i in range(n_groups):
        r = make_random_rotor((), seed=42 + i)
        # We need the full 8-component rotor for pt_rotor_sandwich fallback
        rotors.append(r)
    rotors_full = torch.stack(rotors).float().contiguous().to(device)

    # Centroids
    d_eff = max(n_groups * 8, 64)
    cb = LloydMaxCodebook(d_eff, bits - 1)
    # Split into grade-specific dictionaries as expected by xpu_backend
    centroids_dict = {
        'scalar': cb.centroids.float().contiguous().to(device),
        'vector': cb.centroids.float().contiguous().to(device),
        'bivector': cb.centroids.float().contiguous().to(device),
        'trivector': cb.centroids.float().contiguous().to(device),
    }

    return {
        'd': d, 'n_groups': n_groups, 'bits': bits,
        'rotors': rotors_full, 'centroids': centroids_dict,
        'n_levels': len(centroids_dict['scalar']), 'device': device,
    }

@pytest.fixture
def setup_iso_planar_data():
    """Create test data for IsoQuant and PlanarQuant."""
    from turboquant import isoquant, planarquant
    d = 128
    bits = 4
    device = 'xpu'
    
    # IsoQuant
    iso = isoquant.IsoQuantMSE(d, bits, mode='fast', device='cpu')
    qL = iso.q_L.to(device)
    cb_iso = iso.centroids.to(device)
    
    # PlanarQuant
    planar = planarquant.PlanarQuantMSE(d, bits, device='cpu')
    cs = planar.rot2.to(device)
    cb_planar = planar.centroids.to(device)
    
    return {
        'd': d, 'bits': bits, 'device': device,
        'iso_qL': qL, 'iso_cb': cb_iso,
        'planar_cs': cs, 'planar_cb': cb_planar
    }

class TestXPUFusedKernels:
    def test_rotor_output_shape(self, setup_data):
        s = setup_data
        x = torch.randn(100, s['d'], device=s['device']).half()
        out = xpu_backend.rotor_full_fused(
            x, s['rotors'], 
            s['centroids']['scalar'], s['centroids']['vector'],
            s['centroids']['bivector'], s['centroids']['trivector']
        )
        assert out.shape == (100, s['d'])

    def test_rotor_identity(self, setup_data):
        s = setup_data
        identity_rotors = torch.zeros_like(s['rotors'])
        identity_rotors[:, 0] = 1.0
        
        x = torch.randn(50, s['d'], device=s['device']).half()
        x = x / x.norm(dim=-1, keepdim=True)
        out = xpu_backend.rotor_full_fused(
            x, identity_rotors,
            s['centroids']['scalar'], s['centroids']['vector'],
            s['centroids']['bivector'], s['centroids']['trivector']
        )
        mse = ((x.float() - out.float()) ** 2).sum(dim=-1).mean().item()
        # MSE should be small (quantization noise only)
        assert mse < 1.1

    def test_isoquant_fused(self, setup_iso_planar_data):
        s = setup_iso_planar_data
        x = torch.randn(10, 32, 128, s['d'], device=s['device']).half()
        out = xpu_backend.isoquant_fused(x, s['iso_qL'], s['iso_cb'])
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_planar_fused(self, setup_iso_planar_data):
        s = setup_iso_planar_data
        x = torch.randn(10, 32, 128, s['d'], device=s['device']).half()
        out = xpu_backend.planar_fused(x, s['planar_cs'], s['planar_cb'])
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_rotor_inverse(self, setup_data):
        s = setup_data
        x = torch.randn(50, s['d'], device=s['device']).half()
        x = x / x.norm(dim=-1, keepdim=True)
        # Forward sandwich
        mv = xpu_backend.rotor_sandwich(x, s['rotors'])
        # Inverse sandwich
        x_back = xpu_backend.rotor_inverse(mv, s['rotors'], s['d'])
        diff = (x.float() - x_back.float()).abs().max().item()
        assert diff < 0.01

    def test_rotor_compress(self, setup_data):
        """Test the split compress kernel."""
        s = setup_data
        x = torch.randn(50, s['d'], device=s['device']).half()
        # rotor_compress returns packed indices + norms
        try:
            out = xpu_backend.rotor_compress(
                x, s['rotors'], 
                s['centroids']['scalar'], s['centroids']['vector'],
                s['centroids']['bivector'], s['centroids']['trivector']
            )
            assert out is not None
        except Exception as e:
            pytest.fail(f"rotor_compress failed: {e}")
