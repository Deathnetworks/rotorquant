"""Tests for Lloyd-Max codebook construction on XPU."""
import pytest
import torch
import math
from turboquant.lloyd_max import LloydMaxCodebook, solve_lloyd_max

# Skip if XPU is not available
pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU not available"
)

class TestLloydMaxCodebookXPU:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_correct_number_of_centroids(self, bits):
        # High-level Python objects might still be CPU-resident but used for XPU
        cb = LloydMaxCodebook(128, bits)
        assert cb.n_levels == 2 ** bits
        assert len(cb.centroids) == 2 ** bits

    def test_quantize_dequantize_xpu(self):
        cb = LloydMaxCodebook(128, 3)
        x = torch.randn(100, device='xpu')
        # Ensure the Python method handles XPU tensors (or we move them)
        # LloydMaxCodebook.quantize typically uses torch.bucketize or searchsorted
        idx = cb.quantize(x)
        assert idx.device.type == 'xpu'
        x_hat = cb.dequantize(idx)
        assert x_hat.device.type == 'xpu'
        
        # All reconstructed values should be valid centroids (within FP tolerance)
        centroids_xpu = cb.centroids.to('xpu')
        for v in x_hat.unique():
            # Check if v is close to any centroid
            diff = (centroids_xpu - v).abs().min()
            assert diff < 1e-5

    def test_solve_lloyd_max_various(self):
        """Should converge for various configs."""
        for d in [64, 128]:
            for bits in [2, 3]:
                c, b = solve_lloyd_max(d, bits)
                assert len(c) == 2 ** bits
                assert len(b) == 2 ** bits - 1
