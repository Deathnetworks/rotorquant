"""Tests for RotorQuant: MSE, inner product, KV cache on XPU."""
import pytest
import torch
import math
from turboquant.rotorquant import RotorQuantMSE, RotorQuantProd, RotorQuantKVCache

# Skip if XPU is not available
pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU not available"
)

class TestRotorQuantMSEXPU:
    @pytest.fixture
    def unit_vectors(self):
        torch.manual_seed(42)
        x = torch.randn(500, 128, device='xpu')
        return x / x.norm(dim=-1, keepdim=True)

    def test_output_shape(self, unit_vectors):
        rq = RotorQuantMSE(128, bits=3, seed=42, device='xpu')
        x_hat, indices = rq(unit_vectors)
        assert x_hat.shape == unit_vectors.shape
        assert x_hat.device.type == 'xpu'

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_mse_within_bounds(self, unit_vectors, bits):
        rq = RotorQuantMSE(128, bits=bits, seed=42, device='xpu')
        x_hat, _ = rq(unit_vectors)
        mse = ((unit_vectors - x_hat) ** 2).sum(dim=-1).mean().item()
        assert mse < 2.0, f"MSE {mse} too high for {bits}-bit on XPU"

class TestRotorQuantProdXPU:
    @pytest.fixture
    def unit_vectors(self):
        torch.manual_seed(42)
        x = torch.randn(500, 128, device='xpu')
        return x / x.norm(dim=-1, keepdim=True)

    def test_quantize_returns_dict(self, unit_vectors):
        rq = RotorQuantProd(128, bits=3, seed=42, device='xpu')
        comp = rq.quantize(unit_vectors)
        assert 'mse_indices' in comp
        assert 'qjl_signs' in comp
        # Check that compressed tensors are on XPU
        for k, v in comp.items():
            if isinstance(v, torch.Tensor):
                assert v.device.type == 'xpu'

    def test_dequantize_shape(self, unit_vectors):
        rq = RotorQuantProd(128, bits=3, seed=42, device='xpu')
        comp = rq.quantize(unit_vectors)
        x_hat = rq.dequantize(comp)
        assert x_hat.shape == unit_vectors.shape
        assert x_hat.device.type == 'xpu'

    def test_needle_in_haystack(self):
        torch.manual_seed(42)
        d = 128
        seq_len = 512 # Smaller for quick XPU test
        device = 'xpu'

        keys = torch.randn(seq_len, d, device=device)
        keys = keys / keys.norm(dim=-1, keepdim=True)
        needle_pos = seq_len // 3
        query = keys[needle_pos].clone()

        rq = RotorQuantProd(d, bits=3, seed=42, device=device)
        comp = rq.quantize(keys)
        # expand query to match batch size for inner_product if needed
        # depending on impl, might need unsqueeze
        query_exp = query.unsqueeze(0).expand(seq_len, -1)
        ips = rq.inner_product(query_exp, comp)

        assert ips.argmax().item() == needle_pos

class TestRotorQuantKVCacheXPU:
    def test_append_and_score(self):
        d = 64
        device = 'xpu'
        cache = RotorQuantKVCache(d, d, bits=3, seed=42, device=device)

        keys = torch.randn(32, d, device=device)
        values = torch.randn(32, d, device=device)
        cache.append(keys, values)

        assert len(cache) == 32

        query = torch.randn(32, d, device=device)
        scores = cache.attention_scores(query)
        assert scores.shape[-1] == 32
        assert scores.device.type == 'xpu'
