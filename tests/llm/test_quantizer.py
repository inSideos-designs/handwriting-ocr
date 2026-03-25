import torch
import numpy as np
import pytest

from llm.turboquant.codebook import build_codebook
from llm.turboquant.quantizer import TurboQuantizer, TurboQuantizerWithQJL


@pytest.fixture
def codebook_2bit():
    return build_codebook(dim=64, bits=2)


@pytest.fixture
def codebook_1bit():
    return build_codebook(dim=64, bits=1)


@pytest.fixture
def codebook_3bit():
    return build_codebook(dim=64, bits=3)


@pytest.fixture
def unit_vector():
    x = torch.randn(64)
    return x / torch.norm(x)


class TestTurboQuantizer:
    def test_quantize_returns_indices(self, codebook_2bit, unit_vector):
        q = TurboQuantizer(dim=64, bits=2, codebook=codebook_2bit)
        indices = q.quantize(unit_vector)
        assert indices.shape == (64,)
        assert indices.dtype == torch.int64
        assert indices.min() >= 0
        assert indices.max() < 4  # 2^2 = 4 levels

    def test_dequantize_returns_vector(self, codebook_2bit, unit_vector):
        q = TurboQuantizer(dim=64, bits=2, codebook=codebook_2bit)
        indices = q.quantize(unit_vector)
        x_hat = q.dequantize(indices)
        assert x_hat.shape == (64,)

    def test_reconstruction_error_bounded(self, codebook_2bit, unit_vector):
        q = TurboQuantizer(dim=64, bits=2, codebook=codebook_2bit)
        x_hat = q.quantize_dequantize(unit_vector)
        error = torch.norm(unit_vector - x_hat).item()
        assert error < 1.0  # reconstruction should be reasonable

    def test_deterministic_with_same_seed(self, codebook_2bit, unit_vector):
        q1 = TurboQuantizer(dim=64, bits=2, codebook=codebook_2bit, seed=42)
        q2 = TurboQuantizer(dim=64, bits=2, codebook=codebook_2bit, seed=42)
        idx1 = q1.quantize(unit_vector)
        idx2 = q2.quantize(unit_vector)
        assert torch.equal(idx1, idx2)

    def test_higher_bits_lower_error(self, unit_vector):
        cb2 = build_codebook(dim=64, bits=2)
        cb3 = build_codebook(dim=64, bits=3)
        q2 = TurboQuantizer(dim=64, bits=2, codebook=cb2)
        q3 = TurboQuantizer(dim=64, bits=3, codebook=cb3)
        err2 = torch.norm(unit_vector - q2.quantize_dequantize(unit_vector)).item()
        err3 = torch.norm(unit_vector - q3.quantize_dequantize(unit_vector)).item()
        assert err3 < err2


class TestTurboQuantizerWithQJL:
    def test_quantize_returns_three_components(self, codebook_1bit, unit_vector):
        q = TurboQuantizerWithQJL(dim=64, bits=2, codebook_main=codebook_1bit)
        mse_idx, qjl_signs, res_norm = q.quantize(unit_vector)
        assert mse_idx.shape == (64,)
        assert qjl_signs.shape == (64,)
        assert isinstance(res_norm, float)
        assert all(s in [-1.0, 1.0] for s in qjl_signs.tolist())

    def test_qjl_reduces_error(self):
        cb1 = build_codebook(dim=64, bits=1)
        q_mse = TurboQuantizer(dim=64, bits=1, codebook=cb1, seed=42)
        q_qjl = TurboQuantizerWithQJL(dim=64, bits=2, codebook_main=cb1, seed=42)
        # Average over multiple vectors for a stable comparison
        torch.manual_seed(123)
        total_mse = 0.0
        total_qjl = 0.0
        n_samples = 50
        for _ in range(n_samples):
            x = torch.randn(64)
            x = x / torch.norm(x)
            total_mse += torch.norm(x - q_mse.quantize_dequantize(x)).item() ** 2
            total_qjl += torch.norm(x - q_qjl.quantize_dequantize(x)).item() ** 2
        avg_mse = total_mse / n_samples
        avg_qjl = total_qjl / n_samples
        assert avg_qjl < avg_mse

    def test_reconstruction_preserves_inner_product(self, codebook_1bit):
        dim = 64
        x = torch.randn(dim); x = x / torch.norm(x)
        y = torch.randn(dim); y = y / torch.norm(y)
        q = TurboQuantizerWithQJL(dim=dim, bits=2, codebook_main=codebook_1bit)
        x_hat = q.quantize_dequantize(x)
        true_ip = torch.dot(x, y).item()
        approx_ip = torch.dot(x_hat, y).item()
        # Should be approximately equal (unbiased estimator)
        assert abs(true_ip - approx_ip) < 0.5  # generous bound for dim=64
