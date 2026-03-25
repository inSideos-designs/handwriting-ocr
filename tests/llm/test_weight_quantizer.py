import torch
import torch.nn as nn
import pytest

from llm.turboquant.weight_quantizer import QuantizedLinear, quantize_model


class TestQuantizedLinear:
    def test_from_linear(self):
        linear = nn.Linear(64, 32)
        ql = QuantizedLinear.from_linear(linear, bits=4)
        assert ql.in_features == 64
        assert ql.out_features == 32
        assert ql._quantized is True

    def test_forward_shape(self):
        linear = nn.Linear(64, 32)
        ql = QuantizedLinear.from_linear(linear, bits=4)
        x = torch.randn(8, 64)
        out = ql(x)
        assert out.shape == (8, 32)

    def test_output_close_to_original(self):
        linear = nn.Linear(128, 64)
        ql = QuantizedLinear.from_linear(linear, bits=4)
        x = torch.randn(4, 128)

        with torch.no_grad():
            original_out = linear(x)
            quantized_out = ql(x)

        # Relative error should be modest
        rel_error = torch.norm(original_out - quantized_out) / torch.norm(original_out)
        assert rel_error < 0.5  # generous bound

    def test_higher_bits_lower_error(self):
        linear = nn.Linear(128, 64)
        x = torch.randn(4, 128)

        with torch.no_grad():
            original_out = linear(x)

            ql2 = QuantizedLinear.from_linear(linear, bits=2)
            err2 = torch.norm(original_out - ql2(x)).item()

            ql4 = QuantizedLinear.from_linear(linear, bits=4)
            err4 = torch.norm(original_out - ql4(x)).item()

        assert err4 < err2

    def test_compression_ratio(self):
        linear = nn.Linear(256, 128)
        ql = QuantizedLinear.from_linear(linear, bits=4)
        ratio = ql.compression_ratio()
        assert ratio > 7.0  # 32/4 = 8x theoretical, slightly less with norms

    def test_preserves_bias(self):
        linear = nn.Linear(64, 32, bias=True)
        ql = QuantizedLinear.from_linear(linear, bits=4)
        assert ql.bias is not None
        assert torch.equal(ql.bias.data, linear.bias.data)

    def test_no_bias(self):
        linear = nn.Linear(64, 32, bias=False)
        ql = QuantizedLinear.from_linear(linear, bits=4)
        assert ql.bias is None


class TestQuantizeModel:
    def test_replaces_linear_layers(self):
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        quantize_model(model, bits=4)
        assert isinstance(model[0], QuantizedLinear)
        assert isinstance(model[2], QuantizedLinear)

    def test_skip_layers(self):
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        # Name will be "0" and "2" for sequential
        quantize_model(model, bits=4, skip_layers=["2"])
        assert isinstance(model[0], QuantizedLinear)
        assert isinstance(model[2], nn.Linear)  # skipped

    def test_forward_still_works(self):
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        quantize_model(model, bits=4)
        x = torch.randn(4, 64)
        out = model(x)
        assert out.shape == (4, 16)
