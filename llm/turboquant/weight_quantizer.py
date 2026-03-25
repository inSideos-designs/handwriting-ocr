import torch
import torch.nn as nn
import numpy as np

from llm.turboquant.codebook import build_codebook
from llm.turboquant.quantizer import TurboQuantizer


class QuantizedLinear(nn.Module):
    """
    A quantized replacement for nn.Linear using TurboQuant weight quantization.

    Stores weights as quantization indices + row norms instead of full float32.
    Dequantizes on-the-fly during forward pass.
    """

    def __init__(self, in_features, out_features, bits=4, bias=True, seed=42):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # Build codebook for this dimension
        self.codebook = build_codebook(dim=in_features, bits=bits)

        # Create quantizer (shared rotation matrix for all rows)
        self.quantizer = TurboQuantizer(
            dim=in_features, bits=bits, codebook=self.codebook, seed=seed
        )

        # Storage for quantized weights
        self.register_buffer("weight_indices", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_norms", torch.zeros(out_features, dtype=torch.float16))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self._quantized = False

    @classmethod
    def from_linear(cls, linear, bits=4, seed=42):
        """Create a QuantizedLinear from an existing nn.Linear."""
        has_bias = linear.bias is not None
        ql = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bits=bits,
            bias=has_bias,
            seed=seed,
        )

        if has_bias:
            ql.bias = nn.Parameter(linear.bias.data.clone())

        ql._quantize_weights(linear.weight.data)
        return ql

    def _quantize_weights(self, weight):
        """Quantize weight matrix (vectorized over all rows)."""
        with torch.no_grad():
            w = weight.float()
            norms = torch.norm(w, dim=1)
            self.weight_norms = norms.half()

            # Normalize rows to unit length
            safe_norms = norms.clone()
            safe_norms[safe_norms < 1e-8] = 1.0
            unit_w = w / safe_norms.unsqueeze(1)

            # Rotate all rows at once: (out, in) @ (in, in)^T = (out, in)
            rotated = unit_w @ self.quantizer.rotation.T

            # Quantize: find nearest centroid for each coordinate
            centroids = self.quantizer.centroids
            distances = torch.abs(rotated.unsqueeze(-1) - centroids.unsqueeze(0).unsqueeze(0))
            indices = torch.argmin(distances, dim=-1)
            self.weight_indices = indices.to(torch.int8)

        self._quantized = True

    def _dequantize_weights(self):
        """Reconstruct full weight matrix (vectorized)."""
        indices = self.weight_indices.long()
        centroids = self.quantizer.centroids.to(self.weight_indices.device)
        y_hat = centroids[indices]  # (out, in)
        weight = y_hat @ self.quantizer.rotation.to(self.weight_indices.device)
        weight = weight * self.weight_norms.float().unsqueeze(1)
        return weight

    def forward(self, x):
        """Forward pass with on-the-fly dequantization."""
        weight = self._dequantize_weights()
        return nn.functional.linear(x, weight, self.bias)

    def compression_ratio(self):
        """Calculate compression ratio vs float32."""
        original_bits = self.out_features * self.in_features * 32
        quantized_bits = (
            self.out_features * self.in_features * self.bits  # indices
            + self.out_features * 16  # norms (float16)
        )
        return original_bits / quantized_bits

    def memory_bytes(self):
        """Actual memory usage of quantized representation."""
        index_bytes = self.weight_indices.nelement() * self.weight_indices.element_size()
        norm_bytes = self.weight_norms.nelement() * self.weight_norms.element_size()
        bias_bytes = self.bias.nelement() * self.bias.element_size() if self.bias is not None else 0
        return index_bytes + norm_bytes + bias_bytes


def quantize_model(model, bits=4, seed=42, skip_layers=None):
    """
    Replace all nn.Linear layers in a model with QuantizedLinear.

    Args:
        model: nn.Module to quantize
        bits: quantization bits per coordinate
        seed: random seed
        skip_layers: list of layer name patterns to skip (e.g., ["lm_head"])

    Returns:
        quantized model (modified in-place)
    """
    skip_layers = skip_layers or []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(skip in name for skip in skip_layers):
                continue

            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]

            parent = model
            if parent_name:
                for part in parent_name.split("."):
                    parent = getattr(parent, part)

            quantized = QuantizedLinear.from_linear(module, bits=bits, seed=seed)
            setattr(parent, child_name, quantized)

    return model
