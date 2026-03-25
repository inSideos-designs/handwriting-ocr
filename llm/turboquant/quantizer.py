import torch
import numpy as np


class TurboQuantizer:
    """
    TurboQuant quantizer implementing random rotation + optimal scalar quantization.

    Based on: https://arxiv.org/abs/2504.19874
    """

    def __init__(self, dim, bits, codebook, seed=42):
        self.dim = dim
        self.bits = bits
        self.centroids = torch.tensor(codebook["centroids"], dtype=torch.float32)
        self.boundaries = torch.tensor(codebook["boundaries"], dtype=torch.float32)

        # Generate random rotation matrix via QR decomposition
        rng = torch.Generator().manual_seed(seed)
        random_matrix = torch.randn(dim, dim, generator=rng)
        self.rotation, _ = torch.linalg.qr(random_matrix)

    def quantize(self, x):
        """
        Quantize a vector x (must be unit-normalized).

        Args:
            x: tensor of shape (dim,) with ||x|| = 1

        Returns:
            indices: tensor of shape (dim,) with integer indices into codebook
        """
        # Rotate
        y = self.rotation @ x

        # Quantize each coordinate to nearest centroid
        # y shape: (dim,), centroids shape: (num_levels,)
        distances = torch.abs(y.unsqueeze(-1) - self.centroids.unsqueeze(0))
        indices = torch.argmin(distances, dim=-1)

        return indices

    def dequantize(self, indices):
        """
        Dequantize indices back to a vector.

        Args:
            indices: tensor of shape (dim,) with integer indices

        Returns:
            x_hat: tensor of shape (dim,) — reconstructed vector
        """
        y_hat = self.centroids[indices]
        x_hat = self.rotation.T @ y_hat
        return x_hat

    def quantize_dequantize(self, x):
        """Quantize and immediately dequantize (for testing distortion)."""
        indices = self.quantize(x)
        return self.dequantize(indices)


class TurboQuantizerWithQJL(TurboQuantizer):
    """
    Two-stage TurboQuant: MSE quantizer + QJL residual correction.
    Uses (bits-1) bits for MSE stage and 1 bit for QJL correction.
    """

    def __init__(self, dim, bits, codebook_main, seed=42):
        """
        Args:
            dim: vector dimension
            bits: total bits per coordinate
            codebook_main: codebook for (bits-1) bit MSE stage
            seed: random seed for reproducibility
        """
        super().__init__(dim, bits - 1, codebook_main, seed=seed)
        self.total_bits = bits

        # Generate random Gaussian matrix for QJL, normalized by 1/sqrt(d)
        rng = torch.Generator().manual_seed(seed + 1)
        self.qjl_matrix = torch.randn(dim, dim, generator=rng) / np.sqrt(dim)

    def quantize(self, x):
        """
        Two-stage quantization: MSE + QJL.

        Returns:
            mse_indices: tensor of shape (dim,) — MSE stage indices
            qjl_signs: tensor of shape (dim,) — QJL sign bits (+1/-1)
            residual_norm: float — ||r||_2 for dequantization
        """
        # Stage 1: MSE quantization with (bits-1) bits
        mse_indices = super().quantize(x)
        x_mse = super().dequantize(mse_indices)

        # Compute residual
        residual = x - x_mse
        residual_norm = torch.norm(residual).item()

        # Stage 2: QJL on residual
        qjl_signs = torch.sign(self.qjl_matrix @ residual)
        # Replace zeros with +1
        qjl_signs[qjl_signs == 0] = 1.0

        return mse_indices, qjl_signs, residual_norm

    def dequantize(self, mse_indices, qjl_signs=None, residual_norm=None):
        """
        Two-stage dequantization.

        If qjl_signs and residual_norm are provided, applies QJL correction.
        Otherwise falls back to MSE-only dequantization.
        """
        x_mse = super().dequantize(mse_indices)

        if qjl_signs is not None and residual_norm is not None:
            scale = np.sqrt(np.pi / 2) / self.dim * residual_norm
            x_qjl = scale * (self.qjl_matrix.T @ qjl_signs)
            return x_mse + x_qjl

        return x_mse

    def quantize_dequantize(self, x):
        """Quantize and immediately dequantize."""
        mse_indices, qjl_signs, residual_norm = self.quantize(x)
        return self.dequantize(mse_indices, qjl_signs, residual_norm)
