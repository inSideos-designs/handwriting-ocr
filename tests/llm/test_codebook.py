import numpy as np
import pytest

from llm.turboquant.codebook import (
    beta_pdf,
    gaussian_pdf,
    max_lloyd,
    compute_distortion,
    build_codebook,
)


class TestBetaPdf:
    def test_integrates_to_one(self):
        from scipy.integrate import quad
        for d in [3, 10, 50, 100]:
            integral, _ = quad(lambda x: beta_pdf(x, d), -1, 1)
            assert abs(integral - 1.0) < 1e-6, f"Failed for d={d}: integral={integral}"

    def test_symmetric(self):
        for d in [5, 20, 100]:
            assert abs(beta_pdf(0.3, d) - beta_pdf(-0.3, d)) < 1e-10

    def test_peak_at_zero(self):
        for d in [5, 20, 100]:
            assert beta_pdf(0, d) > beta_pdf(0.5, d)

    def test_rejects_low_dim(self):
        with pytest.raises(ValueError):
            beta_pdf(0, 2)


class TestMaxLloyd:
    def test_uniform_distribution(self):
        pdf = lambda x: 0.5 if -1 <= x <= 1 else 0
        centroids, boundaries = max_lloyd(pdf, 4, -1, 1)
        expected = [-0.75, -0.25, 0.25, 0.75]
        np.testing.assert_allclose(centroids, expected, atol=0.05)

    def test_num_centroids(self):
        pdf = lambda x: gaussian_pdf(x, 0.01)
        for n in [2, 4, 8]:
            centroids, boundaries = max_lloyd(pdf, n, -0.5, 0.5)
            assert len(centroids) == n
            assert len(boundaries) == n + 1

    def test_centroids_ordered(self):
        pdf = lambda x: gaussian_pdf(x, 0.01)
        centroids, _ = max_lloyd(pdf, 8, -0.5, 0.5)
        assert all(centroids[i] < centroids[i+1] for i in range(len(centroids)-1))

    def test_symmetric_centroids_for_symmetric_pdf(self):
        pdf = lambda x: gaussian_pdf(x, 0.01)
        centroids, _ = max_lloyd(pdf, 4, -0.5, 0.5)
        for i in range(len(centroids) // 2):
            assert abs(centroids[i] + centroids[-(i+1)]) < 0.01


class TestBuildCodebook:
    def test_returns_correct_structure(self):
        cb = build_codebook(dim=256, bits=2)
        assert "centroids" in cb
        assert "boundaries" in cb
        assert "distortion" in cb
        assert len(cb["centroids"]) == 4
        assert len(cb["boundaries"]) == 5

    def test_distortion_decreases_with_bits(self):
        d1 = build_codebook(dim=256, bits=1)["distortion"]
        d2 = build_codebook(dim=256, bits=2)["distortion"]
        d3 = build_codebook(dim=256, bits=3)["distortion"]
        assert d1 > d2 > d3

    def test_high_dim_uses_gaussian(self):
        cb = build_codebook(dim=1024, bits=2)
        assert len(cb["centroids"]) == 4
        assert cb["distortion"] > 0
