import numpy as np
from scipy.special import gamma
from scipy.integrate import quad


def beta_pdf(x, d):
    """PDF of a coordinate of a point uniformly distributed on the unit sphere in R^d."""
    if d <= 2:
        raise ValueError("Dimension must be > 2")
    coeff = gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
    return coeff * (1 - x**2) ** ((d - 3) / 2) if abs(x) < 1 else 0.0


def gaussian_pdf(x, variance):
    """Gaussian approximation for high-dimensional case."""
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-x**2 / (2 * variance))


def max_lloyd(pdf, num_levels, x_min=-1.0, x_max=1.0, max_iter=200, tol=1e-8):
    """
    Max-Lloyd algorithm (optimal scalar quantizer) for a given PDF.

    Returns centroids and boundaries that minimize MSE distortion.
    """
    # Initialize centroids uniformly
    centroids = np.linspace(x_min + (x_max - x_min) / (2 * num_levels),
                            x_max - (x_max - x_min) / (2 * num_levels),
                            num_levels)

    for iteration in range(max_iter):
        # Update boundaries (midpoints between centroids)
        boundaries = np.zeros(num_levels + 1)
        boundaries[0] = x_min
        boundaries[-1] = x_max
        for i in range(1, num_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

        # Update centroids (conditional expectations)
        new_centroids = np.zeros(num_levels)
        for i in range(num_levels):
            lo, hi = boundaries[i], boundaries[i + 1]

            numerator, _ = quad(lambda x: x * pdf(x), lo, hi)
            denominator, _ = quad(pdf, lo, hi)

            if denominator > 1e-15:
                new_centroids[i] = numerator / denominator
            else:
                new_centroids[i] = (lo + hi) / 2

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids, boundaries


def compute_distortion(pdf, centroids, boundaries):
    """Compute MSE distortion for a given quantizer."""
    distortion = 0.0
    for i in range(len(centroids)):
        lo, hi = boundaries[i], boundaries[i + 1]
        d, _ = quad(lambda x: (x - centroids[i])**2 * pdf(x), lo, hi)
        distortion += d
    return distortion


def build_codebook(dim, bits):
    """
    Build optimal quantization codebook for TurboQuant.

    For high dimensions, uses Gaussian approximation.
    For lower dimensions, uses exact Beta PDF.
    """
    num_levels = 2 ** bits
    variance = 1.0 / dim

    if dim > 50:
        # High-dimensional: use Gaussian approximation
        # Clip to 4 sigma range
        sigma = np.sqrt(variance)
        x_min, x_max = -4 * sigma, 4 * sigma
        pdf = lambda x: gaussian_pdf(x, variance)
    else:
        # Lower-dimensional: use exact Beta PDF
        x_min, x_max = -1.0, 1.0
        pdf = lambda x: beta_pdf(x, dim)

    centroids, boundaries = max_lloyd(pdf, num_levels, x_min, x_max)
    distortion = compute_distortion(pdf, centroids, boundaries)

    return {
        "centroids": centroids,
        "boundaries": boundaries,
        "distortion": distortion,
        "bits": bits,
        "dim": dim,
    }
