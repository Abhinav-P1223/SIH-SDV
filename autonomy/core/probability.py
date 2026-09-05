"""Probability helpers shared by the risk engine and the trajectory scorer.

Collision probability model (1-D band approximation):

The predicted object position is Gaussian with std-dev sigma(t) around its
mean. The ego footprint plus the object footprint plus the safety margin form
a "collision band" of width W along the line joining the two footprints.
`gap` is the distance from the object's predicted mean to the near edge of that
band (footprint distance minus margin, floored at 0). The probability that the
object actually lies inside the band is

    P = Phi((gap + W) / sigma) - Phi(gap / sigma)

which correctly tends to 0 as sigma grows large (mass spreads out) and to 1
when the predicted footprints already overlap (gap == 0 and sigma small gives
0.5; overlap is forced to 1 by the callers).
"""
from __future__ import annotations

import numpy as np


def norm_cdf(z: np.ndarray | float) -> np.ndarray:
    """Standard normal CDF, Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7), vectorised."""
    z = np.asarray(z, dtype=float)
    sign = np.sign(z)
    x = np.abs(z) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-x * x)
    return 0.5 * (1.0 + sign * erf)


def band_collision_probability(distance: np.ndarray, sigma: np.ndarray, margin: float,
                               band_width: float) -> np.ndarray:
    """P(object inside the ego collision band) per time step; 1 where footprints overlap the margin."""
    distance = np.asarray(distance, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    gap = np.maximum(distance - margin, 0.0)
    p = norm_cdf((gap + band_width) / sigma) - norm_cdf(gap / sigma)
    return np.where(distance <= margin, 1.0, p)
