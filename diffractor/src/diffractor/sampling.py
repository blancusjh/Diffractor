"""Sampling criteria — the closed-form ones.

There is deliberately no closed-form "minimum N" for the near-field angular
spectrum here: a hard aperture at high Fresnel number develops boundary-wave
ripples finer than any a-priori rule predicts, and a naive Nyquist estimate
misses the required grid by an order of magnitude.  The honest procedure is a
convergence study — double the sampling until the answer stops moving — which
is a loop in your script, not an operator in this package.
"""

from __future__ import annotations

__all__ = ["fresnel_min_distance", "fresnel_max_spacing", "next_fft_size"]


def fresnel_min_distance(n: int, dx: float, wavelength: float,
                         n_medium: float = 1.0) -> float:
    """Shortest z at which the single-transform Fresnel method is well
    sampled: the quadratic phase must vary slowly between samples, z ≳ N·dx²/λ_m."""
    return n * dx**2 / (wavelength / n_medium)


def fresnel_max_spacing(n: int, z: float, wavelength: float,
                        n_medium: float = 1.0) -> float:
    """The same criterion solved for the spacing: dx ≤ √(λ_m·z/N)."""
    return ((wavelength / n_medium) * z / n) ** 0.5


def next_fft_size(n: int, *, prefer_fast_len: bool = True) -> int:
    """The next FFT-friendly size ≥ n (5-smooth via SciPy, else a power of 2)."""
    if prefer_fast_len:
        try:
            from scipy.fft import next_fast_len
            return int(next_fast_len(int(n)))
        except Exception:  # pragma: no cover - scipy is a hard dep in practice
            pass
    m = 1
    while m < n:
        m *= 2
    return m
