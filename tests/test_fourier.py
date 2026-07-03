import numpy as np
import pytest

from diffractor import FFT2, IFFT2, frequency_grid, make_grid


def test_fft_ifft_roundtrip():
    rng = np.random.default_rng(0)
    g = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
    np.testing.assert_allclose(IFFT2(FFT2(g)), g, atol=1e-12)


def test_fft2_is_centered():
    # A constant field transforms to a single peak at the center bin.
    n = 32
    g = np.ones((n, n), dtype=complex)
    G = FFT2(g)
    peak = np.unravel_index(np.argmax(np.abs(G)), G.shape)
    assert peak == (n // 2, n // 2)


def test_parseval():
    rng = np.random.default_rng(1)
    g = rng.standard_normal((128, 128))
    G = FFT2(g)
    np.testing.assert_allclose(np.sum(np.abs(g) ** 2), np.sum(np.abs(G) ** 2) / g.size)


def test_frequency_grid_values():
    n, L = 64, 2e-3
    grid = make_grid(n, L)
    fx, fy = frequency_grid(grid)

    dx = L / n
    assert fx.shape == (n, n)
    # Centered: zero frequency at the middle, Nyquist range 1/(2 dx).
    assert fx[0, n // 2] == 0.0
    np.testing.assert_allclose(fx[0, 1] - fx[0, 0], 1.0 / (n * dx))
    np.testing.assert_allclose(fx.min(), -1.0 / (2 * dx))


def test_frequency_grid_rejects_bad_input():
    x, y = make_grid(16, 1.0)
    with pytest.raises(ValueError):
        frequency_grid((x, y[:8, :8]))
    with pytest.raises(ValueError):
        frequency_grid((x[0], y[0]))
