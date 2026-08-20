"""Space and time primitives: Grid (typed samples of space) and Spectrum."""

import numpy as np
import pytest

from diffractor.basis import Basis, CARTESIAN, POLAR, resolve_basis
from diffractor.space import BESSEL_SAMPLES_PER_PERIOD, Grid
from diffractor.spectrum import Spectrum
from diffractor.units import cm, m, mm, nm, um


def test_units_are_metre_based():
    assert m == 1.0
    assert (cm, mm, um, nm) == (1e-2, 1e-3, 1e-6, 1e-9)


def test_resolve_basis_echoes_the_allowed_set():
    assert resolve_basis("cartesian") is CARTESIAN
    assert resolve_basis(POLAR) is POLAR
    with pytest.raises(ValueError, match="'cartesian', 'polar'"):
        resolve_basis("spherical")


def test_cartesian_reciprocal_matches_fftfreq():
    x = (np.arange(64) - 32) * 2.0 * um
    g = Grid.cartesian(x, x)
    k = g.reciprocal()
    assert np.allclose(k.axes[0],
                       2 * np.pi * np.fft.fftshift(np.fft.fftfreq(64, 2.0 * um)))
    with pytest.raises(ValueError, match="polar"):
        g.reciprocal(k_max=1.0)


def test_polar_reciprocal_default_formulas():
    r = np.linspace(0.0, 50 * um, 401)
    g = Grid.polar(r, n_theta=8)
    k = g.reciprocal()
    dr = np.min(np.diff(r))
    k_max = np.pi / dr
    assert np.isclose(k.axes[0][-1], k_max)
    n_k = int(np.ceil(BESSEL_SAMPLES_PER_PERIOD * k_max * r[-1] / (2 * np.pi))) + 1
    assert k.axes[0].size == n_k
    assert k.axes[1] is g.axes[1] or np.array_equal(k.axes[1], g.axes[1])
    k2 = g.reciprocal(k_max=1e5, n_k=17)
    assert k2.axes[0].size == 17 and np.isclose(k2.axes[0][-1], 1e5)


def test_polar_weights_integrate_disc_area():
    R = 3.7 * mm
    for n_theta in (1, 16):
        g = Grid.polar(np.linspace(0.0, R, 1001), n_theta=n_theta)
        assert np.isclose(g.weights().sum(), np.pi * R**2, rtol=1e-9)


def test_cartesian_weights_integrate_window_area():
    g = Grid.cartesian(np.linspace(-1.0, 1.0, 201), np.linspace(-0.5, 0.5, 101))
    assert np.isclose(g.weights().sum(), 2.0 * 1.0)


def test_polar_theta_validation_echoes_requirements():
    with pytest.raises(ValueError, match="2πj/n_θ"):
        Grid.polar(np.linspace(0, 1, 10), theta=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="ascending"):
        Grid.polar(np.array([0.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="theta or n_theta"):
        Grid.polar(np.linspace(0, 1, 4), theta=np.array([0.0]), n_theta=1)


def test_grid_scaled_polar_scales_r_only():
    g = Grid.polar(np.linspace(0, 1, 11), n_theta=4)
    s = g.scaled(3.0)
    assert np.allclose(s.axes[0], 3.0 * g.axes[0])
    assert np.array_equal(s.axes[1], g.axes[1])
    c = Grid.cartesian(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)).scaled(2.0)
    assert np.isclose(c.axes[0][-1], 2.0)


def test_custom_basis_requires_explicit_maps():
    with pytest.raises(TypeError, match="Basis instance"):
        Grid.custom((np.linspace(0, 1, 4), np.linspace(0, 1, 4)), "elliptic")
    ell = Basis("elliptic",
                to_cartesian=lambda u, v: (u * np.cosh(v), u * np.sinh(v)),
                from_cartesian=lambda x, y: (x, y))
    g = Grid.custom((np.linspace(0, 1, 4), np.linspace(0, 1, 3)), ell)
    assert g.basis.name == "elliptic"
    with pytest.raises(NotImplementedError, match="weights"):
        g.weights()


def test_spectrum_constructors_and_iteration():
    s = Spectrum.line(532 * nm)
    assert s.n == 1 and list(s) == [(532 * nm, 1.0)]
    f = Spectrum.flat(400 * nm, 700 * nm, 4)
    assert f.n == 4 and np.allclose(f.weights, 1.0)
    b = Spectrum.blackbody(400 * nm, 700 * nm, 31, temperature=6500.0)
    assert b.n == 31 and np.isclose(b.weights.max(), 1.0) and np.all(b.weights > 0)
    assert np.isclose(b.normalized().weights.sum(), 1.0)


def test_spectrum_validation():
    with pytest.raises(ValueError, match="ascending"):
        Spectrum([2e-6, 1e-6])
    with pytest.raises(ValueError, match="positive"):
        Spectrum([-1e-6])
    with pytest.raises(ValueError, match="one per line"):
        Spectrum([1e-6, 2e-6], [1.0])
