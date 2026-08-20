"""Propagation: one method set, every basis — checked against closed forms."""

import warnings

import numpy as np
import pytest
from scipy.special import j1

from diffractor import (Field, Grid, Medium, MonochromaticField, Spectrum,
                        angular_spectrum, fraunhofer, fresnel,
                        fresnel_validity_distance, rayleigh_sommerfeld,
                        spectral_budget)

LAM = 1.0                       # unit wavelength: everything in λ


def _disc_field(a=15.0, n_r=3001):
    g = Grid.polar(np.linspace(0.0, a, n_r))
    return MonochromaticField(g, np.ones(g.shape), LAM)


def _axial_exact(z, a=15.0):
    """Closed-form on-axis field of a uniformly lit disc (RS-I integrates to
    exactly this)."""
    k = 2 * np.pi / LAM
    R = np.hypot(a, z)
    return np.exp(1j * k * z) - (z / R) * np.exp(1j * k * R)


def test_asm_disc_on_axis_closed_form():
    """A hard disc is the polar quadrature's worst case (its spectrum fills
    the whole propagating band, and the band edge is where kz turns fastest);
    the deep near field belongs to rayleigh_sommerfeld — checked at ~5e-5 in
    its own docstring physics — so the tolerance here is the ASM's honest
    hard-edge accuracy class, not the package's best."""
    f = _disc_field()
    for z in (50.0, 120.0, 300.0, 600.0):
        out = angular_spectrum(f, z)
        assert abs(abs(out.u[0, 0]) - abs(_axial_exact(z))) < 3e-2


def test_asm_cartesian_equals_polar_on_smooth_field():
    """The same supergaussian, cartesian FFT path vs polar Hankel path."""
    a, lam = 12.0, 0.5
    x = (np.arange(1024) - 512) * 0.06
    gc = Grid.cartesian(x, x)
    X, Y = gc.meshes()
    R = np.hypot(X, Y)
    fc = MonochromaticField(gc, np.exp(-(R / a) ** 8), lam)
    rp = np.linspace(0.0, x[-1], 1500)
    fp = MonochromaticField(Grid.polar(rp), np.exp(-(rp[:, None] / a) ** 8), lam)

    z = 150.0
    oc = angular_spectrum(fc, z, pad_factor=2)
    op = angular_spectrum(fp, z)
    I_c = np.abs(oc.u[512:, 512]) ** 2
    I_p = np.interp(x[512:], op.grid.axes[0], np.abs(op.u[:, 0]) ** 2)
    sel = x[512:] < 22.0
    assert np.abs(I_c - I_p)[sel].max() < 1e-4 * I_p.max()


def test_fraunhofer_airy_far_field():
    a, z = 25.0, 3e5
    lam = 0.5
    g = Grid.polar(np.linspace(0.0, a, 2001))
    f = MonochromaticField(g, np.ones(g.shape), lam)
    out_grid = Grid.polar(np.linspace(0.0, 6000.0, 300))
    ff = fraunhofer(f, z, output_grid=out_grid)
    theta = np.arctan2(out_grid.axes[0], z)
    x = (2 * np.pi / lam) * a * np.sin(theta)
    airy = np.where(x == 0, 1.0,
                    (2 * j1(np.where(x == 0, 1, x)) / np.where(x == 0, 1, x)) ** 2)
    I = np.abs(ff.u[:, 0]) ** 2
    assert np.abs(I / I[0] - airy).max() < 1e-4


def test_fresnel_gate_policies():
    f = _disc_field(a=25.0, n_r=501)
    z_bad = 0.5 * fresnel_validity_distance(25.0, LAM)
    with pytest.raises(ValueError, match="validity"):
        fresnel(f, z_bad)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fresnel(f, z_bad, policy="warn")
    assert len(caught) == 1
    fresnel(f, z_bad, policy="force")          # the caller who knows better
    with pytest.raises(ValueError, match="raise.*warn.*force"):
        fresnel(f, z_bad, policy="loose")
    with pytest.raises(ValueError, match="positive"):
        fresnel(f, -10.0, policy="force")


def test_fresnel_polychromatic_requires_output_grid():
    g = Grid.polar(np.linspace(0.0, 25.0, 501))
    f = Field(g, np.ones((*g.shape, 3)), Spectrum.flat(0.4, 0.7, 3))
    with pytest.raises(ValueError, match="output_grid"):
        fresnel(f, 1e5)
    out = fresnel(f, 1e5, output_grid=Grid.polar(np.linspace(0.0, 4000.0, 64)))
    assert out.values.shape == (64, 1, 3)


def test_fresnel_zoom_equals_scaled_grid_mode():
    """The explicit-window (zoom) evaluation, asked for the natural scaled
    grid's own nodes, reproduces the natural-grid result."""
    f = _disc_field(a=25.0, n_r=1201)
    z = 5e4
    natural = fresnel(f, z)
    zoom = fresnel(f, z, output_grid=natural.grid)
    scale = np.abs(natural.u).max()
    assert np.abs(zoom.u - natural.u).max() < 1e-8 * scale


def test_rayleigh_sommerfeld_refuses_non_axisymmetric():
    g = Grid.polar(np.linspace(0.0, 10.0, 64), n_theta=4)
    f = MonochromaticField(g, np.ones(g.shape), LAM)
    with pytest.raises(ValueError, match="axisymmetric"):
        rayleigh_sommerfeld(f, 10.0)
    x = np.linspace(-1, 1, 16)
    fc = MonochromaticField(Grid.cartesian(x, x), np.ones((16, 16)), LAM)
    with pytest.raises(ValueError, match="angular_spectrum"):
        rayleigh_sommerfeld(fc, 10.0)


def test_spectral_budget_smooth_vs_sharp():
    r = np.linspace(0.0, 30.0, 3000)
    g = Grid.polar(r)
    smooth = MonochromaticField(g, np.exp(-(r[:, None] / 10.0) ** 2), LAM)
    sharp = MonochromaticField(g, (r[:, None] <= 10.0).astype(complex), LAM)
    b_smooth = spectral_budget(smooth)
    b_sharp = spectral_budget(sharp)
    assert b_smooth[0] < 1e-4
    assert b_sharp[0] > 10 * max(b_smooth[0], 1e-12)


def test_spectral_axis_matches_per_wavelength_loop():
    """One broadband propagation = n monochromatic propagations."""
    r = np.linspace(0.0, 20.0, 800)
    g = Grid.polar(r)
    spec = Spectrum.flat(0.45, 0.65, 4)
    vals = np.exp(-(r[:, None, None] / 8.0) ** 2) * np.ones((1, 1, 4))
    f = Field(g, vals, spec)
    kg = g.reciprocal(k_max=1.02 * 2 * np.pi / 0.45, n_k=2000)
    out = angular_spectrum(f, 60.0, kgrid=kg)
    for l, (lam, _) in enumerate(spec):
        mono = MonochromaticField(g, vals[..., l], lam)
        out_l = angular_spectrum(mono, 60.0, kgrid=kg)
        assert np.allclose(out.values[..., l], out_l.values[..., 0],
                           rtol=0, atol=1e-12 * np.abs(out_l.values).max())


def test_propagators_preserve_monochromatic_type():
    f = _disc_field(n_r=501)
    for out in (angular_spectrum(f, 100.0),
                fresnel(f, 1e5),
                rayleigh_sommerfeld(f, 100.0)):
        assert type(out) is MonochromaticField
        assert out.wavelength == f.wavelength


def test_polar_options_refused_where_meaningless():
    f = _disc_field(n_r=201)
    with pytest.raises(ValueError, match="quadrature"):
        angular_spectrum(f, 10.0, pad_factor=2)
    with pytest.raises(ValueError, match="quadrature"):
        angular_spectrum(f, 10.0, bandlimit=True)


def test_back_propagation_round_trip():
    """ASM with z then −z is the identity on the propagating band."""
    r = np.linspace(0.0, 20.0, 1000)
    g = Grid.polar(r)
    f = MonochromaticField(g, np.exp(-(r[:, None] / 6.0) ** 2), LAM)
    kg = g.reciprocal(k_max=1.02 * 2 * np.pi, n_k=1500)
    there = angular_spectrum(f, 40.0, kgrid=kg)
    back = angular_spectrum(there, -40.0, kgrid=kg)
    assert np.abs(back.u - f.u).max() < 5e-3
