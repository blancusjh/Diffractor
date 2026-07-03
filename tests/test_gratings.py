import numpy as np
import pytest

from diffraction import (
    Field,
    cross_grating,
    fresnel_zoom_propagator,
    make_grid,
    phase_grating,
    ronchi_grating,
    sinusoidal_amplitude_grating,
    square_aperture,
    thin_lens,
)

L, N = 4e-3, 1024
WL, Z, D = 550e-9, 0.5, 50e-6
GRID = make_grid(N, L)
X, Y = GRID
X_ORDER = WL * Z / D  # position of order 1 in the far field


def _orders(mask_values):
    ap = square_aperture(X, Y, 1e-3)
    U0 = Field(GRID, (mask_values * ap).astype(complex))
    half, nout = 3.2 * X_ORDER, 2048
    out = fresnel_zoom_propagator(U0, z=Z, wavelength=WL, output_half_width=half, output_samples=nout)
    row = np.abs(out.values[nout // 2]) ** 2
    xo = out.grid.x[0, :]

    def energy(m, w=0.4 * X_ORDER):
        return row[np.abs(xo - m * X_ORDER) < w].sum()

    return xo, row, energy


def test_ronchi_transmission_fraction_matches_duty():
    # Hard binary kernel: edges snap to pixels, so the mean duty is only
    # accurate to ~one pixel per period (use antialiased for exact area).
    for duty in (0.3, 0.5, 0.7):
        t = ronchi_grating(X, Y, D, duty=duty)
        np.testing.assert_allclose(t.mean(), duty, atol=2e-2)


def test_ronchi_order_positions_and_suppression():
    xo, row, energy = _orders(ronchi_grating(X, Y, D, duty=0.5))
    # order +1 peak sits at m λ z / d
    near = np.abs(xo - X_ORDER) < 0.5 * X_ORDER
    peak_x = xo[near][np.argmax(row[near])]
    np.testing.assert_allclose(peak_x, X_ORDER, rtol=2e-2)
    # 50% duty kills the even orders
    assert energy(2) / energy(0) < 0.05
    # order 1 carries a substantial fraction (theory ~0.4)
    assert 0.2 < energy(1) / energy(0) < 0.6


def test_sinusoidal_amplitude_only_zero_and_first_orders():
    xo, row, energy = _orders(sinusoidal_amplitude_grating(X, Y, D))
    assert energy(1) / energy(0) > 0.1
    assert energy(2) / energy(0) < 0.02  # higher orders essentially absent


def test_phase_grating_unit_modulus():
    pg = phase_grating(GRID, WL, period=D, height=WL, profile="sinusoidal")
    np.testing.assert_allclose(np.abs(pg.values), 1.0, atol=1e-9)
    assert isinstance(pg, Field)


def test_blazed_grating_steers_into_one_order():
    height = WL / (1.5 - 1.0)  # 2π phase depth -> blaze into a single order
    pg = phase_grating(GRID, WL, period=D, height=height, n1=1.0, n2=1.5, profile="blazed")
    xo, row, energy = _orders(pg.values)
    e = {m: energy(m) for m in (-2, -1, 0, 1, 2)}
    dominant = max(e, key=e.get)
    assert dominant in (-1, 1)  # a first order, not the zero order
    total = sum(e.values())
    assert e[dominant] / total > 0.8  # most energy in that one order
    assert e[dominant] > 10 * e[-dominant]  # strongly asymmetric


def test_grating_plus_lens_focuses_order_to_focal_plane():
    # A grating + lens is a spectrometer: at the lens back focal plane, order m
    # of wavelength WL lands at x_m = m WL f / d (same law as the far field with
    # z -> f), but focused to a sharp spot instead of the broad Fresnel pattern.
    f = 0.25
    x_order = WL * f / D
    ap = square_aperture(X, Y, 1e-3)
    grating = Field(GRID, (ronchi_grating(X, Y, D, duty=0.5) * ap).astype(complex))
    focused = grating * thin_lens(GRID, f, WL)
    half, nout = 3.2 * x_order, 2048
    out = fresnel_zoom_propagator(
        focused, z=f, wavelength=WL, output_half_width=half, output_samples=nout
    )
    row = np.abs(out.values[nout // 2]) ** 2
    xo = out.grid.x[0, :]
    # the +1 order peaks at m WL f / d
    near = np.abs(xo - x_order) < 0.5 * x_order
    peak_x = xo[near][np.argmax(row[near])]
    np.testing.assert_allclose(peak_x, x_order, rtol=2e-2)


def test_cross_grating_is_separable_2d():
    cg = cross_grating(X, Y, D, kind="ronchi")
    gx = ronchi_grating(X, Y, D, orientation="x")
    gy = ronchi_grating(X, Y, D, orientation="y")
    np.testing.assert_allclose(cg, gx * gy)


def test_grating_validation():
    with pytest.raises(ValueError):
        ronchi_grating(X, Y, -1.0)
    with pytest.raises(ValueError):
        ronchi_grating(X, Y, D, duty=1.5)
    with pytest.raises(ValueError):
        ronchi_grating(X, Y, D, orientation="diagonal")
    with pytest.raises(ValueError):
        phase_grating(GRID, WL, period=D, height=WL, profile="bogus")
    with pytest.raises(ValueError):
        cross_grating(X, Y, D, kind="bogus")
