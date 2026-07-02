import numpy as np
import pytest

from diffraction import CartesianSurface, ParabolicSurface, make_grid, thin_element_phase


def test_parabolic_sag():
    surface = ParabolicSurface(focal_length=0.5, z0=0.1)
    rho = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(surface.sag_rho(rho), 0.1 + rho**2 / 2.0)

    x = np.array([[3.0]])
    y = np.array([[4.0]])
    np.testing.assert_allclose(surface.sag(x, y), 0.1 + 25.0 / 2.0)


def test_parabolic_requires_nonzero_focal_length():
    with pytest.raises(ValueError):
        ParabolicSurface(focal_length=0.0)


def test_cartesian_paraxial_curvature():
    # Near the axis the Cartesian oval must reduce to the paraxial diopter:
    # z ≈ ρ² / (2 R) with (n2 - n1) / R = n2 / zi - n1 / zo.
    n1, n2, zo, zi = 1.0, 1.5, -0.8, 1.2
    surface = CartesianSurface(n1=n1, n2=n2, zo=zo, zi=zi)

    curvature = (n2 / zi - n1 / zo) / (n2 - n1)
    rho = 1e-4
    sag = surface.sag(np.array(rho), np.array(0.0))
    np.testing.assert_allclose(sag, 0.5 * curvature * rho**2, rtol=1e-6)


def test_cartesian_oval_is_stigmatic():
    # The defining property: the optical path n1|AQ| + n2|QA'| through any
    # surface point equals the axial path, to machine precision. This is
    # what the (G, O, T, S) series formulas previously failed (1.6 waves
    # of error at f/45 for zo=-50 mm, zi=100 mm).
    n1, n2, zo, zi = 1.0, 1.5, -0.05, 0.1
    surface = CartesianSurface(n1=n1, n2=n2, zo=zo, zi=zi)

    r = np.linspace(0.0, 3.5e-3, 200)  # up to NA_obj ~ 0.07
    z = surface.sag(r, np.zeros_like(r))
    opl = n1 * np.sqrt((z - zo) ** 2 + r**2) + n2 * np.sqrt((zi - z) ** 2 + r**2)
    axial = n1 * abs(zo) + n2 * zi
    # < 1e-12 m of path error, i.e. < 2e-6 waves in the visible
    np.testing.assert_allclose(opl, axial, rtol=0.0, atol=1e-12)


def test_cartesian_oval_stigmatic_for_collimated_input():
    # Same property with the object at quasi-infinity, which stresses the
    # cancellation-free formulation of the solver residual.
    n1, n2, zo, zi = 1.0, 1.5, -1e7, 1.0
    surface = CartesianSurface(n1=n1, n2=n2, zo=zo, zi=zi)

    r = np.linspace(0.0, 5e-3, 100)
    z = surface.sag(r, np.zeros_like(r))
    # Stable evaluation of (|AQ| - |zo|) to avoid catastrophic cancellation
    # in the test itself.
    d1 = np.sqrt((z - zo) ** 2 + r**2)
    excess1 = (z * z - 2 * z * zo + r**2) / (d1 - zo)
    d2 = np.sqrt((zi - z) ** 2 + r**2)
    excess2 = (z * z - 2 * z * zi + r**2) / (d2 + zi)
    np.testing.assert_allclose(n1 * excess1 + n2 * excess2, 0.0, atol=1e-12)


def test_cartesian_sag_is_axisymmetric_and_zero_at_vertex():
    surface = CartesianSurface(n1=1.0, n2=1.5, zo=-1e5, zi=1.0)
    x, y = make_grid(64, 2e-3)
    sag = surface.sag(x, y)
    assert np.isfinite(sag).all()
    np.testing.assert_allclose(surface.sag(np.array(0.0), np.array(0.0)), 0.0, atol=1e-15)
    np.testing.assert_allclose(sag, sag.T, atol=1e-15)


def test_cartesian_validation():
    with pytest.raises(ValueError):
        CartesianSurface(n1=1.0, n2=1.0, zo=-1.0, zi=1.0)
    with pytest.raises(ValueError):
        CartesianSurface(n1=1.0, n2=1.5, zo=0.0, zi=1.0)


def test_cartesian_phase_mask_converges_to_design_image_point():
    # Collimated illumination (zo -> -inf) through the stigmatic surface
    # must leave a converging paraxial wavefront focused at zi in medium
    # n2: t = exp(-i k_m rho^2 / (2 zi)) with k_m = 2 pi n2 / wavelength.
    # This pins down the sign of the thin-element phase.
    n1, n2, zi, wavelength = 1.0, 1.5, 1.0, 500e-9
    surface = CartesianSurface(n1=n1, n2=n2, zo=-1e9, zi=zi)
    grid = make_grid(64, 0.4e-3)  # small window: paraxial regime
    x, y = grid

    mask = surface.phase_mask(grid, wavelength, n1, n2)
    k_m = 2 * np.pi * n2 / wavelength
    expected = np.exp(-1.0j * k_m * (x**2 + y**2) / (2.0 * zi))
    np.testing.assert_allclose(np.angle(mask.values / expected), 0.0, atol=1e-6)


def test_thin_element_phase():
    wavelength = 500e-9
    sag = np.array([0.0, wavelength / 2.0])
    t = thin_element_phase(sag, wavelength, n1=1.0, n2=2.0)

    np.testing.assert_allclose(np.abs(t), 1.0)
    # (n2 - n1) * sag = λ/2 → phase of π, i.e. t = -1.
    np.testing.assert_allclose(t, [1.0, -1.0], atol=1e-12)


def test_surface_phase_mask_matches_thin_element_phase():
    surface = ParabolicSurface(focal_length=0.1)
    grid = make_grid(32, 1e-3)
    x, y = grid
    expected = thin_element_phase(surface.sag(x, y), 500e-9, 1.0, 1.5)
    np.testing.assert_allclose(surface.phase_mask(grid, 500e-9, 1.0, 1.5).values, expected)
