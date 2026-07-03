import numpy as np
import pytest

from diffraction import (
    annular_aperture,
    antialiased,
    circular_aperture,
    elliptical_aperture,
    lattice_aperture,
    lattice_sites,
    make_grid,
    rectangular_aperture,
    slit_aperture,
    square_aperture,
)

GRID = make_grid(256, 2.0)
X, Y = GRID
DX = 2.0 / 256


def test_circular_aperture_area():
    R = 0.5
    mask = circular_aperture(X, Y, R)
    area = mask.sum() * DX * DX
    np.testing.assert_allclose(area, np.pi * R**2, rtol=1e-2)


def test_circular_aperture_center_offset():
    mask = circular_aperture(X, Y, 0.2, center=(0.5, -0.3))
    ys, xs = np.nonzero(mask)
    np.testing.assert_allclose(X[0, xs].mean(), 0.5, atol=DX)
    np.testing.assert_allclose(Y[ys, 0].mean(), -0.3, atol=DX)


def test_rectangular_and_square_aperture():
    mask = rectangular_aperture(X, Y, 1.0, 0.5)
    area = mask.sum() * DX * DX
    # The inclusive <= boundary adds up to one sample row/column per edge.
    np.testing.assert_allclose(area, 0.5, rtol=3e-2)
    np.testing.assert_array_equal(
        square_aperture(X, Y, 0.8), rectangular_aperture(X, Y, 0.8, 0.8)
    )


def test_annular_aperture():
    mask = annular_aperture(X, Y, 0.3, 0.6)
    area = mask.sum() * DX * DX
    np.testing.assert_allclose(area, np.pi * (0.6**2 - 0.3**2), rtol=2e-2)


def test_elliptical_aperture():
    mask = elliptical_aperture(X, Y, 0.8, 0.4)
    area = mask.sum() * DX * DX
    np.testing.assert_allclose(area, np.pi * 0.8 * 0.4, rtol=2e-2)


def test_antialiased_area_is_more_accurate():
    R = 0.5
    hard = circular_aperture(X, Y, R).sum() * DX * DX
    soft = antialiased(circular_aperture, GRID, R, factor=8).values.real.sum() * DX * DX

    exact = np.pi * R**2
    assert abs(soft - exact) < abs(hard - exact)
    np.testing.assert_allclose(soft, exact, rtol=1e-4)


def test_antialiased_returns_field_and_kwargs():
    from diffraction import Field

    field = antialiased(circular_aperture, GRID, 0.5, center=(0.1, 0.0))
    assert isinstance(field, Field)
    assert field.grid is GRID
    t = field.values.real
    assert ((0.0 <= t) & (t <= 1.0)).all()
    # Fully interior and exterior pixels stay binary.
    assert t[128, 128] == 1.0
    assert t[0, 0] == 0.0

    with pytest.raises(ValueError):
        antialiased(circular_aperture, GRID, 0.5, factor=0)


def test_slit_aperture():
    horizontal = slit_aperture(X, Y, 0.2, orientation="x")
    assert horizontal[np.abs(Y) <= 0.1].all()
    assert not horizontal[np.abs(Y) > 0.11].any()

    vertical = slit_aperture(X, Y, 0.2, orientation="y")
    np.testing.assert_array_equal(vertical, horizontal.T)

    with pytest.raises(ValueError):
        slit_aperture(X, Y, 0.2, orientation="diagonal")


def test_lattice_sites_count_and_centering():
    sq = lattice_sites(0.1, lattice="square", size=(3, 4))
    assert sq.shape == (12, 2)
    np.testing.assert_allclose(sq.mean(axis=0), [0.0, 0.0], atol=1e-12)
    hx = lattice_sites(0.1, lattice="hexagonal", size=(4, 4))
    assert hx.shape == (16, 2)
    # hex row height is a√3/2
    ys = np.unique(hx[:, 1].round(9))
    np.testing.assert_allclose(np.diff(ys), 0.1 * np.sqrt(3) / 2, atol=1e-9)


def test_lattice_sites_validation():
    with pytest.raises(ValueError):
        lattice_sites(-1.0)
    with pytest.raises(ValueError):
        lattice_sites(0.1, lattice="triangular")
    with pytest.raises(ValueError):
        lattice_sites(0.1, size=(0, 3))


def test_lattice_aperture_area_and_centering():
    R = 0.03
    mask = lattice_aperture(X, Y, circular_aperture, 0.3, lattice="square", size=(3, 3), R=R)
    # nine well-separated holes -> total area ~ 9 π R²
    area = mask.sum() * DX * DX
    np.testing.assert_allclose(area, 9 * np.pi * R**2, rtol=5e-2)
    # the pattern is centered on the origin (centroid within a pixel)
    total = mask.sum()
    np.testing.assert_allclose((X * mask).sum() / total, 0.0, atol=DX)
    np.testing.assert_allclose((Y * mask).sum() / total, 0.0, atol=DX)


def test_lattice_aperture_composes_with_antialiased():
    from diffraction import Field

    field = antialiased(lattice_aperture, GRID, circular_aperture, 0.4, size=(2, 2), R=0.05)
    assert isinstance(field, Field)
    assert np.all((field.values.real >= 0.0) & (field.values.real <= 1.0))
