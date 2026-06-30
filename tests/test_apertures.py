import numpy as np
import pytest

from diffraction import (
    annular_aperture,
    antialiased,
    circular_aperture,
    elliptical_aperture,
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
    soft = antialiased(circular_aperture, GRID, R, factor=8).sum() * DX * DX

    exact = np.pi * R**2
    assert abs(soft - exact) < abs(hard - exact)
    np.testing.assert_allclose(soft, exact, rtol=1e-4)


def test_antialiased_values_and_kwargs():
    t = antialiased(circular_aperture, GRID, 0.5, center=(0.1, 0.0))
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
