import numpy as np
import pytest

from diffractor import (
    Field,
    MonochromaticField,
    ParabolicSurface,
    circular_aperture,
    make_grid,
    square_aperture,
)


def test_constant_source_fills_grid_uniformly():
    grid = make_grid(64, 2e-3)
    U = MonochromaticField(grid, 1.0)
    np.testing.assert_allclose(U.values, 1.0)
    assert U.values.dtype == np.complex128
    assert U.shape == grid.shape


def test_callable_source_is_evaluated_on_grid():
    grid = make_grid(64, 2e-3)
    U = MonochromaticField(grid, lambda x, y: circular_aperture(x, y, 0.5e-3))
    x, y = grid
    np.testing.assert_allclose(U.values.real, circular_aperture(x, y, 0.5e-3))


def test_source_shape_mismatch_raises():
    grid = make_grid(32, 1e-3)
    with pytest.raises(ValueError):
        MonochromaticField(grid, np.ones((8, 8)))


def test_wavelength_and_n_validation():
    grid = make_grid(16, 1e-3)
    with pytest.raises(ValueError):
        MonochromaticField(grid, 1.0, wavelength=0.0)
    with pytest.raises(ValueError):
        MonochromaticField(grid, 1.0, n=-1.0)


def test_add_aperture_antialiased_gives_grey_edges():
    grid = make_grid(128, 2e-3)
    hard = MonochromaticField(grid, 1.0).add_aperture(circular_aperture, 0.5e-3)
    soft = MonochromaticField(grid, 1.0).add_aperture(
        circular_aperture, 0.5e-3, antialiased=True, factor=8
    )
    # hard mask is strictly binary; antialiased has fractional edge pixels.
    vals_hard = np.unique(hard.values.real)
    assert set(vals_hard.tolist()) <= {0.0, 1.0}
    assert np.any((soft.values.real > 0.0) & (soft.values.real < 1.0))


def test_chaining_returns_self_and_composes():
    grid = make_grid(128, 4e-3)
    base = MonochromaticField(grid, 1.0)
    returned = base.add_aperture(circular_aperture, 1e-3)
    assert returned is base  # add_* mutates and returns self for chaining

    # composing two hard masks multiplies them (logical AND of the supports).
    U = MonochromaticField(grid, 1.0).add_aperture(
        circular_aperture, 1e-3
    ).add_aperture(square_aperture, 1.2e-3)
    x, y = grid
    expected = circular_aperture(x, y, 1e-3) & square_aperture(x, y, 1.2e-3)
    np.testing.assert_allclose(U.values.real, expected.astype(float))


def test_propagate_returns_new_field_carrying_wavelength():
    grid = make_grid(256, 6e-3)
    U0 = MonochromaticField(grid, 1.0, wavelength=532e-9).add_aperture(
        circular_aperture, 0.3e-3, antialiased=True
    )
    Uz = U0.propagate(0.05, method="asm", pad_factor=2)
    assert isinstance(Uz, MonochromaticField)
    assert Uz is not U0
    assert Uz.wavelength == 532e-9
    # energy is conserved by the ASM on the native grid (Parseval).
    e0 = float(U0.intensity.sum())
    ez = float(Uz.intensity.sum())
    np.testing.assert_allclose(ez, e0, rtol=1e-3)


def test_propagate_methods_and_unknown_method():
    grid = make_grid(256, 6e-3)
    U0 = MonochromaticField(grid, 1.0, wavelength=532e-9).add_aperture(
        circular_aperture, 0.3e-3, antialiased=True
    )
    fres = U0.propagate(1.0, method="fresnel")
    fraun = U0.propagate(1.0, method="fraunhofer")
    assert fres.shape == grid.shape and fraun.shape == grid.shape
    with pytest.raises(ValueError):
        U0.propagate(1.0, method="bogus")


def test_add_surface_updates_medium_index():
    grid = make_grid(128, 4e-3)
    U = MonochromaticField(grid, 1.0, wavelength=532e-9)
    assert U.n == 1.0
    U.add_surface(ParabolicSurface(focal_length=0.1), n2=1.5)
    assert U.n == 1.5  # entered the second medium


def test_add_surface_requires_n2_for_parabolic():
    grid = make_grid(64, 2e-3)
    U = MonochromaticField(grid, 1.0)
    with pytest.raises(ValueError):
        U.add_surface(ParabolicSurface(focal_length=0.1))


def test_to_field_roundtrip():
    grid = make_grid(64, 2e-3)
    U = MonochromaticField(grid, 1.0, wavelength=633e-9)
    f = U.to_field()
    assert isinstance(f, Field)
    assert f.grid is grid


def test_longitudinal_uses_stored_wavelength():
    grid = make_grid(128, 2e-3)
    U0 = MonochromaticField(grid, 1.0, wavelength=550e-9).add_aperture(
        circular_aperture, 0.4e-3, antialiased=True
    )
    sec = U0.longitudinal(np.linspace(0.02, 0.05, 5), axis="x")
    assert sec.intensity.shape == (5, 128)
    assert sec.axis == "x"
