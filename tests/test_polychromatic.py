import numpy as np
import pytest

from diffraction import (
    Field,
    antialiased,
    circular_aperture,
    make_grid,
    propagate_polychromatic,
    ronchi_grating,
    square_aperture,
    wavelength_to_rgb,
)


def _aperture():
    grid = make_grid(256, 4e-3)
    return antialiased(circular_aperture, grid, 0.3e-3)


def test_returns_rgb_image_and_grid():
    U0 = _aperture()
    wl = np.linspace(450e-9, 650e-9, 8)
    rgb, grid = propagate_polychromatic(
        U0, wl, z=0.3, output_half_width=2e-3, output_samples=64
    )
    assert rgb.shape == (64, 64, 3)
    assert np.all((rgb >= 0.0) & (rgb <= 1.0))
    assert grid.shape == (64, 64)


def test_single_wavelength_matches_its_color():
    U0 = _aperture()
    rgb, _ = propagate_polychromatic(
        U0, [550e-9], z=0.3, output_half_width=2e-3, output_samples=48
    )
    # brightest pixel should have the hue of 550 nm
    flat = rgb.reshape(-1, 3)
    bright = flat[np.argmax(flat.sum(axis=1))]
    expected = wavelength_to_rgb(550.0)
    # compare normalized hue (direction), brightness aside
    bn = bright / (bright.max() + 1e-12)
    en = expected / (expected.max() + 1e-12)
    np.testing.assert_allclose(bn, en, atol=0.1)


def test_asm_propagator_path():
    grid = make_grid(256, 4e-3)
    U0 = antialiased(circular_aperture, grid, 0.3e-3)
    wl = np.linspace(450e-9, 650e-9, 6)
    rgb, out_grid = propagate_polychromatic(
        U0, wl, z=0.02, propagator="asm", pad_factor=2
    )
    assert rgb.shape == (256, 256, 3)
    assert out_grid is grid  # asm default output grid is the source grid


def test_field_of_callable_for_chromatic_element():
    # A callable source (e.g. a chromatic phase element) is propagated per λ.
    grid = make_grid(256, 4e-3)
    x, y = grid
    base = square_aperture(x, y, 1e-3)

    def field_of(lam):
        # a wavelength-dependent tilt just to exercise the callable path
        phase = np.exp(2j * np.pi * (1e-3 / lam) * 0.0 * x)
        return Field(grid, (base * phase).astype(complex))

    wl = np.linspace(450e-9, 650e-9, 5)
    rgb, _ = propagate_polychromatic(
        field_of, wl, z=0.3, output_half_width=3e-3, output_samples=48
    )
    assert rgb.shape == (48, 48, 3)


def test_validation():
    U0 = _aperture()
    with pytest.raises(ValueError):
        propagate_polychromatic(U0, [], z=0.3, output_half_width=2e-3)
    with pytest.raises(ValueError):
        propagate_polychromatic(
            U0, [550e-9], z=0.3, propagator="bogus", output_half_width=2e-3
        )
