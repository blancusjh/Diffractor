import numpy as np
import pytest

from diffractor import (
    PolychromaticField,
    PolychromaticImage,
    circular_aperture,
    d65_weights,
    make_grid,
    ronchi_grating,
    square_aperture,
    wavelength_to_rgb,
)


def test_records_and_replays_amplitude_masks_per_wavelength():
    grid = make_grid(128, 4e-3)
    wl = np.linspace(450e-9, 650e-9, 5)
    pf = PolychromaticField(grid, 1.0, wavelengths=wl).add_aperture(
        circular_aperture, 0.5e-3, antialiased=True
    )
    # a wavelength-independent amplitude mask -> identical field at every λ.
    f0 = pf.field_at(wl[0]).values
    f1 = pf.field_at(wl[-1]).values
    np.testing.assert_allclose(f0, f1)


def test_lens_phase_is_chromatic():
    grid = make_grid(128, 4e-3)
    wl = np.linspace(450e-9, 650e-9, 3)
    pf = PolychromaticField(grid, 1.0, wavelengths=wl).add_lens(0.05)
    # the thin-lens phase scales with 1/λ, so per-wavelength fields differ.
    assert not np.allclose(pf.field_at(450e-9).values, pf.field_at(650e-9).values)


def test_propagate_returns_image_on_shared_grid():
    grid = make_grid(256, 4e-3)
    wl = np.linspace(450e-9, 650e-9, 8)
    img = (
        PolychromaticField(grid, 1.0, wavelengths=wl, weights=d65_weights(wl * 1e9))
        .add_grating(ronchi_grating, 60e-6, duty=0.5)
        .add_aperture(square_aperture, 1.2e-3)
        .propagate(0.3, method="fresnel_zoom", output_half_width=3e-3, output_samples=64)
    )
    assert isinstance(img, PolychromaticImage)
    assert img.rgb.shape == (64, 64, 3)
    assert img.grid.shape == (64, 64)
    assert np.all((img.rgb >= 0.0) & (img.rgb <= 1.0))


def test_single_wavelength_matches_its_color():
    grid = make_grid(256, 4e-3)
    img = (
        PolychromaticField(grid, 1.0, wavelengths=[550e-9])
        .add_aperture(circular_aperture, 0.3e-3, antialiased=True)
        .propagate(0.3, method="fresnel_zoom", output_half_width=2e-3, output_samples=48)
    )
    flat = img.rgb.reshape(-1, 3)
    bright = flat[np.argmax(flat.sum(axis=1))]
    expected = wavelength_to_rgb(550.0)
    bn = bright / (bright.max() + 1e-12)
    en = expected / (expected.max() + 1e-12)
    np.testing.assert_allclose(bn, en, atol=0.1)


def test_asm_method_and_longitudinal():
    grid = make_grid(256, 4e-3)
    wl = np.linspace(500e-9, 600e-9, 4)
    pf = PolychromaticField(grid, 1.0, wavelengths=wl).add_aperture(
        circular_aperture, 0.3e-3, antialiased=True
    )
    img = pf.propagate(0.02, method="asm", pad_factor=2)
    assert img.rgb.shape == (256, 256, 3)
    assert img.grid is grid  # asm default output grid is the source grid

    sec = pf.longitudinal([0.02, 0.03, 0.04], axis="x")
    assert sec.rgb.shape == (3, 256, 3)
    assert sec.axis == "x"


def test_validation():
    grid = make_grid(32, 1e-3)
    with pytest.raises(ValueError):
        PolychromaticField(grid, 1.0, wavelengths=[])
    with pytest.raises(ValueError):
        PolychromaticField(grid, 1.0, wavelengths=[500e-9], n=-1.0)
    with pytest.raises(ValueError):
        PolychromaticField(grid, 1.0, wavelengths=[-1.0])
    pf = PolychromaticField(grid, 1.0, wavelengths=[550e-9])
    with pytest.raises(ValueError):
        pf.propagate(0.3, method="bogus", output_half_width=1e-3)


def test_weights_override_at_propagate():
    grid = make_grid(64, 2e-3)
    wl = np.linspace(450e-9, 650e-9, 4)
    pf = PolychromaticField(grid, 1.0, wavelengths=wl).add_aperture(
        square_aperture, 0.5e-3
    )
    # flat vs D65 weights should give different composites (no crash either way).
    a = pf.propagate(0.1, method="fresnel_zoom", output_half_width=1e-3, output_samples=32).rgb
    b = pf.propagate(
        0.1, method="fresnel_zoom", output_half_width=1e-3, output_samples=32,
        weights=d65_weights(wl * 1e9),
    ).rgb
    assert a.shape == b.shape == (32, 32, 3)
