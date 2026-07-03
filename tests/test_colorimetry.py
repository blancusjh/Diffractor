import numpy as np
import pytest

from diffraction import (
    blackbody_weights,
    cie_xyz,
    d65_weights,
    spectrum_to_srgb,
    wavelength_to_rgb,
    xyz_to_srgb,
)


def test_cie_xyz_y_peaks_near_555nm():
    lam = np.arange(400, 701, 1.0)
    _, y, _ = cie_xyz(lam)
    peak = lam[np.argmax(y)]
    assert 545 <= peak <= 565  # luminous efficiency peak


def test_wavelength_to_rgb_hues():
    r = wavelength_to_rgb(630.0)  # red
    g = wavelength_to_rgb(530.0)  # green
    b = wavelength_to_rgb(470.0)  # blue
    assert r[0] > r[1] and r[0] > r[2]
    assert g[1] > g[0] and g[1] > g[2]
    assert b[2] > b[0] and b[2] >= b[1] - 1e-9
    for c in (r, g, b):
        assert np.all((c >= 0.0) & (c <= 1.0))


def test_xyz_to_srgb_shape_and_range():
    XYZ = np.random.default_rng(0).random((5, 4, 3))
    rgb = xyz_to_srgb(XYZ)
    assert rgb.shape == (5, 4, 3)
    assert np.all((rgb >= 0.0) & (rgb <= 1.0))
    with pytest.raises(ValueError):
        xyz_to_srgb(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        xyz_to_srgb(np.zeros((4, 3)), gamut="bogus")


def test_flat_spectrum_is_near_neutral():
    wl = np.arange(400, 701, 10.0)
    rgb = spectrum_to_srgb(wl, np.ones((len(wl), 3, 3)))[0, 0]
    assert np.all(rgb > 0.6)
    assert rgb.max() - rgb.min() < 0.3  # roughly balanced


def test_d65_renders_white():
    wl = np.arange(400, 701, 10.0)
    rgb = spectrum_to_srgb(wl, np.ones((len(wl), 2, 2)), weights=d65_weights(wl))[0, 0]
    assert rgb.max() - rgb.min() < 0.1  # near-neutral by construction


def test_single_wavelength_reduces_to_its_color():
    stack = np.ones((1, 3, 3))
    got = spectrum_to_srgb(np.array([530.0]), stack)[0, 0]
    np.testing.assert_allclose(got, wavelength_to_rgb(530.0), atol=1e-9)


def test_spectrum_to_srgb_validation():
    with pytest.raises(ValueError):
        spectrum_to_srgb(np.array([500.0, 600.0]), np.ones((3, 4, 4)))  # K mismatch


def test_d65_and_blackbody_weights():
    wl = np.array([450.0, 550.0, 650.0])
    d = d65_weights(wl)
    assert np.all(d > 0)
    bb = blackbody_weights(wl, 6500.0)
    assert np.all(bb > 0) and bb.max() == pytest.approx(1.0)
    # warmer blackbody weights the red end relatively more than the blue
    warm = blackbody_weights(wl, 3000.0)
    assert warm[2] / warm[0] > bb[2] / bb[0]
    with pytest.raises(ValueError):
        blackbody_weights(wl, -1.0)
