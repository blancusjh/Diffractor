import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from diffraction import (
    Field,
    LongitudinalSection,
    antialiased,
    circular_aperture,
    intensity,
    make_grid,
    plot_intensity,
    plot_longitudinal,
    plot_rgb,
)


@pytest.fixture()
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_intensity_normalizes_to_unit_peak():
    values = np.array([[1.0, 2.0], [0.0, 4.0]], dtype=complex)
    I = intensity(values)
    assert I.max() == 1.0
    np.testing.assert_allclose(I, np.abs(values) ** 2 / 16.0)
    # raw option
    np.testing.assert_allclose(intensity(values, normalize=False), np.abs(values) ** 2)
    # all-zero input does not divide by zero
    np.testing.assert_allclose(intensity(np.zeros((2, 2))), 0.0)


def test_plot_intensity_log_floor_and_extent(ax):
    grid = make_grid(64, 2e-3)
    U = antialiased(circular_aperture, grid, 0.5e-3)
    im = plot_intensity(ax, U, vmin=-4.0)
    # the display range is clipped to the requested decade window
    assert im.get_clim() == (-4.0, 0.0)
    assert im.get_array().max() <= 1e-6  # log10(1 + floor pad) barely above 0
    x0, x1, y0, y1 = im.get_extent()
    assert np.isclose(x0, grid.x.min()) and np.isclose(x1, grid.x.max())


def test_plot_longitudinal_normalize_flag(ax):
    z = np.linspace(0.0, 1.0, 5)
    t = np.linspace(-1e-3, 1e-3, 7)
    intensity_map = np.full((5, 7), 0.01)
    sec = LongitudinalSection(intensity=intensity_map, z=z, t=t, axis="x")

    im_norm = plot_longitudinal(ax, sec, vmin=-4.0)  # default: self-normalized
    assert np.isclose(im_norm.get_array().max(), 0.0, atol=1e-3)

    im_raw = plot_longitudinal(ax, sec, vmin=-4.0, normalize=False)
    # preserved as supplied: 0.01 -> log10 = -2
    assert np.isclose(im_raw.get_array().max(), -2.0, atol=1e-3)
    assert ax.get_xlabel() == "z [m]"
    assert ax.get_ylabel() == "x [m]"


def test_plot_rgb_clips_and_extent(ax):
    grid = make_grid(16, 1e-3)
    rgb = np.random.default_rng(0).uniform(-0.2, 1.3, size=(16, 16, 3))
    im = plot_rgb(ax, rgb, grid)
    shown = im.get_array()
    assert shown.min() >= 0.0 and shown.max() <= 1.0
