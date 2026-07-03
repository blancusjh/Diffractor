import numpy as np
import pytest

from diffraction import (
    Field,
    antialiased,
    circular_aperture,
    longitudinal_field,
    make_grid,
    ronchi_grating,
    thin_lens,
)

WL = 550e-9


def test_section_shape_and_coordinates():
    grid = make_grid(256, 2e-3)
    U0 = antialiased(circular_aperture, grid, 0.4e-3)
    zs = np.linspace(0.02, 0.05, 7)
    sec = longitudinal_field(U0, WL, zs, axis="x")
    assert sec.intensity.shape == (7, 256)
    np.testing.assert_allclose(sec.z, zs)
    np.testing.assert_allclose(sec.t, grid.x[0, :])
    assert sec.axis == "x"


def test_plane_wave_stays_uniform_along_z():
    # A uniform field (only a DC component) propagates unchanged: every plane
    # of the longitudinal section is flat and identical.
    grid = make_grid(128, 1e-3)
    U0 = Field(grid, np.ones(grid.shape, dtype=complex))
    sec = longitudinal_field(U0, WL, np.linspace(0.0, 0.1, 8), axis="x", normalize=False)
    np.testing.assert_allclose(sec.intensity, 1.0, atol=1e-6)


def test_lens_on_axis_intensity_peaks_at_focus():
    f = 0.08
    grid = make_grid(1024, 4e-3)
    U0 = antialiased(circular_aperture, grid, 1.2e-3) * thin_lens(grid, f, WL)
    zs = np.linspace(0.5 * f, 1.5 * f, 61)
    sec = longitudinal_field(U0, WL, zs, axis="x")
    center = sec.t.size // 2
    on_axis = sec.intensity[:, center]  # I(z) along the optical axis
    z_peak = sec.z[int(np.argmax(on_axis))]
    # the axial intensity maximum sits at the focal plane
    assert abs(z_peak - f) < 0.06 * f


def test_axis_x_and_y_agree_for_rotationally_symmetric_input():
    grid = make_grid(512, 3e-3)
    U0 = antialiased(circular_aperture, grid, 0.5e-3)
    zs = np.linspace(0.03, 0.09, 9)
    sx = longitudinal_field(U0, WL, zs, axis="x")
    sy = longitudinal_field(U0, WL, zs, axis="y")
    np.testing.assert_allclose(sx.intensity, sy.intensity, atol=1e-6)


def test_grating_self_images_at_talbot_distance():
    # A Ronchi grating filling an integer number of periods reconstructs its
    # own intensity at z_T = 2 d²/λ (Talbot self-imaging), and is markedly
    # different at z_T / 2 (a half-period lateral shift).
    period = 40e-6
    grid = make_grid(2048, 40 * period)
    x, y = grid
    U0 = Field(grid, ronchi_grating(x, y, period, duty=0.5).astype(complex))
    z_t = 2.0 * period**2 / WL
    sec = longitudinal_field(U0, WL, [0.0, 0.5 * z_t, z_t], axis="x", normalize=False)
    input_line, half_line, talbot_line = sec.intensity

    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    # self-image at z_T is highly correlated with the input; the half-Talbot
    # (shifted) image is not.
    assert corr(input_line, talbot_line) > 0.9
    assert corr(input_line, half_line) < corr(input_line, talbot_line)


def test_axis_validation():
    grid = make_grid(64, 1e-3)
    U0 = antialiased(circular_aperture, grid, 0.2e-3)
    with pytest.raises(ValueError):
        longitudinal_field(U0, WL, [0.01], axis="z")
