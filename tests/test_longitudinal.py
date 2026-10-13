import numpy as np
import pytest

from diffractor import (
    Field,
    MonochromaticField,
    circular_aperture,
    longitudinal_field,
    make_grid,
    ronchi_grating,
)

WL = 550e-9


def _circ(grid, R, *, f=None, wl=WL):
    """Antialiased circular aperture (optionally with a thin lens) as a Field."""
    mf = MonochromaticField(grid, 1.0, wavelength=wl).add_aperture(
        circular_aperture, R, antialiased=True)
    if f is not None:
        mf = mf.add_lens(f)
    return mf.to_field()


def test_section_shape_and_coordinates():
    grid = make_grid(256, 2e-3)
    U0 = _circ(grid, 0.4e-3)
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
    U0 = _circ(grid, 1.2e-3, f=f)
    zs = np.linspace(0.5 * f, 1.5 * f, 61)
    sec = longitudinal_field(U0, WL, zs, axis="x")
    center = sec.t.size // 2
    on_axis = sec.intensity[:, center]  # I(z) along the optical axis
    z_peak = sec.z[int(np.argmax(on_axis))]
    # the axial intensity maximum sits at the focal plane
    assert abs(z_peak - f) < 0.06 * f


def test_axis_x_and_y_agree_for_rotationally_symmetric_input():
    grid = make_grid(512, 3e-3)
    U0 = _circ(grid, 0.5e-3)
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
    U0 = _circ(grid, 0.2e-3)
    with pytest.raises(ValueError):
        longitudinal_field(U0, WL, [0.01], axis="z")


def test_zoomed_line_matches_native_at_native_coordinates():
    # The decoupled output line evaluated exactly at the native x samples must
    # reproduce the native-FFT section (both are exact DFTs).
    grid = make_grid(128, 2e-3)
    U0 = _circ(grid, 0.5e-3)
    zs = [0.03, 0.06]
    native = longitudinal_field(U0, WL, zs, axis="x", normalize=False)
    x = grid.x[0, :]
    # a line spanning the same window with exactly the native sample positions
    zoomed = longitudinal_field(
        U0, WL, zs, axis="x", normalize=False,
        output_half_width=-x[0], output_samples=x.size + 1,
    )
    # zoomed t = linspace(x[0], -x[0], N+1) contains the native x as t[:-1]
    np.testing.assert_allclose(zoomed.t[:-1], x, atol=1e-12)
    np.testing.assert_allclose(zoomed.intensity[:, :-1], native.intensity, rtol=1e-9, atol=1e-15)


def test_zoomed_line_resolves_waist_below_native_spacing():
    # A lens focus whose Airy radius is smaller than the input spacing: the
    # decoupled line must still resolve it (radius ~ 0.61 lambda f / R).
    f = 0.05
    grid = make_grid(256, 2e-3)  # dx ~ 7.8 um
    U0 = _circ(grid, 0.7e-3, f=f)
    airy_radius = 0.61 * WL * f / 0.7e-3  # ~ 2.4 um < dx
    sec = longitudinal_field(
        U0, WL, [f], axis="x",
        output_half_width=6 * airy_radius, output_samples=601,
    )
    line = sec.intensity[0]
    center = line.argmax()
    assert abs(sec.t[center]) < 0.2 * airy_radius  # peak on axis
    # first zero near the Airy radius
    right = line[center:]
    first_min = center + 1 + int(np.argmin(right[1 : int(1.8 * airy_radius / (sec.t[1] - sec.t[0]))]))
    np.testing.assert_allclose(abs(sec.t[first_min]), airy_radius, rtol=0.15)


def test_zoom_validation():
    grid = make_grid(64, 1e-3)
    U0 = _circ(grid, 0.2e-3)
    with pytest.raises(ValueError):
        longitudinal_field(U0, WL, [0.01], output_half_width=-1e-4)
    with pytest.raises(ValueError):
        longitudinal_field(U0, WL, [0.01], output_half_width=1e-4, output_samples=1)
