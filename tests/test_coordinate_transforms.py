"""Physical-coordinate references independent of FFT centering conventions."""

import numpy as np
import pytest

from diffractor import (
    AngularSpectrum, Field, FT2, Grid, IFT2, asm_propagator,
    fraunhofer_propagator, frequency_grid, fresnel_propagator,
    fresnel_zoom_propagator, make_grid,
)


def rectangular_grid(shape, shift=(0.0, 0.0)):
    ny, nx = shape
    x = np.linspace(-1e-3, 1e-3, nx, endpoint=False) + shift[0]
    y = np.linspace(-0.7e-3, 0.7e-3, ny, endpoint=False) + shift[1]
    return Grid(*np.meshgrid(x, y))


def random_field(shape, shift=(0.0, 0.0)):
    grid = rectangular_grid(shape, shift)
    rng = np.random.default_rng(37)
    return Field(grid, rng.normal(size=shape) + 1j * rng.normal(size=shape))


def direct_sum(field, target, sign):
    # Full coordinate sum, deliberately independent of the separable engines.
    phase = np.exp(sign * 2j * np.pi * (
        target.x.ravel()[:, None] * field.grid.x.ravel()[None, :]
        + target.y.ravel()[:, None] * field.grid.y.ravel()[None, :]
    ))
    return (phase @ field.values.ravel()).reshape(target.shape)


@pytest.mark.parametrize("shape", [(7, 9), (8, 10), (7, 10), (8, 9)])
@pytest.mark.parametrize("shift", [(0.0, 0.0), (0.17e-3, -0.23e-3)])
def test_native_transform_matches_physical_sum_and_roundtrip(shape, shift):
    field = random_field(shape, shift)
    spectrum = FT2(field)
    expected = direct_sum(field, frequency_grid(field.grid), -1)
    np.testing.assert_allclose(spectrum.values, expected, atol=2e-12)
    # Metadata must survive spectral filtering and device-preserving copies.
    copied = spectrum.like(spectrum.values.copy()).to("cpu")
    for engine in ("auto", "matrix"):
        back = IFT2(copied, engine=engine)
        assert back.grid is field.grid
        np.testing.assert_allclose(back.values, field.values, atol=2e-12)
    matrix = FT2(field, engine="matrix")
    np.testing.assert_allclose(matrix.values, spectrum.values, atol=2e-12)


@pytest.mark.parametrize("shape", [(7, 9), (8, 10)])
def test_inverse_without_metadata_includes_frequency_origin(shape):
    field = random_field(shape)
    frequencies = frequency_grid(field.grid)
    frequencies = Grid(frequencies.x + 170.0, frequencies.y - 230.0)
    spectrum = Field(frequencies, field.values, domain="frequency")
    target = frequency_grid(frequencies)
    expected = direct_sum(spectrum, target, +1) / spectrum.values.size
    back = IFT2(spectrum)
    np.testing.assert_allclose(back.values, expected, atol=2e-12)
    np.testing.assert_allclose(
        IFT2(spectrum, engine="matrix").values, expected, atol=2e-12,
    )


@pytest.mark.parametrize("shape", [(7, 9), (8, 10)])
def test_czt_matches_coordinate_reference(shape):
    pytest.importorskip("scipy.signal")
    field = random_field(shape, (0.17e-3, -0.23e-3))
    spectrum = FT2(field)
    czt = FT2(field, engine="czt")
    np.testing.assert_allclose(czt.values, spectrum.values, atol=2e-11)
    back = IFT2(czt, engine="czt")
    np.testing.assert_allclose(back.values, field.values, atol=2e-11)
    target = Grid(*np.meshgrid(
        np.linspace(-1700.0, 2200.0, 11), np.linspace(-1400.0, 1800.0, 6),
    ))
    expected = direct_sum(field, target, -1)
    got = FT2(field, kgrid=target, engine="czt")
    np.testing.assert_allclose(got.values, expected, atol=2e-11)


@pytest.mark.parametrize("shape", [(7, 9), (8, 10), (7, 10), (8, 9)])
@pytest.mark.parametrize("pad_factor", [1, 2, 3])
def test_asm_native_and_explicit_coordinates_agree(shape, pad_factor):
    field = random_field(shape, (0.17e-3, -0.23e-3))
    for z in (0.0, 0.003, -0.003):
        native = asm_propagator(
            field, z, 532e-9, pad_factor=pad_factor, bandlimit=False,
        )
        explicit = asm_propagator(
            field, z, 532e-9, output_grid=field.grid, engine="matrix",
            pad_factor=pad_factor, bandlimit=False,
        )
        cached = AngularSpectrum(
            field, 532e-9, output_grid=field.grid,
            pad_factor=pad_factor, bandlimit=False,
        ).propagate(z)
        np.testing.assert_allclose(explicit.values, native.values, atol=3e-12)
        np.testing.assert_allclose(cached.values, native.values, atol=3e-12)
        if z == 0.0:
            np.testing.assert_allclose(explicit.values, field.values, atol=3e-12)


@pytest.mark.parametrize("shape", [(7, 9), (8, 10)])
def test_padding_extends_original_lattice(shape):
    field = random_field(shape, (0.17e-3, -0.23e-3))
    ny, nx = shape
    py, px = ny // 2, nx // 2
    padded = np.pad(field.values, ((py, ny - py), (px, nx - px)))
    dx, dy = field.grid.spacing
    grid = Grid(*np.meshgrid(
        field.grid.x[0, 0] + (np.arange(2 * nx) - px) * dx,
        field.grid.y[0, 0] + (np.arange(2 * ny) - py) * dy,
    ))
    reference = asm_propagator(
        Field(grid, padded), 0.004, 532e-9, bandlimit=False,
    )
    expected = reference.values[py:py + ny, px:px + nx]
    for output_grid in (None, field.grid):
        out = asm_propagator(
            field, 0.004, 532e-9, pad_factor=2, bandlimit=False,
            output_grid=output_grid,
        )
        np.testing.assert_allclose(out.values, expected, atol=3e-12)


@pytest.mark.parametrize("shape", [(7, 9), (8, 10)])
def test_plane_wave_phase_and_integrated_intensity(shape):
    grid = rectangular_grid(shape, (0.17e-3, -0.23e-3))
    dx, dy = grid.spacing
    fx, fy = 1.0 / (shape[1] * dx), -1.0 / (shape[0] * dy)
    values = np.exp(2j * np.pi * (fx * grid.x + fy * grid.y))
    field = Field(grid, values)
    wavelength, n, z = 532e-9, 1.3, 0.003
    kz = 2 * np.pi * np.sqrt((n / wavelength) ** 2 - fx ** 2 - fy ** 2)
    expected = values * np.exp(1j * kz * z)
    for out in (
        asm_propagator(field, z, wavelength, n=n, bandlimit=False),
        AngularSpectrum(field, wavelength, n=n, output_grid=grid, bandlimit=False).propagate(z),
    ):
        np.testing.assert_allclose(out.values, expected, atol=2e-10)
        np.testing.assert_allclose(
            out.intensity.sum() * dx * dy, field.intensity.sum() * dx * dy,
            rtol=2e-12,
        )


@pytest.mark.parametrize("shape", [(7, 9), (8, 10)])
def test_fresnel_fft_agrees_with_direct_integral(shape):
    field = random_field(shape, (0.17e-3, -0.23e-3))
    out = fresnel_propagator(field, 0.3, 532e-9, n=1.3)
    reference = fresnel_zoom_propagator(
        field, 0.3, 532e-9, n=1.3, output_grid=out.grid,
    )
    np.testing.assert_allclose(out.values, reference.values, atol=2e-11)


@pytest.mark.parametrize("samples", [7, 8])
def test_fraunhofer_translation_phase(samples):
    grid = make_grid(samples, 2e-3)
    rng = np.random.default_rng(4)
    values = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
    shift_x, shift_y = 0.17e-3, -0.23e-3
    shifted = Grid(grid.x + shift_x, grid.y + shift_y)
    reference = fraunhofer_propagator(Field(grid, values), 0.3, 532e-9)
    out = fraunhofer_propagator(Field(shifted, values), 0.3, 532e-9)
    fgrid = frequency_grid(grid)
    phase = np.exp(-2j * np.pi * (shift_x * fgrid.x + shift_y * fgrid.y))
    np.testing.assert_allclose(out.values, reference.values * phase, atol=2e-11)


def test_evanescent_plane_wave_has_analytic_decay():
    base = make_grid(9, 1e-6)
    grid = Grid(base.x + 17e-9, base.y - 23e-9)
    wavelength, z, fx = 532e-9, 0.1e-6, 3e6
    field = Field(grid, np.exp(2j * np.pi * fx * grid.x))
    alpha = 2 * np.pi * np.sqrt(fx ** 2 - (1.0 / wavelength) ** 2)
    expected = field.values * np.exp(-alpha * z)
    for out in (
        asm_propagator(field, z, wavelength, include_evanescent=True, bandlimit=False),
        AngularSpectrum(field, wavelength, output_grid=grid,
                        include_evanescent=True, bandlimit=False).propagate(z),
    ):
        np.testing.assert_allclose(out.values, expected, atol=2e-12)
    filtered = asm_propagator(field, z, wavelength, bandlimit=False)
    np.testing.assert_allclose(filtered.values, 0.0, atol=2e-12)


def test_discrete_sum_and_continuous_integral_normalizations():
    field = random_field((7, 10), (0.17e-3, -0.23e-3))
    spectrum = FT2(field)
    dx, dy = field.grid.spacing
    dfx, dfy = spectrum.grid.spacing
    np.testing.assert_allclose(
        np.sum(np.abs(spectrum.values * dx * dy) ** 2) * dfx * dfy,
        np.sum(field.intensity) * dx * dy, rtol=2e-12,
    )
