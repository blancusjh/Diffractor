import numpy as np
import pytest

from diffraction import (
    asm_propagator,
    circular_aperture,
    fraunhofer_propagator,
    fresnel_output_grid,
    fresnel_propagator,
    fresnel_zoom_propagator,
    gaussian_beam,
    make_grid,
)


def beam_width(I: np.ndarray, coord: np.ndarray) -> float:
    """1/e² intensity radius from the second moment of an axisymmetric beam.

    For a Gaussian intensity exp(-2 r² / w²) the 1D marginal gives
    <x²> = w² / 4, so w = 2 sqrt(<x²>).
    """
    total = I.sum()
    x2 = (I * coord**2).sum() / total
    return float(2.0 * np.sqrt(x2))


def analytic_gaussian_width(w0: float, z: float, wavelength: float) -> float:
    zR = np.pi * w0**2 / wavelength
    return w0 * np.sqrt(1.0 + (z / zR) ** 2)


class TestASM:
    def test_zero_distance_is_identity(self):
        grid = make_grid(128, 4e-3)
        x, y = grid
        U0 = circular_aperture(x, y, 1e-3).astype(complex)
        np.testing.assert_allclose(
            asm_propagator(U0, grid, z=0.0, wavelength=532e-9), U0, atol=1e-12
        )

    def test_energy_conservation(self):
        grid = make_grid(256, 4e-3)
        U0 = gaussian_beam(grid, waist=0.4e-3)
        Uz = asm_propagator(U0, grid, z=0.05, wavelength=532e-9)
        np.testing.assert_allclose(
            np.sum(np.abs(Uz) ** 2), np.sum(np.abs(U0) ** 2), rtol=1e-10
        )

    def test_back_propagation_roundtrip(self):
        grid = make_grid(256, 4e-3)
        U0 = gaussian_beam(grid, waist=0.4e-3)
        Uz = asm_propagator(U0, grid, z=0.05, wavelength=532e-9)
        Uback = asm_propagator(Uz, grid, z=-0.05, wavelength=532e-9)
        np.testing.assert_allclose(Uback, U0, atol=1e-8)

    def test_gaussian_beam_width_matches_analytic(self):
        wavelength = 532e-9
        w0 = 0.3e-3
        z = 0.2
        grid = make_grid(512, 8e-3)
        x, _ = grid

        Uz = asm_propagator(gaussian_beam(grid, waist=w0), grid, z=z, wavelength=wavelength)
        measured = beam_width(np.abs(Uz) ** 2, x)
        expected = analytic_gaussian_width(w0, z, wavelength)
        np.testing.assert_allclose(measured, expected, rtol=1e-3)

    def test_padded_propagation_matches_unpadded_for_contained_field(self):
        # A beam that never reaches the window edge must give the same
        # result with and without zero-padding.
        grid = make_grid(256, 8e-3)
        U0 = gaussian_beam(grid, waist=0.3e-3)
        U_plain = asm_propagator(U0, grid, z=0.05, wavelength=532e-9)
        U_padded = asm_propagator(U0, grid, z=0.05, wavelength=532e-9, pad_factor=2)

        assert U_padded.shape == U0.shape
        np.testing.assert_allclose(U_padded, U_plain, atol=1e-6)

    def test_padding_equals_natively_larger_grid(self):
        # Zero-padding must be exactly the accepted wraparound fix: embedding
        # the field in a larger window with the same sampling.
        n, L, z, wavelength = 128, 3e-3, 0.5, 532e-9
        small = make_grid(n, L)
        x, y = small
        U0 = circular_aperture(x, y, 0.3e-3).astype(complex)

        U_padded = asm_propagator(
            U0, small, z=z, wavelength=wavelength, bandlimit=False, pad_factor=4
        )

        large = make_grid(4 * n, 4 * L)
        X, Y = large
        U0_large = circular_aperture(X, Y, 0.3e-3).astype(complex)
        U_reference = asm_propagator(U0_large, large, z=z, wavelength=wavelength, bandlimit=False)
        m = (4 * n - n) // 2
        np.testing.assert_allclose(U_padded, U_reference[m : m + n, m : m + n], atol=1e-12)

    def test_bandlimit_zeroes_undersampled_frequencies(self):
        # A plane wave whose frequency lies beyond the Matsushima limit is
        # aliased by the sampled transfer function: the band limit must
        # remove it, while an unlimited ASM keeps it at full amplitude.
        n, L, z, wavelength = 256, 6e-3, 2.0, 532e-9
        grid = make_grid(n, L)
        x, _ = grid

        df = 1.0 / L
        f_limit = 1.0 / (wavelength * np.sqrt((2 * df * z) ** 2 + 1))  # ≈ 2.8e3 cyc/m
        f0 = 48 * df  # 8e3 cyc/m: beyond the limit, well below Nyquist and 1/λ
        assert f_limit < f0 < 1.0 / (2 * (L / n))
        U0 = np.exp(2j * np.pi * f0 * x)

        U_clean = asm_propagator(U0, grid, z=z, wavelength=wavelength, bandlimit=True)
        U_alias = asm_propagator(U0, grid, z=z, wavelength=wavelength, bandlimit=False)

        np.testing.assert_allclose(np.abs(U_clean), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.abs(U_alias), 1.0, atol=1e-10)

    def test_rejects_bad_pad_factor(self):
        grid = make_grid(64, 1e-3)
        U0 = np.ones((64, 64), dtype=complex)
        with pytest.raises(ValueError):
            asm_propagator(U0, grid, z=0.1, wavelength=532e-9, pad_factor=0)

    def test_evanescent_components_decay(self):
        grid = make_grid(128, 20e-6)  # fine grid so Nyquist exceeds 1/λ
        rng = np.random.default_rng(2)
        U0 = rng.standard_normal((128, 128)) + 0.0j
        wavelength = 532e-9

        U_filtered = asm_propagator(U0, grid, z=1e-6, wavelength=wavelength, bandlimit=False)
        U_evanescent = asm_propagator(
            U0, grid, z=1e-6, wavelength=wavelength, include_evanescent=True, bandlimit=False
        )
        # With decay included the field keeps (slightly) more energy than
        # with a hard cutoff, and both lose energy to the evanescent band.
        E0 = np.sum(np.abs(U0) ** 2)
        E_f = np.sum(np.abs(U_filtered) ** 2)
        E_e = np.sum(np.abs(U_evanescent) ** 2)
        assert E_f < E0
        assert E_f < E_e < E0


class TestFresnel:
    def test_gaussian_beam_width_matches_analytic(self):
        wavelength = 532e-9
        w0 = 0.3e-3
        z = 1.0  # far enough for single-FFT Fresnel output sampling
        grid = make_grid(1024, 12e-3)

        Uz = fresnel_propagator(gaussian_beam(grid, waist=w0), grid, z=z, wavelength=wavelength)
        x_out, _ = fresnel_output_grid(grid, z=z, wavelength=wavelength)

        measured = beam_width(np.abs(Uz) ** 2, x_out)
        expected = analytic_gaussian_width(w0, z, wavelength)
        np.testing.assert_allclose(measured, expected, rtol=1e-3)

    def test_output_grid_scaling(self):
        grid = make_grid(256, 4e-3)
        wavelength, z, n = 532e-9, 0.5, 1.5
        x_out, y_out = fresnel_output_grid(grid, z=z, wavelength=wavelength, n=n)

        dx = 4e-3 / 256
        expected_spacing = (wavelength / n) * z / (256 * dx)
        np.testing.assert_allclose(x_out[0, 1] - x_out[0, 0], expected_spacing)
        np.testing.assert_allclose(y_out[1, 0] - y_out[0, 0], expected_spacing)

    def test_rejects_zero_distance(self):
        grid = make_grid(64, 1e-3)
        U0 = np.ones((64, 64), dtype=complex)
        with pytest.raises(ValueError):
            fresnel_propagator(U0, grid, z=0.0, wavelength=532e-9)

    def test_rejects_bad_medium(self):
        grid = make_grid(64, 1e-3)
        with pytest.raises(ValueError):
            fresnel_output_grid(grid, z=1.0, wavelength=-1.0)
        with pytest.raises(ValueError):
            fresnel_output_grid(grid, z=1.0, wavelength=532e-9, n=0.0)


class TestFresnelZoom:
    def test_gaussian_matches_analytic_amplitude_and_width(self):
        wavelength = 532e-9
        w0 = 0.3e-3
        z = 1.0
        grid = make_grid(1024, 12e-3)

        U_out, (x_out, _) = fresnel_zoom_propagator(
            gaussian_beam(grid, waist=w0),
            grid,
            z=z,
            wavelength=wavelength,
            output_half_width=2e-3,
            output_samples=256,
        )

        assert U_out.shape == (256, 256)
        w_z = analytic_gaussian_width(w0, z, wavelength)
        measured = beam_width(np.abs(U_out) ** 2, x_out)
        np.testing.assert_allclose(measured, w_z, rtol=1e-3)
        # On-axis amplitude of a propagated Gaussian is w0 / w(z).
        c = 256 // 2
        np.testing.assert_allclose(np.abs(U_out[c, c]), w0 / w_z, rtol=1e-3)

    def test_rejects_bad_parameters(self):
        grid = make_grid(64, 1e-3)
        U0 = np.ones((64, 64), dtype=complex)
        with pytest.raises(ValueError):
            fresnel_zoom_propagator(
                U0, grid, z=0.0, wavelength=532e-9, output_half_width=1e-3
            )
        with pytest.raises(ValueError):
            fresnel_zoom_propagator(
                U0, grid, z=1.0, wavelength=532e-9, output_half_width=-1e-3
            )
        with pytest.raises(ValueError):
            fresnel_zoom_propagator(
                U0, grid, z=1.0, wavelength=532e-9, output_half_width=1e-3, output_samples=1
            )


class TestFraunhofer:
    def test_airy_first_zero(self):
        # First dark ring of a circular aperture: r = 1.22 λ z / D.
        wavelength = 532e-9
        D = 0.6e-3
        z = 10.0
        grid = make_grid(2048, 8e-3)
        x, y = grid

        U0 = circular_aperture(x, y, D / 2).astype(complex)
        Uz = fraunhofer_propagator(U0, grid, z=z, wavelength=wavelength)
        x_out, _ = fresnel_output_grid(grid, z=z, wavelength=wavelength)

        # Radial profile along the +x axis from the center.
        center = 2048 // 2
        profile = np.abs(Uz[center, center:]) ** 2
        r = x_out[center, center:]

        first_min = np.argmax((profile[1:-1] < profile[:-2]) & (profile[1:-1] < profile[2:])) + 1
        expected = 1.22 * wavelength * z / D
        np.testing.assert_allclose(r[first_min], expected, rtol=2e-2)
