import numpy as np
import pytest

from diffraction import (
    AngularSpectrum,
    CUPY_AVAILABLE,
    array_module,
    asm_propagator,
    asnumpy,
    circular_aperture,
    gaussian_beam,
    make_grid,
    to_device,
)


def _field(n=128, length=4e-3):
    grid = make_grid(n, length)
    x, y = grid
    U0 = circular_aperture(x, y, 0.8e-3).astype(complex)
    return grid, U0


class TestBackend:
    def test_array_module_is_numpy_for_numpy(self):
        assert array_module(np.zeros(3)) is np

    def test_asnumpy_passthrough_on_cpu(self):
        a = np.arange(4.0)
        np.testing.assert_array_equal(asnumpy(a), a)

    def test_to_device_cpu_roundtrip(self):
        a = np.arange(4.0)
        np.testing.assert_array_equal(to_device(a, "cpu"), a)

    def test_to_device_gpu_without_cupy_raises(self):
        if CUPY_AVAILABLE:
            pytest.skip("CuPy is installed; GPU path is available.")
        with pytest.raises(RuntimeError):
            to_device(np.arange(4.0), "gpu")


class TestAngularSpectrumMatchesFunctional:
    @pytest.mark.parametrize("z", [0.0, 0.02, -0.02])
    def test_matches_default(self, z):
        grid, U0 = _field()
        expected = asm_propagator(U0, grid, z=z, wavelength=532e-9)
        got = AngularSpectrum(U0, grid, wavelength=532e-9).propagate(z)
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_matches_with_padding(self):
        grid, U0 = _field()
        expected = asm_propagator(U0, grid, z=0.03, wavelength=532e-9, pad_factor=2)
        got = AngularSpectrum(U0, grid, wavelength=532e-9, pad_factor=2).propagate(0.03)
        assert got.shape == U0.shape
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_matches_evanescent(self):
        grid, U0 = _field()
        expected = asm_propagator(
            U0, grid, z=1e-5, wavelength=532e-9, include_evanescent=True
        )
        got = AngularSpectrum(
            U0, grid, wavelength=532e-9, include_evanescent=True
        ).propagate(1e-5)
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_matches_without_bandlimit(self):
        grid, U0 = _field()
        expected = asm_propagator(U0, grid, z=0.05, wavelength=532e-9, bandlimit=False)
        got = AngularSpectrum(U0, grid, wavelength=532e-9, bandlimit=False).propagate(
            0.05
        )
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_medium_index(self):
        grid, U0 = _field()
        expected = asm_propagator(U0, grid, z=0.02, wavelength=532e-9, n=1.5)
        got = AngularSpectrum(U0, grid, wavelength=532e-9, n=1.5).propagate(0.02)
        np.testing.assert_allclose(got, expected, atol=1e-12)


class TestAngularSpectrumBatch:
    def test_stack_matches_individual(self):
        grid, U0 = _field()
        prop = AngularSpectrum(U0, grid, wavelength=532e-9)
        zs = [0.0, 0.01, 0.02, 0.03]
        stack = prop.propagate_stack(zs)
        assert len(stack) == len(zs)
        for U, z in zip(stack, zs):
            np.testing.assert_allclose(U, prop.propagate(z), atol=1e-12)

    def test_intensity_stack_normalized_and_real(self):
        grid, U0 = _field()
        prop = AngularSpectrum(U0, grid, wavelength=532e-9)
        frames = prop.intensity_stack([0.01, 0.02], normalize=True)
        for f in frames:
            assert np.isrealobj(f)
            assert f.min() >= 0.0
            np.testing.assert_allclose(f.max(), 1.0, atol=1e-12)

    def test_energy_conserved_per_plane(self):
        grid = make_grid(256, 4e-3)
        U0 = gaussian_beam(grid, waist=0.4e-3)
        prop = AngularSpectrum(U0, grid, wavelength=532e-9)
        e0 = np.sum(np.abs(U0) ** 2)
        for z in (0.02, 0.05):
            Uz = prop.propagate(z)
            np.testing.assert_allclose(np.sum(np.abs(Uz) ** 2), e0, rtol=1e-10)


class TestValidation:
    def test_bad_wavelength(self):
        grid, U0 = _field()
        with pytest.raises(ValueError):
            AngularSpectrum(U0, grid, wavelength=-1.0)

    def test_bad_pad_factor(self):
        grid, U0 = _field()
        with pytest.raises(ValueError):
            AngularSpectrum(U0, grid, wavelength=532e-9, pad_factor=0)

    def test_unknown_device(self):
        grid, U0 = _field()
        with pytest.raises(ValueError):
            AngularSpectrum(U0, grid, wavelength=532e-9, device="quantum")


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy/GPU not available")
class TestGPU:
    def test_gpu_matches_cpu(self):
        grid, U0 = _field()
        cpu = AngularSpectrum(U0, grid, wavelength=532e-9).propagate(0.03)
        gpu = AngularSpectrum(
            U0, grid, wavelength=532e-9, device="gpu"
        ).propagate(0.03)
        np.testing.assert_allclose(asnumpy(gpu), cpu, atol=1e-10)
