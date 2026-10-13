import numpy as np
import pytest

from diffractor import (
    MonochromaticField, coherent_sum, incoherent_intensity, make_grid, plane_wave,
)


def test_equal_sources_interfere_without_mutating_inputs():
    grid = make_grid(16, 1e-3)
    source = MonochromaticField(grid, 2.0, wavelength=532e-9, n=1.5)
    bright = coherent_sum([source, source])
    dark = coherent_sum([source, source], coefficients=[1, -1])
    np.testing.assert_allclose(bright.intensity, 16)
    np.testing.assert_allclose(dark.intensity, 0)
    np.testing.assert_allclose(source.values, 2)
    assert bright.n == source.n and bright.wavelength == source.wavelength
    bright.values[0, 0] = 0
    assert source.values[0, 0] == 2


def test_superposition_commutes_with_propagation():
    grid = make_grid(64, 3e-3)
    a = MonochromaticField(grid, lambda x, y: np.exp(-((x - 0.3e-3)**2 + y**2) / (0.2e-3)**2))
    b = MonochromaticField(grid, lambda x, y: np.exp(-((x + 0.3e-3)**2 + y**2) / (0.2e-3)**2))
    coefficients = [0.4, np.exp(0.7j)]
    together = coherent_sum([a, b], coefficients=coefficients).propagate(0.05)
    separately = coherent_sum([a.propagate(0.05), b.propagate(0.05)], coefficients=coefficients)
    np.testing.assert_allclose(together.values, separately.values, atol=1e-14)


def test_phase_cycle_equals_incoherent_sum():
    grid = make_grid(32, 2e-3)
    a = MonochromaticField(grid, 1)
    b = MonochromaticField(grid, plane_wave(grid, a.wavelength, theta_x=0.001))
    phase_cycle = [coherent_sum([a, b], coefficients=[1, np.exp(1j * p)])
                   for p in np.arange(4) * np.pi / 2]
    averaged = incoherent_intensity(phase_cycle, weights=np.full(4, 0.25))
    np.testing.assert_allclose(averaged, incoherent_intensity([a, b]), atol=1e-14)


def test_incoherent_weights_are_power_weights_and_wavelengths_may_differ():
    grid = make_grid(8, 1e-3)
    a = MonochromaticField(grid, 2, wavelength=450e-9)
    b = MonochromaticField(grid, 3j, wavelength=650e-9)
    np.testing.assert_allclose(incoherent_intensity([a, b], weights=[0.25, 0.75]), 7.75)
    with pytest.raises(ValueError, match="wavelength"):
        coherent_sum([a, b])


@pytest.mark.parametrize("combine", [coherent_sum, incoherent_intensity])
def test_combinations_refuse_different_physical_planes_or_media(combine):
    grid = make_grid(8, 1e-3)
    a = MonochromaticField(grid)
    with pytest.raises(ValueError, match="grid"):
        combine([a, MonochromaticField(make_grid(8, 1.001e-3))])
    with pytest.raises(ValueError, match="medium"):
        combine([a, MonochromaticField(grid, n=1.5)])
    with pytest.raises(ValueError, match="at least one"):
        combine([])


@pytest.mark.parametrize("weights", [[1], [-1, 1], [1, np.nan], [1, np.inf]])
def test_invalid_power_weights_are_rejected(weights):
    a = MonochromaticField(make_grid(8, 1e-3))
    with pytest.raises(ValueError, match="power weight"):
        incoherent_intensity([a, a], weights=weights)


def test_invalid_complex_coefficients_are_rejected():
    a = MonochromaticField(make_grid(8, 1e-3))
    with pytest.raises(ValueError, match="coefficient"):
        coherent_sum([a], coefficients=[np.nan])
