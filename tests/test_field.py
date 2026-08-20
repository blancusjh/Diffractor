"""The Field contract: shape, spectral weighting, and type preservation."""

import numpy as np
import pytest

from diffractor.field import Field, MonochromaticField
from diffractor.optics import Medium
from diffractor.space import Grid
from diffractor.spectrum import Spectrum
from diffractor.units import nm, um

GRID = Grid.polar(np.linspace(0, 50 * um, 65), n_theta=4)


def test_field_shape_contract():
    s = Spectrum.flat(400 * nm, 700 * nm, 3)
    f = Field(GRID, np.ones((*GRID.shape, 3)), s)
    assert f.values.shape == (65, 4, 3) and f.values.dtype == complex
    with pytest.raises(ValueError, match="does not match"):
        Field(GRID, np.ones((65, 4, 2)), s)
    with pytest.raises(ValueError, match="domain"):
        Field(GRID, np.ones((65, 4, 3)), s, domain="fourier")


def test_monochromatic_expands_trailing_axis():
    f = MonochromaticField(GRID, np.ones(GRID.shape), 532 * nm)
    assert f.values.shape == (65, 4, 1)
    assert f.u.shape == GRID.shape
    assert f.wavelength == 532 * nm
    assert f.spectrum.n == 1


def test_like_preserves_monochromatic_type():
    f = MonochromaticField(GRID, np.ones(GRID.shape), 532 * nm,
                           medium=Medium(1.5))
    g = f.like(2.0 * f.values, domain="frequency")
    assert type(g) is MonochromaticField
    assert g.wavelength == f.wavelength and g.medium.n == 1.5
    assert g.domain == "frequency"
    h = g.like(g.values, medium=Medium(2.0))
    assert h.medium.n == 2.0 and h.domain == "frequency"


def test_intensity_weights_spectrum():
    s = Spectrum([500 * nm, 600 * nm], [1.0, 3.0])
    vals = np.ones((*GRID.shape, 2))
    vals[..., 1] = 2.0
    f = Field(GRID, vals, s)
    assert np.allclose(f.intensity, 1.0 * 1.0 + 3.0 * 4.0)
    assert f.spectral_intensity.shape == (*GRID.shape, 2)


def test_power_uses_grid_weights():
    R = 50 * um
    f = MonochromaticField(GRID, np.ones(GRID.shape), 532 * nm)
    assert np.allclose(f.power(), np.pi * R**2, rtol=1e-9)


def test_amplitude_and_phase_are_per_line():
    f = MonochromaticField(GRID, 1j * np.ones(GRID.shape), 532 * nm)
    assert np.allclose(f.amplitude, 1.0)
    assert np.allclose(f.phase, np.pi / 2)
