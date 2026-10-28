"""Broadband intensities must refer to the same physical observation points."""

import numpy as np
import pytest

from diffractor import Field, Grid, make_grid, propagate_polychromatic_longitudinal
import diffractor.physics.polychromatic as polychromatic


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("difference", ["width", "origin", "count", "perpendicular"])
def test_native_longitudinal_rejects_different_coordinates(axis, difference):
    first = make_grid(16, 1e-3)
    other = make_grid(18 if difference == "count" else 16,
                      2e-3 if difference == "width" else 1e-3)
    if difference in ("origin", "perpendicular"):
        shift = 0.1e-3
        if difference == "origin" or axis == "y":
            other = Grid(other.x + shift, other.y)
        else:
            other = Grid(other.x, other.y + shift)

    def build(wavelength):
        grid = first if wavelength < 550e-9 else other
        return Field(grid, np.ones(grid.shape, dtype=complex))

    with pytest.raises(ValueError, match="identical source coordinates.*output_half_width"):
        propagate_polychromatic_longitudinal(
            build, [500e-9, 600e-9], [0.0], axis=axis,
        )


@pytest.mark.parametrize("axis", ["x", "y"])
def test_common_observation_line_registers_different_source_grids(monkeypatch, axis):
    captured = []

    def render(wavelengths, stack, **kwargs):
        captured.append(stack)
        return np.zeros((*stack.shape[1:], 3))

    monkeypatch.setattr(polychromatic, "spectrum_to_srgb", render)

    def build(wavelength):
        count, width = (16, 1e-3) if wavelength < 550e-9 else (19, 2e-3)
        base = make_grid(count, width)
        grid = Grid(base.x + 0.17e-3, base.y - 0.23e-3)
        values = 1.0 + 0.2 * np.exp(2j * np.pi * 1000.0 * grid.x)
        values += 0.3 * np.exp(-2j * np.pi * 1000.0 * grid.y)
        return Field(grid, values)

    offset, half_width = 0.1e-3, 0.25e-3
    out = propagate_polychromatic_longitudinal(
        build, [500e-9, 600e-9], [0.0], axis=axis, offset=offset,
        output_half_width=half_width, output_samples=13, bandlimit=False,
    )
    x, y = (out.t, offset) if axis == "x" else (offset, out.t)
    expected = np.abs(
        1.0 + 0.2 * np.exp(2j * np.pi * 1000.0 * x)
        + 0.3 * np.exp(-2j * np.pi * 1000.0 * y)
    ) ** 2
    assert len(captured) == 1
    assert captured[0].shape == (2, 1, 13)
    for frame in captured[0]:
        np.testing.assert_allclose(frame[0], expected, atol=2e-12)


def test_native_longitudinal_accepts_equal_distinct_grids(monkeypatch):
    def build(wavelength):
        grid = make_grid(15, 1e-3)
        return Field(grid, np.ones(grid.shape, dtype=complex))

    out = propagate_polychromatic_longitudinal(
        build, [500e-9, 600e-9], [0.0, 0.001], pad_factor=2,
    )
    assert out.rgb.shape == (2, 15, 3)
